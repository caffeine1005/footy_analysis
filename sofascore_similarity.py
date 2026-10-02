"""Position-aware similar-player search on Sofascore league stats.

Mirrors `player_similarity.py` / `fbref_test.ipynb`, but scrapes via ScraperFC
Sofascore (same path as `sofascore_stuff/sofascore_scraper.ipynb`) instead of
FBref. Position is a comparison pool: scaler, PCA, role scores and NMF are fit
on the target's group only (DF / MF / FW).

With `--with-maps`, mirrors `player_profile.py`: WhoScored touch/action-map
clustering first, then Sofascore quantitative similarity inside that cluster
(with pass-angle features fused into the stage-2 matrix).

Usage:
    python sofascore_similarity.py --scrape          # fetch + cache CSV, then search
    python sofascore_similarity.py --scrape-ages     # add birth dates to a cached CSV
    python sofascore_similarity.py                   # use cached sofascore_player_stats.csv
    python sofascore_similarity.py --player "Sandro Tonali"
    python sofascore_similarity.py --player "Bruno Fernandes" --with-maps
    python sofascore_similarity.py --player "Pedri" --with-maps --min-age 21 --max-age 25
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import OrderedDict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
from scipy.stats import percentileofscore, rankdata
from sklearn.decomposition import NMF, PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from name_utils import normalize_name, normalized_col
from team_utils import teams_match

DATA_PATH = Path("sofascore_player_stats.csv")
MINUTES_COL = "nineties"
POS_COL = "position"
AGE_COL = "age"
# Date of birth is what gets scraped; `age` is derived from it at load time so a
# cached CSV doesn't go stale as players get older.
DOB_COL = "date_of_birth"
ID_COLS = [
    "league",
    "season",
    "team",
    "player",
    POS_COL,
    "player_id",
    "team_id",
    AGE_COL,
    DOB_COL,
]

# Same Big-5(+)-ish coverage as player_stats.csv / fbref_test.ipynb.
DEFAULT_LEAGUES = [
    "England Premier League",
    "England EFL Championship",
    "Spain La Liga",
    "France Ligue 1",
    "Germany Bundesliga",
    "Italy Serie A",
    "Netherlands Eredivisie",
    "Portugal Primeira Liga",
]

DEFAULT_SEASON = "25/26"

SOFASCORE_API = "https://api.sofascore.com/api/v1"
# Resumable player_id -> ISO birth date cache, so re-running the age scrape is cheap.
DOB_CACHE_PATH = Path("sofascore_player_dob.json")

# Sofascore API position filter label -> pool code used by the similarity stack.
POS_SCRAPE = [
    ("Goalkeepers", "GK"),
    ("Defenders", "DF"),
    ("Midfielders", "MF"),
    ("Forwards", "FW"),
]

ADJACENT = {
    "GK": [],
    "DF": ["MF"],
    "MF": ["DF", "FW"],
    "FW": ["MF"],
}

# Hand-built role templates. Values are substrings matched against column names.
ROLE_BLOCKS = {
    "MF": {
        "creator": [
            "expectedAssists",
            "keyPasses",
            "bigChancesCreated",
            "accurateFinalThirdPasses",
            "passToAssist",
        ],
        "carrier": [
            "successfulDribbles",
            "touches",
            "possessionWonAttThird",
            "totalContest",
        ],
        "ball_winner": [
            "tackles",
            "interceptions",
            "ballRecovery",
            "clearances",
            "dribbledPast",
        ],
        "box": [
            "expectedGoals",
            "goals",
            "totalShots",
            "shotsOnTarget",
            "bigChancesMissed",
        ],
    },
    "FW": {
        "finisher": [
            "expectedGoals",
            "goals",
            "shotsOnTarget",
            "totalShots",
            "bigChancesMissed",
        ],
        "link": [
            "expectedAssists",
            "keyPasses",
            "bigChancesCreated",
            "accurateFinalThirdPasses",
        ],
        "dribbler": ["successfulDribbles", "dispossessed", "touches", "totalContest"],
        "presser": ["tackles", "interceptions", "ballRecovery", "fouls"],
    },
    "DF": {
        "defender": [
            "tackles",
            "interceptions",
            "clearances",
            "outfielderBlocks",
            "ballRecovery",
            "aerialDuelsWon",
            "dribbledPast",
        ],
        "progressor": [
            "accurateLongBalls",
            "accurateFinalThirdPasses",
            "accurateOppositionHalfPasses",
            "accuratePasses",
        ],
        "width": ["accurateCrosses", "totalCross", "successfulDribbles"],
        "box_presence": ["expectedGoals", "goals", "totalShots"],
    },
}

# Drop before feature build (sums / duplicates / keeper-only / id-ish noise).
REDUNDANT_COLS = [
    "goalsAssistsSum",
    "totalDuelsWon",
    "totalDuelsWonPercentage",
    "totalRating",
    "countRating",
    "totwAppearances",
    "scoringFrequency",
    "cleanSheet",
    "goalsConceded",
    "goalsConcededInsideTheBox",
    "goalsConcededOutsideTheBox",
    "goalsPrevented",
    "saves",
    "savesCaught",
    "savesParried",
    "savedShotsFromInsideTheBox",
    "savedShotsFromOutsideTheBox",
    "penaltyFaced",
    "penaltySave",
    "punches",
    "highClaims",
    "crossesNotClaimed",
    "goalKicks",
    "runsOut",
    "successfulRunsOut",
    "appearances",
    "matchesStarted",
    "minutesPlayed",
    # Conversion rates off a handful of attempts a season: noise, not style.
    "penaltyConversion",
    "setPieceConversion",
]

# Season averages, not totals, that don't say so in their name. `load_players`
# minutes-weights these like the `...Percentage` columns; dividing them by 90s
# played (as a total would be) turned `rating` into ~1/minutes.
AVERAGE_COLS = {"rating"}


def is_rate_stat(col: str) -> bool:
    """True for percentage/average columns, False for the per-90 counts."""
    return col.endswith("Percentage") or "%" in col or col in AVERAGE_COLS

DEFAULT_QUERY = {
    "player": "Bruno Fernandes",
    "team": None,
    "league": None,
    "min_90s": 8.0,
    "min_age": None,
    "max_age": None,
    "include_unknown_age": True,
    "allow_adjacent_pos": False,
    "top_n": 9,
    "pca_var": 0.80,
    "nmf_k": 5,
    "rrf_k": 60,
}

# WhoScored / touch-map league labels <-> Sofascore scrape names.
LEAGUE_PAIRS = [
    ("ENG-Premier League", "England Premier League"),
    ("ENG-EFL Championship", "England EFL Championship"),
    ("ESP-La Liga", "Spain La Liga"),
    ("FRA-Ligue 1", "France Ligue 1"),
    ("GER-Bundesliga", "Germany Bundesliga"),
    ("ITA-Serie A", "Italy Serie A"),
    ("NED-Eredivisie", "Netherlands Eredivisie"),
    ("POR-Liga Portugal", "Portugal Primeira Liga"),
]
WHOSCORED_TO_SOFASCORE = {
    normalize_name(ws): sofa for ws, sofa in LEAGUE_PAIRS
}
SOFASCORE_TO_WHOSCORED = {
    normalize_name(sofa): ws for ws, sofa in LEAGUE_PAIRS
}
# Also accept the WhoScored label itself as a key into the reverse map.
SOFASCORE_TO_WHOSCORED.update(
    {normalize_name(ws): ws for ws, _sofa in LEAGUE_PAIRS}
)

STRENGTH_CUTOFF = 75.0
WEAKNESS_CUTOFF = 25.0
TOP_STATS = 8
DEFAULT_N_CLUSTERS = 14
DEFAULT_MAP_SEASON = 2025
DEFAULT_MIN_TOUCHES = 30

# Higher raw rate is worse. Percentile strengths/weaknesses and role averages
# flip these so "p90" always means good and "p10" always means bad; similarity
# matching keeps the raw direction (high-with-high is still similar).
LOWER_IS_BETTER = frozenset({
    "dribbledPast",
    "bigChancesMissed",
    "dispossessed",
    "duelLost",
    "possessionLost",
    "errorLeadToShot",
    "errorLeadToGoal",
    "fouls",
    "yellowCards",
    "redCards",
    "directRedCards",
})


def oriented_percentile(col: str, pct: float) -> float:
    """Map a raw percentile onto a higher-is-better scale for display/labels."""
    return 100.0 - pct if col in LOWER_IS_BETTER else pct

_pool_cache: dict[tuple, dict] = {}
_players: pl.DataFrame | None = None
_feature_cols: list[str] = []


def _map_deps():
    """Lazy-import touch/action-map stack (mplsoccer etc. not needed for stats-only)."""
    import matplotlib.pyplot as plt
    import pass_angle_radar as par
    import player_action_maps as pam
    import touchmap_similarity as tms

    return plt, par, pam, tms


def _target_map_specs(pam, par):
    return [
        ("touch", pam.full_pitch_touch_map, "touch_map"),
        ("pass", pam.pass_map, "pass_map"),
        ("pass_angle", par.pass_angle_radar, "pass_angle_radar"),
        ("takeon", pam.takeon_map, "takeon_map"),
        ("shot", pam.shot_map, "shot_map"),
        ("defensive", pam.defensive_action_map, "defensive_map"),
    ]


def primary_pos(pos) -> str | None:
    if pos is None:
        return None
    token = str(pos).split(",")[0].strip().upper()
    aliases = {"G": "GK", "D": "DF", "M": "MF", "F": "FW"}
    token = aliases.get(token, token)
    return token if token in ADJACENT else None


def _teams_overlap(a: str | None, b: str | None) -> bool:
    return teams_match(a, b)


def sofascore_league_name(league: str | None) -> str | None:
    """Map WhoScored touch-map league labels onto Sofascore scrape names."""
    if not league:
        return None
    mapped = WHOSCORED_TO_SOFASCORE.get(normalize_name(league))
    return mapped or league


def touchmap_league_name(league: str | None) -> str | None:
    """Map Sofascore (or WhoScored) league labels onto WhoScored event folder names."""
    if not league:
        return None
    mapped = SOFASCORE_TO_WHOSCORED.get(normalize_name(league))
    return mapped or league


def pretty_stat(col: str) -> str:
    """Sofascore's camelCase name, marked per 90 unless it is a rate/average."""
    return col if is_rate_stat(col) else f"{col} /90"


def _sofascore_get_json(url: str) -> dict:
    """Fetch a Sofascore API endpoint through ScraperFC's bot-protection-aware getters.

    Sofascore answers plain `requests` with HTTP 403, so the transport has to be
    one of ScraperFC's botasaurus getters. The lightweight request getter works
    for most player pages; the browser getter is the slower fallback.

    When bot protection turns the request getter away it does not raise — it
    returns a perfectly well-formed `{"error": {"code": 403}}` body. Falling back
    only on an exception therefore missed the one case the fallback exists for,
    and every caller read the error body as a player with no data. Treat an
    error payload as the failure it is.
    """
    from ScraperFC.utils import botasaurus_browser_get_json, botasaurus_request_get_json

    def rejected(payload) -> bool:
        return not isinstance(payload, dict) or "error" in payload

    try:
        payload = botasaurus_request_get_json(url)
    except Exception:  # noqa: BLE001 — fall back to a real browser session
        payload = None
    if payload is not None and not rejected(payload):
        return payload

    payload = botasaurus_browser_get_json(url)
    if rejected(payload):
        detail = payload.get("error") if isinstance(payload, dict) else payload
        raise RuntimeError(f"Sofascore refused {url}: {detail}")
    return payload


def _dob_from_player_dict(player: dict) -> str | None:
    """ISO date string from Sofascore's `dateOfBirthTimestamp`, or None."""
    ts = player.get("dateOfBirthTimestamp")
    if ts is None:
        return None
    # Built by offsetting the epoch rather than fromtimestamp() so pre-1970
    # birthdays don't break on Windows.
    moment = datetime.fromtimestamp(0, timezone.utc) + timedelta(seconds=int(ts))
    return moment.date().isoformat()


def collect_ages(
    path: Path | str = DATA_PATH,
    cache_path: Path | str = DOB_CACHE_PATH,
    sleep: float = 0.2,
) -> Path:
    """Add a scraped `date_of_birth` column to an existing Sofascore stats CSV.

    One request per unique `player_id`, cached to `cache_path` so an interrupted
    run resumes instead of re-fetching. `load_players` turns the stored birth
    date into the `age` column, so ages stay current without re-scraping.

    Runs inside a `sofascore_session` for the `X-Captcha` token; when `collect`
    is the caller it already owns one and this reuses it.
    """
    import sofascore_session

    return sofascore_session.run(lambda: _collect_ages(path, cache_path, sleep))


def _collect_ages(
    path: Path | str,
    cache_path: Path | str,
    sleep: float,
) -> Path:
    path, cache_path = Path(path), Path(cache_path)
    frame = pd.read_csv(path)
    if "player_id" not in frame.columns:
        raise ValueError(f"{path} has no player_id column — re-run the stats scrape first")

    cache: dict[str, str | None] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"loaded {len(cache)} cached birth dates from {cache_path}")

    ids = [str(int(v)) for v in frame["player_id"].dropna().unique()]
    todo = [pid for pid in ids if pid not in cache]
    print(f"{len(ids)} unique players, {len(todo)} to fetch")

    failed = 0
    for done, pid in enumerate(todo, start=1):
        try:
            payload = _sofascore_get_json(f"{SOFASCORE_API}/player/{pid}")
        except Exception as exc:  # noqa: BLE001 — one bad player shouldn't end the run
            # Deliberately not cached: a refused request says nothing about the
            # player, and caching it as "no birth date" would make this run's bad
            # luck permanent, since `todo` skips anything already in the cache.
            failed += 1
            if failed <= 5:
                print(f"  skip player {pid}: {exc}")
            elif failed == 6:
                print("  (further fetch failures suppressed)")
            continue
        cache[pid] = _dob_from_player_dict(payload.get("player", {}) or {})
        if done % 50 == 0 or done == len(todo):
            cache_path.write_text(json.dumps(cache, indent=0), encoding="utf-8")
            print(f"  {done}/{len(todo)} fetched")
        if sleep:
            time.sleep(sleep)

    cache_path.write_text(json.dumps(cache, indent=0), encoding="utf-8")
    frame[DOB_COL] = [
        cache.get(str(int(v))) if pd.notna(v) else None for v in frame["player_id"]
    ]
    frame.to_csv(path, index=False)
    known = int(frame[DOB_COL].notna().sum())
    print(f"wrote {DOB_COL} for {known}/{len(frame)} rows -> {path}")
    if failed:
        print(
            f"{failed} player(s) could not be fetched and were left uncached — "
            "re-run with --scrape-ages to retry just those."
        )
    return path


def collect(
    season: str = DEFAULT_SEASON,
    leagues: list[str] | None = None,
    path: Path | str = DATA_PATH,
    accumulation: str = "total",
    with_ages: bool = True,
) -> Path:
    """Scrape Sofascore league player stats and write a flat CSV.

    Wrapped in a `sofascore_session` so ScraperFC's getters carry the `X-Captcha`
    token the API now demands; without it every request comes back 403.
    """
    import sofascore_session

    return sofascore_session.run(
        lambda: _collect(season, leagues, path, accumulation, with_ages)
    )


def _collect(
    season: str,
    leagues: list[str] | None,
    path: Path | str,
    accumulation: str,
    with_ages: bool,
) -> Path:
    import ScraperFC as sfc

    leagues = leagues or DEFAULT_LEAGUES
    ss = sfc.Sofascore()
    frames: list[pd.DataFrame] = []

    for league in leagues:
        print(f"scraping {league} {season} …")
        for api_pos, pos_code in POS_SCRAPE:
            try:
                chunk = ss.scrape_player_league_stats(
                    season,
                    league,
                    accumulation=accumulation,
                    selected_positions=[api_pos],
                )
            except Exception as exc:  # noqa: BLE001 — keep other leagues going
                print(f"  skip {league} / {pos_code}: {exc}")
                continue
            if chunk is None or chunk.empty:
                print(f"  empty {league} / {pos_code}")
                continue
            chunk = chunk.copy()
            chunk["league"] = league
            chunk["season"] = season
            chunk["position"] = pos_code
            frames.append(chunk)
            print(f"  {pos_code}: {len(chunk)} players")

    if not frames:
        raise RuntimeError("no Sofascore player stats scraped — check season/league names")

    out = pd.concat(frames, ignore_index=True)
    # Prefer the tagged scrape position; drop accidental nested leftovers.
    rename = {"player id": "player_id", "team id": "team_id"}
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    path = Path(path)
    out.to_csv(path, index=False)
    print(f"wrote {len(out)} rows -> {path}")
    if with_ages:
        collect_ages(path)
    return path


def chunk_coverage(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Rows per (league, position) in an existing stats CSV.

    `collect` scrapes one chunk per league and position and keeps going when one
    fails, so a run can quietly finish having lost a whole chunk — which reads
    downstream as "that player isn't in Sofascore" for every midfielder in a
    league rather than as a scrape failure.
    """
    frame = pd.read_csv(path)
    have = pd.crosstab(frame["league"], frame["position"])
    for _api, pos in POS_SCRAPE:
        if pos not in have.columns:
            have[pos] = 0
    return have[[pos for _api, pos in POS_SCRAPE]]


def missing_chunks(path: Path | str = DATA_PATH) -> list[tuple[str, str]]:
    """Empty (league, position) chunks for leagues the CSV already covers.

    Restricted to leagues that are present: a league with no rows at all was
    never scraped, which is a choice about scope rather than a gap to be
    quietly filled in.
    """
    have = chunk_coverage(path)
    return [
        (league, pos)
        for league in have.index
        for _api, pos in POS_SCRAPE
        if int(have.loc[league, pos]) == 0
    ]


def top_up(
    path: Path | str = DATA_PATH,
    season: str = DEFAULT_SEASON,
    accumulation: str = "total",
    chunks: list[tuple[str, str]] | None = None,
    with_ages: bool = True,
) -> Path:
    """Re-scrape only the empty (league, position) chunks and merge them in.

    `collect` writes the whole CSV from whatever it just scraped, so narrowing it
    to the broken leagues would delete the working ones. This fetches the missing
    chunks alone and merges, keeping a `.bak` of the file it replaces.
    """
    import sofascore_session

    return sofascore_session.run(
        lambda: _top_up(path, season, accumulation, chunks, with_ages)
    )


def _top_up(
    path: Path | str,
    season: str,
    accumulation: str,
    chunks: list[tuple[str, str]] | None,
    with_ages: bool,
) -> Path:
    import ScraperFC as sfc

    path = Path(path)
    existing = pd.read_csv(path)
    chunks = missing_chunks(path) if chunks is None else chunks

    print(f"scanning {path.name} ...")
    print(chunk_coverage(path).to_string())
    if not chunks:
        print("every league/position chunk has rows — nothing to top up")
        return path
    print("\nmissing: " + ", ".join(f"{lg} / {pos}" for lg, pos in chunks))

    api_for = {pos: api for api, pos in POS_SCRAPE}
    ss = sfc.Sofascore()
    frames = []
    for league, pos in chunks:
        print(f"scraping {league} / {pos} ...", flush=True)
        try:
            chunk = ss.scrape_player_league_stats(
                season, league, accumulation=accumulation,
                selected_positions=[api_for[pos]],
            )
        except Exception as exc:  # noqa: BLE001 — keep the other chunks going
            print(f"  failed {league} / {pos}: {exc}")
            continue
        if chunk is None or chunk.empty:
            print(f"  empty {league} / {pos}")
            continue
        chunk = chunk.copy()
        chunk["league"] = league
        chunk["season"] = season
        chunk["position"] = pos
        chunk = chunk.rename(
            columns={k: v for k, v in {"player id": "player_id", "team id": "team_id"}.items()
                     if k in chunk.columns}
        )
        frames.append(chunk)
        print(f"  {pos}: {len(chunk)} players")

    if not frames:
        print("nothing scraped — CSV left unchanged")
        return path

    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_bytes(path.read_bytes())
    # Reindex to the union of columns first: a scraped chunk can lack a column
    # the file has (`date_of_birth` is added afterwards, not by the scrape), and
    # concatenating frames with all-NA columns is deprecated in pandas.
    columns = list(dict.fromkeys(sum([list(f.columns) for f in [existing, *frames]], [])))
    merged = pd.concat(
        [f.reindex(columns=columns) for f in [existing, *frames]], ignore_index=True
    )
    before = len(merged)
    if "player_id" in merged.columns:
        # A re-scraped chunk supersedes whatever was there for the same player
        # and league, so keep the newest row rather than the first.
        merged = merged.drop_duplicates(subset=["player_id", "league"], keep="last")
    merged.to_csv(path, index=False)
    print(
        f"\nmerged +{len(merged) - len(existing)} rows -> {path} "
        f"({len(existing)} -> {len(merged)}"
        + (f", {before - len(merged)} duplicates dropped" if before != len(merged) else "")
        + f")\nbackup written: {backup}"
    )
    if with_ages:
        collect_ages(path)
    return path


def load_players(path: Path | str = DATA_PATH, as_of: date | None = None) -> pl.DataFrame:
    """Per-player, per-90 table from one season's stats CSV.

    `as_of` is the date ages are measured at. Left as None it is today, which is
    right for the live season; a past season wants the age the player was then,
    not how old they are now.
    """
    raw = pl.read_csv(path, infer_schema_length=None)
    if "player id" in raw.columns:
        raw = raw.rename({"player id": "player_id"})
    if "team id" in raw.columns:
        raw = raw.rename({"team id": "team_id"})

    if "minutesPlayed" not in raw.columns:
        raise ValueError(f"{path} missing minutesPlayed — re-run with --scrape")

    raw = raw.with_columns(
        (pl.col("minutesPlayed").cast(pl.Float64) / 90.0).alias(MINUTES_COL)
    )
    if DOB_COL in raw.columns:
        # Birth date is the scraped truth; age is recomputed on every load so a
        # cached CSV doesn't hand back ages from whenever it was scraped.
        raw = raw.with_columns(
            (
                (
                    pl.lit(as_of or date.today())
                    - pl.col(DOB_COL).cast(pl.String).str.to_date(strict=False)
                ).dt.total_days()
                / 365.25
            )
            .cast(pl.Float64)
            .alias(AGE_COL)
        )
    elif AGE_COL not in raw.columns:
        raw = raw.with_columns(pl.lit(None).cast(pl.Float64).alias(AGE_COL))

    drop = [c for c in REDUNDANT_COLS if c in raw.columns]
    raw = raw.drop(drop)

    if POS_COL not in raw.columns:
        raise ValueError(f"{path} missing position — re-run with --scrape")

    numeric_cols = [c for c in raw.select(cs.numeric()).columns if c not in ID_COLS]
    # The CSV holds season totals. Counts become per 90 below; percentages and
    # averages are minutes-weighted across a player's rows instead.
    pct_cols = [c for c in numeric_cols if is_rate_stat(c)]
    count_cols = [c for c in numeric_cols if c not in pct_cols and c != MINUTES_COL]

    players = (
        raw.sort(MINUTES_COL, descending=True)
        .group_by("player", maintain_order=True)
        .agg(
            pl.first("league"),
            pl.first("season") if "season" in raw.columns else pl.lit(None).alias("season"),
            pl.first("team"),
            pl.first(POS_COL),
            pl.first("player_id") if "player_id" in raw.columns else pl.lit(None).alias("player_id"),
            pl.first("team_id") if "team_id" in raw.columns else pl.lit(None).alias("team_id"),
            pl.first(AGE_COL),
            pl.col(MINUTES_COL).sum(),
            *[pl.col(c).sum() for c in count_cols],
            *[
                (
                    (pl.col(c) * pl.col(MINUTES_COL)).sum()
                    / pl.col(MINUTES_COL).filter(pl.col(c).is_not_null()).sum()
                ).alias(c)
                for c in pct_cols
            ],
        )
    )
    players = players.with_columns(
        [
            pl.when(pl.col(MINUTES_COL) > 0)
            .then(pl.col(c) / pl.col(MINUTES_COL))
            .otherwise(0.0)
            .alias(c)
            for c in count_cols
        ]
    )
    players = players.with_columns(
        [
            pl.when(pl.col(c).is_nan() | pl.col(c).is_infinite())
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in numeric_cols
            if c != MINUTES_COL and c in players.columns
        ]
    )
    players = players.with_columns(
        [pl.col(c).fill_null(pl.col(c).median()) for c in pct_cols if c in players.columns]
    )
    players = players.with_columns(
        [pl.col(c).fill_null(0.0) for c in count_cols if c in players.columns]
    )
    return players.with_columns(
        pl.col(POS_COL)
        .map_elements(primary_pos, return_dtype=pl.String)
        .alias("primary_pos"),
    )


def reset_cache() -> None:
    global _players, _feature_cols, _pool_cache
    _players = None
    _feature_cols = []
    _pool_cache = {}
    _match_indexes.clear()


def _ensure_loaded(path: Path | str = DATA_PATH) -> tuple[pl.DataFrame, list[str]]:
    global _players, _feature_cols
    if _players is None:
        _players = load_players(path)
        skip = {MINUTES_COL, AGE_COL, "player_id", "team_id"}
        _feature_cols = [
            c
            for c in _players.select(cs.numeric()).columns
            if c not in skip
        ]
    return _players, _feature_cols


class _MatchIndex:
    """Accent-folded name, team and league lookups for one players frame.

    `resolve_target`, `_find_sofascore_index` and `_pool_target_idx` are each
    called once per cluster mate — several hundred times per analysis — and each
    call used to run two or three polars filters that rebuilt the whole
    accent-folding expression chain over every row. The folding depends only on
    the frame, so it happens once here and the per-call matching becomes a scan
    over Python lists.
    """

    def __init__(self, df: pl.DataFrame):
        height = df.height
        self.raw_player = df["player"].to_list()
        self.norm_player = [normalize_name(p) for p in self.raw_player]
        self.team = df["team"].to_list() if "team" in df.columns else [None] * height
        leagues = df["league"].to_list() if "league" in df.columns else [None] * height
        # normalize_name(None) is "", which no non-empty needle is a substring
        # of — the same miss a null league gave under `str.contains`.
        self.norm_league = [normalize_name(lg) for lg in leagues]

        self.by_raw_player: dict[str, list[int]] = {}
        for i, raw in enumerate(self.raw_player):
            self.by_raw_player.setdefault(raw, []).append(i)

    def by_name_substring(self, needle: str) -> list[int]:
        return [i for i, name in enumerate(self.norm_player) if needle in name]

    def narrow_league(self, rows: list[int], league: str) -> list[int]:
        needle = normalize_name(sofascore_league_name(league) or league)
        return [i for i in rows if needle in self.norm_league[i]]

    def narrow_team(self, rows: list[int], team: str) -> list[int]:
        return [i for i in rows if _teams_overlap(team, self.team[i])]


# Keyed by id(), with the frame held alongside so a recycled id cannot serve a
# stale index. Small: only the players table and the current pool go through it.
_match_indexes: OrderedDict[int, tuple[pl.DataFrame, _MatchIndex]] = OrderedDict()


def _match_index(df: pl.DataFrame) -> _MatchIndex:
    hit = _match_indexes.get(id(df))
    if hit is not None and hit[0] is df:
        _match_indexes.move_to_end(id(df))
        return hit[1]
    index = _MatchIndex(df)
    _match_indexes[id(df)] = (df, index)
    _match_indexes.move_to_end(id(df))
    while len(_match_indexes) > 4:
        _match_indexes.popitem(last=False)
    return index


def resolve_target(df: pl.DataFrame, q: dict) -> tuple[dict, pl.DataFrame]:
    index = _match_index(df)
    needle = normalize_name(q["player"])
    rows = index.by_name_substring(needle)
    if q.get("league"):
        rows = index.narrow_league(rows, q["league"])
    if q.get("team"):
        by_team = index.narrow_team(rows, q["team"])
        # A team is only ever a narrowing hint: keep the name matches rather
        # than discarding them when the label doesn't line up.
        if by_team:
            rows = by_team
    if not rows:
        raise ValueError(
            f"no player matching {q['player']!r} "
            f"(team={q.get('team')!r}, league={q.get('league')!r})"
        )
    if len(rows) > 1:
        exact = [i for i in rows if index.norm_player[i] == needle]
        if len(exact) == 1:
            rows = exact
    matches = df.with_row_index()[rows]
    row = matches.row(0, named=True)
    if row["primary_pos"] is None:
        raise ValueError(f"could not parse position {row[POS_COL]!r}")
    if row["primary_pos"] == "GK":
        raise ValueError(
            "outfield Sofascore feature set; keepers need a separate keeper feature set"
        )
    return row, matches


def _find_sofascore_index(players: pl.DataFrame, row: dict) -> int | None:
    index = _match_index(players)
    # Exact name equality, not the substring match `resolve_target` does: `row`
    # already came out of this frame.
    rows = index.by_raw_player.get(row["player"], [])
    if row.get("league"):
        narrowed = index.narrow_league(rows, row["league"])
        if narrowed:
            rows = narrowed
    if row.get("team"):
        narrowed = index.narrow_team(rows, row["team"])
        if narrowed:
            rows = narrowed
    if not rows:
        return None
    return rows[0]


def build_sofascore_cluster_pool(
    players: pl.DataFrame,
    cluster_mates,
    target_row: dict,
    min_90s: float | None = None,
) -> tuple[pl.DataFrame, dict]:
    """Map touch-cluster members to Sofascore rows; target is always included."""
    matched_indices: list[int] = []
    unmatched: list[str] = []

    for row in cluster_mates.itertuples(index=False):
        league = getattr(row, "league", None)
        try:
            sofa_row, _ = resolve_target(
                players,
                {"player": row.player, "team": row.team, "league": league},
            )
        except ValueError:
            unmatched.append(
                f"{row.player} ({row.team}" + (f", {league})" if league else ")")
            )
            continue
        if sofa_row[MINUTES_COL] <= 0 and sofa_row["player"] != target_row["player"]:
            continue
        if (
            min_90s is not None
            and sofa_row["player"] != target_row["player"]
            and sofa_row[MINUTES_COL] < min_90s
        ):
            continue
        idx = _find_sofascore_index(players, sofa_row)
        if idx is not None and idx not in matched_indices:
            matched_indices.append(idx)

    target_idx = _find_sofascore_index(players, target_row)
    if target_idx is not None and target_idx not in matched_indices:
        matched_indices.append(target_idx)

    meta = {
        "cluster_touch_size": len(cluster_mates),
        "matched": len(matched_indices),
        "unmatched": unmatched,
    }
    if len(matched_indices) < 2:
        raise ValueError(
            f"only {len(matched_indices)} touch-cluster member(s) matched in Sofascore "
            f"(cluster has {len(cluster_mates)} players with touch data); "
            "need at least 2 for similarity — scrape more leagues or lower min_90s"
        )

    return (
        players.with_row_index()
        .filter(pl.col("index").is_in(matched_indices))
        .drop("index"),
        meta,
    )


def age_eligibility(
    pool: pl.DataFrame,
    min_age: float | None = None,
    max_age: float | None = None,
    include_unknown: bool = True,
) -> tuple[np.ndarray | None, dict]:
    """Boolean mask of pool rows whose age falls in [min_age, max_age].

    Returns `(None, meta)` when no bound is set, so callers can skip masking
    entirely. Players with no known age are kept unless `include_unknown` is
    False — the Sofascore feed omits a birth date for some players, and silently
    dropping them would hide real matches.
    """
    if AGE_COL in pool.columns:
        ages = pool[AGE_COL].cast(pl.Float64).to_numpy().astype(float)
    else:
        ages = np.full(pool.height, np.nan)
    known = np.isfinite(ages)

    meta = {
        "min_age": min_age,
        "max_age": max_age,
        "include_unknown_age": include_unknown,
        "with_age": int(known.sum()),
        "without_age": int(pool.height - known.sum()),
    }
    if min_age is None and max_age is None:
        meta["eligible"] = pool.height
        return None, meta

    # Starts from `known`, so unknown ages stay excluded until opted back in.
    in_range = known.copy()
    if min_age is not None:
        in_range &= ages >= float(min_age)
    if max_age is not None:
        in_range &= ages <= float(max_age)
    if include_unknown:
        in_range |= ~known

    meta["eligible"] = int(in_range.sum())
    return in_range, meta


def format_age_range(age_meta: dict) -> str:
    """Human-readable summary of an applied age filter.

    For example "21-25, 9 eligible, 1 unknown age kept".
    """
    lo, hi = age_meta.get("min_age"), age_meta.get("max_age")
    if lo is None and hi is None:
        return "any age"
    if lo is not None and hi is not None:
        span = f"{lo:g}-{hi:g}"
    elif lo is not None:
        span = f"{lo:g}+"
    else:
        span = f"up to {hi:g}"

    bits = [span]
    eligible = age_meta.get("eligible")
    if eligible is not None:
        bits.append(f"{eligible} eligible")
    unknown = age_meta.get("without_age")
    if unknown:
        kept = "kept" if age_meta.get("include_unknown_age", True) else "dropped"
        bits.append(f"{unknown} unknown age {kept}")
    return ", ".join(bits)


def comparison_pool(df: pl.DataFrame, target: dict, q: dict) -> pl.DataFrame:
    groups = {target["primary_pos"]}
    if q["allow_adjacent_pos"]:
        groups.update(ADJACENT[target["primary_pos"]])
    pool = df.filter(
        pl.col("primary_pos").is_in(sorted(groups))
        & (pl.col(MINUTES_COL) >= q["min_90s"])
    )
    min_age, max_age = q.get("min_age"), q.get("max_age")
    if min_age is not None or max_age is not None:
        in_range = pl.col(AGE_COL).is_not_null()
        if min_age is not None:
            in_range = in_range & (pl.col(AGE_COL) >= float(min_age))
        if max_age is not None:
            in_range = in_range & (pl.col(AGE_COL) <= float(max_age))
        if q.get("include_unknown_age", True):
            in_range = in_range | pl.col(AGE_COL).is_null()
        # The target has to survive its own comparison pool.
        pool = pool.filter(in_range | (pl.col("player") == target["player"]))
    if pool.filter(pl.col("player") == target["player"]).height == 0:
        raise ValueError(
            f"{target['player']} has {target[MINUTES_COL]:.1f} 90s, "
            f"below min_90s={q['min_90s']}"
        )
    if pool.height < 15:
        raise ValueError(
            f"pool too small ({pool.height}); lower min_90s, scrape more leagues, "
            "or allow adjacent positions"
        )
    return pool


def pca_n_components(X: np.ndarray, var_cutoff: float) -> int:
    cap = max(2, min(X.shape[0] - 1, X.shape[1]))
    pca = PCA().fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n = int(np.searchsorted(cum, var_cutoff) + 1)
    return int(np.clip(n, 2, cap))


def role_matrix(
    X_scaled: np.ndarray, cols: list[str], pos_group: str
) -> tuple[np.ndarray, list[str], dict[str, list[str]]]:
    blocks = ROLE_BLOCKS[pos_group]
    names: list[str] = []
    used: dict[str, list[str]] = {}
    parts: list[np.ndarray] = []
    for name, needles in blocks.items():
        idx = [i for i, c in enumerate(cols) if any(n in c for n in needles)]
        if not idx:
            continue
        names.append(name)
        used[name] = [cols[i] for i in idx]
        # Negate lower-is-better columns so e.g. high dribbledPast pulls the
        # ball_winner / defender role score down rather than up.
        signs = np.array(
            [-1.0 if cols[i] in LOWER_IS_BETTER else 1.0 for i in idx],
            dtype=float,
        )
        parts.append((X_scaled[:, idx] * signs).mean(axis=1))
    if not parts:
        raise ValueError(f"no role-block columns matched for {pos_group}")
    return np.column_stack(parts), names, used


def spearman_scores(X: np.ndarray, idx: int) -> np.ndarray:
    ranks = rankdata(X, axis=1).astype(float)
    centered = ranks - ranks.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    z = centered / norms
    scores = z @ z[idx]
    scores[ranks.std(axis=1) == 0] = -np.inf
    return scores


def fit_pool(pool: pl.DataFrame, feature_cols: list[str], q: dict) -> dict:
    X = pool.select(feature_cols).to_numpy()
    X_scaled = RobustScaler().fit_transform(X)

    n_pca = pca_n_components(X_scaled, q["pca_var"])
    pca = PCA(n_pca).fit(X_scaled)
    pcaed = pca.transform(X_scaled)

    X_pos = X_scaled - X_scaled.min(axis=0) + 1e-6
    k = min(int(q["nmf_k"]), X_pos.shape[0] - 1, X_pos.shape[1])
    nmf = NMF(n_components=k, init="nndsvda", max_iter=2000, random_state=42)
    W = nmf.fit_transform(X_pos)
    W_norm = W / np.clip(W.sum(axis=1, keepdims=True), 1e-9, None)

    return {
        "X_scaled": X_scaled,
        "pcaed": pcaed,
        "n_pca": n_pca,
        "pca_var_explained": float(pca.explained_variance_ratio_.sum()),
        "W_norm": W_norm,
        "H": nmf.components_,
    }


def top_table(
    pool: pl.DataFrame, scores: np.ndarray, score_name: str, n: int
) -> pl.DataFrame:
    order = np.argsort(scores)[::-1][:n]
    return pool[order].select(
        "player", "team", "league", POS_COL, "primary_pos", MINUTES_COL
    ).with_columns(pl.Series(score_name, scores[order]))


def search(q: dict | None = None, path: Path | str = DATA_PATH) -> dict:
    q = {**DEFAULT_QUERY, **(q or {})}
    players, feature_cols = _ensure_loaded(path)
    target, matches = resolve_target(players, q)
    pool = comparison_pool(players, target, q)

    cache_key = (
        target["primary_pos"],
        bool(q["allow_adjacent_pos"]),
        float(q["min_90s"]),
        q.get("min_age"),
        q.get("max_age"),
        bool(q.get("include_unknown_age", True)),
        float(q["pca_var"]),
        int(q["nmf_k"]),
    )
    if cache_key not in _pool_cache:
        _pool_cache[cache_key] = fit_pool(pool, feature_cols, q)
    art = _pool_cache[cache_key]

    names = pool["player"].to_list()
    idx = names.index(target["player"])
    X_scaled = art["X_scaled"]
    pcaed = art["pcaed"]
    n = len(names)

    scores: dict[str, np.ndarray] = {}
    cos = cosine_similarity(X_scaled, X_scaled[idx : idx + 1]).ravel().astype(float)
    cos[idx] = -np.inf
    scores["Cosine"] = cos

    knn = NearestNeighbors(n_neighbors=n, metric="euclidean")
    knn.fit(pcaed)
    dist, ind = knn.kneighbors(pcaed[idx : idx + 1], n_neighbors=n)
    knn_scores = np.empty(n, dtype=float)
    knn_scores[ind[0]] = 1.0 / (1.0 + dist[0])
    knn_scores[idx] = -np.inf
    scores["KNN_PCA"] = knn_scores

    sp = spearman_scores(X_scaled, idx)
    sp[idx] = -np.inf
    scores["Spearman"] = sp

    roles, role_names, role_cols = role_matrix(
        X_scaled, feature_cols, target["primary_pos"]
    )
    role_cos = cosine_similarity(roles, roles[idx : idx + 1]).ravel().astype(float)
    role_cos[idx] = -np.inf
    scores["RoleBlocks"] = role_cos

    W_norm = art["W_norm"]
    nmf_cos = cosine_similarity(W_norm, W_norm[idx : idx + 1]).ravel().astype(float)
    nmf_cos[idx] = -np.inf
    scores["NMF"] = nmf_cos

    rrf = np.zeros(n)
    ranks: dict[str, np.ndarray] = {}
    for name, sc in scores.items():
        order = np.argsort(sc)[::-1]
        rnk = np.empty(n)
        rnk[order] = np.arange(1, n + 1)
        ranks[name] = rnk
        rrf += 1.0 / (q["rrf_k"] + rnk)
    rrf[idx] = -np.inf

    return {
        "query": q,
        "target": target,
        "matches": matches,
        "pool": pool,
        "idx": idx,
        "feature_cols": feature_cols,
        "scores": scores,
        "rrf": rrf,
        "ranks": ranks,
        "roles": roles,
        "role_names": role_names,
        "role_cols": role_cols,
        "W_norm": W_norm,
        "H": art["H"],
        "n_pca": art["n_pca"],
        "pca_var_explained": art["pca_var_explained"],
    }


def consensus_table(result: dict) -> pl.DataFrame:
    q = result["query"]
    order = np.argsort(result["rrf"])[::-1][: q["top_n"]]
    return result["pool"][order].select(
        "player", "team", "league", POS_COL, "primary_pos", MINUTES_COL
    ).with_columns(
        [
            pl.Series("rrf_score", result["rrf"][order]),
            *[
                pl.Series(f"{name}_rank", result["ranks"][name][order].astype(int))
                for name in result["scores"]
            ],
        ]
    )


def nmf_mix_table(result: dict) -> pl.DataFrame:
    q = result["query"]
    idx = result["idx"]
    order = np.argsort(result["rrf"])[::-1][: q["top_n"]]
    mix_idx = [idx, *order.tolist()]
    names = [f"type_{j + 1}" for j in range(result["W_norm"].shape[1])]
    return pl.DataFrame(
        {
            "player": [result["pool"]["player"][i] for i in mix_idx],
            **{names[j]: result["W_norm"][mix_idx, j] for j in range(len(names))},
        }
    )


def nmf_loadings(result: dict, top: int = 8) -> list[str]:
    lines = []
    H = result["H"]
    cols = result["feature_cols"]
    for j in range(H.shape[0]):
        hi = np.argsort(H[j])[::-1][:top]
        bits = [f"{cols[i]}={H[j, i]:.2f}" for i in hi]
        lines.append(f"type_{j + 1}: " + "; ".join(bits))
    return lines


def summarize(result: dict) -> None:
    t = result["target"]
    q = result["query"]
    extra = " + adjacent" if q["allow_adjacent_pos"] else ""
    print(
        f"target: {t['player']} ({t['team']}, {t['league']})  "
        f"pos={t[POS_COL]}  pool={t['primary_pos']}{extra}  "
        f"n={result['pool'].height}  "
        f"pca={result['n_pca']} ({100 * result['pca_var_explained']:.0f}% var)  "
        f"90s={t[MINUTES_COL]:.1f}"
    )
    if result["matches"].height > 1:
        print("multiple matches — used the first; set team/league to disambiguate")


def _resolve_touch_player(
    meta: pd.DataFrame,
    player_name: str,
    team: str | None = None,
    league: str | None = None,
) -> int:
    """Resolve a touch-map row; tolerate Sofascore vs WhoScored team/league labels."""
    from name_utils import normalize_series

    matched = meta[
        normalize_series(meta["player"]).str.contains(
            normalize_name(player_name), regex=False
        )
    ]
    if league is not None and "league" in matched.columns and not matched.empty:
        by_league = matched[
            normalize_series(matched["league"]).str.contains(
                normalize_name(league), regex=False
            )
        ]
        if not by_league.empty:
            matched = by_league
    if team is not None and not matched.empty:
        mask = matched["team"].map(lambda t: _teams_overlap(team, t))
        by_team = matched[mask]
        if not by_team.empty:
            matched = by_team
    if matched.empty:
        raise ValueError(
            f"no player matching {player_name!r} (team={team!r}, league={league!r})"
        )
    if len(matched) > 1:
        exact = matched[normalize_series(matched["player"]) == normalize_name(player_name)]
        if len(exact) == 1:
            matched = exact
    if len(matched) > 1:
        listing = ", ".join(
            f"{row.player} ({row.team})" for row in matched.itertuples(index=False)
        )
        raise ValueError(
            f"{player_name!r} matches multiple players: {listing}. "
            "Pass team= and/or league= to disambiguate."
        )
    return matched.index[0]


def _save_target_maps(
    load_events,
    touch_target: dict,
    out_dir: str | Path,
    *,
    events_key: str | None = None,
    dpi: int = 150,
    flip_flanks: bool = False,
) -> dict[str, str]:
    """Render the target's dashboards, reusing any already drawn from the same events.

    Rendering is several seconds, and the pngs depend only on the player and the
    events behind them — so a stamp file records which events they were drawn
    from. A matching stamp means the profile can be shown without rendering
    anything, and without loading the event table at all: `load_events` is a
    callable rather than a frame precisely so that a reuse costs nothing.

    Not every player produces every map (a center-back with no shots raises out
    of `shot_map`), so the stamp is what marks a render pass as complete —
    counting files would re-render those players forever.

    With `flip_flanks`, event y / end_y (and shot mouth/block y) are mirrored
    before drawing so the dashboards match the flipped shape query.
    """
    plt, par, pam, tms = _map_deps()
    out_dir = Path(out_dir)
    slug = pam._slug(touch_target["player"])
    specs = _target_map_specs(pam, par)
    paths = {name: out_dir / f"{slug}_{suffix}.png" for name, _fn, suffix in specs}
    stamp = out_dir / f"{slug}.stamp"
    want = json.dumps(
        {
            "events": events_key,
            "player": touch_target["player"],
            "team": touch_target["team"],
            "flip_flanks": bool(flip_flanks),
        },
        sort_keys=True,
    )

    if events_key is not None and stamp.exists():
        try:
            have = stamp.read_text(encoding="utf-8")
        except OSError:
            have = None
        if have == want:
            saved = {name: str(p) for name, p in paths.items() if p.exists()}
            if saved:
                return saved

    out_dir.mkdir(parents=True, exist_ok=True)
    season_df = load_events()
    if flip_flanks:
        season_df = tms.flip_event_flanks(season_df)
    saved = {}
    for name, fn, suffix in specs:
        try:
            ax = fn(season_df, touch_target["player"], team=touch_target["team"], exact=True)
        except ValueError:
            continue
        path = paths[name]
        ax.figure.savefig(path, dpi=dpi, facecolor=ax.figure.get_facecolor())
        plt.close(ax.figure)
        saved[name] = str(path)
    if events_key is not None:
        try:
            stamp.write_text(want, encoding="utf-8")
        except OSError:  # a stamp we cannot write just means we render again
            pass
    return saved


def _stage2_error(
    exc: ValueError, touch_target: dict, players: pl.DataFrame, stats_path: Path | str
) -> str:
    """Explain a stage-2 miss for a player the event data *did* find.

    "no player matching Bruno Fernandes" reads like a typo, but by this point the
    name came out of the event data — the touch-map stage resolved it. What it
    really means is that the Sofascore stats table has no row for them, and the
    usual reason is a scrape that lost a whole (league, position) chunk: a table
    with 200 midfielders in every league but one is missing that league's
    midfielders, not that one player. Saying which chunk is empty turns a dead
    end into something actionable.
    """
    message = str(exc)
    if not message.startswith("no player matching"):
        return message

    league = touch_target.get("league")
    sofa_league = sofascore_league_name(league) or league
    detail = (
        f"{touch_target['player']} ({touch_target['team']}) is in the event data but has no row "
        f"in the Sofascore stats table ({Path(stats_path).name})."
    )
    if sofa_league is None or POS_COL not in players.columns or "league" not in players.columns:
        return f"{detail} Re-scrape it with `python sofascore_similarity.py --scrape`."

    in_league = players.filter(
        normalized_col("league").str.contains(normalize_name(sofa_league), literal=True)
    )
    if in_league.height == 0:
        return (
            f"{detail} That table has no {sofa_league} players at all — re-scrape with "
            "`python sofascore_similarity.py --scrape`."
        )

    counts = {
        pos: in_league.filter(pl.col(POS_COL) == pos).height for pos in ("GK", "DF", "MF", "FW")
    }
    empty = [pos for pos, n in counts.items() if n == 0]
    if empty:
        have = ", ".join(f"{pos} {n}" for pos, n in counts.items())
        return (
            f"{detail} The scrape of {sofa_league} is incomplete: no {'/'.join(empty)} rows "
            f"({have}), so every {'/'.join(empty)} in that league is missing. Re-scrape with "
            "`python sofascore_similarity.py --scrape`."
        )
    return f"{detail} Re-scrape with `python sofascore_similarity.py --scrape`."


def _once(fn):
    """Call `fn` at most once, holding its result for the rest of this analysis."""
    cache = []

    def wrapper():
        if not cache:
            cache.append(fn())
        return cache[0]

    return wrapper


def _run_touchmap_cluster(
    player_name: str,
    team: str | None,
    league: str | None,
    events_dir: str | Path | None,
    season: int | str,
    min_touches: int,
    n_clusters: int,
    action_features: bool = True,
    save_maps_dir: str | Path | None = None,
) -> tuple[dict, pd.DataFrame, int]:
    """Resolve target and return their touch-map (+ action-map) cluster mates.

    Only the last two steps depend on `player_name`; the grids, the clustering
    and the pass-angle features are shared by every player in the same
    population, so they come from `analysis_cache` and the events themselves are
    loaded lazily — a warm cache never touches them unless maps are being saved.
    """
    import analysis_cache

    _plt, _par, _pam, tms = _map_deps()

    if events_dir is not None:
        events_key = analysis_cache.dir_events_key(events_dir)
        inferred = tms._league_from_dir(events_dir, season)

        def load_events():
            season_df = tms.load_season_events(events_dir)
            if inferred is not None and "league" not in season_df.columns:
                # Shallow: the cached loader already hands back its own frame
                # object, so adding a column here cannot reach the cache.
                season_df = season_df.copy(deep=False)
                season_df["league"] = inferred
            return season_df

        if events_key is not None and inferred is not None:
            # The league tag is added after loading, so it has to be part of the
            # key for the stages built on top of it.
            events_key = f"{events_key}-{inferred}"
    else:
        events_key = analysis_cache.league_events_key(
            list(tms.DEFAULT_LEAGUES), season, "league_games"
        )

        def load_events():
            return tms.load_all_league_events(season=season)

    # Several stages below may each need the events. Whichever asks first pays
    # for the load and the rest reuse it, so a cold run still reads them once.
    load_events = _once(load_events)

    grids_key, meta, X = analysis_cache.cached_grids(
        events_key,
        load_events,
        min_touches=min_touches,
        bins=tms.DEFAULT_BINS,
        smooth_sigma=1.0,
        action_features=action_features,
    )
    labels = analysis_cache.cached_cluster_labels(grids_key, X, n_clusters=n_clusters)

    target_idx = _resolve_touch_player(
        meta, player_name, team=team, league=touchmap_league_name(league)
    )
    target_row = meta.loc[target_idx]
    cluster_label = int(labels[target_idx])
    cluster_mates = meta[labels == cluster_label].reset_index(drop=True)

    touch_target = {
        "player": target_row["player"],
        "team": target_row["team"],
        "league": target_row["league"] if "league" in target_row.index else None,
        "touches": int(target_row["touches"]),
        "cluster": cluster_label,
    }

    # Computed for the whole population and cached, then narrowed to this
    # cluster — the features are per (player, team), so the cluster's own
    # entries are the same either way.
    all_stats_by, all_vec_by = analysis_cache.cached_pass_angle_features(
        grids_key, load_events, meta
    )
    pairs = set(zip(cluster_mates["player"].tolist(), cluster_mates["team"].tolist()))
    stats_by = {k: v for k, v in all_stats_by.items() if k in pairs}
    vec_by = {k: v for k, v in all_vec_by.items() if k in pairs}

    key = (touch_target["player"], touch_target["team"])
    touch_target["pass_angles"] = stats_by.get(key)
    touch_target["pass_angle_stats_by_player"] = stats_by
    touch_target["pass_angle_vec_by_player"] = vec_by

    if save_maps_dir is not None:
        touch_target["maps_saved"] = _save_target_maps(
            load_events, touch_target, save_maps_dir, events_key=events_key
        )

    return touch_target, cluster_mates, cluster_label


def _pool_target_idx(pool: pl.DataFrame, target_row: dict) -> int:
    index = _match_index(pool)
    rows = index.by_raw_player.get(target_row["player"])
    if not rows:
        raise ValueError(f"{target_row['player']!r} is not in the pool")
    for i in rows:
        if target_row.get("league"):
            needle = normalize_name(
                sofascore_league_name(target_row["league"]) or target_row["league"]
            )
            if needle not in index.norm_league[i]:
                continue
        if target_row.get("team") and not _teams_overlap(target_row["team"], index.team[i]):
            continue
        return i
    # Name alone, when no row also agreed on league and team.
    return rows[0]


def _align_pass_angles_to_pool(
    pool: pl.DataFrame,
    cluster_mates: pd.DataFrame,
    vec_by: dict[tuple[str, str], np.ndarray],
    stats_by: dict[tuple[str, str], dict],
) -> tuple[np.ndarray, dict[int, dict]]:
    _plt, par, _pam, _tms = _map_deps()
    n_feat = len(par.PASS_ANGLE_FEATURE_NAMES)
    X = np.full((pool.height, n_feat), np.nan)
    stats_by_idx: dict[int, dict] = {}
    if cluster_mates.empty or not (vec_by or stats_by):
        return X, stats_by_idx

    for row in cluster_mates.itertuples(index=False):
        key = (row.player, row.team)
        vec = vec_by.get(key)
        stats = stats_by.get(key)
        if vec is None and stats is None:
            continue
        league = getattr(row, "league", None)
        try:
            sofa_row, _ = resolve_target(
                pool, {"player": row.player, "team": row.team, "league": league}
            )
            idx = _pool_target_idx(pool, sofa_row)
        except ValueError:
            continue
        if vec is not None:
            X[idx] = vec
        if stats is not None:
            stats_by_idx[idx] = stats
    return X, stats_by_idx


def _augment_pass_angle_features(
    pass_angle_X: np.ndarray, idx: int
) -> np.ndarray | None:
    valid = np.all(np.isfinite(pass_angle_X), axis=1)
    if not valid[idx] or int(valid.sum()) < 2:
        return None
    filled = pass_angle_X.astype(float, copy=True)
    for j in range(filled.shape[1]):
        col = filled[:, j]
        finite = col[np.isfinite(col)]
        med = float(np.median(finite)) if finite.size else 0.0
        col[~np.isfinite(col)] = med
    return filled


def _pass_angle_cosine(pass_angle_X: np.ndarray, idx: int) -> np.ndarray | None:
    valid = np.all(np.isfinite(pass_angle_X), axis=1)
    if not valid[idx] or int(valid.sum()) < 2:
        return None
    scaled = np.zeros_like(pass_angle_X)
    scaled[valid] = RobustScaler().fit_transform(pass_angle_X[valid])
    ang = np.full(pass_angle_X.shape[0], -np.inf)
    hits = np.flatnonzero(valid)
    ang[hits] = cosine_similarity(scaled[hits], scaled[idx : idx + 1]).ravel()
    ang[idx] = -np.inf
    return ang


def search_in_pool(
    pool: pl.DataFrame,
    target_row: dict,
    feature_cols: list[str],
    top_n: int = 10,
    pca_var: float = 0.80,
    nmf_k: int = 5,
    rrf_k: int = 60,
    pass_angle_X: np.ndarray | None = None,
    eligible: np.ndarray | None = None,
    target_idx: int | None = None,
    extra_scores: dict[str, np.ndarray] | None = None,
) -> dict:
    """RRF-ranked Sofascore similarity within a fixed (touch-cluster) pool.

    `eligible` is an optional boolean mask over pool rows. It is applied only
    when picking which ranked players to return, never to the scaler, PCA, NMF
    or percentile baselines, so every similarity score is identical to an
    unmasked run and `top_n` counts eligible players only.

    `target_idx` names the target's pool row outright. A pool spanning several
    seasons can hold the same player once per season, where a name lookup would
    just return whichever season came first.
    `extra_scores` adds rankers from outside the stats table (higher = more
    similar, one value per pool row), fused into the RRF like the built-in ones.
    """
    _plt, par, _pam, _tms = _map_deps()
    idx = _pool_target_idx(pool, target_row) if target_idx is None else int(target_idx)
    pos_group = target_row["primary_pos"]

    sim_pool = pool
    sim_cols = list(feature_cols)
    pass_angle_used = False
    if pass_angle_X is not None:
        filled = _augment_pass_angle_features(pass_angle_X, idx)
        if filled is not None:
            pa_names = [f"pass_angle::{name}" for name in par.PASS_ANGLE_FEATURE_NAMES]
            sim_pool = pool.with_columns(
                [pl.Series(nm, filled[:, j]) for j, nm in enumerate(pa_names)]
            )
            sim_cols = sim_cols + pa_names
            pass_angle_used = True

    art = fit_pool(sim_pool, sim_cols, {"pca_var": pca_var, "nmf_k": nmf_k})
    X_scaled = art["X_scaled"]
    pcaed = art["pcaed"]
    n = pool.height

    scores: dict[str, np.ndarray] = {}
    cos = cosine_similarity(X_scaled, X_scaled[idx : idx + 1]).ravel().astype(float)
    cos[idx] = -np.inf
    scores["Cosine"] = cos

    n_neighbors = min(n, max(top_n + 1, 2))
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(pcaed)
    dist, ind = knn.kneighbors(pcaed[idx : idx + 1], n_neighbors=n_neighbors)
    knn_scores = np.full(n, -np.inf, dtype=float)
    for d, j in zip(dist[0], ind[0]):
        if j != idx:
            knn_scores[j] = 1.0 / (1.0 + d)
    scores["KNN_PCA"] = knn_scores

    sp = spearman_scores(X_scaled, idx)
    sp[idx] = -np.inf
    scores["Spearman"] = sp

    role_X = X_scaled[:, : len(feature_cols)]
    roles, role_names, _ = role_matrix(role_X, feature_cols, pos_group)
    role_cos = cosine_similarity(roles, roles[idx : idx + 1]).ravel().astype(float)
    role_cos[idx] = -np.inf
    scores["RoleBlocks"] = role_cos

    W_norm = art["W_norm"]
    nmf_cos = cosine_similarity(W_norm, W_norm[idx : idx + 1]).ravel().astype(float)
    nmf_cos[idx] = -np.inf
    scores["NMF"] = nmf_cos

    for name, sc in (extra_scores or {}).items():
        sc = np.asarray(sc, dtype=float).copy()
        sc[idx] = -np.inf
        scores[name] = sc

    pass_angle_cos = (
        _pass_angle_cosine(pass_angle_X, idx) if pass_angle_X is not None else None
    )

    rrf = np.zeros(n)
    ranks: dict[str, np.ndarray] = {}
    for name, sc in scores.items():
        order = np.argsort(sc)[::-1]
        rnk = np.empty(n)
        rnk[order] = np.arange(1, n + 1)
        ranks[name] = rnk
        rrf += 1.0 / (rrf_k + rnk)
    rrf[idx] = -np.inf

    ranking = rrf.copy()
    if eligible is not None:
        ranking[~np.asarray(eligible, dtype=bool)] = -np.inf
    order = np.argsort(ranking)[::-1][:top_n]
    # Drop padding when fewer players than `top_n` survive the mask (or the pool
    # is smaller than `top_n`); those slots carry -inf, not real matches.
    order = order[np.isfinite(ranking[order])]

    extra_cols = []
    if pass_angle_cos is not None:
        extra_cols.append(pl.Series("pass_angle_cos", pass_angle_cos[order]))
    consensus = pool[order.tolist()].select(
        "player", "team", "league", POS_COL, "primary_pos", MINUTES_COL, AGE_COL
    ).with_columns(
        pl.Series("rrf_score", rrf[order]),
        *extra_cols,
        *[
            pl.Series(f"{name}_rank", ranks[name][order].astype(int))
            for name in scores
        ],
    )

    return {
        "idx": idx,
        "scores": scores,
        "ranks": ranks,
        "rrf": rrf,
        "order": order,
        "consensus": consensus,
        "n_pca": art["n_pca"],
        "pca_var_explained": art["pca_var_explained"],
        "pass_angle_used": pass_angle_used,
        "pass_angle_cos": pass_angle_cos,
    }


def percentile_profile_values(
    pool: pl.DataFrame,
    feature_cols: list[str],
    row_values: np.ndarray,
    strength_cutoff: float = STRENGTH_CUTOFF,
    weakness_cutoff: float = WEAKNESS_CUTOFF,
    top_k: int = TOP_STATS,
) -> dict:
    pool_values = pool.select(feature_cols).to_numpy()
    n = pool_values.shape[0]
    percentiles: list[tuple[str, float, float]] = []
    for j, col in enumerate(feature_cols):
        col_vals = pool_values[:, j]
        finite = col_vals[np.isfinite(col_vals)]
        if finite.size < 2:
            continue
        val = row_values[j]
        if not np.isfinite(val):
            continue
        pct = oriented_percentile(col, float(percentileofscore(finite, val, kind="rank")))
        percentiles.append((col, float(val), pct))

    strengths = sorted(
        [(c, v, p) for c, v, p in percentiles if p >= strength_cutoff],
        key=lambda x: x[2],
        reverse=True,
    )[:top_k]
    weaknesses = sorted(
        [(c, v, p) for c, v, p in percentiles if p <= weakness_cutoff],
        key=lambda x: x[2],
    )[:top_k]
    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "n_stats": len(percentiles),
        "pool_size": n,
    }


def role_scores_values(
    pool: pl.DataFrame,
    feature_cols: list[str],
    row_values: np.ndarray,
    pos_group: str,
) -> pl.DataFrame:
    X = pool.select(feature_cols).to_numpy()
    scaler = RobustScaler().fit(X)
    row_scaled = scaler.transform(row_values.reshape(1, -1))
    roles, role_names, _ = role_matrix(row_scaled, feature_cols, pos_group)
    return pl.DataFrame(
        {"role": role_names, "score": [float(roles[0, i]) for i in range(len(role_names))]}
    ).sort("score", descending=True)


def profile_from_pool_row(
    pool: pl.DataFrame,
    pool_idx: int,
    feature_cols: list[str],
    pos_group: str,
    strength_cutoff: float = STRENGTH_CUTOFF,
    weakness_cutoff: float = WEAKNESS_CUTOFF,
    top_stats: int = TOP_STATS,
) -> dict:
    row = pool.row(pool_idx, named=True)
    row_values = pool.select(feature_cols).to_numpy()[pool_idx]
    pct = percentile_profile_values(
        pool,
        feature_cols,
        row_values,
        strength_cutoff=strength_cutoff,
        weakness_cutoff=weakness_cutoff,
        top_k=top_stats,
    )
    roles = role_scores_values(pool, feature_cols, row_values, pos_group)
    return {
        "player": row["player"],
        "team": row["team"],
        "league": row["league"],
        "position": row[POS_COL],
        "primary_pos": row["primary_pos"],
        "nineties": float(row[MINUTES_COL]),
        "age": row.get(AGE_COL),
        "percentiles": pct,
        "roles": roles,
        "pool_size": pool.height,
        "pool_idx": pool_idx,
    }


def analyze(
    player_name: str,
    team: str | None = None,
    league: str | None = None,
    *,
    events_dir: str | Path | None = None,
    season: int | str = DEFAULT_MAP_SEASON,
    min_touches: int = DEFAULT_MIN_TOUCHES,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    top_n: int = 10,
    min_90s: float | None = None,
    min_age: float | None = None,
    max_age: float | None = None,
    include_unknown_age: bool = True,
    stats_path: Path | str = DATA_PATH,
    strength_cutoff: float = STRENGTH_CUTOFF,
    weakness_cutoff: float = WEAKNESS_CUTOFF,
    top_stats: int = TOP_STATS,
    pca_var: float = 0.80,
    nmf_k: int = 5,
    rrf_k: int = 60,
    action_features: bool = True,
    save_maps_dir: str | Path | None = None,
) -> dict:
    """Touch (+ action-map) cluster, then Sofascore similarity within that cluster.

    `min_age` / `max_age` narrow which cluster mates are returned. They are
    applied to the ranked results rather than to the comparison pool, so
    percentiles, role scores and similarity numbers match an unfiltered run and
    only the returned set changes.
    """
    touch_target, cluster_mates, cluster_label = _run_touchmap_cluster(
        player_name,
        team,
        league,
        events_dir,
        season,
        min_touches,
        n_clusters,
        action_features=action_features,
        save_maps_dir=save_maps_dir,
    )

    players, feature_cols = _ensure_loaded(stats_path)

    try:
        target_row, _ = resolve_target(
            players,
            {
                "player": touch_target["player"],
                "team": touch_target["team"],
                "league": touch_target.get("league"),
            },
        )
    except ValueError as exc:
        return {
            "touch_target": touch_target,
            "target_profile": {
                "error": _stage2_error(exc, touch_target, players, stats_path),
                "player": touch_target["player"],
            },
            "quant_matches": None,
            "quant_profiles": [],
            "strength_cutoff": strength_cutoff,
            "weakness_cutoff": weakness_cutoff,
            "top_stats": top_stats,
            "min_90s": min_90s,
            "cluster_label": cluster_label,
            "cluster_size": len(cluster_mates),
        }

    try:
        pool, pool_meta = build_sofascore_cluster_pool(
            players, cluster_mates, target_row, min_90s=min_90s
        )
    except ValueError as exc:
        return {
            "touch_target": touch_target,
            "target_profile": {"error": str(exc), "player": touch_target["player"]},
            "quant_matches": None,
            "quant_profiles": [],
            "strength_cutoff": strength_cutoff,
            "weakness_cutoff": weakness_cutoff,
            "top_stats": top_stats,
            "min_90s": min_90s,
            "cluster_label": cluster_label,
            "cluster_size": len(cluster_mates),
            "pool_meta": {
                "cluster_touch_size": len(cluster_mates),
                "matched": 0,
                "unmatched": [],
            },
        }

    pos_group = target_row["primary_pos"]
    vec_by = touch_target.get("pass_angle_vec_by_player") or {}
    stats_by = touch_target.get("pass_angle_stats_by_player") or {}
    pass_angle_X, stats_by_pool_idx = _align_pass_angles_to_pool(
        pool, cluster_mates, vec_by, stats_by
    )

    eligible, age_meta = age_eligibility(
        pool,
        min_age=min_age,
        max_age=max_age,
        include_unknown=include_unknown_age,
    )

    quant = search_in_pool(
        pool,
        target_row,
        feature_cols,
        top_n=top_n,
        pca_var=pca_var,
        nmf_k=nmf_k,
        rrf_k=rrf_k,
        pass_angle_X=pass_angle_X,
        eligible=eligible,
    )

    target_profile = profile_from_pool_row(
        pool,
        quant["idx"],
        feature_cols,
        pos_group,
        strength_cutoff=strength_cutoff,
        weakness_cutoff=weakness_cutoff,
        top_stats=top_stats,
    )
    target_profile["pass_angles"] = stats_by_pool_idx.get(quant["idx"]) or touch_target.get(
        "pass_angles"
    )

    ang_scores = quant.get("pass_angle_cos")
    quant_profiles: list[dict] = []
    for rank, pool_idx in enumerate(quant["order"], start=1):
        prof = profile_from_pool_row(
            pool,
            int(pool_idx),
            feature_cols,
            pos_group,
            strength_cutoff=strength_cutoff,
            weakness_cutoff=weakness_cutoff,
            top_stats=top_stats,
        )
        prof["quant_rank"] = rank
        prof["rrf_score"] = float(quant["rrf"][pool_idx])
        prof["pass_angles"] = stats_by_pool_idx.get(int(pool_idx))
        if ang_scores is not None:
            cos = float(ang_scores[pool_idx])
            prof["pass_angle_cos"] = cos if np.isfinite(cos) else None
        quant_profiles.append(prof)

    return {
        "touch_target": touch_target,
        "target_profile": target_profile,
        "quant_matches": quant["consensus"],
        "quant_search": quant,
        "quant_profiles": quant_profiles,
        "strength_cutoff": strength_cutoff,
        "weakness_cutoff": weakness_cutoff,
        "top_stats": top_stats,
        "min_90s": min_90s,
        "cluster_label": cluster_label,
        "cluster_size": len(cluster_mates),
        "pool_meta": pool_meta,
        "age_meta": age_meta,
    }


def _fmt_stat_line(col: str, value: float, pct: float) -> str:
    return f"  {pretty_stat(col)}: {value:.2f}  (p{pct:.0f})"


def print_profile_block(
    label: str,
    profile: dict,
    strength_cutoff: float,
    weakness_cutoff: float,
    top_stats: int,
    pool_line: str = "",
    extra_header: str = "",
) -> None:
    if profile.get("error"):
        print(f"\n{label}: {profile['player']} — skipped ({profile['error']})")
        return

    age = profile.get("age")
    age_bit = f", age {age:.1f}" if age is not None else ""
    print(
        f"\n{label}: {profile['player']} ({profile['team']}, {profile['league']})"
        f"  {profile['position']}  {profile['nineties']:.1f} 90s{age_bit}{extra_header}"
    )
    if pool_line:
        print(f"  {pool_line}")

    pct = profile["percentiles"]
    strengths = pct["strengths"][:top_stats]
    weaknesses = pct["weaknesses"][:top_stats]

    print(f"\n  Strengths (>= p{strength_cutoff:.0f}):")
    if strengths:
        for col, val, p in strengths:
            print(_fmt_stat_line(col, val, p))
    else:
        print("    (none above cutoff)")

    print(f"\n  Weaknesses (<= p{weakness_cutoff:.0f}):")
    if weaknesses:
        for col, val, p in weaknesses:
            print(_fmt_stat_line(col, val, p))
    else:
        print("    (none below cutoff)")

    print("\n  Role mix (RobustScaler blocks):")
    for row in profile["roles"].iter_rows(named=True):
        print(f"    {row['role']}: {row['score']:+.2f}")

    pass_angles = profile.get("pass_angles")
    if pass_angles and label != "TARGET":
        _plt, par, _pam, _tms = _map_deps()
        cos = profile.get("pass_angle_cos")
        cos_bit = f"  |  cosine vs target {cos:+.3f}" if cos is not None else ""
        print(f"\n  Pass angle tendency{cos_bit}:")
        print(par.format_pass_angle_summary(pass_angles))


def print_map_report(result: dict) -> None:
    _plt, par, _pam, _tms = _map_deps()
    t = result["touch_target"]
    league_bit = f", {t['league']}" if t.get("league") else ""
    cluster_label = result.get("cluster_label", "?")
    cluster_size = result.get("cluster_size", "?")
    pool_meta = result.get("pool_meta", {})
    matched = pool_meta.get("matched", "?")
    min_90s = result.get("min_90s")
    min_bit = f", pool min {min_90s:.0f} 90s" if min_90s is not None else ""

    print(
        f"=== Step 1: touch-map cluster for {t['player']} ({t['team']}{league_bit}) ===\n"
        f"cluster {cluster_label} — {cluster_size} players with touch data, "
        f"{matched} matched in Sofascore{min_bit}"
    )
    unmatched = pool_meta.get("unmatched", [])
    if unmatched:
        n_skip = len(unmatched)
        print(
            f"  ({n_skip} not in Sofascore: {', '.join(unmatched[:5])}"
            + (" …" if n_skip > 5 else "") + ")"
        )

    age_meta = result.get("age_meta") or {}
    if age_meta.get("min_age") is not None or age_meta.get("max_age") is not None:
        print(f"  age filter {format_age_range(age_meta)}")

    maps_saved = t.get("maps_saved")
    if maps_saved:
        print(f"  maps saved: {', '.join(f'{name}={path}' for name, path in maps_saved.items())}")

    pass_angles = t.get("pass_angles")
    print("\n=== Pass angle tendency ===")
    if pass_angles:
        print(par.format_pass_angle_summary(pass_angles))
    else:
        print("  (no directed passes)")

    quant_search = result.get("quant_search")
    if quant_search:
        angle_bit = (
            ", pass-angle features fused"
            if quant_search.get("pass_angle_used")
            else ", pass-angle skipped"
        )
        print(
            f"\n=== Step 2: Sofascore matches within cluster "
            f"(pca={quant_search['n_pca']}, "
            f"{100 * quant_search['pca_var_explained']:.0f}% var{angle_bit}) ==="
        )
        if result.get("quant_matches") is not None:
            print(result["quant_matches"])

    sc = result["strength_cutoff"]
    wc = result["weakness_cutoff"]
    ts = result["top_stats"]
    pool_line = (
        f"percentiles vs {matched} Sofascore players in touch cluster {cluster_label}"
    )

    print("\n=== Strengths / weaknesses ===")
    print_profile_block("TARGET", result["target_profile"], sc, wc, ts, pool_line)

    for prof in result.get("quant_profiles", []):
        extra = f"  |  quant rank #{prof['quant_rank']}, rrf={prof['rrf_score']:.4f}"
        print_profile_block(
            f"MATCH #{prof['quant_rank']}",
            prof,
            sc,
            wc,
            ts,
            pool_line,
            extra_header=extra,
        )


def _montage_from_quant(result: dict) -> pd.DataFrame | None:
    profiles = result.get("quant_profiles")
    if not profiles:
        return None
    similar = pd.DataFrame(
        [
            {
                "rank": p["quant_rank"],
                "player": p["player"],
                "team": p["team"],
                "league": p["league"],
            }
            for p in profiles
        ]
    )
    similar.attrs["target"] = result["touch_target"]
    return similar


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scrape", action="store_true", help="fetch Sofascore stats before search")
    p.add_argument("--season", default=DEFAULT_SEASON, help="Sofascore season, e.g. 25/26")
    p.add_argument(
        "--map-season",
        type=int,
        default=DEFAULT_MAP_SEASON,
        help="WhoScored / touch-map season year (default: %(default)s)",
    )
    p.add_argument(
        "--league",
        action="append",
        dest="leagues",
        help="league name for scrape (repeatable); default is the Big-5+ set",
    )
    p.add_argument("--player", default=DEFAULT_QUERY["player"])
    p.add_argument("--team", default=None)
    p.add_argument(
        "--filter-league",
        default=None,
        help="disambiguate player lookup / touch-map resolve by league",
    )
    p.add_argument("--min-90s", type=float, default=DEFAULT_QUERY["min_90s"])
    p.add_argument("--top-n", type=int, default=DEFAULT_QUERY["top_n"])
    p.add_argument("--adjacent", action="store_true", help="allow adjacent position pools")
    p.add_argument("--data", type=Path, default=DATA_PATH)
    p.add_argument("--scrape-only", action="store_true", help="scrape CSV and exit")
    p.add_argument(
        "--top-up",
        action="store_true",
        help="re-scrape only the empty (league, position) chunks of an existing "
        "CSV and merge them in, then exit; leaves working leagues alone",
    )
    p.add_argument(
        "--check-chunks",
        action="store_true",
        help="print rows per league/position in the CSV and exit (no network)",
    )
    p.add_argument(
        "--with-maps",
        action="store_true",
        help="touch/action-map cluster + Sofascore similarity within cluster "
        "(same pipeline as player_profile.py)",
    )
    p.add_argument("--events-dir", default=None, help="single league events folder")
    p.add_argument("--min-touches", type=int, default=DEFAULT_MIN_TOUCHES)
    p.add_argument("--n-clusters", type=int, default=DEFAULT_N_CLUSTERS)
    p.add_argument(
        "--no-action-features",
        action="store_true",
        help="cluster on touch location only (skip pass/take-on/shot/defensive shape)",
    )
    p.add_argument(
        "--save-maps",
        default=None,
        metavar="DIR",
        help="save target touch/pass/pass-angle/take-on/shot/defensive PNGs",
    )
    p.add_argument(
        "--montage",
        default=None,
        help="save touch-map PNG montage of target + Sofascore matches",
    )
    p.add_argument(
        "--scrape-ages",
        action="store_true",
        help=f"fetch each player's birth date into the {DOB_COL} column of an "
        "existing CSV, then exit; resumable, so re-running only retries what is "
        "still missing",
    )
    p.add_argument(
        "--age-sleep",
        type=float,
        default=0.2,
        help="seconds to wait between birth-date requests (default: %(default)s). "
        "Sofascore starts answering with a 403 challenge after a while; pacing "
        "the run more slowly gets further before that happens",
    )
    p.add_argument("--min-age", type=float, default=None, help="only return matches at or above this age")
    p.add_argument("--max-age", type=float, default=None, help="only return matches at or below this age")
    p.add_argument(
        "--exclude-unknown-age",
        action="store_true",
        help="drop matches with no known birth date when an age range is set",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.check_chunks:
        print(chunk_coverage(args.data).to_string())
        gaps = missing_chunks(args.data)
        print(
            "\nempty chunks: " + (", ".join(f"{lg} / {pos}" for lg, pos in gaps) or "none")
            + ("\nfill them with --top-up" if gaps else "")
        )
        raise SystemExit(0)
    if args.top_up:
        top_up(path=args.data, season=args.season)
        raise SystemExit(0)
    if args.scrape_ages:
        collect_ages(path=args.data, sleep=args.age_sleep)
        raise SystemExit(0)
    if args.scrape or args.scrape_only or not args.data.exists():
        if not args.data.exists() and not args.scrape and not args.scrape_only:
            print(f"{args.data} missing — scraping first")
        collect(season=args.season, leagues=args.leagues, path=args.data)
        if args.scrape_only:
            raise SystemExit(0)

    reset_cache()

    if args.with_maps:
        # Cluster pool already filters by shape; default min_90s is softer.
        map_min_90s = None if args.min_90s == DEFAULT_QUERY["min_90s"] else args.min_90s
        result = analyze(
            args.player,
            team=args.team,
            league=args.filter_league,
            events_dir=args.events_dir,
            season=args.map_season,
            min_touches=args.min_touches,
            n_clusters=args.n_clusters,
            top_n=args.top_n,
            min_90s=map_min_90s,
            min_age=args.min_age,
            max_age=args.max_age,
            include_unknown_age=not args.exclude_unknown_age,
            stats_path=args.data,
            action_features=not args.no_action_features,
            save_maps_dir=args.save_maps,
        )
        print_map_report(result)
        if args.montage:
            montage_df = _montage_from_quant(result)
            if montage_df is not None:
                _plt, _par, _pam, tms = _map_deps()
                tms.plot_similar_montage(
                    montage_df,
                    season=args.map_season,
                    out_path=args.montage,
                )
                print(f"\nmontage saved to {args.montage}")
    else:
        result = search(
            {
                "player": args.player,
                "team": args.team,
                "league": args.filter_league,
                "min_90s": args.min_90s,
                "min_age": args.min_age,
                "max_age": args.max_age,
                "include_unknown_age": not args.exclude_unknown_age,
                "top_n": args.top_n,
                "allow_adjacent_pos": args.adjacent,
            },
            path=args.data,
        )
        summarize(result)
        print(consensus_table(result))
        print(nmf_mix_table(result))
        for line in nmf_loadings(result):
            print(line)
