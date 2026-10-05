"""Player profiles and similar-player search across seasons.

`sofascore_similarity.analyze` works inside one season: one season of WhoScored
events is clustered on touch/action shape, and one Sofascore stats table ranks
the target's cluster mates. This module keeps that two-stage pipeline but makes
every player a *player-season*, so a target from one season can be compared
against players from any other:

    analyze_seasons("Mesut Ozil", target_season=2015, scope_seasons=[2025])
    analyze_seasons("Arda Guler", target_season=2025)            # every season

Stage 1 embeds the touch/action shape vectors of every season involved (the
target's plus the scope) together and takes the target's nearest
player-seasons as the pool (`pool_per_season` per season searched). It used to
take the target's KMeans cluster; hard cluster edges shut out close matches and
the clusters moved with the season scope, k and even the seed — see
`eval_pool_methods.py`. KMeans survives only as a role tag. The vectors are L1-normalised histograms on
the same pitch grid whatever the season, so they are directly comparable. Each
season's vectors are built and cached on their own by `analysis_cache`, so a
season costs its event load once and a later query that pulls it into a
different mix pays only for the (cheap) embedding. A query confined to one
season embeds exactly the population the single-season pipeline does and
hits the same cache entries.

Stage 2 ranks the pool on Sofascore per-90 stats, read from each
season's own CSV (`sofascore_player_stats_1516.csv` ... and the live
`sofascore_player_stats.csv`). Sofascore's historical coverage is uneven —
13/14 and 14/15 carry little beyond goals, assists and cards; expected goals
and assists only start in 22/23; ball recoveries vanish from 16/17 to 22/23 —
and a stat that a season simply lacks reads as zero. Comparing across such a
gap would rank every older player as "never generates xG", so a pool only uses
the stats that every season in it actually records. When that leaves too few
to rank on, the search falls back to ranking by touch/action-shape distance,
which the event data supports in every season.

Ages are measured at the season in question (1 January of its second year) for
past seasons, and today for the live one, so an age filter means "how old they
were that season".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs

import analysis_cache
import sofascore_similarity as sofa
from name_utils import normalize_name

EVENTS_ROOT = "league_games"  # a str: it is part of `league_events_key`'s digest
FIRST_SEASON = 2013
CURRENT_SEASON = sofa.DEFAULT_MAP_SEASON
SEASON_COL = "season_year"

# A stat counts as recorded in a season when at least this share of that
# season's rows are non-zero, relative to the best-covered season. Relative,
# because some stats are rare by nature (red cards, headed goals) and an
# absolute floor would drop them everywhere.
AVAILABLE_SHARE = 0.5
# Below this many shared stats, stage 2 ranks on shape instead (see module doc).
MIN_STAT_FEATURES = 15
# Stage-1 pool: the target's nearest player-seasons in shape space, this many per
# season searched. ~ the size of a 14-way KMeans cluster, which it replaces.
DEFAULT_POOL_PER_SEASON = 250

_POOL_ID_COLS = [
    "player",
    "team",
    "league",
    "season",
    sofa.POS_COL,
    "primary_pos",
    "player_id",
    sofa.MINUTES_COL,
    sofa.AGE_COL,
    SEASON_COL,
]


def season_label(year: int) -> str:
    """2015 -> '15/16'."""
    year = int(year)
    return f"{year % 100:02d}/{(year + 1) % 100:02d}"


def stats_path(year: int) -> Path:
    """The Sofascore stats CSV for a season (same layout as `backfill_history`)."""
    year = int(year)
    if year == CURRENT_SEASON:
        return sofa.DATA_PATH
    return Path(f"sofascore_player_stats_{year % 100:02d}{(year + 1) % 100:02d}.csv")


def _events_key(year: int) -> str | None:
    import touchmap_similarity as tms

    return analysis_cache.league_events_key(list(tms.DEFAULT_LEAGUES), year, EVENTS_ROOT)


def available_seasons() -> list[int]:
    """Seasons with both event data and a stats table, oldest first."""
    import touchmap_similarity as tms

    out = []
    for year in range(FIRST_SEASON, CURRENT_SEASON + 1):
        if not stats_path(year).exists():
            continue
        if any(
            any((Path(EVENTS_ROOT) / tms._league_slug(lg, year)).glob("*.csv"))
            for lg in tms.DEFAULT_LEAGUES
        ):
            out.append(year)
    return out


# ---------------------------------------------------------------------------
# stage 2 tables: one per season
# ---------------------------------------------------------------------------

_lock = threading.RLock()
_season_tables: dict[int, tuple[tuple, pl.DataFrame, list[str]]] = {}
_nonzero_shares: dict[int, tuple[tuple, pd.Series]] = {}


def _file_sig(path: Path) -> tuple:
    st = path.stat()
    return (str(path), st.st_size, st.st_mtime_ns)


def season_players(year: int) -> tuple[pl.DataFrame, list[str]]:
    """`(players, feature_cols)` for one season, loaded once per process.

    Same table `sofascore_similarity` builds, plus the season as an int, and
    with ages taken at that season rather than today.
    """
    year = int(year)
    path = stats_path(year)
    if not path.exists():
        raise ValueError(f"no Sofascore stats for {season_label(year)} ({path} missing)")
    sig = _file_sig(path)
    with _lock:
        hit = _season_tables.get(year)
        if hit is not None and hit[0] == sig:
            return hit[1], hit[2]
        as_of = None if year == CURRENT_SEASON else date(year + 1, 1, 1)
        players = sofa.load_players(path, as_of=as_of).with_columns(
            pl.lit(year, dtype=pl.Int32).alias(SEASON_COL)
        )
        skip = {sofa.MINUTES_COL, sofa.AGE_COL, "player_id", "team_id", SEASON_COL}
        features = [c for c in players.select(cs.numeric()).columns if c not in skip]
        _season_tables[year] = (sig, players, features)
        return players, features


def _nonzero_share(year: int) -> pd.Series:
    """Share of rows with a non-zero value, per numeric column of the raw CSV."""
    path = stats_path(year)
    sig = _file_sig(path)
    with _lock:
        hit = _nonzero_shares.get(year)
        if hit is not None and hit[0] == sig:
            return hit[1]
        num = pd.read_csv(path).select_dtypes("number")
        share = (num.fillna(0) != 0).mean()
        _nonzero_shares[year] = (sig, share)
        return share


def recorded_stats(years: list[int]) -> dict[int, set[str]]:
    """For each season, the stats it actually records (see AVAILABLE_SHARE).

    "Best-covered season" is taken over every available season, not just the
    ones asked about, so whether a season records a stat does not depend on
    which other seasons happen to be in the query.
    """
    every = sorted(set(available_seasons()) | {int(y) for y in years})
    shares = pd.DataFrame({y: _nonzero_share(y) for y in every})
    peak = shares.max(axis=1)
    out = {}
    for y in years:
        col = shares[int(y)]
        out[int(y)] = {c for c in shares.index if peak[c] > 0 and col[c] >= AVAILABLE_SHARE * peak[c]}
    return out


def seasons_with_player(name: str) -> list[int]:
    """Seasons whose stats table has a player whose name contains `name`."""
    needle = normalize_name(name)
    found = []
    for year in available_seasons():
        names = pd.read_csv(stats_path(year), usecols=["player"])["player"].dropna()
        if any(needle in normalize_name(n) for n in names.unique()):
            found.append(year)
    return found


# ---------------------------------------------------------------------------
# stage 1: shape vectors per season, clustered together
# ---------------------------------------------------------------------------


@dataclass
class Population:
    """Every player-season in `years`, with its shape vector and cluster."""

    years: tuple[int, ...]
    meta: pd.DataFrame  # player, team, league, touches, season_year — row-aligned with X
    X: np.ndarray
    labels: np.ndarray  # KMeans role tag only; pools come from `coords`
    coords: np.ndarray  # `touchmap_similarity.embed_shapes` space, row-aligned with X
    pass_stats: dict[tuple, dict]  # (player, team, season_year) -> pass-angle stats
    pass_vecs: dict[tuple, np.ndarray]
    row_of: dict[tuple, int]  # (player, team, season_year) -> row in meta / X


def _season_grids(year: int, *, min_touches: int, action_features: bool, progress=None):
    import touchmap_similarity as tms

    events_key = _events_key(year)
    if events_key is None:
        raise ValueError(f"no event data for {season_label(year)} under {EVENTS_ROOT}/")

    def load():
        if progress:
            progress(f"Reading {season_label(year)} match events (first time only)...")
        # Not persisted: this pass only needs the events long enough to reduce
        # them to grids, and storing each season's frame would push the one in
        # use out of the events cache.
        return tms.load_all_league_events(season=year, persist=False)

    load = sofa._once(load)
    grids_key, meta, X = analysis_cache.cached_grids(
        events_key,
        load,
        min_touches=min_touches,
        bins=tms.DEFAULT_BINS,
        smooth_sigma=1.0,
        action_features=action_features,
    )
    stats_by, vec_by = analysis_cache.cached_pass_angle_features(grids_key, load, meta)
    return grids_key, meta, X, stats_by, vec_by


_populations: OrderedDict[tuple, Population] = OrderedDict()


def population(
    years,
    *,
    min_touches: int = sofa.DEFAULT_MIN_TOUCHES,
    n_clusters: int = sofa.DEFAULT_N_CLUSTERS,
    action_features: bool = True,
    progress=None,
) -> Population:
    """Shape vectors for every player-season in `years`, clustered together."""
    years = tuple(sorted({int(y) for y in years}))
    memo_key = (years, int(min_touches), int(n_clusters), bool(action_features))
    with _lock:
        hit = _populations.get(memo_key)
        if hit is not None:
            _populations.move_to_end(memo_key)
            return hit

        keys, metas, Xs = [], [], []
        pass_stats, pass_vecs = {}, {}
        for n, year in enumerate(years, start=1):
            if progress:
                progress(f"Shape vectors {season_label(year)} ({n}/{len(years)})...")
            grids_key, meta, X, stats_by, vec_by = _season_grids(
                year, min_touches=min_touches, action_features=action_features, progress=progress
            )
            keys.append(grids_key)
            meta = meta.copy()  # the cached frame is shared; do not tag it in place
            meta[SEASON_COL] = year
            metas.append(meta)
            Xs.append(X)
            pass_stats.update({(p, t, year): v for (p, t), v in stats_by.items()})
            pass_vecs.update({(p, t, year): v for (p, t), v in vec_by.items()})

        if progress:
            progress(f"Clustering {sum(len(m) for m in metas):,} player-seasons...")
        if len(years) == 1:
            # Exactly the single-season population, so the same labels cache.
            X_all = Xs[0]
            labels = analysis_cache.cached_cluster_labels(keys[0], X_all, n_clusters=n_clusters)
            coords = analysis_cache.cached_embedding(keys[0], X_all)
        else:
            # float32 halves a ~600MB matrix at 13 seasons; KMeans does not need more.
            X_all = np.vstack([x.astype(np.float32, copy=False) for x in Xs])
            combined = None
            if all(k is not None for k in keys):
                combined = hashlib.sha1(json.dumps(keys).encode()).hexdigest()[:16]
            labels = analysis_cache.cached_cluster_labels(combined, X_all, n_clusters=n_clusters)
            coords = analysis_cache.cached_embedding(combined, X_all)

        meta_all = pd.concat(metas, ignore_index=True)
        row_of = {
            (p, t, int(y)): i
            for i, (p, t, y) in enumerate(
                zip(meta_all["player"], meta_all["team"], meta_all[SEASON_COL])
            )
        }
        pop = Population(
            years, meta_all, X_all, np.asarray(labels), np.asarray(coords), pass_stats, pass_vecs, row_of
        )
        _populations[memo_key] = pop
        # Each one holds its matrix; two covers a profile and a search side by side.
        while len(_populations) > 2:
            _populations.popitem(last=False)
        return pop


def warm(years=None, *, min_touches=sofa.DEFAULT_MIN_TOUCHES, action_features=True) -> None:
    """Build and cache every season's shape vectors and physicality scores, so the app never waits on them."""
    years = years or available_seasons()
    for year in years:
        print(f"[{season_label(year)}] shape vectors ...", flush=True)
        _season_grids(
            year,
            min_touches=min_touches,
            action_features=action_features,
            progress=lambda msg: print(f"  {msg}", flush=True),
        )
        season_players(year)
        import physicality

        physicality.season_scores(year, progress=lambda msg: print(f"  {msg}", flush=True))
    print("done", flush=True)


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------


def _profile(pool: pl.DataFrame, idx: int, features: list[str], pos_group: str, **cutoffs) -> dict:
    """`sofa.profile_from_pool_row`, tolerant of a thin stat set, plus the season."""
    try:
        prof = sofa.profile_from_pool_row(pool, idx, features, pos_group, **cutoffs)
    except ValueError:
        # No role-block stats among `features` (a 13/14-only pool, say): the
        # percentiles still stand on their own.
        row = pool.row(idx, named=True)
        values = pool.select(features).to_numpy()[idx] if features else np.array([])
        pct = (
            sofa.percentile_profile_values(pool, features, values, **{
                "strength_cutoff": cutoffs.get("strength_cutoff", sofa.STRENGTH_CUTOFF),
                "weakness_cutoff": cutoffs.get("weakness_cutoff", sofa.WEAKNESS_CUTOFF),
                "top_k": cutoffs.get("top_stats", sofa.TOP_STATS),
            })
            if features
            else {"strengths": [], "weaknesses": [], "n_stats": 0, "pool_size": pool.height}
        )
        prof = {
            "player": row["player"],
            "team": row["team"],
            "league": row["league"],
            "position": row[sofa.POS_COL],
            "primary_pos": row["primary_pos"],
            "nineties": float(row[sofa.MINUTES_COL]),
            "age": row.get(sofa.AGE_COL),
            "percentiles": pct,
            "roles": None,
            "pool_size": pool.height,
            "pool_idx": idx,
        }
    year = int(pool[SEASON_COL][idx])
    prof["season_year"] = year
    prof["season_label"] = season_label(year)
    player_id = pool["player_id"][idx] if "player_id" in pool.columns else None
    prof["player_id"] = player_id
    return prof


def _same_player(pool: pl.DataFrame, idx: int) -> np.ndarray:
    """Rows that are the same person as row `idx` (any season)."""
    if "player_id" in pool.columns and pool["player_id"][idx] is not None:
        ids = pool["player_id"].to_numpy()
        return ids == pool["player_id"][idx]
    names = [normalize_name(n) for n in pool["player"].to_list()]
    return np.array([n == names[idx] for n in names])


def _error_result(touch_target, message, **extra) -> dict:
    return {
        "touch_target": touch_target,
        "target_profile": {"error": message, "player": touch_target["player"]},
        "quant_profiles": [],
        **extra,
    }


def analyze_seasons(
    player_name: str,
    team: str | None = None,
    league: str | None = None,
    *,
    target_season: int = CURRENT_SEASON,
    scope_seasons=None,
    exclude_target_other_seasons: bool = True,
    min_touches: int = sofa.DEFAULT_MIN_TOUCHES,
    n_clusters: int = sofa.DEFAULT_N_CLUSTERS,
    pool_per_season: int = DEFAULT_POOL_PER_SEASON,
    top_n: int = 10,
    min_90s: float | None = None,
    min_age: float | None = None,
    max_age: float | None = None,
    include_unknown_age: bool = True,
    action_features: bool = True,
    flip_flanks: bool = False,
    save_maps_dir: str | Path | None = None,
    physicality: bool = True,
    match_physicality: bool = False,
    min_physicality_pct: float | None = None,
    strength_cutoff: float = sofa.STRENGTH_CUTOFF,
    weakness_cutoff: float = sofa.WEAKNESS_CUTOFF,
    top_stats: int = sofa.TOP_STATS,
    pca_var: float = 0.80,
    nmf_k: int = 5,
    rrf_k: int = 60,
    progress=None,
) -> dict:
    """Profile `player_name` in `target_season` and rank similar player-seasons.

    `scope_seasons` is where matches may come from; None or empty means every
    available season. The target's own season need not be in it. With
    `exclude_target_other_seasons`, the target's other seasons are kept out of
    the results (they would otherwise top most lists) but still count as part
    of the comparison pool.

    Stage 1's pool is the target's `pool_per_season` x len(scope) nearest
    player-seasons in shape space. `n_clusters` only sets the KMeans role tag
    reported alongside; it no longer decides who can be a match.

    With `flip_flanks`, the target's heatmaps (and pass-angle left/right) are
    mirrored across the pitch before the shape pool is cut — so a right-flank
    inverted winger becomes a left-flank query, and vice versa.

    With `physicality`, every profile carries its `physicality.py` profile for
    that season (built and cached per season on first use). `match_physicality`
    adds closeness on the physicality axes as one more ranker in the fusion;
    `min_physicality_pct` returns only players at or above that percentile of
    their role group, dropping anyone unscored. Like the age range, the filter
    shapes only which ranked players are returned.
    """
    target_season = int(target_season)
    seasons = available_seasons()
    if target_season not in seasons:
        raise ValueError(f"no data for {season_label(target_season)}")
    scope = sorted({int(y) for y in scope_seasons} if scope_seasons else set(seasons))
    missing = [y for y in scope if y not in seasons]
    if missing:
        raise ValueError(f"no data for {', '.join(season_label(y) for y in missing)}")
    years = sorted(set(scope) | {target_season})

    pop = population(
        years,
        min_touches=min_touches,
        n_clusters=n_clusters,
        action_features=action_features,
        progress=progress,
    )

    # -- stage 1: the target's player-season and its cluster ------------------
    in_season = pop.meta[pop.meta[SEASON_COL] == target_season]
    try:
        t_row_idx = sofa._resolve_touch_player(
            in_season, player_name, team=team, league=sofa.touchmap_league_name(league)
        )
    except ValueError as exc:
        if not str(exc).startswith("no player matching"):
            raise
        elsewhere = [y for y in seasons_with_player(player_name) if y != target_season]
        hint = (
            f" They appear in {', '.join(season_label(y) for y in elsewhere)}."
            if elsewhere
            else ""
        )
        raise ValueError(
            f"no player matching {player_name!r} with at least {min_touches} touches "
            f"in {season_label(target_season)}.{hint}"
        ) from None
    t_row = pop.meta.loc[t_row_idx]
    cluster_label = int(pop.labels[t_row_idx])
    touch_key = (t_row["player"], t_row["team"], target_season)
    pass_angles = pop.pass_stats.get(touch_key)
    if flip_flanks and pass_angles is not None:
        import pass_angle_radar as par

        pass_angles = par.mirror_pass_angle_stats(pass_angles)
    touch_target = {
        "player": t_row["player"],
        "team": t_row["team"],
        "league": t_row["league"],
        "touches": int(t_row["touches"]),
        "cluster": cluster_label,
        "season_year": target_season,
        "season_label": season_label(target_season),
        "pass_angles": pass_angles,
        "flip_flanks": bool(flip_flanks),
    }

    if save_maps_dir is not None:
        import touchmap_similarity as tms

        if progress:
            progress("Rendering dashboards...")
        touch_target["maps_saved"] = sofa._save_target_maps(
            sofa._once(lambda: tms.load_all_league_events(season=target_season)),
            touch_target,
            save_maps_dir,
            events_key=_events_key(target_season),
            flip_flanks=flip_flanks,
        )

    import touchmap_similarity as tms

    in_scope = np.flatnonzero(pop.meta[SEASON_COL].isin(scope).to_numpy())
    # When flipping, re-fit PCA on the same X so the mirrored query and the
    # pool share one embedding (cached `pop.coords` has no PCA to project into).
    shape_coords = pop.coords
    query_coord = None
    if flip_flanks:
        if progress:
            progress("Flipping target flanks (left ↔ right)...")
        shape_coords, pca = tms.fit_shape_pca(pop.X)
        mirrored = tms.mirror_heatmaps(pop.X[t_row_idx : t_row_idx + 1])
        query_coord = tms.project_shapes(pca, mirrored)[0]
        near = tms.nearest_rows_to_point(
            shape_coords, query_coord, in_scope, pool_per_season * len(scope)
        )
    else:
        near = tms.nearest_rows(pop.coords, t_row_idx, in_scope, pool_per_season * len(scope))
    mates = pop.meta.iloc[near]
    base = {
        "cluster_label": cluster_label,
        "cluster_size": int(len(mates)),
        "pool_per_season": int(pool_per_season),
        "target_season": target_season,
        "scope_seasons": scope,
        "min_90s": min_90s,
        "strength_cutoff": strength_cutoff,
        "weakness_cutoff": weakness_cutoff,
        "top_stats": top_stats,
        "flip_flanks": bool(flip_flanks),
    }

    # -- stage 2: the target's stats row --------------------------------------
    if progress:
        progress("Matching shape pool to Sofascore stats...")
    players_t, features_t = season_players(target_season)
    try:
        target_row, _ = sofa.resolve_target(
            players_t,
            {"player": t_row["player"], "team": t_row["team"], "league": t_row["league"]},
        )
    except ValueError as exc:
        return _error_result(
            touch_target,
            sofa._stage2_error(exc, touch_target, players_t, stats_path(target_season)),
            **base,
        )
    t_sidx = sofa._find_sofascore_index(players_t, target_row)
    pos_group = target_row["primary_pos"]

    # -- stage 2: pool -> stats rows, season by season ----------------
    # Grouped by season so each season's name index is built once.
    picked: dict[tuple[int, int], tuple] = {(target_season, t_sidx): touch_key}
    unmatched: list[str] = []
    for year, group in mates.groupby(SEASON_COL, sort=True):
        year = int(year)
        players_s, _ = season_players(year)
        for row in group.itertuples(index=False):
            key = (row.player, row.team, year)
            if key == touch_key:
                continue
            try:
                sofa_row, _ = sofa.resolve_target(
                    players_s, {"player": row.player, "team": row.team, "league": row.league}
                )
            except ValueError:
                unmatched.append(f"{row.player} ({row.team}, {season_label(year)})")
                continue
            nineties = sofa_row[sofa.MINUTES_COL]
            if nineties <= 0 or (min_90s is not None and nineties < min_90s):
                continue
            sidx = sofa._find_sofascore_index(players_s, sofa_row)
            if sidx is not None:
                picked.setdefault((year, sidx), key)

    if len(picked) < 2:
        return _error_result(
            touch_target,
            f"only {len(picked)} player-season(s) in the target's shape pool matched Sofascore "
            "stats; widen the season scope or lower Min 90s",
            **base,
            pool_meta={"cluster_touch_size": len(mates), "matched": len(picked), "unmatched": unmatched},
        )

    # -- the pool: shared stats only ------------------------------------------
    pool_years = sorted({y for y, _ in picked})
    recorded = recorded_stats(pool_years)
    per_season_features = {y: set(season_players(y)[1]) for y in pool_years}
    usable = {
        y: {c for c in features_t if c in recorded[y] and c in per_season_features[y]}
        for y in pool_years
    }
    # A season recording too few stats (13/14, 14/15) would cut every other
    # season down to its handful, turning any search that includes it into a
    # shape-only one. So when the target's season is rich and other rich seasons
    # are in play, thin seasons sit this search out; they are only searched (on
    # shape) when the target is from one, or they are all there is.
    thin = {y for y in pool_years if len(usable[y]) < MIN_STAT_FEATURES}
    skipped_seasons: list[int] = []
    if thin and target_season not in thin and any(y not in thin for y in pool_years if y != target_season):
        skipped_seasons = sorted(y for y in thin if any(py == y for py, _ in picked))
        picked = {k: v for k, v in picked.items() if k[0] not in thin}
        pool_years = [y for y in pool_years if y not in thin]
    features = [c for c in features_t if all(c in usable[y] for y in pool_years)]
    dropped = [c for c in features_t if c not in features]

    frames, keys = [], []
    for year in pool_years:
        players_s, _ = season_players(year)
        rows = sorted(sidx for y, sidx in picked if y == year)
        frame = players_s[rows].select(
            [c for c in _POOL_ID_COLS if c in players_s.columns] + features
        )
        frames.append(
            frame.with_columns(
                [pl.col(c).cast(pl.Float64) for c in features]
                + [
                    pl.col(sofa.MINUTES_COL).cast(pl.Float64),
                    pl.col(sofa.AGE_COL).cast(pl.Float64),
                    pl.col("player_id").cast(pl.Int64, strict=False),
                    pl.col("season").cast(pl.String),
                ]
            )
        )
        keys.extend(picked[(year, sidx)] for sidx in rows)
    pool = pl.concat(frames, how="vertical_relaxed")
    t_idx = keys.index(touch_key)

    eligible, age_meta = sofa.age_eligibility(
        pool, min_age=min_age, max_age=max_age, include_unknown=include_unknown_age
    )
    if exclude_target_other_seasons:
        others = _same_player(pool, t_idx)
        others[t_idx] = False
        mask = ~others
        eligible = mask if eligible is None else (np.asarray(eligible, dtype=bool) & mask)

    phys = _pool_physicality(
        keys,
        t_idx,
        enabled=physicality or match_physicality or bool(min_physicality_pct),
        match=match_physicality,
        min_pct=min_physicality_pct,
        progress=progress,
    )
    if phys["eligible"] is not None:
        eligible = phys["eligible"] if eligible is None else (np.asarray(eligible, dtype=bool) & phys["eligible"])

    cutoffs = {
        "strength_cutoff": strength_cutoff,
        "weakness_cutoff": weakness_cutoff,
        "top_stats": top_stats,
    }
    target_profile = _profile(pool, t_idx, features, pos_group, **cutoffs)
    target_profile["pass_angles"] = touch_target["pass_angles"]
    target_profile["physicality"] = phys["rows"][t_idx]

    mode = "stats" if len(features) >= MIN_STAT_FEATURES else "shape"
    ang_scores = None
    if mode == "stats":
        _plt, par, _pam, _tms = sofa._map_deps()
        pass_angle_X = np.full((pool.height, len(par.PASS_ANGLE_FEATURE_NAMES)), np.nan)
        for i, key in enumerate(keys):
            vec = pop.pass_vecs.get(key)
            if vec is not None:
                pass_angle_X[i] = (
                    par.mirror_pass_angle_features(vec) if flip_flanks and key == touch_key else vec
                )
        if progress:
            progress(f"Ranking {pool.height:,} player-seasons on {len(features)} stats...")
        quant = sofa.search_in_pool(
            pool,
            target_row,
            features,
            top_n=top_n,
            pca_var=pca_var,
            nmf_k=nmf_k,
            rrf_k=rrf_k,
            pass_angle_X=pass_angle_X,
            eligible=eligible,
            target_idx=t_idx,
            extra_scores={"Physicality": phys["similarity"]} if phys["similarity"] is not None else None,
        )
        order = [int(i) for i in quant["order"]]
        scores = quant["rrf"]
        ang_scores = quant.get("pass_angle_cos")
        extra = {"n_pca": quant["n_pca"], "pass_angle_used": quant["pass_angle_used"]}
    else:
        # Distance in the same shape space stage 1 picks the pool in,
        # available for every season.
        if progress:
            progress(f"Ranking {pool.height:,} player-seasons on touch/action shape...")
        rows = [pop.row_of[k] for k in keys]
        C = np.asarray(shape_coords[rows], dtype=np.float64)
        target_pt = query_coord if query_coord is not None else C[t_idx]
        dist = np.linalg.norm(C - np.asarray(target_pt, dtype=np.float64), axis=1)
        scores = 1.0 / (1.0 + dist)
        if phys["similarity"] is not None:
            # Two rankers, fused the same way the stats mode fuses its own.
            scores = _rrf([scores, phys["similarity"]], t_idx, rrf_k)
        ranking = scores.copy()
        ranking[t_idx] = -np.inf
        if eligible is not None:
            ranking[~np.asarray(eligible, dtype=bool)] = -np.inf
        order = [int(i) for i in np.argsort(ranking)[::-1][:top_n] if np.isfinite(ranking[i])]
        extra = {}

    profiles = []
    for rank, i in enumerate(order, start=1):
        prof = _profile(pool, i, features, pos_group, **cutoffs)
        prof["quant_rank"] = rank
        prof["score"] = float(scores[i])
        prof["rrf_score"] = float(scores[i])
        prof["pass_angles"] = pop.pass_stats.get(keys[i])
        prof["physicality"] = phys["rows"][i]
        if ang_scores is not None:
            cos = float(ang_scores[i])
            prof["pass_angle_cos"] = cos if np.isfinite(cos) else None
        profiles.append(prof)

    return {
        **base,
        "touch_target": touch_target,
        "target_profile": target_profile,
        "quant_profiles": profiles,
        "pool_meta": {
            "cluster_touch_size": len(mates),
            "matched": pool.height,
            "unmatched": unmatched,
        },
        "age_meta": age_meta,
        "feature_meta": {
            "mode": mode,
            "n_features": len(features),
            "dropped": dropped,
            "pool_seasons": pool_years,
            "skipped_seasons": skipped_seasons,
            **extra,
        },
        "physicality_meta": phys["meta"],
    }


def _rrf(score_lists: list[np.ndarray], t_idx: int, rrf_k: int) -> np.ndarray:
    n = len(score_lists[0])
    out = np.zeros(n)
    for sc in score_lists:
        sc = np.asarray(sc, dtype=float).copy()
        sc[t_idx] = -np.inf
        rank = np.empty(n)
        rank[np.argsort(sc)[::-1]] = np.arange(1, n + 1)
        out += 1.0 / (rrf_k + rank)
    return out


def _pool_physicality(
    keys: list[tuple],
    t_idx: int,
    *,
    enabled: bool,
    match: bool,
    min_pct: float | None,
    progress=None,
) -> dict:
    """Physicality profiles for every pool row, plus the optional ranker and filter.

    Pool keys are WhoScored (player, team, season) identities, the same keys
    `physicality` scores on, so no name matching is needed. Scores are within
    each season's role groups, so comparing across seasons compares a player's
    standing among their peers in that season.
    """
    n = len(keys)
    out = {"rows": [None] * n, "similarity": None, "eligible": None,
           "meta": {"enabled": bool(enabled)}}
    if not enabled:
        return out
    import physicality as phys_model

    failed = []
    for year in sorted({int(k[2]) for k in keys}):
        try:
            table = phys_model.season_scores(year, progress=progress)
        except (ValueError, FileNotFoundError) as exc:
            failed.append(f"{season_label(year)}: {exc}")
            continue
        by_key = {(p, t): i for i, (p, t) in enumerate(zip(table["player"], table["team"]))}
        for i, (p, t, y) in enumerate(keys):
            if int(y) == year and (p, t) in by_key:
                out["rows"][i] = phys_model.profile_dict(table.iloc[by_key[(p, t)]])

    rows = out["rows"]
    target = rows[t_idx]
    meta = out["meta"]
    meta.update({
        "n_scored": sum(r is not None for r in rows),
        "n_pool": n,
        "target_scored": target is not None,
        "failed": failed,
        "min_pct": min_pct,
        "matched": False,
    })

    if min_pct:
        pct = np.array([r["pct"] if r and r["pct"] is not None else -1.0 for r in rows])
        out["eligible"] = pct >= float(min_pct)

    if match and target is not None:
        t = np.array([np.nan if v is None else v for v in target["z_axes"]], dtype=float)
        Z = np.array(
            [[np.nan if v is None else v for v in (r["z_axes"] if r else [None] * len(t))] for r in rows],
            dtype=float,
        )
        axes = np.isfinite(t)
        # A missing axis sits at the role mean (z = 0): no information, not a low score.
        diff = np.nan_to_num(Z[:, axes]) - t[axes]
        sim = 1.0 / (1.0 + np.linalg.norm(diff, axis=1))
        unscored = np.array([r is None for r in rows])
        if (~unscored).any():
            # Unscored players rank mid-table on this ranker rather than last.
            sim[unscored] = np.median(sim[~unscored])
        out["similarity"] = sim
        meta["matched"] = True
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_seasons(text: str | None) -> list[int] | None:
    """'2015', '2015,2025', '2019-2021' or '15/16' -> start years."""
    if not text:
        return None
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "/" in part:
            out.add(2000 + int(part.split("/")[0]))
        elif "-" in part:
            lo, hi = (int(p) for p in part.split("-"))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--warm", action="store_true", help="build every season's cached shape vectors and exit")
    ap.add_argument("--player")
    ap.add_argument("--team")
    ap.add_argument("--league")
    ap.add_argument("--season", type=int, default=CURRENT_SEASON, help="target season start year")
    ap.add_argument("--scope", help="seasons to search, e.g. 2025 or 2019-2021,2025 (default: all)")
    ap.add_argument("--include-self", action="store_true", help="allow the target's other seasons in results")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--min-90s", type=float)
    ap.add_argument("--match-physicality", action="store_true",
                    help="add closeness on the physicality axes as a similarity ranker")
    ap.add_argument("--min-phys-pct", type=float,
                    help="only return players at or above this physicality percentile in their role")
    ap.add_argument(
        "--flip-flanks",
        action="store_true",
        help="mirror the target's heatmaps left↔right before searching (e.g. LW→RW shape)",
    )
    args = ap.parse_args()

    if args.warm:
        warm()
        return 0
    if not args.player:
        ap.error("--player is required (or pass --warm)")

    result = analyze_seasons(
        args.player,
        team=args.team,
        league=args.league,
        target_season=args.season,
        scope_seasons=_parse_seasons(args.scope),
        exclude_target_other_seasons=not args.include_self,
        top_n=args.top_n,
        min_90s=args.min_90s,
        flip_flanks=args.flip_flanks,
        match_physicality=args.match_physicality,
        min_physicality_pct=args.min_phys_pct,
        progress=lambda msg: print(msg, flush=True),
    )
    t = result["touch_target"]
    prof = result["target_profile"]
    if prof.get("error"):
        print(prof["error"])
        return 1
    fm = result["feature_meta"]
    flip_note = "  [flipped L↔R]" if result.get("flip_flanks") else ""
    print(
        f"\ntarget: {t['player']} {t['season_label']} ({t['team']}, {t['league']})  "
        f"role cluster {result['cluster_label']}, {result['pool_meta']['matched']} player-seasons in pool, "
        f"ranked on {fm['mode']} ({fm['n_features']} shared stats){flip_note}"
        + (
            f"; skipped {', '.join(season_label(y) for y in fm['skipped_seasons'])} (too few stats)"
            if fm.get("skipped_seasons")
            else ""
        )
    )
    tph = prof.get("physicality")
    if tph:
        axes = ", ".join(
            f"{a} p{v['pct']:.0f}" for a, v in tph["axes"].items() if v.get("pct") is not None
        )
        print(f"physicality: p{tph['pct']:.0f} in {tph['role_group']} ({axes})")
    for p in result["quant_profiles"]:
        age = f"{p['age']:.1f}" if p.get("age") is not None else "-"
        ph = p.get("physicality") or {}
        phys_txt = f"phys p{ph['pct']:.0f}" if ph.get("pct") is not None else "phys -"
        print(
            f"  #{p['quant_rank']:<3} {p['player']:<28} {p['season_label']}  "
            f"{p['team']:<24} {p['league']:<26} age {age:<5} {phys_txt:<8} score {p['score']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
