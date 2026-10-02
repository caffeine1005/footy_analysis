"""Average team xG difference per player per season.

For each game a player appears in, compute the team's xG differential from that
player's perspective (xG for minus xG against). Season average = mean over
appearances.

Example: Bruno Fernandes plays in a game where Man Utd record 1.0 xG and
concede 0.4 xG → that game contributes +0.6 to his average.

Data sources:
- Match xG: Understat schedule (cached via soccerdata)
- Player appearances: games_players/*.csv, with WhoScored event JSON fallback
- Game metadata: WhoScored event JSON cache or league_games event CSVs

CLI:
    python player_xg_diff.py --all-leagues --season 2025 --plot
    python player_xg_diff.py --all-leagues --from-cache --plot --player "Bruno Fernandes"
    python player_xg_diff.py --league "ENG-Premier League" --season 2025
    python player_xg_diff.py --player "Bruno Fernandes" --min-games 5 --sort vs_team_avg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import soccerdata as sd

from name_utils import normalize_name

DEFAULT_SEASON = 2025
DEFAULT_LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
]


def _league_slug(league: str, season: int | str = DEFAULT_SEASON) -> str:
    return f"{league.replace(' ', '_')}_{season}"

MATCH_XG_DIR = Path("match_xg")
GAMES_PLAYERS_DIR = Path("games_players")
EVENTS_ROOT = Path("league_games")
WHO_SCORED_EVENTS = Path.home() / "soccerdata/data/WhoScored/events"

# WhoScored short names → normalized Understat-style names.
WS_TEAM_ALIASES: dict[str, str] = {
    # Premier League (WhoScored → Understat)
    "man utd": "manchester united",
    "man city": "manchester city",
    "newcastle": "newcastle united",
    "wolves": "wolverhampton wanderers",
    "nottm forest": "nottingham forest",
    "spurs": "tottenham",
    "west ham": "west ham united",
    # La Liga
    "atletico": "atletico madrid",
    "deportivo alaves": "alaves",
    "ath bilbao": "athletic club",
    "athletic": "athletic club",
    "celta": "celta vigo",
    "betis": "real betis",
    "sociedad": "real sociedad",
    # Serie A — mostly already align; keep AC Milan as-is
    "hellas verona": "verona",
    # Bundesliga
    "bayern": "bayern munich",
    "leverkusen": "bayer leverkusen",
    "rbl": "rasenballsport leipzig",
    "rb leipzig": "rasenballsport leipzig",
    "leipzig": "rasenballsport leipzig",
    "fc koln": "fc cologne",
    "koln": "fc cologne",
    "koeln": "fc cologne",
    "mainz": "mainz 05",
    "hamburg": "hamburger sv",
    "stuttgart": "vfb stuttgart",
    "fc heidenheim": "fc heidenheim",
    "heidenheim": "fc heidenheim",
    "mgladbach": "borussia m.gladbach",
    "gladbach": "borussia m.gladbach",
    "dortmund": "borussia dortmund",
    # Ligue 1
    "psg": "paris saint germain",
    "paris sg": "paris saint germain",
    "paris saint-germain": "paris saint germain",
}

UNDERSTAT_LEAGUE_MAP = {
    "ENG-Premier League": "ENG-Premier League",
    "ESP-La Liga": "ESP-La Liga",
    "ITA-Serie A": "ITA-Serie A",
    "GER-Bundesliga": "GER-Bundesliga",
    "FRA-Ligue 1": "FRA-Ligue 1",
}


def _ws_season_code(season: int | str) -> str:
    """2025 calendar season → Understat 2025 / WhoScored 2526."""
    year = int(str(season)[-2:])
    return f"{year}{year + 1}"


def _understat_season(season: int | str) -> str:
    return str(season)


def normalize_team(name: str | None) -> str:
    base = normalize_name(name)
    return WS_TEAM_ALIASES.get(base, base)


def _norm_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.tz_localize(None).dt.normalize()


def _match_key(home: str, away: str) -> str:
    teams = sorted([normalize_team(home), normalize_team(away)])
    return "|".join(teams)


def fetch_match_xg(
    league: str,
    season: int | str = DEFAULT_SEASON,
    *,
    cache: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch or load cached per-game team xG from Understat."""
    if league not in UNDERSTAT_LEAGUE_MAP:
        raise ValueError(
            f"League {league!r} has no Understat mapping. "
            f"Supported: {list(UNDERSTAT_LEAGUE_MAP)}"
        )

    cache_path = MATCH_XG_DIR / f"{_league_slug(league, season)}.csv"
    if cache and cache_path.exists() and not force_refresh:
        df = pd.read_csv(cache_path, parse_dates=["date"])
        return df

    us = sd.Understat(
        leagues=UNDERSTAT_LEAGUE_MAP[league],
        seasons=_understat_season(season),
    )
    sched = us.read_schedule(force_cache=True).reset_index()
    sched["date"] = _norm_date(sched["date"])
    sched["home_team_norm"] = sched["home_team"].map(normalize_team)
    sched["away_team_norm"] = sched["away_team"].map(normalize_team)
    sched["match_key"] = sched.apply(
        lambda r: _match_key(r["home_team"], r["away_team"]), axis=1
    )
    out = sched[
        [
            "league",
            "season",
            "date",
            "home_team",
            "away_team",
            "home_team_norm",
            "away_team_norm",
            "match_key",
            "home_xg",
            "away_xg",
            "has_data",
        ]
    ].copy()

    if cache:
        MATCH_XG_DIR.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache_path, index=False)
    return out


def _lineups_from_event_csvs(
    league: str,
    season: int | str,
    *,
    skip_game_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Infer appearances from cached league_games event CSVs (any recorded event)."""
    game_dir = EVENTS_ROOT / _league_slug(league, season)
    if not game_dir.exists():
        return pd.DataFrame()

    skip = skip_game_ids or set()
    frames = []
    usecols = ["game_id", "team_id", "team", "player_id", "player", "type"]

    for path in sorted(game_dir.glob("*.csv")):
        gid = int(path.stem)
        if gid in skip:
            continue
        try:
            events = pd.read_csv(path, usecols=usecols)
        except (pd.errors.EmptyDataError, ValueError):
            continue
        if events.empty:
            continue
        players = events.dropna(subset=["player_id"]).drop_duplicates(
            ["game_id", "team_id", "player_id"]
        )
        if players.empty:
            continue
        players = players.assign(
            player_name=players["player"],
            minutes_played=pd.NA,
            is_starter=pd.NA,
            source="event_csv",
        )
        frames.append(
            players[
                [
                    "game_id",
                    "team_id",
                    "player_id",
                    "player_name",
                    "minutes_played",
                    "is_starter",
                    "source",
                ]
            ]
        )

    if not frames:
        return pd.DataFrame(
            columns=[
                "game_id",
                "team_id",
                "player_id",
                "player_name",
                "minutes_played",
                "is_starter",
                "source",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _lineups_from_games_players(game_ids: set[int] | None = None) -> pd.DataFrame:
    frames = []
    for path in sorted(GAMES_PLAYERS_DIR.glob("*_players.csv")):
        gid = int(path.stem.split("_")[0])
        if game_ids is not None and gid not in game_ids:
            continue
        df = pd.read_csv(path)
        df = df[df["minutes_played"].fillna(0) > 0].copy()
        df["source"] = "games_players"
        frames.append(df)
    if not frames:
        return pd.DataFrame(
            columns=[
                "game_id",
                "team_id",
                "player_id",
                "player_name",
                "minutes_played",
                "is_starter",
                "source",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _player_played_in_json(player: dict) -> bool:
    stats = player.get("stats") or {}
    if stats.get("touches"):
        return True
    return bool(player.get("isFirstEleven"))


def _lineups_from_whoscored_json(
    league: str,
    season: int | str,
    *,
    skip_game_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (schedule_meta, lineups) parsed from cached WhoScored event JSON."""
    ws_season = _ws_season_code(season)
    event_dir = WHO_SCORED_EVENTS / f"{league}_{ws_season}"
    if not event_dir.exists():
        return pd.DataFrame(), pd.DataFrame()

    skip = skip_game_ids or set()
    meta_rows: list[dict] = []
    lineup_rows: list[pd.DataFrame] = []

    for path in sorted(event_dir.glob("*.json")):
        gid = int(path.stem)
        if gid in skip:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not data or "home" not in data or "away" not in data:
            continue

        start = data.get("startTime")
        if not start:
            continue
        date = _norm_date(pd.Series([start])).iloc[0]

        meta_rows.append(
            {
                "game_id": gid,
                "date": date,
                "home_team": data["home"]["name"],
                "away_team": data["away"]["name"],
                "home_team_id": data["home"]["teamId"],
                "away_team_id": data["away"]["teamId"],
                "home_team_norm": normalize_team(data["home"]["name"]),
                "away_team_norm": normalize_team(data["away"]["name"]),
            }
        )

        players = []
        for side in ("home", "away"):
            team = data[side]
            for pl in team.get("players", []):
                if not _player_played_in_json(pl):
                    continue
                players.append(
                    {
                        "game_id": gid,
                        "team_id": team["teamId"],
                        "player_id": pl["playerId"],
                        "player_name": pl["name"],
                        "minutes_played": pd.NA,
                        "is_starter": pl.get("isFirstEleven", False),
                        "source": "whoscored_json",
                    }
                )
        if players:
            lineup_rows.append(pd.DataFrame(players))

    schedule = pd.DataFrame(meta_rows)
    if lineup_rows:
        lineups = pd.concat(lineup_rows, ignore_index=True)
    else:
        lineups = pd.DataFrame(
            columns=[
                "game_id",
                "team_id",
                "player_id",
                "player_name",
                "minutes_played",
                "is_starter",
                "source",
            ]
        )
    return schedule, lineups


def _schedule_from_league_games(
    league: str,
    season: int | str,
) -> pd.DataFrame:
    """Build a minimal schedule from cached league_games event CSVs."""
    game_dir = EVENTS_ROOT / _league_slug(league, season)
    if not game_dir.exists():
        return pd.DataFrame()

    rows = []
    for path in sorted(game_dir.glob("*.csv")):
        gid = int(path.stem)
        try:
            events = pd.read_csv(path, usecols=["game_id", "type", "team_id", "team"])
        except (pd.errors.EmptyDataError, ValueError):
            continue
        if events.empty:
            continue
        starts = events[events["type"] == "Start"].drop_duplicates("team_id")
        if len(starts) < 2:
            continue
        rows.append(
            {
                "game_id": gid,
                "home_team": starts.iloc[0]["team"],
                "away_team": starts.iloc[1]["team"],
                "home_team_id": int(starts.iloc[0]["team_id"]),
                "away_team_id": int(starts.iloc[1]["team_id"]),
                "home_team_norm": normalize_team(starts.iloc[0]["team"]),
                "away_team_norm": normalize_team(starts.iloc[1]["team"]),
            }
        )
    return pd.DataFrame(rows)


def build_game_schedule(
    league: str,
    season: int | str = DEFAULT_SEASON,
) -> pd.DataFrame:
    """Merge WhoScored JSON metadata with league_games fallback."""
    json_sched, _ = _lineups_from_whoscored_json(league, season)
    csv_sched = _schedule_from_league_games(league, season)

    if json_sched.empty and csv_sched.empty:
        raise FileNotFoundError(
            f"No schedule found for {league} {season}. "
            "Collect league events first (player_touchmaps.collect_league_events)."
        )

    if json_sched.empty:
        sched = csv_sched.copy()
    elif csv_sched.empty:
        sched = json_sched.copy()
    else:
        sched = json_sched.merge(
            csv_sched,
            on="game_id",
            how="outer",
            suffixes=("", "_csv"),
        )
        for col in ("home_team", "away_team", "home_team_id", "away_team_id"):
            csv_col = f"{col}_csv"
            if csv_col in sched.columns:
                sched[col] = sched[col].fillna(sched[csv_col])
                sched = sched.drop(columns=[csv_col])

    sched["match_key"] = sched.apply(
        lambda r: _match_key(r["home_team"], r["away_team"]), axis=1
    )
    sched["league"] = league
    sched["season"] = season
    return sched


def load_player_appearances(
    league: str,
    season: int | str = DEFAULT_SEASON,
    *,
    game_ids: set[int] | None = None,
) -> pd.DataFrame:
    """All player-game rows where minutes_played > 0 (or inferred from events)."""
    gp = _lineups_from_games_players(game_ids)
    covered: set[int] = set(gp["game_id"].unique()) if not gp.empty else set()

    _, json_lineups = _lineups_from_whoscored_json(
        league, season, skip_game_ids=covered
    )
    if not json_lineups.empty:
        if game_ids is not None:
            json_lineups = json_lineups[json_lineups["game_id"].isin(game_ids)]
        covered.update(json_lineups["game_id"].unique())

    csv_lineups = _lineups_from_event_csvs(league, season, skip_game_ids=covered)

    frames = [df for df in (gp, json_lineups, csv_lineups) if not df.empty]
    if not frames:
        raise FileNotFoundError(
            "No player lineups found. Add games_players/{game_id}_players.csv files "
            "or collect league events under league_games/."
        )
    combined = pd.concat(frames, ignore_index=True)
    if game_ids is not None:
        combined = combined[combined["game_id"].isin(game_ids)]
    # Prefer games_players / JSON over event_csv when the same player-game exists.
    source_rank = {"games_players": 0, "whoscored_json": 1, "event_csv": 2}
    combined["_rank"] = combined["source"].map(source_rank).fillna(9)
    combined = (
        combined.sort_values("_rank")
        .drop_duplicates(["game_id", "team_id", "player_id"], keep="first")
        .drop(columns=["_rank"])
    )
    return combined


def attach_match_xg(
    schedule: pd.DataFrame,
    match_xg: pd.DataFrame,
) -> pd.DataFrame:
    """Join schedule rows with Understat home/away xG."""
    xg = match_xg.copy()
    xg["date"] = pd.to_datetime(xg["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    xg_by_key = xg[["match_key", "home_xg", "away_xg"]].drop_duplicates("match_key")
    xg_by_key_date = xg[["match_key", "date", "home_xg", "away_xg"]]

    sched = schedule.copy()
    if "date" not in sched.columns:
        return sched.merge(xg_by_key, on="match_key", how="left")

    sched["date"] = pd.to_datetime(sched["date"], errors="coerce")
    if getattr(sched["date"].dt, "tz", None) is not None:
        sched["date"] = sched["date"].dt.tz_localize(None)
    sched["date"] = sched["date"].dt.normalize()

    has_date = sched["date"].notna()
    dated = sched.loc[has_date].merge(
        xg_by_key_date, on=["match_key", "date"], how="left"
    )
    undated = sched.loc[~has_date].merge(xg_by_key, on="match_key", how="left")
    # Prefer date-matched xG; fill remaining via match_key only.
    if not dated.empty and dated["home_xg"].isna().any():
        fill = dated.loc[dated["home_xg"].isna(), ["game_id", "match_key"]].merge(
            xg_by_key, on="match_key", how="left"
        )
        dated = dated.merge(
            fill[["game_id", "home_xg", "away_xg"]].rename(
                columns={"home_xg": "home_xg_fill", "away_xg": "away_xg_fill"}
            ),
            on="game_id",
            how="left",
        )
        dated["home_xg"] = dated["home_xg"].fillna(dated["home_xg_fill"])
        dated["away_xg"] = dated["away_xg"].fillna(dated["away_xg_fill"])
        dated = dated.drop(columns=["home_xg_fill", "away_xg_fill"])
    return pd.concat([dated, undated], ignore_index=True)


def _team_game_xg(games: pd.DataFrame) -> pd.DataFrame:
    """Expand each game to two rows: one per team with xg_for / xg_against."""
    games = games.dropna(subset=["home_xg", "away_xg"]).copy()
    games["home_xg"] = games["home_xg"].astype(float)
    games["away_xg"] = games["away_xg"].astype(float)

    home = games.assign(
        team_id=games["home_team_id"],
        team=games["home_team"],
        xg_for=games["home_xg"],
        xg_against=games["away_xg"],
    )
    away = games.assign(
        team_id=games["away_team_id"],
        team=games["away_team"],
        xg_for=games["away_xg"],
        xg_against=games["home_xg"],
    )
    cols = ["game_id", "team_id", "team", "xg_for", "xg_against"]
    if "date" in games.columns:
        cols = ["game_id", "date", "team_id", "team", "xg_for", "xg_against"]
    return pd.concat([home[cols], away[cols]], ignore_index=True)


def compute_player_game_xg_diff(
    appearances: pd.DataFrame,
    games_with_xg: pd.DataFrame,
) -> pd.DataFrame:
    team_xg = _team_game_xg(games_with_xg)
    joined = appearances.merge(team_xg, on=["game_id", "team_id"], how="inner")
    joined["xg_diff"] = joined["xg_for"] - joined["xg_against"]
    return joined


def compute_team_season_xg_diff(games_with_xg: pd.DataFrame) -> pd.DataFrame:
    """Team season averages from unique team-games (not player-weighted)."""
    team_xg = _team_game_xg(games_with_xg)
    if team_xg.empty:
        return pd.DataFrame(
            columns=[
                "team_id",
                "team",
                "team_games",
                "team_avg_xg_for",
                "team_avg_xg_against",
                "team_avg_xg_diff",
            ]
        )
    team_xg = team_xg.copy()
    team_xg["xg_diff"] = team_xg["xg_for"] - team_xg["xg_against"]
    return (
        team_xg.groupby(["team_id", "team"], as_index=False)
        .agg(
            team_games=("game_id", "nunique"),
            team_avg_xg_for=("xg_for", "mean"),
            team_avg_xg_against=("xg_against", "mean"),
            team_avg_xg_diff=("xg_diff", "mean"),
        )
    )


def aggregate_player_season_stats(
    player_games: pd.DataFrame,
    team_season: pd.DataFrame,
    *,
    league: str,
    season: int | str,
) -> pd.DataFrame:
    """Season-level average xG diff per player, vs their team's season average."""
    if player_games.empty:
        return pd.DataFrame()

    grouped = (
        player_games.groupby(
            ["player_id", "player_name", "team_id", "team"],
            as_index=False,
        )
        .agg(
            games=("game_id", "nunique"),
            minutes=("minutes_played", lambda s: pd.to_numeric(s, errors="coerce").sum()),
            avg_xg_for=("xg_for", "mean"),
            avg_xg_against=("xg_against", "mean"),
            avg_xg_diff=("xg_diff", "mean"),
            total_xg_diff=("xg_diff", "sum"),
        )
        .assign(league=league, season=season)
    )
    grouped = grouped.merge(
        team_season.drop(columns=["team"], errors="ignore"),
        on="team_id",
        how="left",
    )
    grouped["vs_team_avg"] = grouped["avg_xg_diff"] - grouped["team_avg_xg_diff"]
    return grouped.sort_values("avg_xg_diff", ascending=False)


def build_player_xg_diff_table(
    league: str,
    season: int | str = DEFAULT_SEASON,
    *,
    force_refresh_xg: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline: returns (per-game rows, season aggregates)."""
    match_xg = fetch_match_xg(league, season, force_refresh=force_refresh_xg)
    schedule = build_game_schedule(league, season)
    games = attach_match_xg(schedule, match_xg)
    league_game_ids = set(schedule["game_id"].astype(int))
    appearances = load_player_appearances(
        league, season, game_ids=league_game_ids
    )

    player_games = compute_player_game_xg_diff(appearances, games)
    team_season = compute_team_season_xg_diff(games)
    season_table = aggregate_player_season_stats(
        player_games, team_season, league=league, season=season
    )
    return player_games, season_table


def filter_min_games(df: pd.DataFrame, min_games: int) -> pd.DataFrame:
    return df[df["games"] >= min_games].copy()


def find_player(df: pd.DataFrame, name: str) -> pd.DataFrame:
    key = normalize_name(name)
    mask = df["player_name"].map(normalize_name).str.contains(key, regex=False)
    return df[mask].sort_values("avg_xg_diff", ascending=False)


def save_outputs(
    player_games: pd.DataFrame,
    season_table: pd.DataFrame,
    league: str,
    season: int | str,
) -> tuple[Path, Path]:
    slug = _league_slug(league, season)
    out_dir = Path("player_xg_diff")
    out_dir.mkdir(parents=True, exist_ok=True)
    games_path = out_dir / f"{slug}_games.csv"
    season_path = out_dir / f"{slug}_season.csv"
    player_games.to_csv(games_path, index=False)
    season_table.to_csv(season_path, index=False)
    return games_path, season_path


DISPLAY_COLS = [
    "player_name",
    "team",
    "league",
    "games",
    "avg_xg_diff",
    "team_avg_xg_diff",
    "vs_team_avg",
]

BG_COLOR = "#0C0D0E"
FG_COLOR = "#E8E8E8"
GRID_COLOR = "#2A2A2A"
ACCENT_COLOR = "#FF4C4C"
LEAGUE_COLORS = {
    "ENG-Premier League": "#3D85C6",
    "ESP-La Liga": "#E69138",
    "ITA-Serie A": "#6AA84F",
    "GER-Bundesliga": "#C27BA0",
    "FRA-Ligue 1": "#76A5AF",
}

Y_METRIC_LABELS = {
    "vs_team_avg": "vs team avg xG diff",
    "avg_xg_diff": "avg xG diff",
}


def plot_games_vs_metric(
    df: pd.DataFrame,
    *,
    y_metric: str = "vs_team_avg",
    min_games: int = 5,
    highlight: str | None = None,
    label_top: int = 12,
    title: str | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
):
    """Scatter: games played (x) vs chosen metric (y).

    Defaults to `vs_team_avg` — the player lift over their team's season
    average xG differential. Labels the top players by that metric plus any
    highlighted player.
    """
    import matplotlib.pyplot as plt

    if y_metric not in Y_METRIC_LABELS:
        raise ValueError(f"y_metric must be one of {list(Y_METRIC_LABELS)}")

    plot_df = filter_min_games(df, min_games).dropna(subset=["games", y_metric]).copy()
    if plot_df.empty:
        raise ValueError("No players left to plot after filtering.")

    fig, ax = plt.subplots(figsize=(12, 8), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    leagues = (
        list(plot_df["league"].dropna().unique())
        if "league" in plot_df.columns
        else []
    )
    if leagues:
        for league in leagues:
            subset = plot_df[plot_df["league"] == league]
            ax.scatter(
                subset["games"],
                subset[y_metric],
                s=28,
                alpha=0.55,
                c=LEAGUE_COLORS.get(league, ACCENT_COLOR),
                edgecolors="none",
                label=league,
                zorder=2,
            )
    else:
        ax.scatter(
            plot_df["games"],
            plot_df[y_metric],
            s=28,
            alpha=0.55,
            c=ACCENT_COLOR,
            edgecolors="none",
            zorder=2,
        )

    ax.axhline(0, color=FG_COLOR, linewidth=0.8, alpha=0.45, zorder=1)

    labels = plot_df.nlargest(label_top, y_metric)
    if highlight:
        hits = find_player(plot_df, highlight)
        if not hits.empty:
            key_cols = (
                ["player_id", "team_id"]
                if {"player_id", "team_id"}.issubset(labels.columns)
                else ["player_name", "team"]
            )
            labels = pd.concat([labels, hits]).drop_duplicates(subset=key_cols)
            ax.scatter(
                hits["games"],
                hits[y_metric],
                s=90,
                facecolors="none",
                edgecolors=ACCENT_COLOR,
                linewidths=1.8,
                zorder=4,
            )

    for _, row in labels.iterrows():
        ax.annotate(
            row["player_name"],
            (row["games"], row[y_metric]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
            color=FG_COLOR,
            alpha=0.95,
            zorder=5,
        )

    ax.set_xlabel("Games played", color=FG_COLOR, fontsize=11)
    ax.set_ylabel(Y_METRIC_LABELS[y_metric], color=FG_COLOR, fontsize=11)
    ax.set_title(
        title
        or f"Games played vs {Y_METRIC_LABELS[y_metric]} (min {min_games} games)",
        color=FG_COLOR,
        fontsize=13,
        pad=12,
    )
    ax.tick_params(colors=FG_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    if len(leagues) > 1:
        legend = ax.legend(frameon=False, fontsize=8, loc="lower right")
        for text in legend.get_texts():
            text.set_color(FG_COLOR)

    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
        print(f"Plot saved -> {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute average team xG difference per player per season."
    )
    parser.add_argument("--league", default=None)
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON)
    parser.add_argument("--min-games", type=int, default=5)
    parser.add_argument("--player", default=None, help="Filter/highlight one player")
    parser.add_argument("--top", type=int, default=25, help="Rows to print")
    parser.add_argument(
        "--sort",
        default="avg_xg_diff",
        choices=["avg_xg_diff", "vs_team_avg"],
        help="Ranking column (default: avg_xg_diff)",
    )
    parser.add_argument("--refresh-xg", action="store_true")
    parser.add_argument(
        "--all-leagues",
        action="store_true",
        help="Run for every Big-5 league with Understat coverage",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a games-played vs metric scatter plot",
    )
    parser.add_argument(
        "--plot-metric",
        default="vs_team_avg",
        choices=["vs_team_avg", "avg_xg_diff"],
        help="Y-axis metric for --plot (default: vs_team_avg)",
    )
    parser.add_argument(
        "--plot-out",
        default=None,
        help="Output PNG path (default under player_xg_diff/)",
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Plot/print from existing season CSVs instead of rebuilding",
    )
    args = parser.parse_args()

    if args.all_leagues or args.league is None:
        leagues = list(DEFAULT_LEAGUES)
    else:
        leagues = [args.league]
    leagues = [lg for lg in leagues if lg in UNDERSTAT_LEAGUE_MAP]

    all_season = []
    for league in leagues:
        print(f"\n=== {league} {args.season} ===")
        season_path = (
            Path("player_xg_diff") / f"{_league_slug(league, args.season)}_season.csv"
        )
        if args.from_cache and season_path.exists():
            season_table = pd.read_csv(season_path)
            print(f"Loaded cache -> {season_path}")
        else:
            player_games, season_table = build_player_xg_diff_table(
                league,
                args.season,
                force_refresh_xg=args.refresh_xg,
            )
            games_path, season_path = save_outputs(
                player_games, season_table, league, args.season
            )
            print(f"Games with player xG: {len(player_games)} rows -> {games_path}")
            print(f"Season table: {len(season_table)} players -> {season_path}")
            if not season_table.empty:
                matched = season_table["team_avg_xg_diff"].notna().mean()
                print(f"Team-avg join rate: {matched:.0%}")

        ranked = filter_min_games(season_table, args.min_games)
        ranked = ranked.sort_values(args.sort, ascending=False)
        all_season.append(ranked)

        if args.player:
            hits = find_player(season_table, args.player)
            if hits.empty:
                print(f"No rows for player {args.player!r}")
            else:
                print(hits[DISPLAY_COLS].to_string(index=False))
        else:
            print(
                f"\nTop {args.top} by {args.sort} "
                f"(min {args.min_games} games):"
            )
            print(ranked.head(args.top)[DISPLAY_COLS].to_string(index=False))

        if args.plot and len(leagues) == 1:
            out = (
                Path(args.plot_out)
                if args.plot_out
                else Path("player_xg_diff")
                / f"{_league_slug(league, args.season)}_{args.plot_metric}.png"
            )
            plot_games_vs_metric(
                season_table,
                y_metric=args.plot_metric,
                min_games=args.min_games,
                highlight=args.player,
                title=(
                    f"{league} {args.season}: games vs "
                    f"{Y_METRIC_LABELS[args.plot_metric]}"
                ),
                out_path=out,
            )

    if len(all_season) > 1:
        combined = pd.concat(all_season, ignore_index=True)
        combined = combined.sort_values(args.sort, ascending=False)
        out = Path("player_xg_diff") / f"big5_{args.season}.csv"
        combined.to_csv(out, index=False)
        print(f"\nCombined Big-5 rankings -> {out}")
        if args.plot:
            plot_out = (
                Path(args.plot_out)
                if args.plot_out
                else Path("player_xg_diff")
                / f"big5_{args.season}_{args.plot_metric}.png"
            )
            plot_games_vs_metric(
                combined,
                y_metric=args.plot_metric,
                min_games=args.min_games,
                highlight=args.player,
                title=(
                    f"Big 5 {args.season}: games vs "
                    f"{Y_METRIC_LABELS[args.plot_metric]}"
                ),
                out_path=plot_out,
            )


if __name__ == "__main__":
    main()
