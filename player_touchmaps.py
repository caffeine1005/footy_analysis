"""League-wide touch maps and average-position heatmaps for any player.

Builds on the WhoScored scraping loop in whoscored_team_colection.ipynb:
`collect_league_events` gathers a full league season's event data (every
club, every player — scraping only the games not already cached under
`save_dir`), and `load_season_events` reads whatever is already on disk
without touching the scraper. From there, `final_third_touch_map` plots
every final-third touch a player made across the season, and
`avg_position_heatmap` plots where a player spent their time in one game
(or across the whole season if `game_id` is left as None).

A single game's event csv already contains both clubs' players (WhoScored
records the full match, not just one side), so a directory built from one
team's fixtures (e.g. `utd_games/`) only has full-season coverage for that
team — opponents only show up in the games they played against it. Point
`collect_league_events`/`load_season_events` at a directory built from the
*whole* schedule (`team=None`) to check any player in the league.

WhoScored/Opta coordinates run 0-100 on both axes with each team always
attacking towards x=100, regardless of which half or side they started on.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import soccerdata as sd
from mplsoccer import Pitch, VerticalPitch
from scipy.ndimage import gaussian_filter

from whoscored_patch import apply_whoscored_json_patch

FINAL_THIRD_X = 200 / 3  # opta x-scale (0-100)

BG_COLOR = "#0C0D0E"
LINE_COLOR = "#BBBBBB"
ACCENT_COLOR = "#FF4C4C"


def collect_league_events(
    league: str = "ENG-Premier League",
    season: int | str = 2026,
    team: str | None = None,
    save_dir: str | Path | None = None,
    browser_path: str = "/usr/bin/chromium",
) -> pd.DataFrame:
    """Scrape (or reuse cached) WhoScored event data for a league season.

    With `team=None` (the default) every fixture in the league's schedule is
    collected, so any club's player can be looked up. Pass `team` to narrow
    this to just that team's fixtures (much faster — this is how
    `utd_games/` was originally built, and opponents will still show up in
    those specific games).

    Games already saved as `{save_dir}/{game_id}.csv` are read straight from
    disk; anything missing is pulled via soccerdata's WhoScored reader and
    written out so future calls don't re-scrape it. A full league season is
    ~380 games for a 20-team division — this will take a while the first
    time and is safe to interrupt and resume since each game is cached as
    it's scraped.
    """
    apply_whoscored_json_patch()
    if save_dir is None:
        save_dir = f"league_games/{league}_{season}".replace(" ", "_")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ws = sd.WhoScored(leagues=league, seasons=season, path_to_browser=browser_path)
    schedule = ws.read_schedule()
    if team is not None:
        schedule = schedule[
            (schedule["home_team"] == team) | (schedule["away_team"] == team)
        ]

    game_ids = schedule["game_id"].tolist()
    frames = []
    for i, game_id in enumerate(game_ids, start=1):
        path = save_dir / f"{game_id}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
            continue
        print(f"[{i}/{len(game_ids)}] scraping game {game_id}...", flush=True)
        game = ws.read_events(match_id=game_id).reset_index(drop=True)
        game.to_csv(path, index=False)
        frames.append(game)

    return pd.concat(frames, ignore_index=True)


def load_season_events(save_dir: str | Path = "utd_games") -> pd.DataFrame:
    """Concatenate every cached game csv in `save_dir` without touching the scraper."""
    paths = sorted(glob.glob(str(Path(save_dir) / "*.csv")))
    if not paths:
        raise FileNotFoundError(f"no game csvs found in {save_dir}")
    return pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)


def player_events(
    df: pd.DataFrame,
    player_name: str,
    team: str | None = None,
    require_unique: bool = True,
    exact: bool = False,
) -> pd.DataFrame:
    """Case-insensitive partial-match filter on the `player` column.

    League-wide data raises the odds of two different players matching the
    same partial name (common surnames, different clubs). By default this
    raises if the match isn't a single player — narrow with a fuller name or
    `team` to disambiguate, or pass `require_unique=False` to skip the check.
    Pass `exact=True` to match the `player` column exactly instead (used by
    the batch renderer, where names already come straight from the data).
    """
    if exact:
        matched = df[df["player"] == player_name]
    else:
        matched = df[df["player"].str.contains(player_name, case=False, na=False, regex=False)]
    if team is not None:
        matched = matched[matched["team"] == team]
    if matched.empty:
        raise ValueError(f"no events found for player matching {player_name!r} (team={team!r})")
    if require_unique:
        options = matched[["player", "team"]].drop_duplicates()
        if len(options) > 1:
            listing = ", ".join(f"{p} ({t})" for p, t in options.itertuples(index=False))
            raise ValueError(
                f"{player_name!r} matches multiple players: {listing}. "
                "Pass a fuller name or `team=` to disambiguate."
            )
    return matched


def final_third_touch_map(
    season_df: pd.DataFrame,
    player_name: str,
    team: str | None = None,
    exact: bool = False,
    ax=None,
    font=None,
    pitch_color: str = BG_COLOR,
    accent_color: str = ACCENT_COLOR,
):
    """Every final-third touch by `player_name` across all games in `season_df`."""
    events = player_events(season_df, player_name, team=team, exact=exact)
    touches = events[(events["is_touch"] == True) & (events["x"] >= FINAL_THIRD_X)]
    if touches.empty:
        raise ValueError(f"no final-third touches found for {player_name!r}")

    pitch = VerticalPitch(
        pitch_type="opta", half=True, pitch_color=pitch_color,
        line_color=LINE_COLOR, linewidth=1, line_zorder=2,
    )
    if ax is None:
        fig, ax = pitch.draw(figsize=(8, 8))
        fig.patch.set_facecolor(pitch_color)
    else:
        pitch.draw(ax=ax)
    ax.set_facecolor(pitch_color)
    ax.set_ylim(FINAL_THIRD_X, 100)

    if len(touches) >= 5:
        pitch.kdeplot(
            touches["x"], touches["y"], ax=ax, cmap="Reds", fill=True,
            levels=100, alpha=0.6, zorder=1,
        )
    pitch.scatter(
        touches["x"], touches["y"], ax=ax, s=45, color="white",
        edgecolors=accent_color, linewidth=1, alpha=0.85, zorder=2,
    )

    n_games = touches["game_id"].nunique()
    title_kwargs = {"font": font} if font is not None else {}
    ax.set_title(
        f"{player_name} — Final Third Touches ({len(touches)} across {n_games} games)",
        color="white", fontsize=13, **title_kwargs,
    )
    return ax


def full_pitch_touch_map(
    season_df: pd.DataFrame,
    player_name: str,
    team: str | None = None,
    exact: bool = False,
    ax=None,
    font=None,
    pitch_color: str = BG_COLOR,
    accent_color: str = ACCENT_COLOR,
):
    """Every touch by `player_name` anywhere on the pitch, across all games in `season_df`.

    Same style as `final_third_touch_map` (KDE density + scatter dots on a
    dark pitch) but over the full 0-100 x-range instead of just x >= 66.7.
    """
    events = player_events(season_df, player_name, team=team, exact=exact)
    touches = events[events["is_touch"] == True]
    if touches.empty:
        raise ValueError(f"no touches found for {player_name!r}")

    pitch = VerticalPitch(
        pitch_type="opta", pitch_color=pitch_color,
        line_color=LINE_COLOR, linewidth=1, line_zorder=2,
    )
    if ax is None:
        fig, ax = pitch.draw(figsize=(8, 11.5))
        fig.patch.set_facecolor(pitch_color)
    else:
        pitch.draw(ax=ax)
    ax.set_facecolor(pitch_color)

    if len(touches) >= 5:
        pitch.kdeplot(
            touches["x"], touches["y"], ax=ax, cmap="Reds", fill=True,
            levels=100, alpha=0.6, zorder=1,
        )
    pitch.scatter(
        touches["x"], touches["y"], ax=ax, s=45, color="white",
        edgecolors=accent_color, linewidth=1, alpha=0.85, zorder=2,
    )

    n_games = touches["game_id"].nunique()
    title_kwargs = {"font": font} if font is not None else {}
    ax.set_title(
        f"{player_name} — Touch Map ({len(touches)} across {n_games} games)",
        color="white", fontsize=13, **title_kwargs,
    )
    return ax


def avg_position_heatmap(
    season_df: pd.DataFrame,
    player_name: str,
    team: str | None = None,
    game_id: int | None = None,
    exact: bool = False,
    ax=None,
    font=None,
    pitch_color: str = BG_COLOR,
    accent_color: str = ACCENT_COLOR,
):
    """Heatmap of touch locations for one game (`game_id`), or the whole season if None."""
    events = player_events(season_df, player_name, team=team, exact=exact)
    if game_id is not None:
        events = events[events["game_id"] == game_id]
    touches = events[events["is_touch"] == True]
    if touches.empty:
        scope = f"game {game_id}" if game_id is not None else "the season"
        raise ValueError(f"no touches found for {player_name!r} in {scope}")

    pitch = Pitch(
        pitch_type="opta", pitch_color=pitch_color, line_color=LINE_COLOR,
        linewidth=1, line_zorder=2,
    )
    if ax is None:
        fig, ax = pitch.draw(figsize=(10, 7))
        fig.patch.set_facecolor(pitch_color)
    else:
        pitch.draw(ax=ax)
    ax.set_facecolor(pitch_color)

    bin_stat = pitch.bin_statistic(touches["x"], touches["y"], statistic="count", bins=(6, 5))
    bin_stat["statistic"] = gaussian_filter(bin_stat["statistic"], 1)
    pitch.heatmap(bin_stat, ax=ax, cmap="hot", zorder=0, alpha=0.85)

    avg_x, avg_y = touches["x"].median(), touches["y"].median()
    pitch.scatter(
        avg_x, avg_y, ax=ax, s=500, marker="o", color=pitch_color,
        edgecolors="white", linewidth=2.5, zorder=3,
    )

    n_games = touches["game_id"].nunique()
    scope_label = f"Game {game_id}" if game_id is not None else f"{n_games}-game season"
    title_kwargs = {"font": font} if font is not None else {}
    ax.set_title(
        f"{player_name} — Average Position Heatmap ({scope_label})",
        color="white", fontsize=13, **title_kwargs,
    )
    return ax


def _slug(text: str) -> str:
    return re.sub(r"[^\w\-]+", "_", text).strip("_") or "unknown"


def generate_all_player_maps(
    season_df: pd.DataFrame,
    out_dir: str | Path = "player_maps",
    font=None,
    min_touches: int = 10,
    skip_existing: bool = True,
    dpi: int = 150,
) -> pd.DataFrame:
    """Render + save a final-third touch map and average-position heatmap for every player.

    Iterates every distinct (player, team) pair in `season_df` with at least
    `min_touches` recorded touches, saving
    `{out_dir}/final_third/{team}/{player}_final_third.png` and
    `{out_dir}/{team}/{player}_avg_position.png` — final-third maps live in
    their own subfolder, separate from the average-position heatmaps.
    Players with zero final-third touches (e.g. most goalkeepers) still get
    an average-position heatmap, just no touch map. With `skip_existing=True`
    (default), a player/file already written is left alone — safe to re-run
    or resume a partial run.

    Returns a summary dataframe (also written to `{out_dir}/summary.csv`) of
    what was written, skipped, or errored per player, for later review.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    touches_all = season_df[(season_df["is_touch"] == True) & season_df["player"].notna()]
    counts = touches_all.groupby(["player", "team"]).size()
    pairs = sorted(counts[counts >= min_touches].index.tolist())

    rows = []
    for i, (player_name, team) in enumerate(pairs, start=1):
        ft_dir = out_dir / "final_third" / _slug(team)
        avg_dir = out_dir / _slug(team)
        ft_dir.mkdir(parents=True, exist_ok=True)
        avg_dir.mkdir(parents=True, exist_ok=True)
        ft_path = ft_dir / f"{_slug(player_name)}_final_third.png"
        avg_path = avg_dir / f"{_slug(player_name)}_avg_position.png"

        row = {
            "player": player_name,
            "team": team,
            "touches": int(counts[(player_name, team)]),
            "final_third_saved": False,
            "avg_position_saved": False,
            "error": None,
        }
        print(f"[{i}/{len(pairs)}] {player_name} ({team})", flush=True)

        if skip_existing and ft_path.exists():
            row["final_third_saved"] = True
        else:
            try:
                ax = final_third_touch_map(season_df, player_name, team=team, exact=True, font=font)
                ax.figure.savefig(ft_path, dpi=dpi, facecolor=ax.figure.get_facecolor())
                plt.close(ax.figure)
                row["final_third_saved"] = True
            except ValueError:
                pass  # no final-third touches for this player — expected for most keepers
            except Exception as e:  # noqa: BLE001
                row["error"] = f"final_third: {e}"

        if skip_existing and avg_path.exists():
            row["avg_position_saved"] = True
        else:
            try:
                ax = avg_position_heatmap(season_df, player_name, team=team, exact=True, font=font)
                ax.figure.savefig(avg_path, dpi=dpi, facecolor=ax.figure.get_facecolor())
                plt.close(ax.figure)
                row["avg_position_saved"] = True
            except Exception as e:  # noqa: BLE001
                row["error"] = f"{row['error']}; avg_position: {e}" if row["error"] else f"avg_position: {e}"

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    return summary


def build_league_player_maps(
    league: str = "ENG-Premier League",
    season: int | str = 2025,
    events_dir: str | Path | None = None,
    maps_dir: str | Path | None = None,
    min_touches: int = 10,
    browser_path: str = "/usr/bin/chromium",
    font=None,
) -> pd.DataFrame:
    """End to end: scrape a full league season, then render every player's maps.

    Safe to interrupt and re-run — both the game-event cache and the
    rendered-image cache are skip-if-exists.
    """
    league_slug = league.replace(" ", "_")
    if events_dir is None:
        events_dir = f"league_games/{league_slug}_{season}"
    if maps_dir is None:
        maps_dir = f"player_maps/{league_slug}_{season}"

    season_df = collect_league_events(
        league=league, season=season, team=None, save_dir=events_dir, browser_path=browser_path
    )
    print(f"collected {len(season_df)} events across {season_df['game_id'].nunique()} games", flush=True)

    return generate_all_player_maps(season_df, out_dir=maps_dir, font=font, min_touches=min_touches)


def generate_all_touch_maps(
    season_df: pd.DataFrame,
    out_dir: str | Path = "touch_maps",
    font=None,
    min_touches: int = 10,
    skip_existing: bool = True,
    dpi: int = 150,
) -> pd.DataFrame:
    """Render + save a full-pitch touch map for every player.

    Same style and iteration as `generate_all_player_maps` (one file per
    distinct (player, team) pair with at least `min_touches` touches,
    `skip_existing` resumability, a `summary.csv`), but `full_pitch_touch_map`
    instead of the final-third + average-position pair — saved to
    `{out_dir}/{team}/{player}_touch_map.png`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    touches_all = season_df[(season_df["is_touch"] == True) & season_df["player"].notna()]
    counts = touches_all.groupby(["player", "team"]).size()
    pairs = sorted(counts[counts >= min_touches].index.tolist())

    rows = []
    for i, (player_name, team) in enumerate(pairs, start=1):
        player_dir = out_dir / _slug(team)
        player_dir.mkdir(parents=True, exist_ok=True)
        path = player_dir / f"{_slug(player_name)}_touch_map.png"

        row = {
            "player": player_name,
            "team": team,
            "touches": int(counts[(player_name, team)]),
            "touch_map_saved": False,
            "error": None,
        }
        print(f"[{i}/{len(pairs)}] {player_name} ({team})", flush=True)

        if skip_existing and path.exists():
            row["touch_map_saved"] = True
        else:
            try:
                ax = full_pitch_touch_map(season_df, player_name, team=team, exact=True, font=font)
                ax.figure.savefig(path, dpi=dpi, facecolor=ax.figure.get_facecolor())
                plt.close(ax.figure)
                row["touch_map_saved"] = True
            except Exception as e:  # noqa: BLE001
                row["error"] = str(e)

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    return summary


def build_league_touch_maps(
    league: str = "ENG-Premier League",
    season: int | str = 2025,
    events_dir: str | Path | None = None,
    maps_dir: str | Path | None = None,
    min_touches: int = 10,
    browser_path: str = "/usr/bin/chromium",
    font=None,
) -> pd.DataFrame:
    """End to end: scrape a full league season, then render every player's full-pitch touch map.

    Safe to interrupt and re-run — both the game-event cache and the
    rendered-image cache are skip-if-exists.
    """
    league_slug = league.replace(" ", "_")
    if events_dir is None:
        events_dir = f"league_games/{league_slug}_{season}"
    if maps_dir is None:
        maps_dir = f"touch_maps/{league_slug}_{season}"

    season_df = collect_league_events(
        league=league, season=season, team=None, save_dir=events_dir, browser_path=browser_path
    )
    print(f"collected {len(season_df)} events across {season_df['game_id'].nunique()} games", flush=True)

    return generate_all_touch_maps(season_df, out_dir=maps_dir, font=font, min_touches=min_touches)


if __name__ == "__main__":
    try:
        from pyfonts import load_google_font

        _font = load_google_font("DotGothic16")
        if hasattr(_font, "_pyfonts_provider_metadata"):
            del _font._pyfonts_provider_metadata
    except Exception:  # noqa: BLE001
        _font = None

    result = build_league_player_maps(season=2025, font=_font)
    n_ok = ((result["final_third_saved"]) | (result["avg_position_saved"])).sum()
    print(f"done: {n_ok}/{len(result)} players have at least one map saved")
    print(result[result["error"].notna()])
