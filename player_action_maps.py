"""Event-based action maps for any player: passes, take-ons, shots, defensive actions.

Same data source and visual style as `player_touchmaps.py` (cached WhoScored
event csvs, Opta 0-100 pitch, every team always attacking towards x=100) —
but where that module only looks at `is_touch` rows, these read the richer
`type` / `outcome_type` / `blocked_x` / `goal_mouth_y` columns to plot what a
player *did* with the ball (passing, dribbling, shooting) and against it
(tackles, interceptions, clearances, ...), not just where they touched it.

CLI (one player, saved as four separate PNGs):

    python player_action_maps.py "Bruno Fernandes" --events-dir league_games/ENG-Premier_League_2025 --out-dir bruno_maps

Batch job (every player in a whole cached league season — `touch_maps/`
already covers the touch map itself, see `player_touchmaps.py`):

    python collect_action_maps.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from mplsoccer import VerticalPitch

from player_touchmaps import (
    ACCENT_COLOR,
    BG_COLOR,
    LINE_COLOR,
    _slug,
    full_pitch_touch_map,
    load_season_events,
    player_events,
    resolve_player_identity,
)

SUCCESS_COLOR = "#00D2A0"
FAIL_COLOR = "#FF4C4C"
GOAL_COLOR = "#FFD400"
BLOCK_COLOR = "#999999"

# Aerial/Challenge/BallRecovery can be won by either side; `outcome_type`
# already reflects whether *this* player came out on top, so no extra
# filtering is needed beyond the type list. BlockedPass is excluded here —
# WhoScored logs it against the *passer* whose ball got blocked, not the
# blocking defender, so it belongs on the pass map instead.
DEFENSIVE_TYPES = ["Tackle", "Interception", "Clearance", "Aerial", "BallRecovery", "Challenge"]
PASS_TYPES = ["Pass", "BlockedPass"]
TAKEON_TYPES = ["TakeOn"]


def _title(ax, text: str, font=None) -> None:
    title_kwargs = {"font": font} if font is not None else {}
    ax.set_title(text, color="white", fontsize=12, **title_kwargs)


# Outcome-splitting helpers, one per action type. Each mirrors exactly the
# success/fail split that type's map renders as color (green/red, gold
# star, grey block, ...) — factored out here so `touchmap_similarity.py` can
# turn that same color-coded information into numeric features instead of
# pixels, rather than re-deriving (and risking drift from) the plotting logic.


def _pass_outcomes(passes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(completed, incomplete, blocked) — matches `pass_map`'s green/red/red-X coloring."""
    has_end = passes["end_x"].notna()
    completed = passes[has_end & (passes["outcome_type"] == "Successful")]
    incomplete = passes[has_end & (passes["outcome_type"] == "Unsuccessful")]
    blocked = passes[~has_end]
    return completed, incomplete, blocked


def _takeon_outcomes(takeons: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(won, lost) — matches `takeon_map`'s green/red coloring."""
    won = takeons[takeons["outcome_type"] == "Successful"]
    lost = takeons[takeons["outcome_type"] == "Unsuccessful"]
    return won, lost


def _shot_outcome_kind(row: pd.Series) -> str:
    """'goal' / 'ontarget' / 'offtarget' / 'blocked' — matches `_shot_style`'s coloring.

    `type` decides goal/saved first: WhoScored's `blocked_x`/`blocked_y` is a
    generic "where the shot was stopped" coordinate populated for keeper
    saves too, not just outfield blocks, so checking it before `SavedShot`
    would misclassify every save as blocked.
    """
    if row["type"] == "Goal":
        return "goal"
    if row["type"] == "SavedShot":
        return "ontarget"
    if pd.notna(row["blocked_x"]):
        return "blocked"
    return "offtarget"


def _shot_outcomes(
    shots: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(goals, on-target saves, off-target misses, blocked) — matches `shot_map`'s coloring."""
    if shots.empty:
        return shots, shots, shots, shots
    kind = shots.apply(_shot_outcome_kind, axis=1)
    return (
        shots[kind == "goal"],
        shots[kind == "ontarget"],
        shots[kind == "offtarget"],
        shots[kind == "blocked"],
    )


def _defensive_outcomes(actions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(won, lost) — matches `defensive_action_map`'s green/red coloring."""
    won = actions[actions["outcome_type"] == "Successful"]
    lost = actions[actions["outcome_type"] == "Unsuccessful"]
    return won, lost


def pass_map(
    season_df,
    player_name: str,
    team: str | None = None,
    exact: bool = False,
    ax=None,
    font=None,
    pitch_color: str = BG_COLOR,
    success_color: str = SUCCESS_COLOR,
    fail_color: str = FAIL_COLOR,
    show_unsuccessful: bool = True,
):
    """Every pass by `player_name`: an arrow from origin to end point.

    Green = completed, red = incomplete (including blocked attempts, which
    have no end coordinate and are drawn as an X at the origin instead). A
    small dot marks the origin of every arrow so start and end are both
    explicit, not just implied by the arrow's direction.
    """
    events = player_events(season_df, player_name, team=team, exact=exact)
    passes = events[events["type"].isin(PASS_TYPES)]
    if passes.empty:
        raise ValueError(f"no passes found for {player_name!r}")

    pitch = VerticalPitch(
        pitch_type="opta", pitch_color=pitch_color, line_color=LINE_COLOR,
        linewidth=1, line_zorder=2,
    )
    if ax is None:
        fig, ax = pitch.draw(figsize=(8, 11.5))
        fig.patch.set_facecolor(pitch_color)
    else:
        pitch.draw(ax=ax)
    ax.set_facecolor(pitch_color)

    completed, incomplete, blocked = _pass_outcomes(passes)

    if not show_unsuccessful:
        incomplete = incomplete.iloc[0:0]
        blocked = blocked.iloc[0:0]

    for subset, color, alpha, z in (
        (incomplete, fail_color, 0.55, 2),
        (completed, success_color, 0.75, 3),
    ):
        if len(subset):
            pitch.arrows(
                subset["x"], subset["y"], subset["end_x"], subset["end_y"],
                ax=ax, color=color, width=1.3, headwidth=5, headlength=5,
                alpha=alpha, zorder=z,
            )
            pitch.scatter(
                subset["x"], subset["y"], ax=ax, s=16, color=color,
                edgecolors="white", linewidth=0.4, alpha=alpha + 0.1, zorder=z + 1,
            )
    if len(blocked):
        pitch.scatter(
            blocked["x"], blocked["y"], ax=ax, s=70, marker="x",
            color=fail_color, linewidth=2, zorder=3,
        )

    n_games = passes["game_id"].nunique()
    n_completed, n_total = len(completed), len(passes)
    pct = 100 * n_completed / n_total if n_total else 0.0
    _title(
        ax,
        f"{player_name} — Pass Map ({n_completed}/{n_total}, {pct:.0f}%, {n_games} games)",
        font=font,
    )
    return ax


def _dribble_segments(player_events_df: pd.DataFrame) -> pd.DataFrame:
    """One row per take-on with an inferred end point.

    WhoScored logs a `TakeOn` as a single (x, y) skill-move location with no
    end coordinate. For a *successful* take-on we approximate how far it
    carried the player by using their own next action in the same game
    (the next pass/carry/shot necessarily starts from wherever the dribble
    ended). A *failed* take-on didn't advance the ball, so its end point is
    just its own start point — plotted as a stopped run, not an arrow.
    """
    ev = player_events_df.sort_values(
        ["game_id", "period", "minute", "second"], kind="stable"
    ).reset_index(drop=True)
    takeons = ev[ev["type"].isin(TAKEON_TYPES)]

    rows = []
    for idx, row in takeons.iterrows():
        end_x, end_y = row["x"], row["y"]
        if row["outcome_type"] == "Successful":
            same_game_after = ev[(ev["game_id"] == row["game_id"]) & (ev.index > idx)]
            if len(same_game_after):
                nxt = same_game_after.iloc[0]
                end_x, end_y = nxt["x"], nxt["y"]
        rows.append(
            {
                "x": row["x"], "y": row["y"], "end_x": end_x, "end_y": end_y,
                "outcome_type": row["outcome_type"], "game_id": row["game_id"],
            }
        )
    return pd.DataFrame(rows)


def takeon_map(
    season_df,
    player_name: str,
    team: str | None = None,
    exact: bool = False,
    ax=None,
    font=None,
    pitch_color: str = BG_COLOR,
    success_color: str = SUCCESS_COLOR,
    fail_color: str = FAIL_COLOR,
):
    """Every take-on (dribble attempt) by `player_name`.

    Green arrow = beat the defender, running to wherever the player's next
    action started (an estimate of how far the take-on carried them); red X
    = lost the ball right there.
    """
    events = player_events(season_df, player_name, team=team, exact=exact)
    if events[events["type"].isin(TAKEON_TYPES)].empty:
        raise ValueError(f"no take-ons found for {player_name!r}")
    takeons = _dribble_segments(events)

    pitch = VerticalPitch(
        pitch_type="opta", pitch_color=pitch_color, line_color=LINE_COLOR,
        linewidth=1, line_zorder=2,
    )
    if ax is None:
        fig, ax = pitch.draw(figsize=(8, 11.5))
        fig.patch.set_facecolor(pitch_color)
    else:
        pitch.draw(ax=ax)
    ax.set_facecolor(pitch_color)

    won, lost = _takeon_outcomes(takeons)

    if len(lost):
        pitch.scatter(
            lost["x"], lost["y"], ax=ax, s=90, marker="X", color=fail_color,
            edgecolors="white", linewidth=0.8, alpha=0.85, zorder=2,
        )
    if len(won):
        pitch.arrows(
            won["x"], won["y"], won["end_x"], won["end_y"], ax=ax, color=success_color,
            width=1.6, headwidth=5, headlength=5, alpha=0.8, zorder=3,
        )
        pitch.scatter(
            won["x"], won["y"], ax=ax, s=30, color=success_color,
            edgecolors="white", linewidth=0.5, alpha=0.9, zorder=4,
        )

    n_games = takeons["game_id"].nunique()
    n_won, n_total = len(won), len(takeons)
    pct = 100 * n_won / n_total if n_total else 0.0
    _title(
        ax,
        f"{player_name} — Take-Ons ({n_won}/{n_total}, {pct:.0f}%, {n_games} games)",
        font=font,
    )
    return ax


_SHOT_STYLE_BY_KIND = {
    "goal": (GOAL_COLOR, "*", 320),
    "blocked": (BLOCK_COLOR, "s", 90),
    "ontarget": (SUCCESS_COLOR, "o", 130),
    "offtarget": (FAIL_COLOR, "o", 100),
}


def _shot_style(row: pd.Series) -> tuple[str, str, int]:
    """(color, marker, size) for one shot row."""
    return _SHOT_STYLE_BY_KIND[_shot_outcome_kind(row)]


def shot_map(
    season_df,
    player_name: str,
    team: str | None = None,
    exact: bool = False,
    ax=None,
    font=None,
    pitch_color: str = BG_COLOR,
    accent_color: str = ACCENT_COLOR,
):
    """Every shot by `player_name` on the attacking half.

    Gold star = goal, teal circle = saved (on target), red circle = missed
    (off target), grey square = blocked — each with an arrow from the shot's
    origin to where it was headed (goal mouth for on-target attempts, the
    blocker's location for blocked ones), so start and end are both explicit.
    """
    events = player_events(season_df, player_name, team=team, exact=exact)
    shots = events[events["is_shot"] == True]  # noqa: E712
    if shots.empty:
        raise ValueError(f"no shots found for {player_name!r}")

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

    for _, row in shots.iterrows():
        color, marker, size = _shot_style(row)
        blocked = pd.notna(row["blocked_x"])
        end_x = row["blocked_x"] if blocked else 100.0
        end_y = row["blocked_y"] if blocked else row["goal_mouth_y"]
        if pd.notna(end_y):
            pitch.arrows(
                row["x"], row["y"], end_x, end_y, ax=ax, color=color,
                width=1.1, headwidth=4, headlength=4, alpha=0.55, zorder=1,
            )
        pitch.scatter(
            row["x"], row["y"], ax=ax, s=size, marker=marker, color=color,
            edgecolors="white", linewidth=0.8, alpha=0.9, zorder=2,
        )

    n_games = shots["game_id"].nunique()
    n_goals = int((shots["type"] == "Goal").sum())
    _title(
        ax,
        f"{player_name} — Shot Map ({n_goals}G / {len(shots)} shots, {n_games} games)",
        font=font,
    )
    return ax


def defensive_action_map(
    season_df,
    player_name: str,
    team: str | None = None,
    exact: bool = False,
    ax=None,
    font=None,
    pitch_color: str = BG_COLOR,
    success_color: str = SUCCESS_COLOR,
    fail_color: str = FAIL_COLOR,
):
    """Every tackle / interception / clearance / aerial / recovery / challenge by `player_name`.

    Same touch-map look (KDE density + white dots on a dark pitch) as
    `full_pitch_touch_map`, but each dot's edge is colored by outcome:
    green = won (successful tackle, interception, clearance, aerial duel,
    ...), red = lost.
    """
    events = player_events(season_df, player_name, team=team, exact=exact)
    actions = events[events["type"].isin(DEFENSIVE_TYPES)]
    if actions.empty:
        raise ValueError(f"no defensive actions found for {player_name!r}")

    pitch = VerticalPitch(
        pitch_type="opta", pitch_color=pitch_color, line_color=LINE_COLOR,
        linewidth=1, line_zorder=2,
    )
    if ax is None:
        fig, ax = pitch.draw(figsize=(8, 11.5))
        fig.patch.set_facecolor(pitch_color)
    else:
        pitch.draw(ax=ax)
    ax.set_facecolor(pitch_color)

    if len(actions) >= 5:
        pitch.kdeplot(
            actions["x"], actions["y"], ax=ax, cmap="Blues", fill=True,
            levels=100, alpha=0.5, zorder=1,
        )

    won, lost = _defensive_outcomes(actions)
    if len(lost):
        pitch.scatter(
            lost["x"], lost["y"], ax=ax, s=45, color="white",
            edgecolors=fail_color, linewidth=1.6, alpha=0.9, zorder=2,
        )
    if len(won):
        pitch.scatter(
            won["x"], won["y"], ax=ax, s=45, color="white",
            edgecolors=success_color, linewidth=1.6, alpha=0.9, zorder=3,
        )

    n_games = actions["game_id"].nunique()
    n_won, n_total = len(won), len(actions)
    pct = 100 * n_won / n_total if n_total else 0.0
    _title(
        ax,
        f"{player_name} — Defensive Actions ({n_won}/{n_total} won, {pct:.0f}%, {n_games} games)",
        font=font,
    )
    return ax


# (label, plotting fn, filename suffix) — shared by the single-player CLI
# below and by `player_profile.py`'s Step 1 target-map saving.
MAP_SPECS = [
    ("pass", pass_map, "pass_map"),
    ("takeon", takeon_map, "takeon_map"),
    ("shot", shot_map, "shot_map"),
    ("defensive", defensive_action_map, "defensive_map"),
]


def _generate_all_maps(
    season_df,
    plot_fn,
    out_dir: str | Path,
    font=None,
    min_touches: int = 10,
    skip_existing: bool = True,
    dpi: int = 150,
    filename_suffix: str = "map",
    player: str | None = None,
    team: str | None = None,
) -> pd.DataFrame:
    """Render + save `plot_fn`'s map for every (player, team) with >= `min_touches` touches.

    Same iteration, skip-if-exists resumability, and `summary.csv` output as
    `player_touchmaps.generate_all_touch_maps` — one file per player at
    `{out_dir}/{team}/{player}_{filename_suffix}.png`. A player with no
    events of this action type (a keeper's shots, a center-back's take-ons)
    is skipped without being treated as an error.

    Pass `player=` (and optionally `team=`) to render only that resolved
    identity; `min_touches` is ignored in that case.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    touches_all = season_df[(season_df["is_touch"] == True) & season_df["player"].notna()]  # noqa: E712
    counts = touches_all.groupby(["player", "team"]).size()
    if player is not None:
        resolved_player, resolved_team = resolve_player_identity(season_df, player, team=team)
        pairs = [(resolved_player, resolved_team)]
    else:
        pairs = sorted(counts[counts >= min_touches].index.tolist())

    rows = []
    for i, (player_name, player_team) in enumerate(pairs, start=1):
        player_dir = out_dir / _slug(player_team)
        player_dir.mkdir(parents=True, exist_ok=True)
        path = player_dir / f"{_slug(player_name)}_{filename_suffix}.png"

        touch_count = (
            int(counts[(player_name, player_team)])
            if (player_name, player_team) in counts.index
            else 0
        )
        row = {
            "player": player_name, "team": player_team, "touches": touch_count,
            "saved": False, "error": None,
        }
        print(f"[{i}/{len(pairs)}] {filename_suffix}: {player_name} ({player_team})", flush=True)

        if skip_existing and path.exists():
            row["saved"] = True
        else:
            try:
                ax = plot_fn(season_df, player_name, team=player_team, exact=True, font=font)
                ax.figure.savefig(path, dpi=dpi, facecolor=ax.figure.get_facecolor())
                plt.close(ax.figure)
                row["saved"] = True
            except ValueError:
                pass
            except Exception as e:  # noqa: BLE001
                row["error"] = str(e)

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    return summary


def generate_all_pass_maps(
    season_df, out_dir: str | Path = "pass_maps", font=None,
    min_touches: int = 10, skip_existing: bool = True, dpi: int = 150,
    player: str | None = None, team: str | None = None,
) -> pd.DataFrame:
    """Pass map for every player. See `_generate_all_maps`."""
    return _generate_all_maps(
        season_df, pass_map, out_dir, font, min_touches, skip_existing, dpi, "pass_map",
        player=player, team=team,
    )


def generate_all_takeon_maps(
    season_df, out_dir: str | Path = "takeon_maps", font=None,
    min_touches: int = 10, skip_existing: bool = True, dpi: int = 150,
    player: str | None = None, team: str | None = None,
) -> pd.DataFrame:
    """Take-on/dribble map for every player. See `_generate_all_maps`."""
    return _generate_all_maps(
        season_df, takeon_map, out_dir, font, min_touches, skip_existing, dpi, "takeon_map",
        player=player, team=team,
    )


def generate_all_shot_maps(
    season_df, out_dir: str | Path = "shot_maps", font=None,
    min_touches: int = 10, skip_existing: bool = True, dpi: int = 150,
    player: str | None = None, team: str | None = None,
) -> pd.DataFrame:
    """Shot map for every player. See `_generate_all_maps`."""
    return _generate_all_maps(
        season_df, shot_map, out_dir, font, min_touches, skip_existing, dpi, "shot_map",
        player=player, team=team,
    )


def generate_all_defensive_maps(
    season_df, out_dir: str | Path = "defensive_maps", font=None,
    min_touches: int = 10, skip_existing: bool = True, dpi: int = 150,
    player: str | None = None, team: str | None = None,
) -> pd.DataFrame:
    """Defensive-action map for every player. See `_generate_all_maps`."""
    return _generate_all_maps(
        season_df, defensive_action_map, out_dir, font, min_touches, skip_existing, dpi,
        "defensive_map", player=player, team=team,
    )


def build_league_action_maps(
    league: str = "ENG-Premier League",
    season: int | str = 2025,
    events_dir: str | Path | None = None,
    maps_root: str | Path = ".",
    min_touches: int = 10,
    font=None,
    player: str | None = None,
    team: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Pass/take-on/shot/defensive-action maps for every player in one cached league season.

    Reuses the WhoScored event csvs already cached under `league_games/`
    (built by `player_touchmaps.collect_league_events` / `touch_maps.py`) —
    doesn't scrape anything itself, so the league must already be collected.
    Mirrors `player_touchmaps.build_league_touch_maps`'s directory layout,
    one subtree per action type: `{maps_root}/{pass,takeon,shot,defensive}_maps/{league}_{season}/`.

    Pass `player=` to render only that one resolved identity.
    """
    league_slug = league.replace(" ", "_")
    if events_dir is None:
        events_dir = f"league_games/{league_slug}_{season}"
    season_df = load_season_events(events_dir)
    print(
        f"loaded {len(season_df)} events across {season_df['game_id'].nunique()} games "
        f"for {league} {season}",
        flush=True,
    )

    generators = {
        "pass": generate_all_pass_maps,
        "takeon": generate_all_takeon_maps,
        "shot": generate_all_shot_maps,
        "defensive": generate_all_defensive_maps,
    }
    summaries = {}
    for name, generate_fn in generators.items():
        out_dir = Path(maps_root) / f"{name}_maps" / f"{league_slug}_{season}"
        summaries[name] = generate_fn(
            season_df,
            out_dir=out_dir,
            font=font,
            min_touches=min_touches,
            player=player,
            team=team,
        )
    return summaries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pass / take-on / shot / defensive-action maps for one player, as separate PNGs."
    )
    parser.add_argument("player", help="Target player name (partial, case-insensitive match)")
    parser.add_argument("--team", default=None, help="Disambiguate if the name matches multiple players")
    parser.add_argument("--events-dir", default="utd_games", help="Cached season events directory")
    parser.add_argument(
        "--out-dir", default=None,
        help="Save each map as its own PNG in this directory instead of showing them",
    )
    args = parser.parse_args()

    season_df = load_season_events(args.events_dir)
    for name, fn, suffix in MAP_SPECS:
        try:
            ax = fn(season_df, args.player, team=args.team)
        except ValueError as e:
            print(f"{name}: {e}")
            continue
        if args.out_dir:
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{_slug(args.player)}_{suffix}.png"
            ax.figure.savefig(path, dpi=150, facecolor=ax.figure.get_facecolor())
            print(f"{name} map saved to {path}")
        else:
            plt.show()
