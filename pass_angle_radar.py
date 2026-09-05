"""Pass-angle radar: directional tendency of a player's passes.

Uses the same WhoScored event csvs as `player_action_maps.py`. Each pass
with an end coordinate is converted to a real-world angle (Opta 0-100
scaled onto a 105×68 m pitch, team always attacking towards x=100):

    0°   = forward (towards the opponent's goal)
    90°  = the player's right
    180° = backward
    270° = the player's left

Those angles are binned into an 8-spoke radar so the shape is the
*tendency* (share of passes in each direction), not raw volume.

CLI (one player):

    python pass_angle_radar.py "Bruno Fernandes"
    python pass_angle_radar.py "Bruno Fernandes" --team "Man Utd" --out bruno_angles.png

Batch (every player in every collected league — default when run with no args,
same skip-if-exists layout as `touch_maps.py`):

    python pass_angle_radar.py
    python pass_angle_radar.py --league "ENG-Premier League"
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from player_action_maps import PASS_TYPES, SUCCESS_COLOR
from player_touchmaps import (
    BG_COLOR,
    LINE_COLOR,
    _slug,
    find_player_in_leagues,
    load_season_events,
    player_events,
    resolve_player_identity,
)
from touchmap_similarity import DEFAULT_LEAGUES, DEFAULT_SEASON

# Opta is 0-100 on both axes; angles need real metres or a 10-unit "diagonal"
# in x is much longer on the grass than the same 10 units in y.
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
MIN_PASS_LENGTH_M = 0.5

BIN_LABELS = (
    "Forward",
    "Fwd-Right",
    "Right",
    "Back-Right",
    "Backward",
    "Back-Left",
    "Left",
    "Fwd-Left",
)
N_BINS = len(BIN_LABELS)
BIN_WIDTH_DEG = 360.0 / N_BINS  # 45°
# Quadrants are 90° slices centred on the four cardinals.
QUAD_LABELS = ("forward", "right", "backward", "left")

# Compact numeric view of the radar for similarity (stage-2 RRF, etc.).
# 8-bin shares capture the shape; concentration / length / completion are
# the extra knobs; mean direction is encoded as (cos, sin) so 359° is
# next to 1° instead of wrapping around.
PASS_ANGLE_FEATURE_NAMES = (
    *(f"share_{label}" for label in BIN_LABELS),
    "concentration",
    "mean_length_m",
    "completion_rate",
    "mean_angle_cos",
    "mean_angle_sin",
)

DEFAULT_MIN_PASSES = 20
DEFAULT_OUT_ROOT = "pass_angle_radars"


def _complete_image(path: Path) -> bool:
    """True only if `path` is a non-empty file — 0-byte stubs from a
    crashed `savefig` must not count as already rendered."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _load_font():
    try:
        from pyfonts import load_google_font

        font = load_google_font("DotGothic16")
        if hasattr(font, "_pyfonts_provider_metadata"):
            del font._pyfonts_provider_metadata
        return font
    except Exception:  # noqa: BLE001
        return None


def _title_kwargs(font) -> dict:
    return {"font": font} if font is not None else {}


def passes_with_end(events: pd.DataFrame) -> pd.DataFrame:
    """Pass rows that have a usable end coordinate (completed + incomplete).

    Blocked attempts have no `end_x`/`end_y` in WhoScored, so they can't
    contribute an angle — same rows `pass_map` draws as an X at the origin.
    """
    passes = events[events["type"].isin(PASS_TYPES)]
    return passes[passes["end_x"].notna() & passes["end_y"].notna()].copy()


def add_pass_angles(passes: pd.DataFrame) -> pd.DataFrame:
    """Attach `length_m` and `angle_deg` (0=forward, clockwise=right, [0, 360))."""
    out = passes.copy()
    dx_m = (out["end_x"] - out["x"]) * (PITCH_LENGTH_M / 100.0)
    dy_right_m = (out["y"] - out["end_y"]) * (PITCH_WIDTH_M / 100.0)
    length = np.hypot(dx_m, dy_right_m)
    angle = np.degrees(np.arctan2(dy_right_m.to_numpy(), dx_m.to_numpy()))
    out["length_m"] = length
    out["angle_deg"] = np.mod(angle, 360.0)
    return out[out["length_m"] >= MIN_PASS_LENGTH_M]


def _bin_index(angle_deg: np.ndarray) -> np.ndarray:
    """Map degrees in [0, 360) onto 8 bins centred on the compass points."""
    shifted = np.mod(angle_deg + BIN_WIDTH_DEG / 2.0, 360.0)
    return np.minimum((shifted / BIN_WIDTH_DEG).astype(int), N_BINS - 1)


def _circular_stats(angle_deg: np.ndarray) -> tuple[float, float]:
    """(mean angle in [0, 360), mean resultant length 0–1)."""
    rad = np.deg2rad(angle_deg)
    c, s = float(np.mean(np.cos(rad))), float(np.mean(np.sin(rad)))
    r = float(np.hypot(c, s))
    mean = float(np.mod(np.degrees(np.arctan2(s, c)), 360.0))
    return mean, r


def summarize_pass_angles(angled: pd.DataFrame) -> dict:
    """Numerical pass-angle profile from rows that already have `angle_deg`."""
    n = len(angled)
    if n == 0:
        raise ValueError("no passes with a usable angle")

    completed = (angled["outcome_type"] == "Successful").to_numpy()
    n_completed = int(completed.sum())
    angles = angled["angle_deg"].to_numpy()
    lengths = angled["length_m"].to_numpy()
    bins = _bin_index(angles)
    mean_angle, concentration = _circular_stats(angles)

    bin_rows = []
    for i, label in enumerate(BIN_LABELS):
        in_bin = bins == i
        n_bin = int(in_bin.sum())
        n_ok = int((in_bin & completed).sum())
        bin_rows.append(
            {
                "label": label,
                "n": n_bin,
                "share": n_bin / n,
                "completed": n_ok,
                "completion_pct": (100.0 * n_ok / n_bin) if n_bin else 0.0,
                "mean_length_m": float(lengths[in_bin].mean()) if n_bin else 0.0,
            }
        )

    # 90° quadrants centred on the same cardinals as the 8-bin radar.
    quad_shifted = np.mod(angles + 45.0, 360.0)
    quad_idx = np.minimum((quad_shifted / 90.0).astype(int), 3)
    quadrants = {name: float((quad_idx == i).mean()) for i, name in enumerate(QUAD_LABELS)}

    n_games = int(angled["game_id"].nunique()) if "game_id" in angled.columns else None
    return {
        "n_angled": n,
        "n_games": n_games,
        "n_completed": n_completed,
        "completion_pct": 100.0 * n_completed / n,
        "mean_length_m": float(angled["length_m"].mean()),
        "mean_angle_deg": mean_angle,
        "concentration": concentration,
        "bins": bin_rows,
        "quadrants": quadrants,
    }


def player_pass_angle_frame(
    season_df: pd.DataFrame,
    player_name: str,
    team: str | None = None,
    exact: bool = False,
) -> tuple[pd.DataFrame, int]:
    """`(angled_passes, n_pass_events)` including blocked attempts in the count."""
    events = player_events(season_df, player_name, team=team, exact=exact)
    n_passes = int(events["type"].isin(PASS_TYPES).sum())
    angled = add_pass_angles(passes_with_end(events))
    if angled.empty:
        raise ValueError(f"no directed passes found for {player_name!r}")
    return angled, n_passes


def pass_angle_stats(
    season_df: pd.DataFrame,
    player_name: str,
    team: str | None = None,
    exact: bool = False,
) -> dict:
    """Directional tendency numbers for one player."""
    angled, n_passes = player_pass_angle_frame(
        season_df, player_name, team=team, exact=exact
    )
    stats = summarize_pass_angles(angled)
    resolved = angled[["player", "team"]].drop_duplicates().iloc[0]
    stats["player"] = str(resolved["player"])
    stats["team"] = str(resolved["team"])
    stats["n_passes"] = n_passes
    return stats


def pass_angle_feature_vector(stats: dict) -> np.ndarray:
    """Length-`PASS_ANGLE_FEATURE_NAMES` vector from a `summarize_pass_angles` dict."""
    shares = [float(b["share"]) for b in stats["bins"]]
    mean_rad = np.deg2rad(float(stats["mean_angle_deg"]))
    return np.array(
        [
            *shares,
            float(stats["concentration"]),
            float(stats["mean_length_m"]),
            float(stats["completion_pct"]) / 100.0,
            float(np.cos(mean_rad)),
            float(np.sin(mean_rad)),
        ],
        dtype=float,
    )


def pass_angle_features_for_players(
    season_df: pd.DataFrame,
    pairs: list[tuple[str, str]],
) -> tuple[dict[tuple[str, str], dict], dict[tuple[str, str], np.ndarray]]:
    """Pass-angle stats + feature vectors for each `(player, team)` in `pairs`.

    One pass over the event table (filtered to those names), then a groupby —
    much cheaper than calling `pass_angle_stats` per player on a full season.
    Pairs with no directed passes are omitted.
    """
    if not pairs:
        return {}, {}
    names = {p for p, _ in pairs}
    teams = {t for _, t in pairs}
    events = season_df[
        season_df["player"].isin(names) & season_df["team"].isin(teams)
    ]
    angled_all = add_pass_angles(passes_with_end(events))
    if angled_all.empty:
        return {}, {}

    pair_set = set(pairs)
    n_passes = (
        events[events["type"].isin(PASS_TYPES)]
        .groupby(["player", "team"], sort=False)
        .size()
    )

    stats_by: dict[tuple[str, str], dict] = {}
    vec_by: dict[tuple[str, str], np.ndarray] = {}
    for (player, team), grp in angled_all.groupby(["player", "team"], sort=False):
        key = (str(player), str(team))
        if key not in pair_set:
            continue
        stats = summarize_pass_angles(grp)
        stats["player"] = key[0]
        stats["team"] = key[1]
        stats["n_passes"] = int(n_passes.get(key, len(grp)))
        stats_by[key] = stats
        vec_by[key] = pass_angle_feature_vector(stats)
    return stats_by, vec_by


def format_pass_angle_summary(stats: dict) -> str:
    """A few compact lines for CLI / `player_profile` reports."""
    q = stats["quadrants"]
    bin_bits = "  ".join(
        f"{b['label']} {100 * b['share']:.0f}%" for b in stats["bins"]
    )
    games = f", {stats['n_games']} games" if stats.get("n_games") else ""
    lines = [
        f"  {stats['n_angled']} directed passes ({stats['n_completed']}/{stats['n_angled']}, "
        f"{stats['completion_pct']:.0f}%{games})",
        f"  Forward {100 * q['forward']:.0f}%  |  Right {100 * q['right']:.0f}%  |  "
        f"Backward {100 * q['backward']:.0f}%  |  Left {100 * q['left']:.0f}%",
        f"  {bin_bits}",
        f"  mean length {stats['mean_length_m']:.1f}m   "
        f"concentration {stats['concentration']:.2f} (0=even, 1=one direction)",
    ]
    return "\n".join(lines)


def pass_angle_radar(
    season_df,
    player_name: str,
    team: str | None = None,
    exact: bool = False,
    ax=None,
    font=None,
    pitch_color: str = BG_COLOR,
    accent_color: str = SUCCESS_COLOR,
    stats: dict | None = None,
):
    """8-spoke radar of the share of `player_name`'s passes in each direction.

    Forward is at 12 o'clock; clockwise is the player's right (same
    attacking-up orientation as the vertical action maps). Returns `ax`.
    """
    if stats is None:
        stats = pass_angle_stats(season_df, player_name, team=team, exact=exact)

    shares = np.array([b["share"] for b in stats["bins"]], dtype=float)
    theta = np.deg2rad(np.arange(0.0, 360.0, BIN_WIDTH_DEG))
    theta_closed = np.append(theta, theta[0])
    shares_closed = np.append(shares, shares[0])

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
        fig.patch.set_facecolor(pitch_color)
    ax.set_facecolor(pitch_color)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    rmax = max(0.35, float(shares.max()) * 1.15)
    ax.set_ylim(0, rmax)
    yticks = [t for t in (0.10, 0.20, 0.30, 0.40, 0.50) if t < rmax]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(t * 100)}%" for t in yticks], color=LINE_COLOR, fontsize=8)
    ax.set_xticks(theta)
    ax.set_xticklabels(BIN_LABELS, color="white", fontsize=10, **_title_kwargs(font))

    ax.plot(theta_closed, shares_closed, color=accent_color, linewidth=2.2, zorder=3)
    ax.fill(theta_closed, shares_closed, color=accent_color, alpha=0.32, zorder=2)
    ax.scatter(theta, shares, s=36, color=accent_color, edgecolors="white", linewidths=0.6, zorder=4)

    for th, share in zip(theta, shares):
        ax.annotate(
            f"{100 * share:.0f}%",
            xy=(th, share),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="white",
            fontsize=9,
            **_title_kwargs(font),
        )

    ax.spines["polar"].set_color(LINE_COLOR)
    ax.grid(color=LINE_COLOR, alpha=0.35, linewidth=0.7)
    ax.tick_params(axis="x", pad=12, colors="white")
    ax.tick_params(axis="y", colors=LINE_COLOR)

    n_games = stats.get("n_games")
    games_bit = f", {n_games} games" if n_games else ""
    ax.set_title(
        f"{stats.get('player', player_name)} — Pass Angle Tendency "
        f"({stats['n_angled']} passes, {stats['completion_pct']:.0f}%{games_bit})",
        color="white",
        fontsize=12,
        pad=18,
        **_title_kwargs(font),
    )
    return ax


def generate_all_pass_angle_radars(
    season_df,
    out_dir: str | Path = DEFAULT_OUT_ROOT,
    font=None,
    min_passes: int = DEFAULT_MIN_PASSES,
    skip_existing: bool = True,
    dpi: int = 150,
    player: str | None = None,
    team: str | None = None,
) -> pd.DataFrame:
    """Radar PNG for every (player, team) with enough directed passes.

    Same skip-if-exists layout as the action-map batch jobs:
    `{out_dir}/{team}/{player}_pass_angle_radar.png`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if player is not None:
        resolved_player, resolved_team = resolve_player_identity(season_df, player, team=team)
        pairs = [(resolved_player, resolved_team)]
    else:
        passes = season_df[
            season_df["type"].isin(PASS_TYPES)
            & season_df["player"].notna()
            & season_df["end_x"].notna()
        ]
        counts = passes.groupby(["player", "team"]).size()
        pairs = sorted(counts[counts >= min_passes].index.tolist())

    rows = []
    for i, (player_name, player_team) in enumerate(pairs, start=1):
        player_dir = out_dir / _slug(player_team)
        player_dir.mkdir(parents=True, exist_ok=True)
        path = player_dir / f"{_slug(player_name)}_pass_angle_radar.png"
        row = {
            "player": player_name,
            "team": player_team,
            "saved": False,
            "error": None,
            "path": str(path),
        }
        print(f"[{i}/{len(pairs)}] pass_angle: {player_name} ({player_team})", flush=True)

        if skip_existing and _complete_image(path):
            row["saved"] = True
            rows.append(row)
            continue
        tmp = path.with_name(path.name + ".tmp")
        try:
            ax = pass_angle_radar(
                season_df, player_name, team=player_team, exact=True, font=font
            )
            ax.figure.savefig(
                tmp, dpi=dpi, facecolor=ax.figure.get_facecolor(), format="png"
            )
            plt.close(ax.figure)
            if not _complete_image(tmp):
                raise RuntimeError("savefig wrote an empty PNG")
            tmp.replace(path)
            row["saved"] = True
        except ValueError as e:
            # Expected when a player has no usable directed passes.
            # Other ValueErrors (bad save format, matplotlib, …) must not
            # look like a successful skip.
            msg = str(e)
            if "no directed" not in msg.lower() and "no events found" not in msg.lower():
                row["error"] = msg
        except Exception as e:  # noqa: BLE001
            row["error"] = str(e)
        finally:
            plt.close("all")
            tmp.unlink(missing_ok=True)
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    return summary


def build_league_pass_angle_radars(
    league: str = "ENG-Premier League",
    season: int | str = DEFAULT_SEASON,
    events_dir: str | Path | None = None,
    maps_root: str | Path = ".",
    min_passes: int = DEFAULT_MIN_PASSES,
    font=None,
    player: str | None = None,
    team: str | None = None,
) -> pd.DataFrame:
    """Radars for every (or one) player in a cached league season."""
    league_slug = league.replace(" ", "_")
    if events_dir is None:
        events_dir = f"league_games/{league_slug}_{season}"
    season_df = load_season_events(events_dir)
    print(
        f"loaded {len(season_df)} events across {season_df['game_id'].nunique()} games "
        f"for {league} {season}",
        flush=True,
    )
    out_dir = Path(maps_root) / DEFAULT_OUT_ROOT / f"{league_slug}_{season}"
    return generate_all_pass_angle_radars(
        season_df,
        out_dir=out_dir,
        font=font,
        min_passes=min_passes,
        player=player,
        team=team,
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Radar chart of a player's pass-angle tendency (or every player's)."
    )
    parser.add_argument(
        "player",
        nargs="?",
        default=None,
        help="Target player (partial, case-insensitive). Omit to batch-render every player.",
    )
    parser.add_argument("--team", default=None, help="Disambiguate if the name matches several players")
    parser.add_argument("--league", default=None, help="Restrict to one league")
    parser.add_argument("--events-dir", default=None, help="Cached season events directory")
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON)
    parser.add_argument("--out", default=None, help="Save the single-player radar to this PNG")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Render a radar for every player in the chosen league(s). Implied when no player is given.",
    )
    parser.add_argument("--min-passes", type=int, default=DEFAULT_MIN_PASSES)
    parser.add_argument("--maps-root", default=".")
    args = parser.parse_args()

    if args.player is None:
        args.all = True

    font = _load_font()

    if args.all:
        if args.events_dir:
            season_df = load_season_events(args.events_dir)
            out_dir = Path(args.maps_root) / DEFAULT_OUT_ROOT
            summary = generate_all_pass_angle_radars(
                season_df,
                out_dir=out_dir,
                font=font,
                min_passes=args.min_passes,
                player=args.player,
                team=args.team,
            )
            print(f"saved {int(summary['saved'].sum())}/{len(summary)} radars under {out_dir}")
            return 0

        leagues = [args.league] if args.league else list(DEFAULT_LEAGUES)
        player, team = args.player, args.team
        if player is not None:
            search = [args.league] if args.league else list(DEFAULT_LEAGUES)
            league, player, team = find_player_in_leagues(
                player, season=args.season, leagues=search, team=team
            )
            leagues = [league]
            print(f"resolved player: {player} ({team}) in {league} {args.season}", flush=True)

        for league in leagues:
            print(f"\n===== {league} {args.season} =====", flush=True)
            try:
                summary = build_league_pass_angle_radars(
                    league=league,
                    season=args.season,
                    font=font,
                    min_passes=args.min_passes,
                    maps_root=args.maps_root,
                    player=player,
                    team=team,
                )
            except Exception as e:  # noqa: BLE001
                print(f"!! {league} failed: {e}", flush=True)
                continue
            print(
                f"{league}: {int(summary['saved'].sum())}/{len(summary)} radars saved",
                flush=True,
            )
        return 0

    if args.events_dir:
        season_df = load_season_events(args.events_dir)
        resolved_player, resolved_team = resolve_player_identity(
            season_df, args.player, team=args.team
        )
    else:
        search = [args.league] if args.league else list(DEFAULT_LEAGUES)
        league, resolved_player, resolved_team = find_player_in_leagues(
            args.player, season=args.season, leagues=search, team=args.team
        )
        events_dir = f"league_games/{league.replace(' ', '_')}_{args.season}"
        season_df = load_season_events(events_dir)
        print(f"resolved player: {resolved_player} ({resolved_team}) in {league}", flush=True)

    stats = pass_angle_stats(season_df, resolved_player, team=resolved_team, exact=True)
    print(f"{stats['player']} ({stats['team']}) — pass angle tendency")
    print(format_pass_angle_summary(stats))

    ax = pass_angle_radar(
        season_df, resolved_player, team=resolved_team, exact=True, font=font, stats=stats
    )
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(path, dpi=150, facecolor=ax.figure.get_facecolor())
        plt.close(ax.figure)
        print(f"radar saved to {path}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
