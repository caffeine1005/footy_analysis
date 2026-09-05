"""Batch job: pass-angle radar for every player, every collected league.

Same layout as the other map trees (`touch_maps/`, `pass_maps/`, ...):

    pass_angle_radars/{league}_{season}/{team}/{player}_pass_angle_radar.png

Reuses the WhoScored event csvs already cached under `league_games/` —
doesn't scrape anything, so each league must already be collected there.

Safe to interrupt and resume: rendered images are skip-if-exists, and a
failure in one league doesn't block the others.

Run directly:

    python collect_pass_angle_radars.py
    python collect_pass_angle_radars.py "Bruno Fernandes"
    python collect_pass_angle_radars.py "Bruno Fernandes" --season 2025 --team "Man Utd"
    python collect_pass_angle_radars.py --season 2025 --league "ENG-Premier League"
"""

from __future__ import annotations

import argparse
import sys

from pass_angle_radar import DEFAULT_MIN_PASSES, build_league_pass_angle_radars
from player_touchmaps import find_player_in_leagues
from touchmap_similarity import DEFAULT_LEAGUES, DEFAULT_SEASON


def _load_font():
    try:
        from pyfonts import load_google_font

        font = load_google_font("DotGothic16")
        if hasattr(font, "_pyfonts_provider_metadata"):
            del font._pyfonts_provider_metadata
        return font
    except Exception:  # noqa: BLE001
        return None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build pass-angle radars from cached league events. "
            "Optionally restrict to one player and/or season."
        )
    )
    parser.add_argument(
        "player",
        nargs="?",
        default=None,
        help="Exact or unique player name — only build a radar for that player",
    )
    parser.add_argument(
        "--team",
        default=None,
        help="Disambiguate if the name matches multiple players",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=DEFAULT_SEASON,
        help=f"Season year (default: {DEFAULT_SEASON})",
    )
    parser.add_argument(
        "--league",
        default=None,
        help=(
            "Restrict to one league (e.g. 'ENG-Premier League'). "
            "Default: all collected leagues, or the league where --player is found."
        ),
    )
    parser.add_argument(
        "--min-passes",
        type=int,
        default=DEFAULT_MIN_PASSES,
        help=f"Skip players with fewer directed passes (default: {DEFAULT_MIN_PASSES})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    font = _load_font()

    player = args.player
    team = args.team
    season = args.season

    if player is not None:
        search_leagues = [args.league] if args.league else list(DEFAULT_LEAGUES)
        try:
            league, player, team = find_player_in_leagues(
                player, season=season, leagues=search_leagues, team=team
            )
        except ValueError as e:
            print(f"!! {e}", flush=True)
            return 1
        leagues = [league]
        print(f"resolved player: {player} ({team}) in {league} {season}", flush=True)
    else:
        leagues = [args.league] if args.league else list(DEFAULT_LEAGUES)

    for league in leagues:
        print(f"\n===== {league} {season} =====", flush=True)
        try:
            summary = build_league_pass_angle_radars(
                league=league,
                season=season,
                font=font,
                min_passes=args.min_passes,
                player=player,
                team=team,
            )
        except Exception as e:  # noqa: BLE001
            print(f"!! {league} failed: {e}", flush=True)
            continue
        n_ok = int(summary["saved"].sum())
        print(f"{league} pass_angle: {n_ok}/{len(summary)} players saved", flush=True)
        errors = summary[summary["error"].notna()]
        if not errors.empty:
            print(errors, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
