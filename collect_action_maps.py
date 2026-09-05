"""Batch job: pass / take-on / shot / defensive-action maps for every player, every collected league.

Touch maps already exist for every player across all 7 leagues (see
`touch_maps.py` / `collect_more_touchmaps.py`, output under `touch_maps/`).
This does the same for the four other action types, reusing the WhoScored
event csvs already cached under `league_games/` — it doesn't scrape
anything, so each league must already be collected there.

Safe to interrupt and resume: rendered images are skip-if-exists (per map
type, per player), and a failure in one league doesn't block the others.

Run directly:

    python collect_action_maps.py
    python collect_action_maps.py "Bruno Fernandes"
    python collect_action_maps.py "Bruno Fernandes" --season 2025 --team "Man Utd"
    python collect_action_maps.py --season 2024 --league "ENG-Premier League"
"""

from __future__ import annotations

import argparse
import sys

from player_action_maps import build_league_action_maps
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
            "Build pass/take-on/shot/defensive-action maps from cached league events. "
            "Optionally restrict to one player and/or season."
        )
    )
    parser.add_argument(
        "player",
        nargs="?",
        default=None,
        help="Exact or unique player name — only build maps for that player",
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
            summaries = build_league_action_maps(
                league=league,
                season=season,
                font=font,
                player=player,
                team=team,
            )
        except Exception as e:  # noqa: BLE001
            print(f"!! {league} failed: {e}", flush=True)
            continue
        for name, summary in summaries.items():
            n_ok = summary["saved"].sum()
            print(f"{league} {name}: {n_ok}/{len(summary)} players saved", flush=True)
            errors = summary[summary["error"].notna()]
            if not errors.empty:
                print(errors, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
