"""Batch job: full-pitch touch maps for every remaining top league.

Same as `touch_maps.py` (which already covered ENG-Premier League), but
looped over the rest of the "big 5" plus Eredivisie and Liga Portugal.
Each league is fully independent and skip-if-exists at both the event-scrape
and rendered-image layer, so this is safe to interrupt/resume, and a
failure in one league doesn't block the others.

Run directly:

    python collect_more_touchmaps.py
    python collect_more_touchmaps.py "Joshua Kimmich"
    python collect_more_touchmaps.py "Joshua Kimmich" --season 2025 --league "GER-Bundesliga"
    python collect_more_touchmaps.py --season 2024 --league "FRA-Ligue 1"
"""

from __future__ import annotations

import argparse
import sys

from player_touchmaps import build_league_touch_maps, find_player_in_leagues
from touchmap_similarity import DEFAULT_LEAGUES, DEFAULT_SEASON

# Default batch set: leagues not already covered by touch_maps.py (Prem).
LEAGUES = [
    "GER-Bundesliga",
    "FRA-Ligue 1",
    "NED-Eredivisie",
    "POR-Liga Portugal",
]


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
            "Build full-pitch touch maps (scrape/reuse league events, then render). "
            "Optionally restrict to one player and/or season."
        )
    )
    parser.add_argument(
        "player",
        nargs="?",
        default=None,
        help="Exact or unique player name — only build a map for that player",
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
            "Restrict to one league (e.g. 'GER-Bundesliga'). "
            "Default batch: remaining top leagues; with --player, searches all "
            "collected leagues unless --league is set."
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
            # No cached hit — if a league was given, scrape/build that league for the query.
            if args.league is None:
                print(f"!! {e}", flush=True)
                print(
                    "Tip: pass --league to scrape/build that league for this player.",
                    flush=True,
                )
                return 1
            league = args.league
            print(
                f"no cached match; building {league} {season} for query {player!r}",
                flush=True,
            )
        else:
            print(f"resolved player: {player} ({team}) in {league} {season}", flush=True)
        leagues = [league]
    else:
        leagues = [args.league] if args.league else list(LEAGUES)

    for league in leagues:
        print(f"\n===== {league} {season} =====", flush=True)
        try:
            result = build_league_touch_maps(
                league=league,
                season=season,
                font=font,
                player=player,
                team=team,
            )
        except Exception as e:  # noqa: BLE001
            print(f"!! {league} failed: {e}", flush=True)
            continue
        n_ok = result["touch_map_saved"].sum()
        print(f"{league} done: {n_ok}/{len(result)} players have a touch map saved", flush=True)
        errors = result[result["error"].notna()]
        if not errors.empty:
            print(errors, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
