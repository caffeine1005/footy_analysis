"""Backfill historical seasons: WhoScored match events and Sofascore league stats.

The app's pipeline needs both halves for a season — event csvs under
`league_games/{league}_{year}/` for the touch/action-map clustering, and a
Sofascore stats table for the quantitative stage — and they have to be the same
season, or a player's touch map and stats describe different years.

Seasons are named by their starting year throughout, the way soccerdata and the
existing folders already do: 2014 is 2014/15, stored in `league_games/*_2014`
and `sofascore_player_stats_1415.csv`.

Both halves are resumable. Events are written one csv per game and a game
already on disk is never fetched again; a Sofascore season whose table already
has every (league, position) chunk is skipped, and one with gaps is topped up
rather than re-scraped.

    python backfill_history.py status
    python backfill_history.py stats                       # 13/14 .. 24/25, all 7 leagues
    python backfill_history.py events --max-games 3        # quick timing trial
    python backfill_history.py events --prune-raw          # the long one
    python backfill_history.py events --from 2024 --to 2024 --league "ENG-Premier League"
"""

from __future__ import annotations

import argparse
import shutil
import time
import traceback
from pathlib import Path

import pandas as pd

import sofascore_similarity as sofa
from touchmap_similarity import DEFAULT_LEAGUES, _league_slug

EVENTS_ROOT = Path("league_games")
FIRST_SEASON = 2013
# 2025/26 is the live season and already collected by the existing scripts.
LAST_HISTORIC_SEASON = 2024
CURRENT_SEASON = 2025
RAW_CACHE = Path.home() / "soccerdata" / "data" / "WhoScored" / "events"

BROWSER_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/usr/bin/chromium",
    "/usr/bin/google-chrome",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def sofascore_season(year: int) -> str:
    """2015 -> '15/16'."""
    return f"{year % 100:02d}/{(year + 1) % 100:02d}"


def soccerdata_season(year: int) -> str:
    """2015 -> '1516', soccerdata's cache folder suffix."""
    return f"{year % 100:02d}{(year + 1) % 100:02d}"


def stats_path(year: int) -> Path:
    if year == CURRENT_SEASON:
        return sofa.DATA_PATH
    return Path(f"sofascore_player_stats_{soccerdata_season(year)}.csv")


def events_dir(league: str, year: int) -> Path:
    return EVENTS_ROOT / _league_slug(league, year)


def default_browser() -> str | None:
    found = next((p for p in BROWSER_CANDIDATES if Path(p).exists()), None)
    return found or shutil.which("chromium") or shutil.which("google-chrome")


# ---------------------------------------------------------------------------
# Sofascore
# ---------------------------------------------------------------------------


def _missing_stats_chunks(path: Path, leagues: list[str]) -> list[tuple[str, str]]:
    """(league, position) chunks with no rows, over the leagues asked for.

    Unlike `sofa.missing_chunks` this also counts a league that is absent
    altogether, since the backfill decides the league set rather than the file.
    """
    have = sofa.chunk_coverage(path)
    gaps = []
    for league in leagues:
        for _api, pos in sofa.POS_SCRAPE:
            n = int(have.loc[league, pos]) if league in have.index else 0
            if n == 0:
                gaps.append((league, pos))
    return gaps


def backfill_stats(years: list[int], leagues: list[str]) -> None:
    import sofascore_session

    def run():
        for year in years:
            season = sofascore_season(year)
            path = stats_path(year)
            if path.exists():
                gaps = _missing_stats_chunks(path, leagues)
                if not gaps:
                    print(f"[stats {season}] complete -> {path}", flush=True)
                    continue
                print(f"[stats {season}] topping up {len(gaps)} chunk(s) in {path}", flush=True)
                sofa.top_up(path=path, season=season, chunks=gaps)
                continue
            print(f"[stats {season}] scraping -> {path}", flush=True)
            try:
                sofa.collect(season=season, leagues=leagues, path=path)
            except Exception as exc:  # noqa: BLE001 - one season must not end the run
                print(f"[stats {season}] failed: {exc}", flush=True)
                continue
            gaps = _missing_stats_chunks(path, leagues)
            if gaps:
                print(f"[stats {season}] retrying {len(gaps)} empty chunk(s)", flush=True)
                sofa.top_up(path=path, season=season, chunks=gaps)

    # One browser session for every season, instead of one per `collect`.
    sofascore_session.run(run)


# ---------------------------------------------------------------------------
# WhoScored
# ---------------------------------------------------------------------------


def _played(schedule: pd.DataFrame) -> pd.DataFrame:
    """Drop fixtures that never produced events (postponed and never replayed, etc.)."""
    if "status" in schedule.columns:
        # soccerdata keeps WhoScored's numeric status; 6 is "full time".
        done = schedule[schedule["status"] == 6]
        if not done.empty:
            return done
    return schedule


def _have_game(path: Path) -> bool:
    # A header-only csv is a game that came back without events, not a game done.
    return path.exists() and path.stat().st_size > 1024


def backfill_events(
    years: list[int],
    leagues: list[str],
    browser: str,
    *,
    max_games: int | None = None,
    prune_raw: bool = False,
) -> None:
    import soccerdata as sd

    from whoscored_patch import apply_whoscored_json_patch

    apply_whoscored_json_patch()
    budget = max_games
    for year in years:
        for league in leagues:
            if budget is not None and budget <= 0:
                return
            save_dir = events_dir(league, year)
            save_dir.mkdir(parents=True, exist_ok=True)
            tag = f"[events {league} {sofascore_season(year)}]"
            try:
                ws = sd.WhoScored(leagues=league, seasons=year, path_to_browser=browser)
                schedule = _played(ws.read_schedule())
            except Exception as exc:  # noqa: BLE001
                print(f"{tag} schedule failed: {exc}", flush=True)
                continue

            game_ids = [int(g) for g in schedule["game_id"].tolist()]
            todo = [g for g in game_ids if not _have_game(save_dir / f"{g}.csv")]
            print(f"{tag} {len(game_ids)} games, {len(todo)} to scrape", flush=True)
            failed = 0
            started = time.monotonic()
            for n, game_id in enumerate(todo, start=1):
                if budget is not None:
                    if budget <= 0:
                        break
                    budget -= 1
                try:
                    game = ws.read_events(match_id=game_id).reset_index(drop=True)
                except Exception as exc:  # noqa: BLE001 - skip it, a later run retries
                    failed += 1
                    print(f"{tag} game {game_id} failed: {exc}", flush=True)
                    continue
                if game.empty:
                    # soccerdata logs "No events found" and hands back an empty
                    # frame when the page loads without its match data. Saving
                    # that would mark the game done for good.
                    failed += 1
                    print(f"{tag} game {game_id} came back empty — will retry next run", flush=True)
                    continue
                path = save_dir / f"{game_id}.csv"
                tmp = path.with_suffix(".csv.part")
                game.to_csv(tmp, index=False)
                # Renamed into place, so an interrupted write never leaves a
                # truncated csv that the resume check would treat as done.
                tmp.replace(path)
                if prune_raw:
                    raw = RAW_CACHE / f"{league}_{soccerdata_season(year)}" / f"{game_id}.json"
                    raw.unlink(missing_ok=True)
                if n % 10 == 0 or n == len(todo):
                    rate = (time.monotonic() - started) / n
                    left = (len(todo) - n) * rate / 60
                    print(
                        f"{tag} {n}/{len(todo)}  {rate:.1f}s/game  ~{left:.0f} min left here",
                        flush=True,
                    )
            if failed:
                print(f"{tag} {failed} game(s) failed — re-run to retry them", flush=True)
            try:
                ws._driver.quit()  # noqa: SLF001 - soccerdata keeps one browser per reader
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def status(years: list[int], leagues: list[str]) -> pd.DataFrame:
    rows = []
    for year in years:
        row = {"season": sofascore_season(year)}
        for league in leagues:
            d = events_dir(league, year)
            row[league.split("-")[0]] = sum(_have_game(p) for p in d.glob("*.csv")) if d.exists() else 0
        path = stats_path(year)
        if path.exists():
            frame = pd.read_csv(path, usecols=["league"])
            row["stats_rows"] = len(frame)
            row["stats_gaps"] = len(_missing_stats_chunks(path, [sofa.sofascore_league_name(l) for l in leagues]))
        else:
            row["stats_rows"] = 0
            row["stats_gaps"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["status", "stats", "events"])
    ap.add_argument("--from", dest="first", type=int, default=FIRST_SEASON, help="first season start year")
    ap.add_argument("--to", dest="last", type=int, default=LAST_HISTORIC_SEASON, help="last season start year")
    ap.add_argument(
        "--league",
        action="append",
        dest="leagues",
        help="WhoScored league label, e.g. 'ENG-Premier League' (repeatable; default all 7)",
    )
    ap.add_argument("--browser", default=None, help="Chrome/Chromium executable for WhoScored")
    ap.add_argument("--max-games", type=int, default=None, help="stop after this many games (trial runs)")
    ap.add_argument(
        "--prune-raw",
        action="store_true",
        help="delete soccerdata's raw JSON for a game once its csv is written; "
        "the csv is the copy everything reads, and the JSON roughly doubles disk use",
    )
    args = ap.parse_args()

    leagues = args.leagues or list(DEFAULT_LEAGUES)
    # Newest first: each finished season is usable on its own, and recent ones
    # are the likelier comparison targets.
    years = list(range(args.last, args.first - 1, -1))

    if args.what == "status":
        years = list(range(CURRENT_SEASON, args.first - 1, -1))
        print(status(years, leagues).to_string(index=False))
        return 0
    if args.what == "stats":
        backfill_stats(years, [sofa.sofascore_league_name(l) for l in leagues])
        return 0

    browser = args.browser or default_browser()
    if browser is None:
        ap.error("no Chrome/Chromium found — pass --browser")
    try:
        backfill_events(years, leagues, browser, max_games=args.max_games, prune_raw=args.prune_raw)
    except KeyboardInterrupt:
        print("\ninterrupted — re-run the same command to resume", flush=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
