"""Batch job: full-pitch touch maps for every remaining top league.

Same as `touch_maps.py` (which already covered ENG-Premier League), but
looped over the rest of the "big 5" plus Eredivisie and Liga Portugal.
Each league is fully independent and skip-if-exists at both the event-scrape
and rendered-image layer, so this is safe to interrupt/resume, and a
failure in one league doesn't block the others.

Run directly:

    python collect_more_touchmaps.py
"""

from __future__ import annotations

from player_touchmaps import build_league_touch_maps

LEAGUES = [
    "GER-Bundesliga",
    "FRA-Ligue 1",
    "NED-Eredivisie",
    "POR-Liga Portugal",
]

if __name__ == "__main__":
    try:
        from pyfonts import load_google_font

        font = load_google_font("DotGothic16")
        if hasattr(font, "_pyfonts_provider_metadata"):
            del font._pyfonts_provider_metadata
    except Exception:  # noqa: BLE001
        font = None

    for league in LEAGUES:
        print(f"\n===== {league} =====", flush=True)
        try:
            result = build_league_touch_maps(league=league, season=2025, font=font)
        except Exception as e:  # noqa: BLE001
            print(f"!! {league} failed: {e}", flush=True)
            continue
        n_ok = result["touch_map_saved"].sum()
        print(f"{league} done: {n_ok}/{len(result)} players have a touch map saved", flush=True)
        errors = result[result["error"].notna()]
        if not errors.empty:
            print(errors, flush=True)
