"""Batch job: full-pitch touch map for every player in a league season.

Scrapes (or reuses cached) event data for the whole league schedule via
`player_touchmaps.collect_league_events`, then renders a full-pitch touch
map — same dark-pitch KDE + scatter style as `final_third_touch_map`, just
over the whole 0-100 x-range instead of the attacking third — for every
player, saved under `touch_maps/`.

Run directly:

    python touch_maps.py

Safe to interrupt and resume: both the scraped game cache
(`league_games/`) and the rendered-image cache (`touch_maps/`) are
skip-if-exists.
"""

from __future__ import annotations

from player_touchmaps import build_league_touch_maps

if __name__ == "__main__":
    try:
        from pyfonts import load_google_font

        font = load_google_font("DotGothic16")
        if hasattr(font, "_pyfonts_provider_metadata"):
            del font._pyfonts_provider_metadata
    except Exception:  # noqa: BLE001
        font = None

    result = build_league_touch_maps(season=2025, font=font)
    n_ok = result["touch_map_saved"].sum()
    print(f"done: {n_ok}/{len(result)} players have a touch map saved")
    print(result[result["error"].notna()])
