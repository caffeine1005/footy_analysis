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
"""

from __future__ import annotations

from player_action_maps import build_league_action_maps
from touchmap_similarity import DEFAULT_LEAGUES, DEFAULT_SEASON

if __name__ == "__main__":
    try:
        from pyfonts import load_google_font

        font = load_google_font("DotGothic16")
        if hasattr(font, "_pyfonts_provider_metadata"):
            del font._pyfonts_provider_metadata
    except Exception:  # noqa: BLE001
        font = None

    for league in DEFAULT_LEAGUES:
        print(f"\n===== {league} =====", flush=True)
        try:
            summaries = build_league_action_maps(league=league, season=DEFAULT_SEASON, font=font)
        except Exception as e:  # noqa: BLE001
            print(f"!! {league} failed: {e}", flush=True)
            continue
        for name, summary in summaries.items():
            n_ok = summary["saved"].sum()
            print(f"{league} {name}: {n_ok}/{len(summary)} players saved", flush=True)
            errors = summary[summary["error"].notna()]
            if not errors.empty:
                print(errors, flush=True)
