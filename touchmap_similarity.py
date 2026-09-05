"""Find players with the most similar on-pitch "shape" to a target player.

Reuses the cached WhoScored event data behind touch_maps.py /
player_touchmaps.py (`load_season_events`). By default this pools every
league collected under `league_games/` (the big 5 plus Eredivisie and Liga
Portugal). Each (player, team, league)'s season of touches is reduced to a
smoothed, L1-normalized 2D histogram over the same Opta 0-100 pitch grid
used to render `touch_maps/*.png` — i.e. the same touch-map information,
just as a numeric vector instead of a rendered image. Normalizing strips
out raw touch volume so the vector captures *where* a player operates, not
how heavily involved in play they were.

KMeans clusters every player on that shape (after a Hellinger-style sqrt
transform + PCA, since the vectors are probability distributions), and
`find_similar` ranks the target's cluster-mates by distance to the target —
falling back to the next-nearest clusters if the home cluster has fewer than
`top_n` other members.

CLI:

    python touchmap_similarity.py "Bruno Fernandes" --montage bruno_similar.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Pitch
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

from player_action_maps import (
    DEFENSIVE_TYPES,
    PASS_TYPES,
    TAKEON_TYPES,
    _defensive_outcomes,
    _pass_outcomes,
    _shot_outcomes,
    _takeon_outcomes,
)
from player_touchmaps import _slug, load_season_events

DEFAULT_BINS = (12, 8)  # (x_bins, y_bins) across the opta 0-100 pitch
DEFAULT_MIN_TOUCHES = 30
DEFAULT_SEASON = 2025
DEFAULT_LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
    "NED-Eredivisie",
    "POR-Liga Portugal",
]
BG_COLOR = "#0C0D0E"


def _league_slug(league: str, season: int | str = DEFAULT_SEASON) -> str:
    return f"{league.replace(' ', '_')}_{season}"


def _league_from_dir(path: str | Path, season: int | str = DEFAULT_SEASON) -> str | None:
    """Recover 'ENG-Premier League' from a dir named ENG-Premier_League_2025."""
    name = Path(path).name
    suffix = f"_{season}"
    if name.endswith(suffix):
        return name[: -len(suffix)].replace("_", " ")
    return None


def load_all_league_events(
    leagues: list[str] | None = None,
    season: int | str = DEFAULT_SEASON,
    events_root: str | Path = "league_games",
) -> pd.DataFrame:
    """Concatenate cached game csvs for every collected league, tagging each row with `league`."""
    leagues = leagues or list(DEFAULT_LEAGUES)
    frames = []
    missing = []
    for league in leagues:
        save_dir = Path(events_root) / _league_slug(league, season)
        try:
            df = load_season_events(save_dir)
        except FileNotFoundError:
            missing.append(str(save_dir))
            continue
        df = df.copy()
        df["league"] = league
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"no game csvs found for leagues {leagues} under {events_root}"
            + (f" (missing: {', '.join(missing)})" if missing else "")
        )
    if missing:
        print(f"warning: skipped leagues with no cached games: {', '.join(missing)}", flush=True)
    loaded = [f["league"].iloc[0] for f in frames]
    print(f"loaded events from {len(loaded)} leagues: {', '.join(loaded)}", flush=True)
    return pd.concat(frames, ignore_index=True)


def build_touch_grids(
    season_df: pd.DataFrame,
    min_touches: int = DEFAULT_MIN_TOUCHES,
    bins: tuple[int, int] = DEFAULT_BINS,
    smooth_sigma: float = 1.0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Reduce every (player, team, league)'s season touches to a flattened touch-shape vector.

    Returns `(meta, X)`: `meta` has one row per (player, team) — and `league`
    when the events carry one — with a `touches` count, row-aligned with `X`
    (n_players x n_bins) — a smoothed, L1-normalized spatial histogram, same
    grid for every player so the vectors are directly comparable.
    """
    touches_all = season_df[(season_df["is_touch"] == True) & season_df["player"].notna()]
    pitch = Pitch(pitch_type="opta")
    group_cols = ["player", "team"]
    has_league = "league" in touches_all.columns
    if has_league:
        group_cols = ["player", "team", "league"]

    rows = []
    vectors = []
    for key, g in touches_all.groupby(group_cols):
        if len(g) < min_touches:
            continue
        bin_stat = pitch.bin_statistic(g["x"], g["y"], statistic="count", bins=bins)
        grid = gaussian_filter(bin_stat["statistic"], smooth_sigma)
        total = grid.sum()
        if total <= 0:
            continue
        vectors.append((grid / total).ravel())
        if has_league:
            player, team, league = key
            rows.append({"player": player, "team": team, "league": league, "touches": len(g)})
        else:
            player, team = key
            rows.append({"player": player, "team": team, "touches": len(g)})

    if not rows:
        raise ValueError(f"no player had >= {min_touches} touches")

    meta = pd.DataFrame(rows).reset_index(drop=True)
    X = np.vstack(vectors)
    return meta, X


def _bin_vectors(
    events: pd.DataFrame,
    group_cols: list[str],
    bins: tuple[int, int],
    smooth_sigma: float,
    pitch: Pitch,
) -> dict[tuple, np.ndarray]:
    """Same smoothed, L1-normalized spatial histogram as `build_touch_grids`, keyed by group."""
    vectors = {}
    for key, g in events.groupby(group_cols):
        bin_stat = pitch.bin_statistic(g["x"], g["y"], statistic="count", bins=bins)
        grid = gaussian_filter(bin_stat["statistic"], smooth_sigma)
        total = grid.sum()
        if total <= 0:
            continue
        vectors[key] = (grid / total).ravel()
    return vectors


ACTION_TYPE_SPECS = {
    "pass": {"types": PASS_TYPES},
    "takeon": {"types": TAKEON_TYPES},
    "shot": {"is_shot": True},
    "defensive": {"types": DEFENSIVE_TYPES},
}


def build_action_grids(
    season_df: pd.DataFrame,
    meta: pd.DataFrame,
    bins: tuple[int, int] = DEFAULT_BINS,
    smooth_sigma: float = 1.0,
) -> dict[str, np.ndarray]:
    """Pass / take-on / shot / defensive-action shape vectors, row-aligned with `meta`.

    `meta` is the touch-based population from `build_touch_grids` (i.e. the
    min-touches-filtered player list clustering is actually run on). A
    player with no events of a given type (a keeper's take-ons, a
    center-back's shots) gets an all-zero vector for that block instead of
    being dropped — it contributes no signal to that block rather than
    excluding the player from the clustering population.
    """
    has_league = "league" in meta.columns
    group_cols = ["player", "team", "league"] if has_league else ["player", "team"]
    pitch = Pitch(pitch_type="opta")
    zero = np.zeros(bins[0] * bins[1])

    blocks: dict[str, np.ndarray] = {}
    for name, spec in ACTION_TYPE_SPECS.items():
        if spec.get("is_shot"):
            events = season_df[season_df["is_shot"] == True]  # noqa: E712
        else:
            events = season_df[
                season_df["type"].isin(spec["types"]) & season_df["player"].notna()
            ]
        vectors = _bin_vectors(events, group_cols, bins, smooth_sigma, pitch)
        blocks[name] = np.vstack(
            [vectors.get(tuple(row[c] for c in group_cols), zero) for _, row in meta.iterrows()]
        )
    return blocks


# Outcome buckets per action type, in a fixed order matching each helper's
# return tuple in `player_action_maps.py` (which in turn matches that type's
# map coloring: pass_map's green/red/red-X, takeon_map's and
# defensive_action_map's green/red, shot_map's gold-star/teal/red/grey).
ACTION_OUTCOME_LABELS = {
    "pass": ("completed", "incomplete", "blocked"),
    "takeon": ("won", "lost"),
    "shot": ("goal", "ontarget", "offtarget", "blocked"),
    "defensive": ("won", "lost"),
}


def _outcome_event_sets(season_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split each action type's events into its map's outcome buckets, keyed `"{type}_{outcome}"`.

    E.g. `"pass_completed"`, `"pass_incomplete"`, `"pass_blocked"`,
    `"shot_goal"`, ... — the same win/loss/goal information
    `player_action_maps.py` conveys through marker color.
    """
    passes = season_df[season_df["type"].isin(PASS_TYPES) & season_df["player"].notna()]
    takeons = season_df[season_df["type"].isin(TAKEON_TYPES) & season_df["player"].notna()]
    shots = season_df[season_df["is_shot"] == True]  # noqa: E712
    defensive = season_df[season_df["type"].isin(DEFENSIVE_TYPES) & season_df["player"].notna()]

    splits = {
        "pass": _pass_outcomes(passes),
        "takeon": _takeon_outcomes(takeons),
        "shot": _shot_outcomes(shots),
        "defensive": _defensive_outcomes(defensive),
    }
    return {
        f"{action}_{label}": subset
        for action, subsets in splits.items()
        for label, subset in zip(ACTION_OUTCOME_LABELS[action], subsets)
    }


def build_outcome_grids(
    season_df: pd.DataFrame,
    meta: pd.DataFrame,
    bins: tuple[int, int] = DEFAULT_BINS,
    smooth_sigma: float = 1.0,
) -> dict[str, np.ndarray]:
    """Outcome-split shape vectors (pass completed/incomplete/blocked, take-on
    won/lost, shot goal/on-target/off-target/blocked, defensive won/lost),
    row-aligned with `meta`.

    Where `build_action_grids` gives one shape per action type regardless of
    how it turned out, this splits each type the way its map colors it —
    the same success/fail/goal information a rendered map conveys visually
    through marker color, fed in here as numeric histograms instead of
    pixels. A player with no events in a given outcome bucket (e.g. never
    had a pass blocked) gets an all-zero vector for that block, same
    convention as `build_action_grids`.
    """
    has_league = "league" in meta.columns
    group_cols = ["player", "team", "league"] if has_league else ["player", "team"]
    pitch = Pitch(pitch_type="opta")
    zero = np.zeros(bins[0] * bins[1])

    blocks: dict[str, np.ndarray] = {}
    for name, events in _outcome_event_sets(season_df).items():
        vectors = _bin_vectors(events, group_cols, bins, smooth_sigma, pitch)
        blocks[name] = np.vstack(
            [vectors.get(tuple(row[c] for c in group_cols), zero) for _, row in meta.iterrows()]
        )
    return blocks


def build_combined_grids(
    season_df: pd.DataFrame,
    min_touches: int = DEFAULT_MIN_TOUCHES,
    bins: tuple[int, int] = DEFAULT_BINS,
    smooth_sigma: float = 1.0,
    include_outcomes: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Touch shape + pass/take-on/shot/defensive-action shapes, concatenated per player.

    Extends `build_touch_grids`'s "where do they operate" signal with "what
    do they do there": one more smoothed, L1-normalized histogram block per
    action type (`build_action_grids`), all on the same min-touches-filtered
    population. With `include_outcomes` (default), also appends one block
    per outcome bucket within each type (`build_outcome_grids` — e.g.
    completed vs incomplete vs blocked passes), so the vector captures not
    just where a player passes/dribbles/shoots/defends but whether it works
    there — the same success/fail coloring the action maps render visually.
    All blocks share the same pitch grid, so directly comparable.
    `cluster_players` then groups players by positional *and* behavioral
    shape instead of touch location alone.
    """
    meta, X_touch = build_touch_grids(
        season_df, min_touches=min_touches, bins=bins, smooth_sigma=smooth_sigma
    )
    action_blocks = build_action_grids(season_df, meta, bins=bins, smooth_sigma=smooth_sigma)
    blocks = [X_touch, *(action_blocks[name] for name in ACTION_TYPE_SPECS)]
    if include_outcomes:
        outcome_blocks = build_outcome_grids(season_df, meta, bins=bins, smooth_sigma=smooth_sigma)
        blocks.extend(outcome_blocks[name] for name in outcome_blocks)
    X = np.hstack(blocks)
    return meta, X


def cluster_players(
    X: np.ndarray,
    n_clusters: int = 14,
    pca_var: float = 0.90,
    random_state: int = 42,
) -> dict:
    """KMeans-cluster players on a Hellinger-style (sqrt) transform of their touch histograms.

    Taking the elementwise sqrt of a probability vector turns Euclidean
    distance into (twice) the Hellinger distance — a metric well suited to
    comparing distributions, unlike raw-count Euclidean distance.
    """
    X_hell = np.sqrt(X)

    pca_full = PCA(random_state=random_state).fit(X_hell)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    cap = max(2, min(X_hell.shape) - 1)
    n_components = int(np.clip(np.searchsorted(cum, pca_var) + 1, 2, cap))
    pca = PCA(n_components=n_components, random_state=random_state).fit(X_hell)
    coords = pca.transform(X_hell)

    n_clusters = max(2, min(n_clusters, X_hell.shape[0] - 1))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit(coords)

    return {
        "X_hell": X_hell,
        "coords": coords,
        "pca": pca,
        "n_components": n_components,
        "kmeans": kmeans,
        "labels": kmeans.labels_,
    }


def _player_label(row) -> str:
    league = row["league"] if "league" in row.index and pd.notna(row["league"]) else None
    if league:
        return f"{row['player']} ({row['team']}, {league})"
    return f"{row['player']} ({row['team']})"


def resolve_player(
    meta: pd.DataFrame,
    player_name: str,
    team: str | None = None,
    league: str | None = None,
) -> int:
    """Case-insensitive partial-match lookup, mirroring touch_maps.player_events."""
    matched = meta[meta["player"].str.contains(player_name, case=False, na=False, regex=False)]
    if team is not None:
        matched = matched[matched["team"].str.contains(team, case=False, na=False, regex=False)]
    if league is not None and "league" in matched.columns:
        matched = matched[matched["league"].str.contains(league, case=False, na=False, regex=False)]
    if matched.empty:
        raise ValueError(
            f"no player matching {player_name!r} (team={team!r}, league={league!r})"
        )
    if len(matched) > 1:
        listing = ", ".join(_player_label(row) for _, row in matched.iterrows())
        raise ValueError(
            f"{player_name!r} matches multiple players: {listing}. "
            "Pass team= and/or league= to disambiguate."
        )
    return matched.index[0]


def find_similar(
    meta: pd.DataFrame,
    cluster_result: dict,
    player_name: str,
    team: str | None = None,
    league: str | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Rank the target's cluster-mates by positional similarity.

    Walks outward from the target's own cluster to the next-nearest ones (by
    centroid distance) until at least `top_n` other players are available,
    then returns the `top_n` closest by distance to the target in PCA space.
    The returned frame's `.attrs["target"]` holds the resolved target's own
    (player, team, touches).
    """
    idx = resolve_player(meta, player_name, team, league=league)
    coords = cluster_result["coords"]
    labels = cluster_result["labels"]
    kmeans = cluster_result["kmeans"]

    target_label = labels[idx]
    centroid_order = np.argsort(
        euclidean_distances(
            kmeans.cluster_centers_, kmeans.cluster_centers_[target_label : target_label + 1]
        ).ravel()
    )

    candidates: list[int] = []
    for label in centroid_order:
        members = [m for m in np.where(labels == label)[0] if m != idx]
        candidates.extend(m for m in members if m not in candidates)
        if len(candidates) >= top_n:
            break

    dists = euclidean_distances(coords[candidates], coords[idx : idx + 1]).ravel()
    order = np.argsort(dists)[:top_n]
    chosen = [candidates[i] for i in order]

    cols = ["player", "team", "touches"]
    if "league" in meta.columns:
        cols = ["player", "team", "league", "touches"]
    out = meta.iloc[chosen][cols].reset_index(drop=True)
    out["cluster"] = labels[chosen]
    out["same_cluster_as_target"] = labels[chosen] == target_label
    out["distance"] = dists[order]
    out.insert(0, "rank", np.arange(1, len(out) + 1))

    target_row = meta.loc[idx]
    out.attrs["target"] = {
        "player": target_row["player"],
        "team": target_row["team"],
        "league": target_row["league"] if "league" in target_row.index else None,
        "touches": int(target_row["touches"]),
        "cluster": int(target_label),
    }
    return out


def build_and_search(
    player_name: str,
    team: str | None = None,
    league: str | None = None,
    events_dir: str | Path | None = None,
    leagues: list[str] | None = None,
    season: int | str = DEFAULT_SEASON,
    min_touches: int = DEFAULT_MIN_TOUCHES,
    bins: tuple[int, int] = DEFAULT_BINS,
    n_clusters: int = 14,
    top_n: int = 10,
) -> pd.DataFrame:
    """End to end: load cached season events, build touch-shape features, cluster, and rank.

    With no `events_dir`, pools every league in `leagues` (default: the 7
    collected top leagues). Pass `events_dir` to restrict to one cached
    season directory.
    """
    if events_dir is not None:
        season_df = load_season_events(events_dir)
        inferred = _league_from_dir(events_dir, season)
        if inferred is not None and "league" not in season_df.columns:
            season_df = season_df.copy()
            season_df["league"] = inferred
    else:
        season_df = load_all_league_events(leagues=leagues, season=season)
    meta, X = build_touch_grids(season_df, min_touches=min_touches, bins=bins)
    cluster_result = cluster_players(X, n_clusters=n_clusters)
    return find_similar(
        meta, cluster_result, player_name, team=team, league=league, top_n=top_n
    )


def plot_similar_montage(
    similar: pd.DataFrame,
    maps_dir: str | Path | None = None,
    maps_root: str | Path = "touch_maps",
    season: int | str = DEFAULT_SEASON,
    out_path: str | Path | None = None,
    ncols: int = 4,
):
    """Arrange the target's pre-rendered touch map next to its top similar players', for a visual check.

    Reuses the PNGs already saved by `touch_maps.py`. With no `maps_dir`,
    each player's image is looked up under `maps_root/{league}_{season}/`;
    pass `maps_dir` to force a single league folder. A missing image (player
    not yet rendered) shows a placeholder instead of failing the whole
    montage. Requires `similar` to carry `.attrs["target"]` (set by
    `find_similar`).
    """
    target = similar.attrs["target"]
    maps_dir = Path(maps_dir) if maps_dir is not None else None
    maps_root = Path(maps_root)
    has_league = "league" in similar.columns

    def _path(p, t, league=None):
        root = maps_dir
        if root is None:
            root = maps_root / _league_slug(league, season) if league else maps_root
        return root / _slug(t) / f"{_slug(p)}_touch_map.png"

    entries = [
        (target["player"], target["team"], target.get("league"), "TARGET")
    ] + [
        (
            row.player,
            row.team,
            getattr(row, "league", None) if has_league else None,
            f"#{row.rank}",
        )
        for row in similar.itertuples()
    ]
    n = len(entries)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 5.5 * nrows), facecolor=BG_COLOR)
    axes = np.atleast_1d(axes).ravel()

    for ax, (p, t, league, tag) in zip(axes, entries):
        ax.axis("off")
        path = _path(p, t, league)
        if path.exists():
            ax.imshow(mpimg.imread(path))
        else:
            ax.text(0.5, 0.5, f"{p}\n(no image)", ha="center", va="center", color="white")
        subtitle = f"{p} ({t}, {league})" if league else f"{p} ({t})"
        ax.set_title(f"{tag}: {subtitle}", color="white", fontsize=10)
    for ax in axes[n:]:
        ax.axis("off")

    fig.patch.set_facecolor(BG_COLOR)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Find the players with the most similar touch-map / positional footprint."
    )
    parser.add_argument("player", help="Target player name (partial, case-insensitive match)")
    parser.add_argument("--team", default=None, help="Disambiguate if the name matches multiple players")
    parser.add_argument(
        "--league",
        default=None,
        help="Disambiguate if the name matches multiple players across leagues "
        "(partial match, e.g. Prem, La Liga, Serie A)",
    )
    parser.add_argument(
        "--events-dir",
        default=None,
        help="Restrict to one cached season directory. Default: pool all 7 collected leagues.",
    )
    parser.add_argument(
        "--maps-dir",
        default=None,
        help="Force a single touch-map folder for the montage. Default: look up each "
        "player under touch_maps/{league}_{season}/.",
    )
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON)
    parser.add_argument("--min-touches", type=int, default=DEFAULT_MIN_TOUCHES)
    parser.add_argument("--n-clusters", type=int, default=14)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--montage", default=None, help="Path to save a PNG montage of the target + similar players' touch maps"
    )
    args = parser.parse_args()

    similar = build_and_search(
        args.player,
        team=args.team,
        league=args.league,
        events_dir=args.events_dir,
        season=args.season,
        min_touches=args.min_touches,
        n_clusters=args.n_clusters,
        top_n=args.top_n,
    )
    target = similar.attrs["target"]
    league_bit = f", {target['league']}" if target.get("league") else ""
    print(
        f"target: {target['player']} ({target['team']}{league_bit}) — "
        f"{target['touches']} touches, cluster {target['cluster']}"
    )
    print(similar.to_string(index=False))

    if args.montage:
        plot_similar_montage(
            similar,
            maps_dir=args.maps_dir,
            season=args.season,
            out_path=args.montage,
        )
        print(f"montage saved to {args.montage}")
