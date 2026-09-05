"""Positional cluster + quantitative FBref similarity.

Step 1 — touch-map clustering (`touchmap_similarity`): assign the target to a
positional cluster based on where they operate on the pitch.

Step 2 — FBref stats (`player_similarity` / `fbref_test.ipynb`): within that
cluster, find the most quantitatively similar players (cosine / PCA-KNN /
Spearman / role-blocks / NMF, fused with RRF), then profile each match's
strengths and weaknesses vs the cluster pool.

Pass-angle tendency (`pass_angle_radar`) is fused directly into that
second stage: each cluster mate's 8-spoke radar (plus concentration, mean
length, completion, circular mean direction) is median-imputed and
appended to the FBref feature matrix, so the Cosine / KNN-PCA / Spearman /
NMF rankers all treat pass direction as real inputs (RoleBlocks stays
FBref-only). A cosine of those same vectors is still printed next to the
target and each match as a readout.

CLI:

    python player_profile.py "Bruno Fernandes" --team "Man Utd"
    python player_profile.py "Lewis Hall" --top-n 5
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
from scipy.stats import percentileofscore
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

import pass_angle_radar as par
import player_action_maps as pam
import touchmap_similarity as tms
from name_utils import normalize_name, normalized_col, strip_accents
from player_similarity import (
    DATA_PATH,
    MINUTES_COL,
    POS_COL,
    fit_pool,
    load_players,
    role_matrix,
    spearman_scores,
)

DEFAULT_MIN_90S = None  # optional filter on cluster pool members (target always kept)
STRENGTH_CUTOFF = 75.0
WEAKNESS_CUTOFF = 25.0
TOP_STATS = 8
DEFAULT_PCA_VAR = 0.80
DEFAULT_NMF_K = 5
DEFAULT_RRF_K = 60


def pretty_stat(col: str) -> str:
    """Turn \"('Expected', 'xAG')\" into \"xAG\" (or \"Expected · xAG\")."""
    try:
        parts = ast.literal_eval(col)
        if isinstance(parts, tuple):
            bits = [str(p) for p in parts if p]
            return bits[-1] if len(bits) == 1 else " · ".join(bits)
    except (SyntaxError, ValueError):
        pass
    return col


def _team_tokens(name: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", strip_accents(name).lower()))


def _teams_overlap(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return True
    ta, tb = _team_tokens(a), _team_tokens(b)
    if not ta or not tb:
        return True
    if ta & tb:
        return True
    a_norm = re.sub(r"[^a-z0-9]", "", strip_accents(a).lower())
    b_norm = re.sub(r"[^a-z0-9]", "", strip_accents(b).lower())
    return a_norm in b_norm or b_norm in a_norm


def resolve_fbref_row(
    players: pl.DataFrame,
    player: str,
    team: str | None = None,
    league: str | None = None,
) -> tuple[dict, pl.DataFrame]:
    """Match a touch-map row to one FBref player (name + optional team/league)."""
    matches = players.with_row_index().filter(
        normalized_col("player").str.contains(normalize_name(player), literal=True)
    )
    if league:
        matches = matches.filter(
            normalized_col("league").str.contains(normalize_name(league), literal=True)
        )
    if team and matches.height > 1:
        by_team = matches.filter(
            pl.col("team").map_elements(
                lambda t: _teams_overlap(team, t), return_dtype=pl.Boolean
            )
        )
        if by_team.height >= 1:
            matches = by_team
    if matches.height == 0:
        raise ValueError(
            f"no FBref row for {player!r} (team={team!r}, league={league!r})"
        )
    if matches.height > 1:
        exact = matches.filter(normalized_col("player") == normalize_name(player))
        if exact.height == 1:
            matches = exact
    row = matches.row(0, named=True)
    if row["primary_pos"] is None:
        raise ValueError(f"could not parse position for {row['player']!r}")
    return row, matches



def _find_fbref_index(players: pl.DataFrame, row: dict) -> int | None:
    matched = players.with_row_index().filter(pl.col("player") == row["player"])
    if row.get("league"):
        matched = matched.filter(
            pl.col("league").str.contains("(?i)" + str(row["league"]))
        )
    if row.get("team"):
        matched = matched.filter(
            pl.col("team").map_elements(
                lambda t: _teams_overlap(row["team"], t), return_dtype=pl.Boolean
            )
        )
    if matched.height == 0:
        matched = players.with_row_index().filter(pl.col("player") == row["player"])
    if matched.height == 0:
        return None
    return int(matched["index"][0])


def build_fbref_cluster_pool(
    players: pl.DataFrame,
    cluster_mates,
    target_row: dict,
    min_90s: float | None = None,
) -> tuple[pl.DataFrame, dict]:
    """Map touch-cluster members to FBref rows; target is always included."""
    matched_indices: list[int] = []
    unmatched: list[str] = []

    for row in cluster_mates.itertuples(index=False):
        league = getattr(row, "league", None)
        try:
            fbref_row, _ = resolve_fbref_row(players, row.player, row.team, league)
        except ValueError:
            unmatched.append(
                f"{row.player} ({row.team}" + (f", {league})" if league else ")")
            )
            continue
        if (
            fbref_row[MINUTES_COL] <= 0
            and fbref_row["player"] != target_row["player"]
        ):
            continue
        if (
            min_90s is not None
            and fbref_row["player"] != target_row["player"]
            and fbref_row[MINUTES_COL] < min_90s
        ):
            continue
        idx = _find_fbref_index(players, fbref_row)
        if idx is not None and idx not in matched_indices:
            matched_indices.append(idx)

    target_idx = _find_fbref_index(players, target_row)
    if target_idx is not None and target_idx not in matched_indices:
        matched_indices.append(target_idx)

    meta = {
        "cluster_touch_size": len(cluster_mates),
        "matched": len(matched_indices),
        "unmatched": unmatched,
    }
    if len(matched_indices) < 2:
        raise ValueError(
            f"only {len(matched_indices)} touch-cluster member(s) matched in FBref "
            f"(cluster has {len(cluster_mates)} players with touch data); "
            "need at least 2 for percentile comparison"
        )

    return (
        players.with_row_index()
        .filter(pl.col("index").is_in(matched_indices))
        .drop("index"),
        meta,
    )


# (name, plotting fn, filename suffix) for the target's own action maps —
# `pam.full_pitch_touch_map` covers the touch map itself.
TARGET_MAP_SPECS = [
    ("touch", pam.full_pitch_touch_map, "touch_map"),
    ("pass", pam.pass_map, "pass_map"),
    ("pass_angle", par.pass_angle_radar, "pass_angle_radar"),
    ("takeon", pam.takeon_map, "takeon_map"),
    ("shot", pam.shot_map, "shot_map"),
    ("defensive", pam.defensive_action_map, "defensive_map"),
]


def _save_target_maps(
    season_df: pd.DataFrame, touch_target: dict, out_dir: str | Path, dpi: int = 150
) -> dict[str, str]:
    """Save the target's touch/pass/angle/take-on/shot/defensive maps as PNGs.

    A map type the target has no events for (e.g. a center-back's shot map)
    is skipped rather than failing the others. Returns `{name: saved_path}`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = pam._slug(touch_target["player"])

    saved = {}
    for name, fn, suffix in TARGET_MAP_SPECS:
        try:
            ax = fn(season_df, touch_target["player"], team=touch_target["team"], exact=True)
        except ValueError:
            continue
        path = out_dir / f"{slug}_{suffix}.png"
        ax.figure.savefig(path, dpi=dpi, facecolor=ax.figure.get_facecolor())
        plt.close(ax.figure)
        saved[name] = str(path)
    return saved


def _run_touchmap_cluster(
    player_name: str,
    team: str | None,
    league: str | None,
    events_dir: str | Path | None,
    season: int | str,
    min_touches: int,
    n_clusters: int,
    action_features: bool = True,
    save_maps_dir: str | Path | None = None,
) -> tuple[dict, pd.DataFrame, int]:
    """Resolve target and return their touch-map (+ action-map) cluster mates.

    With `action_features` (default), the clustering vector is touch shape
    concatenated with pass/take-on/shot/defensive-action shape, both overall
    and split by outcome — completed/incomplete/blocked passes, won/lost
    take-ons, goal/on-target/off-target/blocked shots, won/lost defensive
    actions (see `touchmap_similarity.build_combined_grids`) — i.e. the same
    success/fail information those maps render as marker color, so the
    cluster groups players by behavioral role and effectiveness, not just
    pitch position. Pass `action_features=False` to fall back to
    touch-location-only clustering.

    With `save_maps_dir`, also saves the target's touch/pass/pass-angle/
    take-on/shot/defensive-action maps as separate PNGs in that directory,
    reusing the same `season_df` already loaded for clustering.
    """
    if events_dir is not None:
        season_df = tms.load_season_events(events_dir)
        inferred = tms._league_from_dir(events_dir, season)
        if inferred is not None and "league" not in season_df.columns:
            season_df = season_df.copy()
            season_df["league"] = inferred
    else:
        season_df = tms.load_all_league_events(season=season)

    if action_features:
        meta, X = tms.build_combined_grids(season_df, min_touches=min_touches)
    else:
        meta, X = tms.build_touch_grids(season_df, min_touches=min_touches)
    cluster_result = tms.cluster_players(X, n_clusters=n_clusters)
    target_idx = tms.resolve_player(meta, player_name, team=team, league=league)
    target_row = meta.loc[target_idx]
    cluster_label = int(cluster_result["labels"][target_idx])
    cluster_mates = meta[cluster_result["labels"] == cluster_label].reset_index(drop=True)

    touch_target = {
        "player": target_row["player"],
        "team": target_row["team"],
        "league": target_row["league"] if "league" in target_row.index else None,
        "touches": int(target_row["touches"]),
        "cluster": cluster_label,
    }

    pairs = list(
        zip(cluster_mates["player"].tolist(), cluster_mates["team"].tolist())
    )
    stats_by, vec_by = par.pass_angle_features_for_players(season_df, pairs)
    key = (touch_target["player"], touch_target["team"])
    touch_target["pass_angles"] = stats_by.get(key)
    touch_target["pass_angle_stats_by_player"] = stats_by
    touch_target["pass_angle_vec_by_player"] = vec_by

    if save_maps_dir is not None:
        touch_target["maps_saved"] = _save_target_maps(season_df, touch_target, save_maps_dir)

    return touch_target, cluster_mates, cluster_label


def percentile_profile_values(
    pool: pl.DataFrame,
    feature_cols: list[str],
    row_values: np.ndarray,
    strength_cutoff: float = STRENGTH_CUTOFF,
    weakness_cutoff: float = WEAKNESS_CUTOFF,
    top_k: int = TOP_STATS,
) -> dict:
    """Percentile rank of one player's stats against a reference pool."""
    pool_values = pool.select(feature_cols).to_numpy()
    n = pool_values.shape[0]

    percentiles: list[tuple[str, float, float]] = []
    for j, col in enumerate(feature_cols):
        col_vals = pool_values[:, j]
        finite = col_vals[np.isfinite(col_vals)]
        if finite.size < 2:
            continue
        val = row_values[j]
        if not np.isfinite(val):
            continue
        pct = float(percentileofscore(finite, val, kind="rank"))
        percentiles.append((col, float(val), pct))

    strengths = sorted(
        [(c, v, p) for c, v, p in percentiles if p >= strength_cutoff],
        key=lambda x: x[2],
        reverse=True,
    )[:top_k]
    weaknesses = sorted(
        [(c, v, p) for c, v, p in percentiles if p <= weakness_cutoff],
        key=lambda x: x[2],
    )[:top_k]

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "n_stats": len(percentiles),
        "pool_size": n,
    }


def percentile_profile(
    pool: pl.DataFrame,
    feature_cols: list[str],
    row_idx: int,
    strength_cutoff: float = STRENGTH_CUTOFF,
    weakness_cutoff: float = WEAKNESS_CUTOFF,
    top_k: int = TOP_STATS,
) -> dict:
    row_values = pool.select(feature_cols).to_numpy()[row_idx]
    return percentile_profile_values(
        pool, feature_cols, row_values, strength_cutoff, weakness_cutoff, top_k
    )


def role_scores_values(
    pool: pl.DataFrame,
    feature_cols: list[str],
    row_values: np.ndarray,
    pos_group: str,
) -> pl.DataFrame:
    """Role-block scores for one row, scaled against the reference pool."""
    X = pool.select(feature_cols).to_numpy()
    scaler = RobustScaler().fit(X)
    row_scaled = scaler.transform(row_values.reshape(1, -1))
    roles, role_names, _ = role_matrix(row_scaled, feature_cols, pos_group)
    return pl.DataFrame(
        {"role": role_names, "score": [float(roles[0, i]) for i in range(len(role_names))]}
    ).sort("score", descending=True)


def role_scores(
    pool: pl.DataFrame,
    feature_cols: list[str],
    row_idx: int,
    pos_group: str,
) -> pl.DataFrame:
    row_values = pool.select(feature_cols).to_numpy()[row_idx]
    return role_scores_values(pool, feature_cols, row_values, pos_group)


def _pool_target_idx(pool: pl.DataFrame, target_row: dict) -> int:
    for i, row in enumerate(pool.iter_rows(named=True)):
        if row["player"] != target_row["player"]:
            continue
        if target_row.get("league") and not re.search(
            "(?i)" + str(target_row["league"]), row["league"]
        ):
            continue
        if target_row.get("team") and not _teams_overlap(target_row["team"], row["team"]):
            continue
        return i
    names = pool["player"].to_list()
    return names.index(target_row["player"])


def _align_pass_angles_to_pool(
    pool: pl.DataFrame,
    cluster_mates: pd.DataFrame,
    vec_by: dict[tuple[str, str], np.ndarray],
    stats_by: dict[tuple[str, str], dict],
) -> tuple[np.ndarray, dict[int, dict]]:
    """Map WhoScored pass-angle features onto FBref `pool` rows.

    Same name/team/league resolver as `build_fbref_cluster_pool`. Players
    with no directed-pass vector stay as NaN; `search_in_pool` median-imputes
    them before fusing the features into the stage-2 matrix.
    """
    n_feat = len(par.PASS_ANGLE_FEATURE_NAMES)
    X = np.full((pool.height, n_feat), np.nan)
    stats_by_idx: dict[int, dict] = {}
    if cluster_mates.empty or not (vec_by or stats_by):
        return X, stats_by_idx

    for row in cluster_mates.itertuples(index=False):
        key = (row.player, row.team)
        vec = vec_by.get(key)
        stats = stats_by.get(key)
        if vec is None and stats is None:
            continue
        league = getattr(row, "league", None)
        try:
            fbref_row, _ = resolve_fbref_row(pool, row.player, row.team, league)
            idx = _pool_target_idx(pool, fbref_row)
        except ValueError:
            continue
        if vec is not None:
            X[idx] = vec
        if stats is not None:
            stats_by_idx[idx] = stats
    return X, stats_by_idx


def _augment_pass_angle_features(
    pass_angle_X: np.ndarray, idx: int
) -> np.ndarray | None:
    """Median-impute the pass-angle matrix so it can be fused into stage-2 inputs.

    Missing rows (players without a directed-pass vector) are filled with
    each feature's median over the players who do have one, i.e. a neutral
    value that neither helps nor hurts their similarity. Returns the filled
    ``(n, k)`` matrix, or ``None`` when the target has no vector or fewer
    than two players do (nothing to compare against).
    """
    valid = np.all(np.isfinite(pass_angle_X), axis=1)
    if not valid[idx] or int(valid.sum()) < 2:
        return None
    filled = pass_angle_X.astype(float, copy=True)
    for j in range(filled.shape[1]):
        col = filled[:, j]
        finite = col[np.isfinite(col)]
        med = float(np.median(finite)) if finite.size else 0.0
        col[~np.isfinite(col)] = med
    return filled


def _pass_angle_cosine(pass_angle_X: np.ndarray, idx: int) -> np.ndarray | None:
    """Cosine of Robust-scaled pass-tendency vectors vs row `idx`, or None.

    Used only for the report readout now that the pass-angle features are
    fused into the stage-2 feature matrix (see `search_in_pool`).
    """
    valid = np.all(np.isfinite(pass_angle_X), axis=1)
    if not valid[idx] or int(valid.sum()) < 2:
        return None
    scaled = np.zeros_like(pass_angle_X)
    scaled[valid] = RobustScaler().fit_transform(pass_angle_X[valid])
    ang = np.full(pass_angle_X.shape[0], -np.inf)
    hits = np.flatnonzero(valid)
    ang[hits] = cosine_similarity(scaled[hits], scaled[idx : idx + 1]).ravel()
    ang[idx] = -np.inf
    return ang


def search_in_pool(
    pool: pl.DataFrame,
    target_row: dict,
    feature_cols: list[str],
    top_n: int = 10,
    pca_var: float = DEFAULT_PCA_VAR,
    nmf_k: int = DEFAULT_NMF_K,
    rrf_k: int = DEFAULT_RRF_K,
    pass_angle_X: np.ndarray | None = None,
) -> dict:
    """RRF-ranked quantitative similarity within a fixed FBref pool.

    When `pass_angle_X` is aligned to `pool` and the target has a finite
    row, the pass-tendency features are median-imputed and appended to the
    FBref feature matrix (early fusion), so the Cosine / KNN-PCA / Spearman
    / NMF rankers are all computed on them as real inputs. RoleBlocks stays
    FBref-only. A cosine of the raw pass-tendency vectors is still returned
    (`pass_angle_cos`) purely as a report readout, not as an RRF channel.
    """
    idx = _pool_target_idx(pool, target_row)
    pos_group = target_row["primary_pos"]

    # Early fusion: append the (median-imputed) pass-angle features to the
    # FBref columns so they feed the similarity rankers as genuine inputs.
    sim_pool = pool
    sim_cols = list(feature_cols)
    pass_angle_used = False
    if pass_angle_X is not None:
        filled = _augment_pass_angle_features(pass_angle_X, idx)
        if filled is not None:
            pa_names = [f"pass_angle::{name}" for name in par.PASS_ANGLE_FEATURE_NAMES]
            sim_pool = pool.with_columns(
                [pl.Series(nm, filled[:, j]) for j, nm in enumerate(pa_names)]
            )
            sim_cols = sim_cols + pa_names
            pass_angle_used = True

    q = {"pca_var": pca_var, "nmf_k": nmf_k}
    art = fit_pool(sim_pool, sim_cols, q)

    X_scaled = art["X_scaled"]
    pcaed = art["pcaed"]
    n = pool.height

    scores: dict[str, np.ndarray] = {}
    cos = cosine_similarity(X_scaled, X_scaled[idx : idx + 1]).ravel().astype(float)
    cos[idx] = -np.inf
    scores["Cosine"] = cos

    n_neighbors = min(n, max(top_n + 1, 2))
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(pcaed)
    dist, ind = knn.kneighbors(pcaed[idx : idx + 1], n_neighbors=n_neighbors)
    knn_scores = np.full(n, -np.inf, dtype=float)
    for d, j in zip(dist[0], ind[0]):
        if j != idx:
            knn_scores[j] = 1.0 / (1.0 + d)
    scores["KNN_PCA"] = knn_scores

    sp = spearman_scores(X_scaled, idx)
    sp[idx] = -np.inf
    scores["Spearman"] = sp

    # RoleBlocks stays FBref-only — the pass-angle columns (appended last)
    # are not part of any role block, so slice them back off here.
    role_X = X_scaled[:, : len(feature_cols)]
    roles, role_names, _ = role_matrix(role_X, feature_cols, pos_group)
    role_cos = cosine_similarity(roles, roles[idx : idx + 1]).ravel().astype(float)
    role_cos[idx] = -np.inf
    scores["RoleBlocks"] = role_cos

    W_norm = art["W_norm"]
    nmf_cos = cosine_similarity(W_norm, W_norm[idx : idx + 1]).ravel().astype(float)
    nmf_cos[idx] = -np.inf
    scores["NMF"] = nmf_cos

    # Not an RRF channel: the pass-angle tendency is already fused into the
    # feature matrix above. This cosine is kept only for the report readout.
    pass_angle_cos = (
        _pass_angle_cosine(pass_angle_X, idx) if pass_angle_X is not None else None
    )

    rrf = np.zeros(n)
    ranks: dict[str, np.ndarray] = {}
    for name, sc in scores.items():
        order = np.argsort(sc)[::-1]
        rnk = np.empty(n)
        rnk[order] = np.arange(1, n + 1)
        ranks[name] = rnk
        rrf += 1.0 / (rrf_k + rnk)
    rrf[idx] = -np.inf

    order = np.argsort(rrf)[::-1][:top_n]
    extra_cols = []
    if pass_angle_cos is not None:
        extra_cols.append(pl.Series("pass_angle_cos", pass_angle_cos[order]))
    consensus = pool[order.tolist()].select(
        "player", "team", "league", POS_COL, "primary_pos", "age"
    ).with_columns(
        pl.Series("rrf_score", rrf[order]),
        *extra_cols,
        *[
            pl.Series(f"{name}_rank", ranks[name][order].astype(int))
            for name in scores
        ],
    )

    return {
        "idx": idx,
        "scores": scores,
        "ranks": ranks,
        "rrf": rrf,
        "order": order,
        "consensus": consensus,
        "n_pca": art["n_pca"],
        "pca_var_explained": art["pca_var_explained"],
        "pass_angle_used": pass_angle_used,
        "pass_angle_cos": pass_angle_cos,
    }


def profile_from_pool_row(
    pool: pl.DataFrame,
    pool_idx: int,
    feature_cols: list[str],
    pos_group: str,
    strength_cutoff: float = STRENGTH_CUTOFF,
    weakness_cutoff: float = WEAKNESS_CUTOFF,
    top_stats: int = TOP_STATS,
) -> dict:
    """Strengths/weaknesses for a player already in the cluster pool."""
    row = pool.row(pool_idx, named=True)
    row_values = pool.select(feature_cols).to_numpy()[pool_idx]
    pct = percentile_profile_values(
        pool,
        feature_cols,
        row_values,
        strength_cutoff=strength_cutoff,
        weakness_cutoff=weakness_cutoff,
        top_k=top_stats,
    )
    roles = role_scores_values(pool, feature_cols, row_values, pos_group)
    return {
        "player": row["player"],
        "team": row["team"],
        "league": row["league"],
        "position": row[POS_COL],
        "primary_pos": row["primary_pos"],
        "nineties": float(row[MINUTES_COL]),
        "age": row.get("age"),
        "percentiles": pct,
        "roles": roles,
        "pool_size": pool.height,
        "pool_idx": pool_idx,
    }


def analyze(
    player_name: str,
    team: str | None = None,
    league: str | None = None,
    *,
    events_dir: str | Path | None = None,
    season: int | str = tms.DEFAULT_SEASON,
    min_touches: int = tms.DEFAULT_MIN_TOUCHES,
    n_clusters: int = 14,
    top_n: int = 10,
    min_90s: float | None = DEFAULT_MIN_90S,
    stats_path: Path | str = DATA_PATH,
    strength_cutoff: float = STRENGTH_CUTOFF,
    weakness_cutoff: float = WEAKNESS_CUTOFF,
    top_stats: int = TOP_STATS,
    pca_var: float = DEFAULT_PCA_VAR,
    nmf_k: int = DEFAULT_NMF_K,
    rrf_k: int = DEFAULT_RRF_K,
    action_features: bool = True,
    save_maps_dir: str | Path | None = None,
) -> dict:
    """Assign touch (+ action-map) cluster, then find + profile quantitatively similar cluster mates.

    `save_maps_dir`, if given, saves the target's touch/pass/pass-angle/
    take-on/shot/defensive-action maps as separate PNGs (see `_save_target_maps`).
    Pass-angle tendency is computed for every cluster mate and fused into
    the stage-2 feature matrix as extra inputs (early fusion); the target's
    own numbers also live on `touch_target["pass_angles"]`.
    """
    touch_target, cluster_mates, cluster_label = _run_touchmap_cluster(
        player_name,
        team,
        league,
        events_dir,
        season,
        min_touches,
        n_clusters,
        action_features=action_features,
        save_maps_dir=save_maps_dir,
    )

    players = load_players(stats_path)
    feature_cols = [
        c
        for c in players.select(cs.numeric()).columns
        if c not in (MINUTES_COL, "age", "('born', '')")
    ]

    try:
        target_row, _ = resolve_fbref_row(
            players,
            touch_target["player"],
            touch_target["team"],
            touch_target.get("league"),
        )
    except ValueError as exc:
        return {
            "touch_target": touch_target,
            "target_profile": {"error": str(exc), "player": touch_target["player"]},
            "quant_matches": None,
            "quant_profiles": [],
            "strength_cutoff": strength_cutoff,
            "weakness_cutoff": weakness_cutoff,
            "top_stats": top_stats,
            "min_90s": min_90s,
            "cluster_label": cluster_label,
            "cluster_size": len(cluster_mates),
        }

    try:
        pool, pool_meta = build_fbref_cluster_pool(
            players, cluster_mates, target_row, min_90s=min_90s
        )
    except ValueError as exc:
        return {
            "touch_target": touch_target,
            "target_profile": {"error": str(exc), "player": touch_target["player"]},
            "quant_matches": None,
            "quant_profiles": [],
            "strength_cutoff": strength_cutoff,
            "weakness_cutoff": weakness_cutoff,
            "top_stats": top_stats,
            "min_90s": min_90s,
            "cluster_label": cluster_label,
            "cluster_size": len(cluster_mates),
            "pool_meta": {
                "cluster_touch_size": len(cluster_mates),
                "matched": 0,
                "unmatched": [],
            },
        }

    pos_group = target_row["primary_pos"]
    vec_by = touch_target.get("pass_angle_vec_by_player") or {}
    stats_by = touch_target.get("pass_angle_stats_by_player") or {}
    pass_angle_X, stats_by_pool_idx = _align_pass_angles_to_pool(
        pool, cluster_mates, vec_by, stats_by
    )

    quant = search_in_pool(
        pool,
        target_row,
        feature_cols,
        top_n=top_n,
        pca_var=pca_var,
        nmf_k=nmf_k,
        rrf_k=rrf_k,
        pass_angle_X=pass_angle_X,
    )

    target_profile = profile_from_pool_row(
        pool,
        quant["idx"],
        feature_cols,
        pos_group,
        strength_cutoff=strength_cutoff,
        weakness_cutoff=weakness_cutoff,
        top_stats=top_stats,
    )
    target_profile["pass_angles"] = stats_by_pool_idx.get(quant["idx"]) or touch_target.get(
        "pass_angles"
    )

    ang_scores = quant.get("pass_angle_cos")
    quant_profiles: list[dict] = []
    for rank, pool_idx in enumerate(quant["order"], start=1):
        prof = profile_from_pool_row(
            pool,
            int(pool_idx),
            feature_cols,
            pos_group,
            strength_cutoff=strength_cutoff,
            weakness_cutoff=weakness_cutoff,
            top_stats=top_stats,
        )
        prof["quant_rank"] = rank
        prof["rrf_score"] = float(quant["rrf"][pool_idx])
        prof["pass_angles"] = stats_by_pool_idx.get(int(pool_idx))
        if ang_scores is not None:
            cos = float(ang_scores[pool_idx])
            prof["pass_angle_cos"] = cos if np.isfinite(cos) else None
        quant_profiles.append(prof)

    return {
        "touch_target": touch_target,
        "target_profile": target_profile,
        "quant_matches": quant["consensus"],
        "quant_search": quant,
        "quant_profiles": quant_profiles,
        "strength_cutoff": strength_cutoff,
        "weakness_cutoff": weakness_cutoff,
        "top_stats": top_stats,
        "min_90s": min_90s,
        "cluster_label": cluster_label,
        "cluster_size": len(cluster_mates),
        "pool_meta": pool_meta,
    }


def _fmt_stat_line(col: str, value: float, pct: float) -> str:
    name = pretty_stat(col)
    return f"  {name}: {value:.2f}/90  (p{pct:.0f})"


def print_profile_block(
    label: str,
    profile: dict,
    strength_cutoff: float,
    weakness_cutoff: float,
    top_stats: int,
    pool_line: str = "",
    extra_header: str = "",
) -> None:
    if profile.get("error"):
        print(f"\n{label}: {profile['player']} — skipped ({profile['error']})")
        return

    age = profile.get("age")
    age_bit = f", age {age:.1f}" if age is not None else ""
    header = (
        f"\n{label}: {profile['player']} ({profile['team']}, {profile['league']})"
        f"  {profile['position']}  {profile['nineties']:.1f} 90s{age_bit}{extra_header}"
    )
    print(header)
    if pool_line:
        print(f"  {pool_line}")

    pct = profile["percentiles"]
    strengths = pct["strengths"][:top_stats]
    weaknesses = pct["weaknesses"][:top_stats]

    print(f"\n  Strengths (>= p{strength_cutoff:.0f}):")
    if strengths:
        for col, val, p in strengths:
            print(_fmt_stat_line(col, val, p))
    else:
        print("    (none above cutoff)")

    print(f"\n  Weaknesses (<= p{weakness_cutoff:.0f}):")
    if weaknesses:
        for col, val, p in weaknesses:
            print(_fmt_stat_line(col, val, p))
    else:
        print("    (none below cutoff)")

    print("\n  Role mix (RobustScaler blocks):")
    for row in profile["roles"].iter_rows(named=True):
        print(f"    {row['role']}: {row['score']:+.2f}")

    # Target tendency is printed once above step 2; repeats here for matches.
    pass_angles = profile.get("pass_angles")
    if pass_angles and label != "TARGET":
        cos = profile.get("pass_angle_cos")
        cos_bit = f"  |  cosine vs target {cos:+.3f}" if cos is not None else ""
        print(f"\n  Pass angle tendency{cos_bit}:")
        print(par.format_pass_angle_summary(pass_angles))


def print_report(result: dict) -> None:
    t = result["touch_target"]
    league_bit = f", {t['league']}" if t.get("league") else ""
    cluster_label = result.get("cluster_label", "?")
    cluster_size = result.get("cluster_size", "?")
    pool_meta = result.get("pool_meta", {})
    matched = pool_meta.get("matched", "?")
    min_90s = result.get("min_90s")
    min_bit = f", pool min {min_90s:.0f} 90s" if min_90s is not None else ""

    print(
        f"=== Step 1: touch-map cluster for {t['player']} ({t['team']}{league_bit}) ===\n"
        f"cluster {cluster_label} — {cluster_size} players with touch data, "
        f"{matched} matched in FBref{min_bit}"
    )
    unmatched = pool_meta.get("unmatched", [])
    if unmatched:
        n_skip = len(unmatched)
        print(
            f"  ({n_skip} not in FBref: {', '.join(unmatched[:5])}"
            + (" …" if n_skip > 5 else "") + ")"
        )

    maps_saved = t.get("maps_saved")
    if maps_saved:
        print(f"  maps saved: {', '.join(f'{name}={path}' for name, path in maps_saved.items())}")

    pass_angles = t.get("pass_angles")
    if pass_angles:
        print("\n=== Pass angle tendency ===")
        print(par.format_pass_angle_summary(pass_angles))
    else:
        print("\n=== Pass angle tendency ===")
        print("  (no directed passes)")

    quant_search = result.get("quant_search")
    if quant_search:
        angle_bit = (
            ", pass-angle features fused"
            if quant_search.get("pass_angle_used")
            else ", pass-angle skipped"
        )
        print(
            f"\n=== Step 2: quantitative matches within cluster "
            f"(pca={quant_search['n_pca']}, "
            f"{100 * quant_search['pca_var_explained']:.0f}% var{angle_bit}) ==="
        )
        if result.get("quant_matches") is not None:
            print(result["quant_matches"])

    sc = result["strength_cutoff"]
    wc = result["weakness_cutoff"]
    ts = result["top_stats"]
    pool_line = (
        f"percentiles vs {matched} FBref players in touch cluster {cluster_label}"
    )

    print("\n=== Strengths / weaknesses ===")
    print_profile_block("TARGET", result["target_profile"], sc, wc, ts, pool_line)

    for prof in result.get("quant_profiles", []):
        extra = f"  |  quant rank #{prof['quant_rank']}, rrf={prof['rrf_score']:.4f}"
        print_profile_block(
            f"MATCH #{prof['quant_rank']}",
            prof,
            sc,
            wc,
            ts,
            pool_line,
            extra_header=extra,
        )


def _montage_from_quant(result: dict) -> pd.DataFrame | None:
    """Build a touch-map montage frame from quantitative matches."""
    profiles = result.get("quant_profiles")
    if not profiles:
        return None
    rows = [
        {
            "rank": p["quant_rank"],
            "player": p["player"],
            "team": p["team"],
            "league": p["league"],
        }
        for p in profiles
    ]
    similar = pd.DataFrame(rows)
    similar.attrs["target"] = result["touch_target"]
    return similar


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Touch-map cluster assignment + quantitative FBref similarity "
            "within cluster (pass-angle tendency is a stage-2 RRF channel)."
        )
    )
    parser.add_argument("player", help="Target player (partial, case-insensitive)")
    parser.add_argument("--team", default=None, help="Disambiguate touch-map / FBref lookup")
    parser.add_argument("--league", default=None, help="Disambiguate across leagues")
    parser.add_argument("--events-dir", default=None, help="Single league events folder")
    parser.add_argument("--season", type=int, default=tms.DEFAULT_SEASON)
    parser.add_argument("--min-touches", type=int, default=tms.DEFAULT_MIN_TOUCHES)
    parser.add_argument("--n-clusters", type=int, default=14)
    parser.add_argument(
        "--no-action-features",
        action="store_true",
        help="Cluster on touch location only, skipping pass/take-on/shot/defensive-action shape",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Quantitatively similar cluster mates to return and profile",
    )
    parser.add_argument(
        "--min-90s",
        type=float,
        default=None,
        help="Drop cluster pool members below this many 90s (target always kept)",
    )
    parser.add_argument("--pca-var", type=float, default=DEFAULT_PCA_VAR)
    parser.add_argument("--nmf-k", type=int, default=DEFAULT_NMF_K)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--stats-path", default=str(DATA_PATH))
    parser.add_argument("--strength-cutoff", type=float, default=STRENGTH_CUTOFF)
    parser.add_argument("--weakness-cutoff", type=float, default=WEAKNESS_CUTOFF)
    parser.add_argument("--top-stats", type=int, default=TOP_STATS)
    parser.add_argument(
        "--montage",
        default=None,
        help="Save touch-map PNG montage of target + quantitative matches",
    )
    parser.add_argument(
        "--save-maps",
        default=None,
        metavar="DIR",
        help="Save the target's touch/pass/pass-angle/take-on/shot/defensive-action "
        "maps as separate PNGs in this directory",
    )
    args = parser.parse_args()

    result = analyze(
        args.player,
        team=args.team,
        league=args.league,
        events_dir=args.events_dir,
        season=args.season,
        min_touches=args.min_touches,
        n_clusters=args.n_clusters,
        top_n=args.top_n,
        min_90s=args.min_90s,
        stats_path=args.stats_path,
        strength_cutoff=args.strength_cutoff,
        weakness_cutoff=args.weakness_cutoff,
        top_stats=args.top_stats,
        pca_var=args.pca_var,
        nmf_k=args.nmf_k,
        rrf_k=args.rrf_k,
        action_features=not args.no_action_features,
        save_maps_dir=args.save_maps,
    )
    print_report(result)

    if args.montage:
        montage_df = _montage_from_quant(result)
        if montage_df is not None:
            tms.plot_similar_montage(
                montage_df,
                season=args.season,
                out_path=args.montage,
            )
            print(f"\nmontage saved to {args.montage}")


if __name__ == "__main__":
    main()
