"""Ground-truth-free benchmark for the shape-similarity space.

The test: a player who stays at the same club from one season to the next
mostly keeps the same role, so his season-N vector should be among the
nearest season-(N-1) vectors to it. For every such pair we rank the player's
own previous season among *all* of that season's player-seasons and report:

  hit@1 / hit@10   previous season is the closest / in the closest 10
  median rank      where it lands
  in pool          it lands inside the stage-1 pool (kmeans cluster or kNN top-N)
  switched side    the same numbers on the subset of players who moved flank
                   (mean touch y crossed the middle), where a side-agnostic
                   metric should help

It is not a perfect oracle — players change roles and managers change systems —
but it treats every variant the same way, so it can rank them.

    python benchmark_similarity.py                # all variants, all seasons
    python benchmark_similarity.py --tune         # also learn block weights
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import season_similarity as ss
import sofascore_similarity as sofa
import touchmap_similarity as tms

GRID = (8, 12)  # (rows = pitch width y, cols = pitch length x), as mplsoccer bins (12, 8)
CELLS = GRID[0] * GRID[1]
WEIGHTS_PATH = Path("similarity_block_weights.json")


# ---------------------------------------------------------------------------
# feature layout
# ---------------------------------------------------------------------------


block_names = tms.feature_block_names


def split_blocks(X: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    names = block_names()
    grids = [X[:, i * CELLS:(i + 1) * CELLS] for i in range(len(names))]
    rates = X[:, len(names) * CELLS:]
    assert rates.shape[1] == len(tms.CREATION_RATE_COLS), "X layout changed"
    return grids, rates


def mirror(X: np.ndarray) -> np.ndarray:
    """Flip every heatmap block across the pitch's long axis (left <-> right flank)."""
    return tms.mirror_heatmaps(X)


def coarsen(g: np.ndarray, f: int) -> np.ndarray:
    """Sum f x f cells: (n, 96) -> (n, 96 / f^2). Mass is preserved."""
    n = len(g)
    r, c = GRID
    return g.reshape(n, r // f, f, c // f, f).sum(axis=(2, 4)).reshape(n, -1)


def feature_space(
    X: np.ndarray,
    block_w: np.ndarray | None = None,
    scales: tuple[float, ...] = (1.0,),
) -> np.ndarray:
    """Hellinger space with optional per-block weights and extra coarse scales.

    `scales[0]` weights the native 12x8 grid, `scales[1]` a 6x4 version and
    `scales[2]` a 3x2 version of every block. Comparing heatmaps at several
    resolutions is a cheap stand-in for earth mover's distance: a player whose
    touches sit one cell over still matches at the coarse scale.
    """
    grids, rates = split_blocks(X)
    w = np.ones(len(grids) + 1) if block_w is None else block_w
    parts = []
    for s, f in zip(scales, (1, 2, 4)):
        if s == 0:
            continue
        for g, wb in zip(grids, w[:-1]):
            gg = g if f == 1 else coarsen(g, f)
            parts.append(np.sqrt(gg) * (wb * s))
    parts.append(np.sqrt(rates) * w[-1])
    return np.hstack(parts).astype(np.float32)


def embed(F: np.ndarray, pca_var: float = 0.90, seed: int = 42) -> PCA:
    return PCA(n_components=pca_var, svd_solver="full", random_state=seed).fit(F)


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


def load_all(years) -> tuple[pd.DataFrame, np.ndarray]:
    metas, Xs = [], []
    for y in years:
        _, meta, X, _, _ = ss._season_grids(
            y, min_touches=sofa.DEFAULT_MIN_TOUCHES, action_features=True
        )
        meta = meta.copy()
        meta[ss.SEASON_COL] = y
        metas.append(meta)
        Xs.append(X.astype(np.float32, copy=False))
    return pd.concat(metas, ignore_index=True), np.vstack(Xs)


def side_offset(X: np.ndarray) -> np.ndarray:
    """Touch-weighted row centroid minus the middle: sign = which flank."""
    touch = split_blocks(X)[0][0].reshape(-1, *GRID).sum(axis=2)  # (n, rows)
    rows = np.arange(GRID[0])
    return (touch * rows).sum(axis=1) / np.maximum(touch.sum(axis=1), 1e-12) - (GRID[0] - 1) / 2


def season_pairs(meta: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
    """(query row, answer row) for same player + same team in consecutive seasons."""
    key = meta[["player", "team"]].astype(str).agg("|".join, axis=1)
    season = meta[ss.SEASON_COL].to_numpy()
    row_of = {(k, s): i for i, (k, s) in enumerate(zip(key, season))}
    q, a = [], []
    for i, (k, s) in enumerate(zip(key, season)):
        j = row_of.get((k, s - 1))
        if j is not None:
            q.append(i)
            a.append(j)
    pairs = pd.DataFrame({"q": q, "a": a, "season": season[q]})
    off = side_offset(X)
    # Moved flank: clearly on one side, then clearly on the other.
    pairs["switched"] = (np.sign(off[pairs["q"]]) != np.sign(off[pairs["a"]])) & (
        np.minimum(np.abs(off[pairs["q"]]), np.abs(off[pairs["a"]])) > 0.5
    )
    return pairs


def duplicated_seasons(meta: pd.DataFrame, X: np.ndarray) -> list[int]:
    """Seasons whose vectors are (near) copies of the season before.

    It has happened: every league_games/*_2021 folder held the 20/21 matches.
    Such a season makes every pair touching it trivially rank 1, and counts
    the copied season twice in the PCA fit, so it has to go before anything.
    """
    pairs = season_pairs(meta, X)
    gap = np.abs(X[pairs["q"]] - X[pairs["a"]]).sum(axis=1)
    return [int(s) for s, g in pd.Series(gap).groupby(pairs["season"].to_numpy()) if np.median(g) < 0.01]


def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T


def answer_ranks(
    coords: np.ndarray,
    meta: pd.DataFrame,
    pairs: pd.DataFrame,
    coords_mirror: np.ndarray | None = None,
) -> np.ndarray:
    """Rank (1 = nearest) of each pair's answer among the answer's whole season."""
    season = meta[ss.SEASON_COL].to_numpy()
    ranks = np.empty(len(pairs), dtype=np.int64)
    for s, grp in pairs.groupby("season"):
        cand = np.flatnonzero(season == s - 1)
        pos = {r: k for k, r in enumerate(cand)}
        D = _sqdist(coords[grp["q"].to_numpy()], coords[cand])
        if coords_mirror is not None:
            D = np.minimum(D, _sqdist(coords[grp["q"].to_numpy()], coords_mirror[cand]))
        target = D[np.arange(len(grp)), [pos[a] for a in grp["a"]]]
        ranks[grp.index.to_numpy()] = (D < target[:, None]).sum(axis=1) + 1
    return ranks


def summarize(ranks: np.ndarray, pairs: pd.DataFrame, pool_hit: np.ndarray | None = None) -> dict:
    sw = pairs["switched"].to_numpy()
    out = {
        "hit@1": float((ranks == 1).mean()),
        "hit@10": float((ranks <= 10).mean()),
        "median_rank": float(np.median(ranks)),
        "mean_log_rank": float(np.log(ranks).mean()),
        "switched_hit@10": float((ranks[sw] <= 10).mean()) if sw.any() else float("nan"),
        "switched_median": float(np.median(ranks[sw])) if sw.any() else float("nan"),
    }
    if pool_hit is not None:
        out["in_pool"] = float(pool_hit.mean())
    return out


def kmeans_pool_hit(
    coords, meta, pairs, ranks, k=sofa.DEFAULT_N_CLUSTERS, seed=42
) -> tuple[np.ndarray, np.ndarray]:
    """(answer in the query's kmeans cluster, answer in a kNN pool of the same size).

    The kNN pool for each query is sized to its cluster's membership in the
    answer's season, so the comparison is like for like.
    """
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(coords).labels_
    season = meta[ss.SEASON_COL].to_numpy()
    size = pd.Series(1, index=pd.MultiIndex.from_arrays([season, labels])).groupby(level=[0, 1]).size()
    q, a = pairs["q"].to_numpy(), pairs["a"].to_numpy()
    n = size.reindex(pd.MultiIndex.from_arrays([season[a], labels[q]])).fillna(0).to_numpy()
    return labels[q] == labels[a], ranks <= n


def fmt(name: str, r: dict) -> str:
    pool = f"{r['in_pool']:.1%}" if "in_pool" in r else "   -  "
    return (f"{name:<38} hit@1 {r['hit@1']:.1%} | hit@10 {r['hit@10']:.1%} | "
            f"median {r['median_rank']:>5.0f} | in pool {pool} | "
            f"switched hit@10 {r['switched_hit@10']:.1%} (median {r['switched_median']:.0f})")


# ---------------------------------------------------------------------------
# block-weight tuning
# ---------------------------------------------------------------------------


def tune_weights(
    X: np.ndarray,
    meta: pd.DataFrame,
    pairs: pd.DataFrame,
    scales: tuple[float, ...],
    per_season: int = 400,
    grid=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
    passes: int = 2,
    seed: int = 0,
) -> np.ndarray:
    """Coordinate search on per-block weights, maximizing -mean log rank.

    Runs in the un-PCA'd Hellinger space so each block's squared distance can
    be precomputed once and re-weighted for free: D = sum_b w_b^2 * D_b.
    Only `pairs` passed in are used, so pass the training seasons only.
    """
    rng = np.random.default_rng(seed)
    season = meta[ss.SEASON_COL].to_numpy()
    n_blocks = len(block_names()) + 1
    unit = np.eye(n_blocks)

    # Per block: (queries x candidates) squared distances, one stack per season pair.
    stacks, targets = [], []
    for s, grp in pairs.groupby("season"):
        grp = grp.iloc[rng.permutation(len(grp))[:per_season]]
        cand = np.flatnonzero(season == s - 1)
        pos = {r: k for k, r in enumerate(cand)}
        Db = []
        for b in range(n_blocks):
            F = feature_space(X, block_w=unit[b], scales=scales)
            Db.append(_sqdist(F[grp["q"].to_numpy()], F[cand]).astype(np.float32))
        stacks.append(np.stack(Db))  # (blocks, Q, C)
        targets.append(np.array([pos[a] for a in grp["a"]]))

    def score(w):
        w2 = (w ** 2)[:, None, None]
        logs = []
        for Db, t in zip(stacks, targets):
            D = (Db * w2).sum(axis=0)
            tgt = D[np.arange(len(t)), t]
            logs.append(np.log((D < tgt[:, None]).sum(axis=1) + 1))
        return -np.concatenate(logs).mean()

    w = np.ones(n_blocks)
    best = score(w)
    names = [*block_names(), "creation_rates"]
    for p in range(passes):
        for b in range(n_blocks):
            for g in grid:
                trial = w.copy()
                trial[b] = g
                if not trial.any():
                    continue
                s = score(trial)
                if s > best + 1e-4:
                    best, w = s, trial
        print(f"   pass {p + 1}: mean log rank {-best:.3f}  "
              + ", ".join(f"{n}={v:g}" for n, v in zip(names, w) if v != 1.0), flush=True)
    return w


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*")
    ap.add_argument("--tune", action="store_true", help="learn block weights on early seasons")
    ap.add_argument("--test-from", type=int, default=2021, help="first season of the held-out test set")
    ap.add_argument("--knn", type=int, default=150, help="kNN pool size per season searched")
    args = ap.parse_args()

    t0 = time.time()
    years = args.seasons or ss.available_seasons()
    meta, X = load_all(years)
    dup = duplicated_seasons(meta, X)
    if dup:
        print("WARNING: season data duplicated from the season before — "
              + ", ".join(f"{ss.season_label(s)} == {ss.season_label(s - 1)}" for s in dup)
              + "; dropping it", flush=True)
        keep = ~meta[ss.SEASON_COL].isin(dup).to_numpy()
        meta, X = meta[keep].reset_index(drop=True), X[keep]
    pairs = season_pairs(meta, X)
    test = pairs["season"].to_numpy() >= args.test_from
    print(f"{len(meta):,} player-seasons, {len(pairs):,} same-club consecutive pairs "
          f"({pairs['switched'].sum():,} switched flank); test = {test.sum():,} pairs from "
          f"{ss.season_label(args.test_from)} on ({time.time() - t0:.0f}s)\n")

    results = {}

    def run(name, F, mirrored=False):
        pca = embed(F)
        C = pca.transform(F)
        Cm = pca.transform(F_mirror_of[name]) if mirrored else None
        ranks = answer_ranks(C, meta, pairs, Cm)
        tp = pairs[test].reset_index(drop=True)
        r = ranks[test]
        results[name] = summarize(r, tp, r <= args.knn)
        print(fmt(name, results[name]), f"[{C.shape[1]} dims]", flush=True)
        return C, ranks

    F_mirror_of = {}

    print("=" * 110)
    print(f"HELD-OUT TEST SEASONS (>= {ss.season_label(args.test_from)}), 'in pool' = answer in the "
          f"top-{args.knn} kNN pool, unless noted")
    print("-" * 110)

    # 1. baseline space, kmeans vs knn pool
    F0 = feature_space(X)
    C0, r0 = run("baseline", F0)
    km, knn_same = kmeans_pool_hit(C0, meta, pairs, r0)
    print(f"{'':<38} stage-1 pool, same size: kmeans cluster {km[test].mean():.1%} "
          f"vs kNN {knn_same[test].mean():.1%} contain the answer")

    # 2. mirroring: min distance over the candidate and its flipped self
    F_mirror_of["mirror"] = feature_space(mirror(X))
    run("mirror", F0, mirrored=True)

    # 3. multi-scale (EMD stand-in)
    scale_runs = {(1.0,): "baseline"}
    for scales in ((1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.5, 0.5)):
        scale_runs[scales] = f"multiscale {scales}"
        run(scale_runs[scales], feature_space(X, scales=scales))

    # 4. tuned block weights (trained on early seasons only)
    if args.tune:
        print("\nTuning block weights on seasons before the test set ...", flush=True)
        best = max(scale_runs, key=lambda s: -results[scale_runs[s]]["mean_log_rank"])
        scales = best
        train = pairs[~test].reset_index(drop=True)
        w = tune_weights(X, meta, train, scales)
        WEIGHTS_PATH.write_text(json.dumps(
            {"scales": list(scales), "weights": dict(zip([*block_names(), "creation_rates"], w.tolist()))},
            indent=2,
        ))
        run(f"tuned weights, scales {scales}", feature_space(X, block_w=w, scales=scales))
        F_mirror_of[f"tuned + mirror"] = feature_space(mirror(X), block_w=w, scales=scales)
        run("tuned + mirror", feature_space(X, block_w=w, scales=scales), mirrored=True)

    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
