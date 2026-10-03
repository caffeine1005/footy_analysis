"""Compare two ways of choosing the stage-1 candidate pool for similar-player search.

- kmeans: the target's KMeans cluster (what `season_similarity.analyze_seasons` does now)
- knn:    the target's N nearest player-seasons in the same sqrt+PCA shape space

Both use exactly the geometry of `touchmap_similarity.cluster_players` (Hellinger
transform, PCA to 90% variance), so the only difference is how the pool is cut.

    python eval_pool_methods.py                  # every available season
    python eval_pool_methods.py --seasons 2023 2024 2025

Reports:
  1. Boundary misses — share of each player's true nearest neighbours that
     their cluster excludes (kNN excludes none by construction).
  2. Pool sizes — cluster sizes vs a fixed N.
  3. Scope stability — the same season's players, pool picked from a
     one-season fit vs an all-seasons fit (the profile/similarity-page gap).
  4. k and seed sensitivity — how much the pool moves when only k or the
     random seed changes.
  5. Named examples — nearest players each target's cluster shuts out.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

import season_similarity as ss
import sofascore_similarity as sofa

EXAMPLES = [
    "Mohamed Salah", "Bruno Fernandes", "Bukayo Saka", "Rodri", "Declan Rice",
    "Trent Alexander-Arnold", "Virgil van Dijk", "Erling Haaland", "Martin Ødegaard",
    "Florian Wirtz", "Jude Bellingham", "Achraf Hakimi",
]


def embed(X: np.ndarray, pca_var: float = 0.90, seed: int = 42) -> np.ndarray:
    """Same transform as `touchmap_similarity.cluster_players`, minus the KMeans."""
    Xh = np.sqrt(X.astype(np.float64, copy=False))
    full = PCA(random_state=seed).fit(Xh)
    n = int(np.clip(np.searchsorted(np.cumsum(full.explained_variance_ratio_), pca_var) + 1,
                    2, max(2, min(Xh.shape) - 1)))
    return PCA(n_components=n, random_state=seed).fit_transform(Xh)


def kmeans_labels(coords: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(coords).labels_


def knn_index(coords: np.ndarray, n: int) -> np.ndarray:
    """(rows, n) nearest other rows, closest first."""
    nn = NearestNeighbors(n_neighbors=n + 1).fit(coords)
    idx = nn.kneighbors(coords, return_distance=False)
    return idx[:, 1:]


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def cluster_pools(labels: np.ndarray, rows: np.ndarray) -> list[set]:
    """Pool (as a set of row ids, restricted to `rows`) for each row in `rows`."""
    by_label = {}
    for r in rows:
        by_label.setdefault(labels[r], set()).add(int(r))
    return [by_label[labels[r]] - {int(r)} for r in rows]


def knn_pools(coords: np.ndarray, rows: np.ndarray, n: int) -> list[set]:
    """N nearest among `rows` only (so pools are comparable across fits)."""
    sub = coords[rows]
    idx = knn_index(sub, min(n, len(rows) - 1))
    return [set(int(rows[j]) for j in nbrs) for nbrs in idx]


def summarize(values, label: str) -> str:
    v = np.asarray(values, dtype=float)
    return (f"{label:<34} mean {v.mean():.2f} | median {np.median(v):.2f} | "
            f"p10 {np.percentile(v, 10):.2f} | p90 {np.percentile(v, 90):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*")
    ap.add_argument("--focus", type=int, default=None, help="season for tests 3-5 (default: latest)")
    ap.add_argument("--k", type=int, default=sofa.DEFAULT_N_CLUSTERS)
    ap.add_argument("--n", type=int, default=150, help="kNN pool size")
    args = ap.parse_args()

    years = args.seasons or ss.available_seasons()
    focus = args.focus or max(years)
    t0 = time.time()

    metas, Xs = [], []
    for y in years:
        print(f"[{ss.season_label(y)}] shape vectors ...", flush=True)
        _, meta, X, _, _ = ss._season_grids(
            y, min_touches=sofa.DEFAULT_MIN_TOUCHES, action_features=True,
            progress=lambda m: print(f"   {m}", flush=True),
        )
        meta = meta.copy()
        meta[ss.SEASON_COL] = y
        metas.append(meta)
        Xs.append(X.astype(np.float32, copy=False))
    meta_all = pd.concat(metas, ignore_index=True)
    X_all = np.vstack(Xs)
    print(f"\n{len(meta_all):,} player-seasons, {X_all.shape[1]} features, "
          f"{len(years)} seasons ({time.time() - t0:.0f}s)\n")

    # ---- all-seasons fit: what the similarity page does with nothing ticked
    coords_all = embed(X_all)
    labels_all = kmeans_labels(coords_all, args.k)
    all_rows = np.arange(len(meta_all))

    # 1. boundary misses ------------------------------------------------------
    print("=" * 78)
    print(f"1. BOUNDARY MISSES  (all-seasons fit, k={args.k}, {coords_all.shape[1]} PCA dims)")
    print("   share of a player's true nearest neighbours that their cluster excludes")
    print("-" * 78)
    for m in (10, 25, 50):
        nbrs = knn_index(coords_all, m)
        miss = (labels_all[nbrs] != labels_all[:, None]).mean(axis=1)
        print(summarize(miss, f"top-{m} neighbours outside cluster"))
        if m == 25:
            miss25 = miss
    print(f"   players missing >=half their top-25:  {(miss25 >= 0.5).mean():.1%}")

    # 2. pool sizes -----------------------------------------------------------
    sizes = np.bincount(labels_all)
    print("\n" + "=" * 78)
    print("2. POOL SIZES  (all-seasons fit)")
    print("-" * 78)
    print(f"   kmeans cluster sizes: min {sizes.min():,} | median {int(np.median(sizes)):,} | "
          f"max {sizes.max():,}  (ratio {sizes.max() / sizes.min():.1f}x)")
    print(f"   knn pool: always {args.n}")

    # 3. scope stability ------------------------------------------------------
    f_rows = np.flatnonzero(meta_all[ss.SEASON_COL].to_numpy() == focus)
    coords_one = embed(X_all[f_rows])
    labels_one = kmeans_labels(coords_one, args.k)
    local = np.arange(len(f_rows))

    km_one = cluster_pools(labels_one, local)
    km_all = cluster_pools(labels_all, f_rows)
    km_all = [{int(np.searchsorted(f_rows, r)) for r in s} for s in km_all]

    n_one = min(args.n // 3, len(f_rows) - 1)  # one season is ~1/len(years) of the rows
    knn_one = knn_pools(coords_one, local, n_one)
    knn_all = knn_pools(coords_all, f_rows, n_one)
    knn_all = [{int(np.searchsorted(f_rows, r)) for r in s} for s in knn_all]

    print("\n" + "=" * 78)
    print(f"3. SCOPE STABILITY  ({ss.season_label(focus)} players: fit on {ss.season_label(focus)} "
          f"alone vs fit on all {len(years)} seasons)")
    print("   Jaccard overlap of the same player's pool (among that season's players)")
    print("-" * 78)
    print(summarize([jaccard(a, b) for a, b in zip(km_one, km_all)], "kmeans"))
    print(summarize([jaccard(a, b) for a, b in zip(knn_one, knn_all)], f"knn (n={n_one})"))

    # 4. k / seed sensitivity -------------------------------------------------
    print("\n" + "=" * 78)
    print(f"4. SENSITIVITY  ({ss.season_label(focus)} fit, pool vs the k={args.k} seed=42 pool)")
    print("-" * 78)
    for k in (args.k - 4, args.k + 4):
        alt = cluster_pools(kmeans_labels(coords_one, k), local)
        print(summarize([jaccard(a, b) for a, b in zip(km_one, alt)], f"kmeans k={k}"))
    seed_j = []
    for seed in (0, 1, 2):
        alt = cluster_pools(kmeans_labels(coords_one, args.k, seed=seed), local)
        seed_j.extend(jaccard(a, b) for a, b in zip(km_one, alt))
    print(summarize(seed_j, "kmeans other seeds (0,1,2)"))
    for n in (n_one - 15, n_one + 15):
        alt = knn_pools(coords_one, local, n)
        print(summarize([jaccard(a, b) for a, b in zip(knn_one, alt)], f"knn n={n}"))

    # 5. examples -------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"5. EXAMPLES  ({ss.season_label(focus)} fit, k={args.k}): closest 10 players, "
          "[x] = shut out by the cluster")
    print("-" * 78)
    fmeta = meta_all.iloc[f_rows].reset_index(drop=True)
    nbrs10 = knn_index(coords_one, 10)
    fsizes = np.bincount(labels_one)
    for name in EXAMPLES:
        hit = fmeta.index[fmeta["player"] == name]
        if hit.empty:
            continue
        i = int(hit[0])
        near = nbrs10[i]
        out = [j for j in near if labels_one[j] != labels_one[i]]
        cells = [f"{'[x] ' if labels_one[j] != labels_one[i] else ''}{fmeta.at[j, 'player']}"
                 for j in near]
        print(f"{name} ({fmeta.at[i, 'team']}) — cluster {labels_one[i]} "
              f"(size {fsizes[labels_one[i]]}), {len(out)}/10 shut out")
        print("    " + ", ".join(cells))
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
