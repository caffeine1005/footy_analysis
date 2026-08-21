"""Position-aware similar-player search on the FBref table from fbref_test.ipynb.

Position is a comparison pool: scaler, PCA, role scores and NMF are fit on the
target's group only (DF / MF / FW). Change QUERY and call search().
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs
from scipy.stats import rankdata
from sklearn.decomposition import NMF, PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

DATA_PATH = Path("player_stats.csv")
POS_COL = "('pos', '')"
MINUTES_COL = "('Playing Time', '90s')"
AGE_COL = "('age', '')"
ID_COLS = ["league", "team", "player", POS_COL, AGE_COL, "('born', '')"]

REDUNDANT_COLS = [
    "('Performance', 'G+A')",
    "('Performance', 'G-PK')",
    "('Expected', 'npxG+xAG')",
    "('Expected', 'A-xAG')",
    "('Progression', 'PrgC')",
    "('Progression', 'PrgP')",
    "('Progression', 'PrgR')",
    "('Ast', '')",
    "('xAG', '')",
    "('Touches', 'Touches')",
    "('Touches', 'Live')",
    "('SCA', 'SCA')",
    "('GCA', 'GCA')",
    "('Total', 'Cmp')",
    "('Total', 'Att')",
    "('Tackles', 'Tkl')",
    "('Tkl+Int', '')",
    "('Blocks', 'Blocks')",
    "('Challenges', 'Lost')",
]

ADJACENT = {
    "GK": [],
    "DF": ["MF"],
    "MF": ["DF", "FW"],
    "FW": ["MF"],
}

# Hand-built role templates. Values are substrings matched against column names.
ROLE_BLOCKS = {
    "MF": {
        "creator": ["xAG", "xA", "KP", "PPA", "PrgP", "PassLive", "SCA Types"],
        "carrier": ["PrgC", "Take-Ons", "CPA", "PrgDist", "Carries', '1/3"],
        "ball_winner": ["TklW", "Int", "Clr", "Challenges", "Blocks", "Tackles', 'Mid"],
        "box": ["npxG", "Gls", "Att Pen", "CPA"],
    },
    "FW": {
        "finisher": ["npxG", "Gls", "Att Pen"],
        "link": ["xAG", "xA", "PrgR", "Rec", "PassLive", "KP"],
        "dribbler": ["Take-Ons", "CPA", "PrgC", "Att 3rd"],
        "presser": ["Tackles', 'Att", "Challenges", "Int"],
    },
    "DF": {
        "defender": ["TklW", "Int", "Clr", "Blocks', 'Sh", "Challenges", "Def Pen", "Def 3rd"],
        "progressor": ["PrgP", "PrgC", "PrgDist", "Total', 'PrgDist", "1/3"],
        "width": ["CrsPA", "PPA", "Att 3rd", "Take-Ons"],
        "box_presence": ["Att Pen", "npxG", "Gls"],
    },
}

DEFAULT_QUERY = {
    "player": "Bruno Fernandes",
    "team": None,
    "league": None,
    "min_90s": 8.0,
    "max_age": None,  # only compare to players at or below this age (years)
    "allow_adjacent_pos": False,
    "top_n": 9,
    "pca_var": 0.80,
    "nmf_k": 5,
    "rrf_k": 60,
}

_pool_cache: dict[tuple, dict] = {}
_players: pl.DataFrame | None = None
_feature_cols: list[str] = []


def primary_pos(pos) -> str | None:
    if pos is None:
        return None
    token = str(pos).split(",")[0].strip().upper()
    return token if token in ADJACENT else None


def parse_age(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if "-" in text:
        years, days = text.split("-", 1)
        try:
            return int(years) + int(days) / 365.0
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def load_players(path: Path | str = DATA_PATH) -> pl.DataFrame:
    raw = pl.read_csv(path)
    raw = raw.drop([c for c in REDUNDANT_COLS if c in raw.columns])

    numeric_cols = [c for c in raw.select(cs.numeric()).columns if c not in ID_COLS]
    pct_cols = [c for c in numeric_cols if "%" in c]
    count_cols = [c for c in numeric_cols if c not in pct_cols and c != MINUTES_COL]

    players = (
        raw.sort(MINUTES_COL, descending=True)
        .group_by("player", maintain_order=True)
        .agg(
            pl.first("league"),
            pl.first("team"),
            pl.first(POS_COL),
            pl.first(AGE_COL),
            pl.col(MINUTES_COL).sum(),
            *[pl.col(c).sum() for c in count_cols],
            *[
                (
                    (pl.col(c) * pl.col(MINUTES_COL)).sum()
                    / pl.col(MINUTES_COL).filter(pl.col(c).is_not_null()).sum()
                ).alias(c)
                for c in pct_cols
            ],
        )
    )
    players = players.with_columns(
        [(pl.col(c) / pl.col(MINUTES_COL)).alias(c) for c in count_cols]
    )
    players = players.with_columns(
        [
            pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c)
            for c in numeric_cols
            if c != MINUTES_COL
        ]
    )
    players = players.with_columns(
        [pl.col(c).fill_null(pl.col(c).median()) for c in pct_cols]
    )
    players = players.with_columns(
        [pl.col(c).fill_null(0.0) for c in count_cols]
    )
    return players.with_columns(
        pl.col(POS_COL).map_elements(primary_pos, return_dtype=pl.String).alias("primary_pos"),
        pl.col(AGE_COL)
        .map_elements(parse_age, return_dtype=pl.Float64)
        .alias("age"),
    )


def _ensure_loaded(path: Path | str = DATA_PATH) -> tuple[pl.DataFrame, list[str]]:
    global _players, _feature_cols
    if _players is None:
        _players = load_players(path)
        _feature_cols = [
            c
            for c in _players.select(cs.numeric()).columns
            if c not in (MINUTES_COL, "age", "('born', '')")
        ]
    return _players, _feature_cols


def resolve_target(df: pl.DataFrame, q: dict) -> tuple[dict, pl.DataFrame]:
    matches = df.with_row_index().filter(
        pl.col("player").str.contains("(?i)" + q["player"])
    )
    if q.get("team"):
        matches = matches.filter(pl.col("team").str.contains("(?i)" + q["team"]))
    if q.get("league"):
        matches = matches.filter(pl.col("league").str.contains("(?i)" + q["league"]))
    if matches.height == 0:
        raise ValueError(
            f"no player matching {q['player']!r} "
            f"(team={q.get('team')!r}, league={q.get('league')!r})"
        )
    row = matches.row(0, named=True)
    if row["primary_pos"] is None:
        raise ValueError(f"could not parse position {row[POS_COL]!r}")
    if row["primary_pos"] == "GK":
        raise ValueError(
            "this table is outfield FBref stats; keepers need a separate keeper feature set"
        )
    return row, matches


def comparison_pool(df: pl.DataFrame, target: dict, q: dict) -> pl.DataFrame:
    groups = {target["primary_pos"]}
    if q["allow_adjacent_pos"]:
        groups.update(ADJACENT[target["primary_pos"]])
    pool = df.filter(
        pl.col("primary_pos").is_in(sorted(groups))
        & (pl.col(MINUTES_COL) >= q["min_90s"])
    )
    max_age = q.get("max_age")
    if max_age is not None:
        pool = pool.filter(
            pl.col("age").is_not_null()
            & (
                (pl.col("age") <= float(max_age))
                | (pl.col("player") == target["player"])
            )
        )
    if pool.filter(pl.col("player") == target["player"]).height == 0:
        raise ValueError(
            f"{target['player']} has {target[MINUTES_COL]:.1f} 90s, "
            f"below min_90s={q['min_90s']}"
        )
    if pool.height < 15:
        raise ValueError(
            f"pool too small ({pool.height}); lower min_90s or allow adjacent positions"
        )
    return pool


def pca_n_components(X: np.ndarray, var_cutoff: float) -> int:
    cap = max(2, min(X.shape[0] - 1, X.shape[1]))
    pca = PCA().fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n = int(np.searchsorted(cum, var_cutoff) + 1)
    return int(np.clip(n, 2, cap))


def role_matrix(
    X_scaled: np.ndarray, cols: list[str], pos_group: str
) -> tuple[np.ndarray, list[str], dict[str, list[str]]]:
    blocks = ROLE_BLOCKS[pos_group]
    names: list[str] = []
    used: dict[str, list[str]] = {}
    parts: list[np.ndarray] = []
    for name, needles in blocks.items():
        idx = [i for i, c in enumerate(cols) if any(n in c for n in needles)]
        if not idx:
            continue
        names.append(name)
        used[name] = [cols[i] for i in idx]
        parts.append(X_scaled[:, idx].mean(axis=1))
    if not parts:
        raise ValueError(f"no role-block columns matched for {pos_group}")
    return np.column_stack(parts), names, used


def spearman_scores(X: np.ndarray, idx: int) -> np.ndarray:
    ranks = rankdata(X, axis=1).astype(float)
    centered = ranks - ranks.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    z = centered / norms
    scores = z @ z[idx]
    scores[ranks.std(axis=1) == 0] = -np.inf
    return scores


def fit_pool(pool: pl.DataFrame, feature_cols: list[str], q: dict) -> dict:
    X = pool.select(feature_cols).to_numpy()
    X_scaled = RobustScaler().fit_transform(X)

    n_pca = pca_n_components(X_scaled, q["pca_var"])
    pca = PCA(n_pca).fit(X_scaled)
    pcaed = pca.transform(X_scaled)

    X_pos = X_scaled - X_scaled.min(axis=0) + 1e-6
    k = min(int(q["nmf_k"]), X_pos.shape[0] - 1, X_pos.shape[1])
    nmf = NMF(n_components=k, init="nndsvda", max_iter=2000, random_state=42)
    W = nmf.fit_transform(X_pos)
    W_norm = W / np.clip(W.sum(axis=1, keepdims=True), 1e-9, None)

    return {
        "X_scaled": X_scaled,
        "pcaed": pcaed,
        "n_pca": n_pca,
        "pca_var_explained": float(pca.explained_variance_ratio_.sum()),
        "W_norm": W_norm,
        "H": nmf.components_,
    }


def top_table(
    pool: pl.DataFrame, scores: np.ndarray, score_name: str, n: int
) -> pl.DataFrame:
    order = np.argsort(scores)[::-1][:n]
    return pool[order].select(
        "player", "team", "league", POS_COL, "primary_pos", "age"
    ).with_columns(pl.Series(score_name, scores[order]))


def search(q: dict | None = None, path: Path | str = DATA_PATH) -> dict:
    q = {**DEFAULT_QUERY, **(q or {})}
    players, feature_cols = _ensure_loaded(path)
    target, matches = resolve_target(players, q)
    pool = comparison_pool(players, target, q)

    cache_key = (
        target["primary_pos"],
        bool(q["allow_adjacent_pos"]),
        float(q["min_90s"]),
        q.get("max_age"),
        float(q["pca_var"]),
        int(q["nmf_k"]),
    )
    if cache_key not in _pool_cache:
        _pool_cache[cache_key] = fit_pool(pool, feature_cols, q)
    art = _pool_cache[cache_key]

    names = pool["player"].to_list()
    idx = names.index(target["player"])
    X_scaled = art["X_scaled"]
    pcaed = art["pcaed"]
    n = len(names)

    scores: dict[str, np.ndarray] = {}
    cos = cosine_similarity(X_scaled, X_scaled[idx : idx + 1]).ravel().astype(float)
    cos[idx] = -np.inf
    scores["Cosine"] = cos

    knn = NearestNeighbors(n_neighbors=n, metric="euclidean")
    knn.fit(pcaed)
    dist, ind = knn.kneighbors(pcaed[idx : idx + 1], n_neighbors=n)
    knn_scores = np.empty(n, dtype=float)
    knn_scores[ind[0]] = 1.0 / (1.0 + dist[0])
    knn_scores[idx] = -np.inf
    scores["KNN_PCA"] = knn_scores

    sp = spearman_scores(X_scaled, idx)
    sp[idx] = -np.inf
    scores["Spearman"] = sp

    roles, role_names, role_cols = role_matrix(
        X_scaled, feature_cols, target["primary_pos"]
    )
    role_cos = cosine_similarity(roles, roles[idx : idx + 1]).ravel().astype(float)
    role_cos[idx] = -np.inf
    scores["RoleBlocks"] = role_cos

    W_norm = art["W_norm"]
    nmf_cos = cosine_similarity(W_norm, W_norm[idx : idx + 1]).ravel().astype(float)
    nmf_cos[idx] = -np.inf
    scores["NMF"] = nmf_cos

    rrf = np.zeros(n)
    ranks: dict[str, np.ndarray] = {}
    for name, sc in scores.items():
        order = np.argsort(sc)[::-1]
        rnk = np.empty(n)
        rnk[order] = np.arange(1, n + 1)
        ranks[name] = rnk
        rrf += 1.0 / (q["rrf_k"] + rnk)
    rrf[idx] = -np.inf

    return {
        "query": q,
        "target": target,
        "matches": matches,
        "pool": pool,
        "idx": idx,
        "feature_cols": feature_cols,
        "scores": scores,
        "rrf": rrf,
        "ranks": ranks,
        "roles": roles,
        "role_names": role_names,
        "role_cols": role_cols,
        "W_norm": W_norm,
        "H": art["H"],
        "n_pca": art["n_pca"],
        "pca_var_explained": art["pca_var_explained"],
    }


def consensus_table(result: dict) -> pl.DataFrame:
    q = result["query"]
    order = np.argsort(result["rrf"])[::-1][: q["top_n"]]
    return result["pool"][order].select(
        "player", "team", "league", POS_COL, "primary_pos", "age"
    ).with_columns(
        [
            pl.Series("rrf_score", result["rrf"][order]),
            *[
                pl.Series(f"{name}_rank", result["ranks"][name][order].astype(int))
                for name in result["scores"]
            ],
        ]
    )


def nmf_mix_table(result: dict) -> pl.DataFrame:
    q = result["query"]
    idx = result["idx"]
    order = np.argsort(result["rrf"])[::-1][: q["top_n"]]
    mix_idx = [idx, *order.tolist()]
    names = [f"type_{j + 1}" for j in range(result["W_norm"].shape[1])]
    return pl.DataFrame(
        {
            "player": [result["pool"]["player"][i] for i in mix_idx],
            **{names[j]: result["W_norm"][mix_idx, j] for j in range(len(names))},
        }
    )


def role_recipe(result: dict) -> pl.DataFrame:
    idx = result["idx"]
    return pl.DataFrame(
        [
            {"role": name, "score": float(result["roles"][idx, i])}
            for i, name in enumerate(result["role_names"])
        ]
    )


def nmf_loadings(result: dict, top: int = 8) -> list[str]:
    lines = []
    H = result["H"]
    cols = result["feature_cols"]
    for j in range(H.shape[0]):
        hi = np.argsort(H[j])[::-1][:top]
        bits = [f"{cols[i]}={H[j, i]:.2f}" for i in hi]
        lines.append(f"type_{j + 1}: " + "; ".join(bits))
    return lines


def summarize(result: dict) -> None:
    t = result["target"]
    q = result["query"]
    extra = " + adjacent" if q["allow_adjacent_pos"] else ""
    age_bits = []
    if t.get("age") is not None:
        age_bits.append(f"age={t['age']:.1f}")
    if q.get("max_age") is not None:
        age_bits.append(f"max_age={q['max_age']}")
    age_txt = ("  " + "  ".join(age_bits)) if age_bits else ""
    print(
        f"target: {t['player']} ({t['team']}, {t['league']})  "
        f"pos={t[POS_COL]}  pool={t['primary_pos']}{extra}  "
        f"n={result['pool'].height}  "
        f"pca={result['n_pca']} ({100 * result['pca_var_explained']:.0f}% var)"
        f"{age_txt}"
    )
    if result["matches"].height > 1:
        print("multiple matches — used the first; set team/league to disambiguate")


if __name__ == "__main__":
    result = search()
    summarize(result)
    print(consensus_table(result))
    print(nmf_mix_table(result))
    for line in nmf_loadings(result):
        print(line)
