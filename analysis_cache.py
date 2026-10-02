"""Disk and in-process caches for the player-independent stages of the pipeline.

Everything that happens before "which player did you ask about" is a pure
function of the cached match csvs plus a handful of numeric settings:
concatenating ~2,300 match files (~45s cold), reducing them to one shape vector
per player (~33s), KMeans over those vectors (~2s) and the pass-angle features.
The desktop app re-ran all of it on every profile, every similar-player search
and every tight-space build, even when nothing but the player name had changed.

Each stage is memoized twice here: on disk under `.desktop_cache/`, so a fresh
process skips the work, and in this process, so a second lookup in the same
session costs nothing at all.

Cache keys hash the inputs. For events that is the name, size and mtime of every
csv that would be read, so re-scraping a league invalidates it on its own; for
the later stages it is the events key plus the settings that stage reads. The
events cache stores the concatenated frame exactly as `pd.concat(read_csv(...))`
produced it, dtypes included, so a cached run and an uncached run see the same
frame.

Set `FOOTY_NO_CACHE=1` to bypass all of it, or call `clear()` to drop what is
held in memory.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import pickle
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_ROOT = Path(".desktop_cache")
EVENTS_DIR = CACHE_ROOT / "events"
GRIDS_DIR = CACHE_ROOT / "grids"

# The events frame is ~1.7GB, so only the most recent one is held in memory.
# The later stages are small enough to keep a handful of.
_EVENTS_MEMO_SIZE = 1
_SMALL_MEMO_SIZE = 8
# Event pickles kept on disk, least-recently-used dropped beyond this.
_EVENTS_KEEP_ON_DISK = 3

_events_memo: OrderedDict[str, pd.DataFrame] = OrderedDict()
_small_memo: OrderedDict[str, object] = OrderedDict()


def enabled() -> bool:
    return os.environ.get("FOOTY_NO_CACHE", "").strip().lower() not in {"1", "true", "yes", "on"}


def clear() -> None:
    """Drop everything held in memory (the disk caches are left alone)."""
    _events_memo.clear()
    _small_memo.clear()


def release_events() -> None:
    """Drop just the cached events frame, for when memory matters more than speed."""
    _events_memo.clear()


def _memo_get(store: OrderedDict, key: str):
    if key in store:
        store.move_to_end(key)
        return store[key]
    return None


def _memo_put(store: OrderedDict, key: str, value, limit: int) -> None:
    store[key] = value
    store.move_to_end(key)
    while len(store) > limit:
        store.popitem(last=False)


def _digest(payload) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def _tmp_path(path: Path) -> Path:
    """A private temp name for `path`.

    The desktop app can have a profile and a similar-player search in flight at
    once, so two threads may write the same cache entry at the same time. A
    shared temp name would let them interleave into one corrupt file; with a
    name per writer the loser of the `os.replace` race simply gets overwritten
    by an equivalent file.
    """
    return path.with_suffix(f"{path.suffix}.{os.getpid()}.{threading.get_ident()}.tmp")


def _write_pickle(path: Path, obj) -> None:
    """Write via a temp file, so an interrupted write cannot leave a half-read cache."""
    tmp = _tmp_path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001 - a cache we cannot write is not an error
        tmp.unlink(missing_ok=True)
        print(f"warning: could not write cache {path.name}: {exc}", flush=True)


def _write_npy(path: Path, arr: np.ndarray) -> None:
    tmp = _tmp_path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Through a file object: `np.save` appends `.npy` to a path that lacks it,
        # which would leave the temp file under a name we never rename.
        with tmp.open("wb") as fh:
            np.save(fh, arr)
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        print(f"warning: could not write cache {path.name}: {exc}", flush=True)


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------


def _dir_signature(save_dir: str | Path) -> list | None:
    """(name, size, mtime) for every csv in `save_dir`, or None if there are none.

    Cheap enough to run on every call — it stats the files rather than reading
    them — and it changes whenever a match is added, rescraped or removed, which
    is exactly when a cached frame stops being valid.
    """
    paths = sorted(glob.glob(str(Path(save_dir) / "*.csv")))
    if not paths:
        return None
    out = []
    for p in paths:
        try:
            st = os.stat(p)
        except OSError:
            return None
        out.append([Path(p).name, st.st_size, st.st_mtime_ns])
    return out


def dir_events_key(save_dir: str | Path) -> str | None:
    sig = _dir_signature(save_dir)
    if sig is None:
        return None
    return _digest({"dir": str(Path(save_dir)), "files": sig})


def league_events_key(leagues: list[str], season: int | str, events_root: str | Path) -> str | None:
    """Key for a pooled multi-league load, or None if no league has any csvs."""
    from touchmap_similarity import _league_slug

    # An ordered list rather than a mapping: the leagues are concatenated in the
    # order given, so two calls that name the same leagues in a different order
    # produce differently-ordered frames and must not share a cache entry.
    parts = []
    for league in leagues:
        sig = _dir_signature(Path(events_root) / _league_slug(league, season))
        if sig is not None:
            parts.append([league, sig])
    if not parts:
        return None
    return _digest({"root": str(events_root), "season": str(season), "leagues": parts})


def _events_path(key: str) -> Path:
    return EVENTS_DIR / f"{key}.pkl"


def cached_events(key: str | None, build, *, persist: bool = True) -> pd.DataFrame:
    """Return the events frame for `key`, calling `build()` only on a miss.

    Callers get their own DataFrame object (a shallow copy), so adding or
    replacing a column on the result — which `_run_touchmap_cluster` does — does
    not reach back into the cache.

    `persist=False` still reads an existing cache but does not store a miss, on
    disk or in memory. It is for one-off passes over many seasons that only need
    each season's events long enough to reduce them to grids: storing each one
    would write ~1.5GB per season and push out the season that is actually in
    use.
    """
    if key is None or not enabled():
        return build()

    if not persist:
        hit = _memo_get(_events_memo, key)
        if hit is not None:
            return hit.copy(deep=False)
        path = _events_path(key)
        if path.exists():
            try:
                with path.open("rb") as fh:
                    df = pickle.load(fh)
            except Exception:  # noqa: BLE001
                df = None
            if isinstance(df, pd.DataFrame):
                print(f"events: {len(df):,} rows from cache", flush=True)
                return df
        return build()

    hit = _memo_get(_events_memo, key)
    if hit is not None:
        return hit.copy(deep=False)

    path = _events_path(key)
    if path.exists():
        df = None
        try:
            with path.open("rb") as fh:
                df = pickle.load(fh)
        except Exception:  # noqa: BLE001 - a bad cache should cost a reload, not a crash
            df = None
        if isinstance(df, pd.DataFrame):
            print(f"events: {len(df):,} rows from cache", flush=True)
            _touch(path)
            _memo_put(_events_memo, key, df, _EVENTS_MEMO_SIZE)
            return df.copy(deep=False)

    df = build()
    _write_pickle(path, df)
    _prune_events()
    _memo_put(_events_memo, key, df, _EVENTS_MEMO_SIZE)
    return df.copy(deep=False)


def _touch(path: Path) -> None:
    """Mark a cache file as just used, so pruning can drop the least recent."""
    try:
        os.utime(path, None)
    except OSError:
        pass


def _prune_events() -> None:
    """Keep only the few most recently used event pickles.

    Each one is up to ~1.7GB, and re-scraping a league strands the previous
    signature's file for good — so without this the cache directory grows by a
    couple of gigabytes every time the data is refreshed. A handful is kept so
    that alternating between, say, the pooled leagues and a single-league
    directory does not make each switch a rebuild.
    """
    try:
        files = sorted(
            EVENTS_DIR.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        for stale in files[_EVENTS_KEEP_ON_DISK:]:
            stale.unlink()
            print(f"events: dropped stale cache {stale.name}", flush=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# grids, cluster labels, pass angles
# ---------------------------------------------------------------------------


def _stage_key(parent_key: str, **settings) -> str:
    return _digest({"parent": parent_key, **settings})


def cached_grids(
    events_key: str | None,
    load_events,
    *,
    min_touches: int,
    bins: tuple[int, int],
    smooth_sigma: float,
    action_features: bool,
) -> tuple[str | None, pd.DataFrame, np.ndarray]:
    """`(key, meta, X)` for the shape-vector population, built only on a miss.

    `load_events` is a callable rather than a frame so that a hit here never
    pays for loading the events at all — which is what makes a repeat
    similar-player search cheap.
    """
    import touchmap_similarity as tms

    def build():
        season_df = load_events()
        if action_features:
            return tms.build_combined_grids(
                season_df, min_touches=min_touches, bins=bins, smooth_sigma=smooth_sigma
            )
        return tms.build_touch_grids(
            season_df, min_touches=min_touches, bins=bins, smooth_sigma=smooth_sigma
        )

    if events_key is None or not enabled():
        meta, X = build()
        return None, meta, X

    key = _stage_key(
        events_key,
        stage="grids",
        min_touches=min_touches,
        bins=list(bins),
        smooth_sigma=smooth_sigma,
        action_features=action_features,
        # Adding a feature block changes X's width, so it must miss the cache.
        layout=tms.feature_layout() if action_features else None,
    )

    hit = _memo_get(_small_memo, key)
    if hit is not None:
        meta, X = hit
        return key, meta, X

    meta_path = GRIDS_DIR / f"{key}_meta.pkl"
    x_path = GRIDS_DIR / f"{key}_X.npy"
    if meta_path.exists() and x_path.exists():
        meta, X = None, None
        try:
            with meta_path.open("rb") as fh:
                meta = pickle.load(fh)
            X = np.load(x_path)
        except Exception:  # noqa: BLE001
            meta, X = None, None
        if isinstance(meta, pd.DataFrame) and isinstance(X, np.ndarray) and len(meta) == len(X):
            _memo_put(_small_memo, key, (meta, X), _SMALL_MEMO_SIZE)
            return key, meta, X

    meta, X = build()
    _write_pickle(meta_path, meta)
    _write_npy(x_path, X)
    _memo_put(_small_memo, key, (meta, X), _SMALL_MEMO_SIZE)
    return key, meta, X


def cached_cluster_labels(
    grids_key: str | None,
    X: np.ndarray,
    *,
    n_clusters: int,
    pca_var: float = 0.90,
    random_state: int = 42,
) -> np.ndarray:
    """KMeans labels row-aligned with the grid population.

    Only the labels are cached: `_run_touchmap_cluster` reads nothing else off
    the clustering, and the fitted PCA/KMeans objects are not worth persisting.
    """
    import touchmap_similarity as tms

    def build():
        return tms.cluster_players(
            X, n_clusters=n_clusters, pca_var=pca_var, random_state=random_state
        )["labels"]

    if grids_key is None or not enabled():
        return build()

    key = _stage_key(
        grids_key,
        stage="labels",
        n_clusters=n_clusters,
        pca_var=pca_var,
        random_state=random_state,
    )
    hit = _memo_get(_small_memo, key)
    if hit is not None:
        return hit

    path = GRIDS_DIR / f"{key}_labels.npy"
    if path.exists():
        labels = None
        try:
            labels = np.load(path)
        except Exception:  # noqa: BLE001
            labels = None
        if isinstance(labels, np.ndarray) and len(labels) == len(X):
            _memo_put(_small_memo, key, labels, _SMALL_MEMO_SIZE)
            return labels

    labels = build()
    _write_npy(path, labels)
    _memo_put(_small_memo, key, labels, _SMALL_MEMO_SIZE)
    return labels


def cached_embedding(
    grids_key: str | None,
    X: np.ndarray,
    *,
    pca_var: float = 0.90,
    random_state: int = 42,
) -> np.ndarray:
    """`touchmap_similarity.embed_shapes` coords, row-aligned with the grid population."""
    import touchmap_similarity as tms

    def build():
        return tms.embed_shapes(X, pca_var=pca_var, random_state=random_state)

    if grids_key is None or not enabled():
        return build()

    key = _stage_key(
        grids_key,
        stage="embedding",
        pca_var=pca_var,
        random_state=random_state,
        weights=tms.SHAPE_BLOCK_WEIGHTS,
    )
    hit = _memo_get(_small_memo, key)
    if hit is not None:
        return hit

    path = GRIDS_DIR / f"{key}_coords.npy"
    if path.exists():
        coords = None
        try:
            coords = np.load(path)
        except Exception:  # noqa: BLE001
            coords = None
        if isinstance(coords, np.ndarray) and len(coords) == len(X):
            _memo_put(_small_memo, key, coords, _SMALL_MEMO_SIZE)
            return coords

    coords = build()
    _write_npy(path, coords)
    _memo_put(_small_memo, key, coords, _SMALL_MEMO_SIZE)
    return coords


def cached_pass_angle_features(
    grids_key: str | None,
    load_events,
    meta: pd.DataFrame,
) -> tuple[dict, dict]:
    """Pass-angle stats and feature vectors for *every* player in `meta`.

    The uncached call is made per cluster, but the features are purely per
    `(player, team)` — so computing the whole population once and letting each
    cluster read its own keys out of the dicts gives that cluster the same
    answer, and means a cluster nobody has looked at yet still costs nothing.
    """
    import pass_angle_radar as par

    pairs = list(zip(meta["player"].tolist(), meta["team"].tolist()))

    if grids_key is None or not enabled():
        return par.pass_angle_features_for_players(load_events(), pairs)

    key = _stage_key(grids_key, stage="pass_angles")
    hit = _memo_get(_small_memo, key)
    if hit is not None:
        return hit

    path = GRIDS_DIR / f"{key}_pass_angles.pkl"
    if path.exists():
        payload = None
        try:
            with path.open("rb") as fh:
                payload = pickle.load(fh)
        except Exception:  # noqa: BLE001
            payload = None
        if isinstance(payload, tuple) and len(payload) == 2:
            _memo_put(_small_memo, key, payload, _SMALL_MEMO_SIZE)
            return payload

    payload = par.pass_angle_features_for_players(load_events(), pairs)
    _write_pickle(path, payload)
    _memo_put(_small_memo, key, payload, _SMALL_MEMO_SIZE)
    return payload
