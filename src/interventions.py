"""Markov-DCT features of transformed images together with the metadata of the file each transform writes.

Used by run_revision.py for the separated-intervention grid (re-encoding only, downscaling only, both), the
metadata baseline after normalisation, and the matched-content encoding-swap test.
"""
from __future__ import annotations

import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import markov_features as mf


def _worker(job):
    path, spec, T = job
    try:
        bgr = mf.read_bgr(path)
        if bgr is None:
            return None
        x, file_info = mf.apply_transform_with_file(bgr, spec)
        f = mf.markov_dct_features(x, T)
        container, nbytes, quality = file_info if file_info else ("", 0, np.nan)
        return f.astype(np.float32), container, nbytes, quality, x.shape[1], x.shape[0]
    except Exception:
        return None


def extract_with_files(paths, spec: str, cache: Path, T: int = mf.T_DEFAULT, workers: int | None = None) -> dict:
    """{'X', 'ok', 'container', 'file_bytes', 'quality', 'width', 'height'} for `spec` applied to every path."""
    cache = Path(cache)
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        res = {k: d[k] for k in d.files}
        if res["ok"].all():
            return res
        return _retry(paths, spec, T, res, cache)
    n = len(paths)
    X = np.zeros((n, mf.feature_dim(T)), dtype=np.float32)
    ok = np.zeros(n, dtype=bool)
    container = np.full(n, "", dtype="<U8")
    nbytes = np.zeros(n, dtype=np.int64)
    quality = np.full(n, np.nan)
    width = np.zeros(n, dtype=np.int32)
    height = np.zeros(n, dtype=np.int32)
    t0 = time.time()
    with Pool(workers or max(1, min(6, (os.cpu_count() or 2) - 1))) as pool:
        for i, r in enumerate(pool.imap(_worker, [(str(p), spec, T) for p in paths], chunksize=8)):
            if r is not None:
                X[i], container[i], nbytes[i], quality[i], width[i], height[i] = r
                ok[i] = True
            if (i + 1) % 2000 == 0 or i + 1 == n:
                print(f"  [{spec}] {i + 1}/{n} images, {time.time() - t0:.0f} s", flush=True)
    res = dict(X=X, ok=ok, container=container, file_bytes=nbytes, quality=quality, width=width, height=height)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **res)
    return res if ok.all() else _retry(paths, spec, T, res, cache)


def _retry(paths, spec, T, res, cache):
    """Images that failed inside the worker pool (transient memory pressure) are recomputed one by one."""
    for i in np.where(~res["ok"])[0]:
        r = _worker((str(paths[i]), spec, T))
        if r is None:
            print(f"  [{spec}] still failing: {paths[i]}", flush=True)
            continue
        res["X"][i], res["container"][i], res["file_bytes"][i], res["quality"][i], res["width"][i], res["height"][i] = r
        res["ok"][i] = True
    print(f"  [{spec}] retried, {int((~res['ok']).sum())} failures left", flush=True)
    np.savez_compressed(cache, **res)
    return res


def metadata_from_files(d: dict) -> np.ndarray:
    """The 9 metadata features of forensics_data.metadata_matrix, computed on the normalised files."""
    c = d["container"]
    px = np.maximum(d["width"].astype(np.float64) * d["height"], 1)
    b = np.maximum(d["file_bytes"].astype(np.float64), 1)
    return np.column_stack([c == "JPEG", c == "TIFF", c == "PNG", c == "BMP", np.log(b), np.log(px), b / px,
                            d["width"] / np.maximum(d["height"], 1), np.nan_to_num(d["quality"], nan=100.0)]).astype(np.float64)
