"""Markov transition features of block-DCT differences, as defined in the group-8 notebook.

The definition is kept identical to `cnn-markov-dct.ipynb` so that the original result can be
reproduced: YCrCb (OpenCV channel order) -> orthonormal 8x8 block DCT on float pixels ->
intra-block and inter-block differences (horizontal, vertical) -> clip to [-T, T] and truncate
to int -> horizontal and vertical transition probability matrices (TPMs).
Output: 3 channels x 8 TPMs x (2T+1)^2 values = 1944 for T = 4, ordered
[intra_h:h, intra_h:v, intra_v:h, intra_v:v, inter_h:h, inter_h:v, inter_v:h, inter_v:v] per channel.
"""
from __future__ import annotations

import os
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy.fftpack import dct

T_DEFAULT = 4
BLOCK = 8
TPM_NAMES = ["intra_h:h", "intra_h:v", "intra_v:h", "intra_v:v", "inter_h:h", "inter_h:v", "inter_v:h", "inter_v:v"]
CHANNEL_NAMES = ["Y", "Cr", "Cb"]


def feature_dim(T: int = T_DEFAULT) -> int:
    return len(CHANNEL_NAMES) * len(TPM_NAMES) * (2 * T + 1) ** 2


def _tpm(diff: np.ndarray, T: int, direction: str) -> np.ndarray:
    d = np.clip(diff, -T, T).astype(int) + T
    dim = 2 * T + 1
    if direction == "h":
        cur, nxt = d[:, :-1].ravel(), d[:, 1:].ravel()
    else:
        cur, nxt = d[:-1, :].ravel(), d[1:, :].ravel()
    m = np.bincount(cur * dim + nxt, minlength=dim * dim).reshape(dim, dim).astype(float)
    rows = m.sum(axis=1, keepdims=True)
    rows[rows == 0] = 1
    return (m / rows).ravel()


def markov_dct_features(bgr: np.ndarray, T: int = T_DEFAULT, block: int = BLOCK) -> np.ndarray:
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    feats = []
    for ch in range(3):
        comp = ycc[:, :, ch].astype(np.float32)
        h, w = comp.shape
        h, w = h - h % block, w - w % block
        comp = comp[:h, :w]
        blocks = comp.reshape(h // block, block, w // block, block).transpose(0, 2, 1, 3)
        coef = dct(dct(blocks, axis=-1, norm="ortho"), axis=-2, norm="ortho")
        img = coef.transpose(0, 2, 1, 3).reshape(h, w)
        diffs = (img[:, :-1] - img[:, 1:], img[:-1, :] - img[1:, :],
                 img[:, :-block] - img[:, block:], img[:-block, :] - img[block:, :])
        for dmat in diffs:
            feats.append(_tpm(dmat, T, "h"))
            feats.append(_tpm(dmat, T, "v"))
    return np.concatenate(feats)


def apply_transform(bgr: np.ndarray, spec: str) -> np.ndarray:
    """Chain of operations separated by '+': 'orig', 'jpegQ' (re-encode at quality Q), 'resizeS' (scale S),
    'qt<table><subsampling>' (re-encode with a recorded CASIA quantisation table, e.g. 'qttp444'; see qtable_encode)."""
    return apply_transform_with_file(bgr, spec)[0]


def apply_transform_with_file(bgr: np.ndarray, spec: str):
    """apply_transform that also returns the file the last step would write: (container, bytes, JPEG quality).
    A chain that ends with a resize is stored losslessly (PNG); 'orig' returns None (the original file is used)."""
    out, last = bgr, None
    for tok in spec.split("+"):
        if tok == "orig":
            continue
        if tok.startswith("resize"):
            s = float(tok[len("resize"):])
            out = cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            last = "lossless"
        elif tok.startswith("jpeg"):
            q = int(tok[len("jpeg"):])
            ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            last = ("JPEG", int(buf.size), float(q))
        elif tok.startswith("qt"):
            out, last = qtable_encode(out, tok)
        else:
            raise ValueError(f"unknown transform token: {tok}")
    if last == "lossless":
        ok, buf = cv2.imencode(".png", out, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        last = ("PNG", int(buf.size), 100.0)
    return out, last


_QTABLES = None
QTABLE_FILE = Path(__file__).resolve().parent.parent / "results_rev" / "casia_signatures.json"


def qtable_encode(bgr: np.ndarray, tok: str):
    """'qt' + table key ('au': most common CASIA authentic JPEG table, 'tp': most common CASIA tampered JPEG table)
    + chroma subsampling ('444' or '420'). Encoded with Pillow, which accepts explicit quantisation tables."""
    import io
    import json

    from PIL import Image
    global _QTABLES
    if _QTABLES is None:
        _QTABLES = json.loads(QTABLE_FILE.read_text())
    key, ss = tok[2:4], tok[4:]
    tables = [_QTABLES["tables"][key][k] for k in sorted(_QTABLES["tables"][key], key=int)]
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(buf, "JPEG", qtables=tables,
                                                               subsampling={"444": 0, "420": 2}[ss])
    data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    quality = _QTABLES["quality"][key]
    return cv2.imdecode(data, cv2.IMREAD_COLOR), ("JPEG", int(data.size), float(quality))


def read_bgr(path: str) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def _worker(job):
    path, spec, T = job
    try:
        bgr = read_bgr(path)
        if bgr is None:
            return None
        x = apply_transform(bgr, spec)
        t0 = time.perf_counter()
        f = markov_dct_features(x, T)
        return f.astype(np.float32), time.perf_counter() - t0
    except Exception:
        return None


def extract(paths, spec: str | list[str] = "orig", T: int = T_DEFAULT, cache: Path | None = None,
            workers: int | None = None):
    """Features for every path. `spec` is one transform for all images or one transform per image.
    Returns X, ok-mask, seconds-per-image (feature step only)."""
    if cache is not None and Path(cache).exists():
        d = np.load(cache)
        return d["X"], d["ok"], d["sec"]
    n = len(paths)
    specs = [spec] * n if isinstance(spec, str) else list(spec)
    if len(specs) != n:
        raise ValueError("one transform per image is required")
    label = spec if isinstance(spec, str) else "per-image transforms"
    X = np.zeros((n, feature_dim(T)), dtype=np.float32)
    ok = np.zeros(n, dtype=bool)
    sec = np.full(n, np.nan, dtype=np.float32)
    t_start = time.time()
    # Large IMD2020 photos need ~0.7 GB per worker; six workers keep peak RAM well under 16 GB.
    with Pool(workers or max(1, min(6, (os.cpu_count() or 2) - 1))) as pool:
        for i, r in enumerate(pool.imap(_worker, [(str(p), s, T) for p, s in zip(paths, specs)], chunksize=8)):
            if r is not None:
                X[i], sec[i] = r
                ok[i] = True
            if (i + 1) % 2000 == 0 or i + 1 == n:
                print(f"  [{label}] {i + 1}/{n} images, {time.time() - t_start:.0f} s", flush=True)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X, ok=ok, sec=sec)
    return X, ok, sec
