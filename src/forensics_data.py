"""Index of the merged benchmark used by group 8 (CASIA 2.0 + Columbia + IMD2020), with the file-level
metadata needed for bias-controlled evaluation (real container format, JPEG quality, size, source groups).
"""
from __future__ import annotations

import csv
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

SOURCES = {
    "casia": {0: "casia2/CASIA2/Au", 1: "casia2/CASIA2/Tp"},
    "columbia": {0: "columbia/Columbia Uncompressed Image Splicing Detection/4cam_auth/4cam_auth",
                 1: "columbia/Columbia Uncompressed Image Splicing Detection/4cam_splc/4cam_splc"},
    "imd2020": {0: "imd2020/IMD2020/Authentic", 1: "imd2020/IMD2020/img"},
}
# Extensions accepted by the group's notebook (BMP is silently excluded there).
TEAM_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

# IJG standard luminance quantisation table; only its sum is used, so coefficient order does not matter.
_STD_LUMA_SUM = sum([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                     14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
                     92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99])


def estimate_jpeg_quality(img: Image.Image) -> float:
    """Approximate IJG quality factor from the luminance table; 100 for lossless containers."""
    q = getattr(img, "quantization", None)
    if not q:
        return 100.0
    scale = 100.0 * sum(q[min(q)]) / _STD_LUMA_SUM
    quality = (200.0 - scale) / 2.0 if scale <= 100 else 5000.0 / scale
    return float(np.clip(quality, 1, 100))


_CASIA_ID = re.compile(r"([a-z]{3})(\d{5})")


def casia_source_ids(name: str) -> list[str]:
    """Au_ani_00001.jpg -> ['ani00001']; Tp_D_CND_M_N_ani00018_sec00096_00138.tif -> ['ani00018', 'sec00096']."""
    stem = Path(name).stem
    if stem.startswith("Au_"):
        parts = stem.split("_")
        return [parts[1] + parts[2]] if len(parts) >= 3 else [stem]
    return _CASIA_ID.findall(stem.lower()) and ["".join(m) for m in _CASIA_ID.findall(stem.lower())] or [stem]


def build_index(root, exts=TEAM_EXTS) -> list[dict]:
    root = Path(root)
    rows = []
    for source, folders in SOURCES.items():
        for label, rel in folders.items():
            folder = root / rel
            if not folder.is_dir():
                raise FileNotFoundError(folder)
            for name in sorted(os.listdir(folder)):
                if not name.lower().endswith(exts):
                    continue
                path = folder / name
                row = dict(path=str(path), name=name, source=source, label=label, ext=Path(name).suffix.lower(),
                           file_bytes=path.stat().st_size)
                try:
                    with Image.open(path) as im:
                        row.update(container=(im.format or "").upper(), width=im.width, height=im.height,
                                   mode=im.mode, jpeg_quality=estimate_jpeg_quality(im))
                except Exception:
                    row.update(container="UNREADABLE", width=0, height=0, mode="", jpeg_quality=np.nan)
                if source == "casia":
                    row["group_ids"] = casia_source_ids(name)
                elif source == "imd2020":
                    row["group_ids"] = [f"imd2020:{name.split('_')[0]}"]
                else:
                    row["group_ids"] = [f"{source}:{Path(name).stem}"]
                rows.append(row)
    return rows


def assign_groups(rows: list[dict]) -> np.ndarray:
    """Connected components over shared source ids, so an authentic image and every forgery built from it share a group."""
    parent: dict[str, str] = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for r in rows:
        ids = [f"{r['source']}:{i}" if r["source"] == "casia" else i for i in r["group_ids"]]
        for other in ids[1:]:
            parent[find(ids[0])] = find(other)
        find(ids[0])
    roots, groups = {}, []
    for r in rows:
        key = f"{r['source']}:{r['group_ids'][0]}" if r["source"] == "casia" else r["group_ids"][0]
        groups.append(roots.setdefault(find(key), len(roots)))
    return np.asarray(groups)


def metadata_matrix(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Non-forensic descriptors: container format, file size, resolution, compression level."""
    names = ["is_jpeg", "is_tiff", "is_png", "is_bmp", "log_file_bytes", "log_pixels", "bytes_per_pixel",
             "aspect", "jpeg_quality"]
    M = np.array([[r["container"] == "JPEG", r["container"] == "TIFF", r["container"] == "PNG", r["container"] == "BMP",
                   np.log(max(r["file_bytes"], 1)), np.log(max(r["width"] * r["height"], 1)),
                   r["file_bytes"] / max(r["width"] * r["height"], 1), r["width"] / max(r["height"], 1),
                   r["jpeg_quality"] if r["jpeg_quality"] == r["jpeg_quality"] else 100.0] for r in rows], dtype=np.float64)
    return M, names


def save_index(rows: list[dict], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["path", "name", "source", "label", "ext", "container", "width", "height", "mode", "file_bytes",
            "jpeg_quality", "group_ids"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "group_ids": "|".join(r["group_ids"])})
