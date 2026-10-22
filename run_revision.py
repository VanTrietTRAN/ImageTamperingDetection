"""Experiments added for the TSE 2026 camera-ready revision (reviewer requests R1-R7).

    python run_revision.py --data <datasets_forgery> [--stages signatures features deep ladder grid swap cross ...]
                           [--deep-python <python with torch-directml>] [--deep-device dml]

Every stage writes results_rev/<stage>.json and is skipped when it exists (unless --force). Shared protocol for all
new stages: source-disjoint StratifiedGroupKFold (5 folds) repeated with seeds 0-4, metrics computed inside each test
fold and averaged over folds, then mean +- SD over the five repetitions; group-bootstrap CIs resample source groups.

  signatures  JPEG encoding signatures (exact quantisation tables, chroma subsampling) of every JPEG file; signature
              rules; the CASIA tables used by the matched-content test (results_rev/casia_signatures.json)
  features    Markov-DCT features + metadata of the normalised file for every intervention (GRID, AUG_EXTRA, SWAP)
  deep        frozen-ResNet18 embeddings for GRID and SWAP
  ladder      cumulative ladder: random -> +source-disjoint -> +JPEG-only -> +history erasure, all detectors,
              metadata baseline on the normalised files, per-dataset AUC
  grid        separated interventions: transform applied to training and test images (matched), per dataset;
              robustness: standard / degradation-augmented training on original files, transformed test files
  swap        matched-content encoding swap on held-out CASIA images (same pixels, controlled final encoding)
  cross       cross-dataset generalisation, all detectors, 5 seeds, class counts, sensitivity/specificity
  gboot       group-bootstrap confidence intervals and paired differences for the central comparisons
  splits      split definitions (random split and grouped folds for every seed) for release
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE))

import forensics_data as fd  # noqa: E402
import interventions as iv  # noqa: E402
import markov_features as mf  # noqa: E402
import metrics as mt  # noqa: E402
import run_experiments as rx  # noqa: E402

OUT = HERE / "results_rev"
FEAT = OUT / "features"
OLD = HERE / "results" / "features"
SEEDS = [0, 1, 2, 3, 4]
# 3 x 3 grid of scale {1, 0.75, 0.5} x final encoding {none, JPEG 95, JPEG 75}, plus JPEG 85 for the robustness curve.
GRID = ["jpeg95", "jpeg85", "jpeg75", "resize0.75", "resize0.75+jpeg95", "resize0.75+jpeg75",
        "resize0.5", "resize0.5+jpeg95", "resize0.5+jpeg75"]
AUG_EXTRA = ["jpeg65", "resize0.75+jpeg85", "resize0.5+jpeg90"]  # remaining members of rx.AUGMENT_POOL
SWAP = ["qtau420", "qtau444", "qttp420", "qttp444"]
HISTORY_ERASED = rx.HISTORY_ERASED
SOURCES = ["casia", "columbia", "imd2020"]


def save(obj, path: Path):
    rx.save(obj, path)


def load_benchmark(data: Path):
    rows = fd.build_index(data)
    X, ok, _ = mf.extract([r["path"] for r in rows], "orig", cache=OLD / "orig.npz")
    keep = np.where(ok & np.array([r["container"] != "UNREADABLE" for r in rows]))[0]
    rows, X = [rows[i] for i in keep], X[keep]
    y = np.array([r["label"] for r in rows])
    M, _ = fd.metadata_matrix(rows)
    groups = fd.assign_groups(rows)
    return rows, X, y, M, groups


# ------------------------------------------------------------------------------------------------------- signatures
def jpeg_signature(path: str):
    from PIL import Image, JpegImagePlugin
    with Image.open(path) as im:
        if im.format != "JPEG":
            return None
        q = {int(k): list(v) for k, v in im.quantization.items()}
        exact = hashlib.md5(json.dumps(sorted(q.items())).encode()).hexdigest()[:10]
        return dict(qtable=exact, subsampling={0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}.get(JpegImagePlugin.get_sampling(im), "other"),
                    tables=q, quality=fd.estimate_jpeg_quality(im))


def signature_rule(sig, y, groups=None, seed=0):
    """Majority label per signature value, 5-fold CV (grouped when `groups` is given); unseen -> training majority."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    sig, y = np.asarray(sig), np.asarray(y)
    split = (StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(sig, y, groups) if groups is not None
             else StratifiedKFold(5, shuffle=True, random_state=seed).split(sig, y))
    pred = np.zeros(len(y), int)
    accs = []
    for tr, te in split:
        rule = collections.defaultdict(list)
        for s, l in zip(sig[tr], y[tr]):
            rule[s].append(l)
        rule = {s: int(np.mean(v) >= 0.5) for s, v in rule.items()}
        fb = int(y[tr].mean() >= 0.5)
        pred[te] = [rule.get(s, fb) for s in sig[te]]
        accs.append(float((pred[te] == y[te]).mean()))
    rep = mt.binary_report(y, pred)
    return dict(n=len(y), n_tampered=int(y.sum()), accuracy=rep["accuracy"], accuracy_fold_mean=float(np.mean(accs)),
                balanced_accuracy=rep["balanced_accuracy"], majority_rate=float(max(y.mean(), 1 - y.mean())),
                distinct=len(set(sig.tolist())))


def stage_signatures(rows, y, groups):
    sigs = [jpeg_signature(r["path"]) if r["container"] == "JPEG" else None for r in rows]
    src = np.array([r["source"] for r in rows])
    jp = np.array([s is not None for s in sigs])
    qt = np.array([s["qtable"] if s else "" for s in sigs])
    ss = np.array([s["subsampling"] if s else "" for s in sigs])
    cont = np.array([r["container"] for r in rows])
    # Most common exact table + subsampling of CASIA authentic and tampered JPEG files.
    chosen = {}
    for key, lab in (("au", 0), ("tp", 1)):
        m = jp & (src == "casia") & (y == lab)
        top = collections.Counter(zip(qt[m], ss[m])).most_common(1)[0][0]
        i = int(np.where(m & (qt == top[0]) & (ss == top[1]))[0][0])
        chosen[key] = dict(qtable=top[0], subsampling=top[1], example=rows[i]["name"], tables=sigs[i]["tables"],
                           quality=sigs[i]["quality"],
                           n_authentic=int((jp & (src == "casia") & (y == 0) & (qt == top[0])).sum()),
                           n_tampered=int((jp & (src == "casia") & (y == 1) & (qt == top[0])).sum()))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "casia_signatures.json").write_text(json.dumps(dict(
        tables={k: {str(t): v for t, v in c["tables"].items()} for k, c in chosen.items()},
        quality={k: c["quality"] for k, c in chosen.items()},
        chosen={k: {kk: vv for kk, vv in c.items() if kk != "tables"} for k, c in chosen.items()}), indent=1))
    # Round-trip check: Pillow writes exactly the recorded tables (same order as it reads them).
    import io

    from PIL import Image
    rgb = np.zeros((64, 64, 3), np.uint8) + 128
    roundtrip = {}
    for key in ("au", "tp"):
        tabs = json.loads((OUT / "casia_signatures.json").read_text())["tables"][key]
        b = io.BytesIO()
        Image.fromarray(rgb).save(b, "JPEG", qtables=[tabs[k] for k in sorted(tabs, key=int)], subsampling=0)
        with Image.open(io.BytesIO(b.getvalue())) as im:
            roundtrip[key] = all(list(im.quantization[int(k)]) == v for k, v in tabs.items())
    res = dict(chosen={k: {kk: vv for kk, vv in c.items() if kk != "tables"} for k, c in chosen.items()},
               roundtrip_tables_identical=roundtrip, rules={}, crosstab={})
    for name, m in {"casia_jpeg": jp & (src == "casia"), "imd2020_jpeg": jp & (src == "imd2020"), "all_jpeg": jp}.items():
        for prop, sig in {"qtable": qt, "subsampling": ss, "qtable+subsampling": np.char.add(np.char.add(qt, "|"), ss)}.items():
            res["rules"][f"{name}|{prop}"] = dict(stratified=signature_rule(sig[m], y[m]),
                                                  source_disjoint=signature_rule(sig[m], y[m], groups[m]))
        tab = collections.Counter(zip(ss[m], y[m]))
        res["crosstab"][f"{name}|subsampling"] = {f"{s}|label={l}": n for (s, l), n in sorted(tab.items())}
    for name, (m, sig) in {"all|container": (np.ones(len(y), bool), cont),
                           "columbia|size": (src == "columbia", np.array([f"{r['width']}x{r['height']}" for r in rows])),
                           "all|source": (np.ones(len(y), bool), src)}.items():
        res["rules"][name] = dict(stratified=signature_rule(sig[m], y[m]), source_disjoint=signature_rule(sig[m], y[m], groups[m]))
    import csv
    with open(OUT / "jpeg_signatures.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "source", "label", "container", "qtable", "subsampling"])
        for r, a, b in zip(rows, qt, ss):
            w.writerow([r["name"], r["source"], r["label"], r["container"], a, b])
    for k, v in res["rules"].items():
        print(f"[signatures] {k:32s} strat {100 * v['stratified']['accuracy']:.1f}%  grouped "
              f"{100 * v['source_disjoint']['accuracy']:.1f}%  majority {100 * v['stratified']['majority_rate']:.1f}%", flush=True)
    print("[signatures] crosstab", json.dumps(res["crosstab"]), "roundtrip", roundtrip, flush=True)
    save(res, OUT / "signatures.json")


# --------------------------------------------------------------------------------------------------------- features
def markov_view(rows, spec):
    """Markov-DCT features and metadata matrix of the normalised files for one intervention ('orig' = original files)."""
    paths = [r["path"] for r in rows]
    if spec == "orig":
        X, _, _ = mf.extract(paths, "orig", cache=OLD / "orig.npz")
        return X, fd.metadata_matrix(rows)[0]
    d = iv.extract_with_files(paths, spec, cache=FEAT / f"markov_{spec}.npz")
    if not d["ok"].all():
        raise RuntimeError(f"{int((~d['ok']).sum())} images failed for {spec}")
    return d["X"], iv.metadata_from_files(d)


def swap_view(rows, spec, idx):
    paths = [rows[i]["path"] for i in idx]
    d = iv.extract_with_files(paths, spec, cache=FEAT / f"markov_swap_{spec}.npz")
    return d["X"], iv.metadata_from_files(d)


def deep_view(rows, spec, args, idx=None, tag=None):
    paths = [r["path"] for r in rows] if idx is None else [rows[i]["path"] for i in idx]
    name = tag or spec
    cache = FEAT / f"deep_{name}.npz"
    legacy = {"orig": OLD / "deep_orig.npz", HISTORY_ERASED: OLD / "deep_history_erased.npz"}
    if not cache.exists() and idx is None and spec in legacy:
        cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(legacy[spec], cache)
    d = rx.deep_features(paths, spec, cache, args.deep_python, args.deep_device)
    if not d["ok"].all():
        raise RuntimeError(f"frozen-ResNet18 inputs failed for {int((~d['ok']).sum())} images ({name})")
    return {"logreg_resnet_rgb": d["rgb"], "logreg_resnet_eladct": d["eladct"]}


def casia_idx(rows):
    return np.array([i for i, r in enumerate(rows) if r["source"] == "casia"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--stages", nargs="+", default=["signatures", "features", "deep"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--deep-python", default=sys.executable)
    ap.add_argument("--deep-device", default="auto", choices=["auto", "cpu", "dml"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--grid-models", default="all", choices=["all", "markov", "deep"])
    ap.add_argument("--grid-conds", nargs="+", default=None, help="grid training conditions handled by this process")
    args = ap.parse_args()

    def todo(stage):
        return stage in args.stages and (args.force or not (OUT / f"{stage}.json").exists())

    rows, X, y, M, groups = load_benchmark(args.data)
    print(f"{len(rows)} images ({int(y.sum())} tampered), {len(set(groups))} source groups", flush=True)

    if todo("signatures"):
        stage_signatures(rows, y, groups)

    if todo("features"):
        t0 = time.time()
        for spec in GRID + AUG_EXTRA:
            markov_view(rows, spec)
        ci = casia_idx(rows)
        for spec in SWAP:
            swap_view(rows, spec, ci)
        # The cached history-erased features of run_experiments.py must be reproduced exactly.
        old = np.load(OLD / "history_erased.npz")["X"]
        new = markov_view(rows, HISTORY_ERASED)[0]
        save(dict(seconds=time.time() - t0, history_erased_max_abs_diff=float(np.abs(old - new).max())),
             OUT / "features.json")

    if todo("deep"):
        t0 = time.time()
        for spec in ["orig"] + GRID:
            deep_view(rows, spec, args)
        ci = casia_idx(rows)
        for spec in SWAP:
            deep_view(rows, spec, args, idx=ci, tag=f"swap_{spec}")
        save(dict(seconds=time.time() - t0), OUT / "deep.json")

    if set(args.stages) - {"signatures", "features", "deep"}:
        import revision_stages as rs  # modelling stages (kept separate so feature extraction can run first)
        rs.run(args, todo, rows, X, y, M, groups)


if __name__ == "__main__":
    main()
