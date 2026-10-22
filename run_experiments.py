"""Bias-controlled evaluation of Markov-DCT tampering detection on the merged CASIA 2.0 + Columbia + IMD2020 benchmark.

    python run_experiments.py --data <datasets_forgery folder> --out results [--seeds 0 1 2 3 4] [--stages audit p1 ...]
                              [--deep-python <python with torch-directml>] [--deep-device auto|cpu|dml]

Stages (each writes results/<stage>.json and is skipped when that file exists, unless --force):
  audit  container / JPEG-quality / resolution composition per source and class
  p1     random stratified 80/20 split (the group's protocol), several seeds
  p2     source-disjoint grouped 5-fold cross-validation on pooled out-of-fold scores (CASIA forgeries grouped with
         the authentic images they were built from)
  p3     container-controlled: JPEG files only
  p4     history-erased: every image downscaled 0.5x and re-encoded at JPEG 75 before feature extraction
  p5     robustness to messaging-app processing, and training with randomly degraded copies
  p6     cross-dataset generalisation
  p7     feature ablations (channels, intra/inter-block TPMs)
  p8     CPU cost of feature extraction and inference
  p9     p1-p6 repeated with frozen ImageNet ResNet18 embeddings of the group's RGB and ELA+DCT inputs + logistic
         regression (a GPU-free stand-in for the group's ResNet18 runs; see src/deep_features.py)
  p10    residual-signal controls: source-prior baseline, within-source AUC with and without compression history,
         and train/test inside CASIA JPEG and inside IMD2020
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

import forensics_data as fd  # noqa: E402
import markov_features as mf  # noqa: E402
import markov_models as mm  # noqa: E402
import metrics as mt  # noqa: E402

DEGRADATIONS = ["jpeg95", "jpeg85", "jpeg75", "resize0.5", "resize0.5+jpeg75"]
HISTORY_ERASED = "resize0.5+jpeg75"
AUGMENT_POOL = ["jpeg95", "jpeg85", "jpeg75", "jpeg65", "resize0.75+jpeg85", "resize0.5+jpeg90", "resize0.5+jpeg75"]
MAIN_MODELS = ["cnn_keras", "cnn_aligned", "hgb_markov", "logreg_markov", "hgb_metadata"]
MAIN_PAIRS = [("cnn_keras", "cnn_aligned"), ("cnn_aligned", "hgb_markov"), ("hgb_markov", "hgb_metadata")]
METRIC_KEYS = ("accuracy", "balanced_accuracy", "auc", "f1_macro", "f1_tampered")
CROSS_DATASET = [(("casia", "columbia"), "imd2020"), (("imd2020",), "casia"), (("casia", "imd2020"), "columbia")]
DEEP_MODELS = ["logreg_resnet_rgb", "logreg_resnet_eladct"]
P10_MODELS = ["cnn_aligned", "hgb_markov", "logreg_resnet_rgb", "logreg_resnet_eladct"]
DEEP_PAIRS = [("logreg_resnet_rgb", "logreg_resnet_eladct"), ("logreg_resnet_eladct", "hgb_markov"),
              ("logreg_resnet_eladct", "hgb_metadata")]


def save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def stratified_split(y, seed, test_size=0.2):
    from sklearn.model_selection import train_test_split
    return train_test_split(np.arange(len(y)), test_size=test_size, random_state=seed, stratify=y)


def fit_and_score(model, Xtr, ytr, tests: dict, seed, Mtr=None, Mtests=None):
    """Train one model and return ({test_name: scores}, fit info)."""
    # Split helpers can return sorted indices (grouped or per-source selections); the CNN recipe takes its
    # validation set from the tail of the training arrays, so rows must be shuffled first.
    perm = np.random.default_rng(seed).permutation(len(ytr))
    Xtr, ytr = Xtr[perm], ytr[perm]
    if Mtr is not None:
        Mtr = Mtr[perm]
    if model == "source_prior":
        # Majority class of each source dataset among the training rows; reads neither pixels nor file metadata.
        codes = Xtr[:, 0]
        majority = {c: float(ytr[codes == c].mean() >= 0.5) for c in np.unique(codes)}
        return {k: np.array([majority.get(c, 0.0) for c in X[:, 0]]) for k, X in tests.items()}, {}
    if model.startswith("cnn_"):
        net, meta = mm.train_cnn(Xtr, ytr, seed=seed, layout=model[4:])
        return {k: mm.cnn_scores(net, X) for k, X in tests.items()}, {k: v for k, v in meta.items() if k != "history"}
    if model == "hgb_metadata":
        est = mm.fit_sklearn("hgb", Mtr, ytr, seed)
        return {k: est.predict_proba(M)[:, 1] for k, M in Mtests.items()}, dict(fit_seconds=est.fit_seconds_)
    est = mm.fit_sklearn(model.split("_")[0], Xtr, ytr, seed)
    return {k: est.predict_proba(X)[:, 1] for k, X in tests.items()}, dict(fit_seconds=est.fit_seconds_)


def deep_features(paths, spec, cache: Path, python: str, device: str) -> dict:
    """Frozen-ResNet18 embeddings computed by src/deep_features.py, run with `python` (e.g. a torch-directml venv)."""
    if not cache.exists():
        jobs = cache.with_name(cache.stem + ".jobs.json")
        jobs.parent.mkdir(parents=True, exist_ok=True)
        jobs.write_text(json.dumps(dict(paths=[str(p) for p in paths], spec=spec)), encoding="utf-8")
        subprocess.run([python, str(HERE / "src" / "deep_features.py"), "--jobs", str(jobs), "--cache", str(cache),
                        "--device", device], check=True)
        jobs.unlink()
    d = np.load(cache)
    return {k: d[k] for k in d.files}


def subgroup_masks(rows, idx):
    src = np.array([rows[i]["source"] for i in idx])
    cont = np.array([rows[i]["container"] for i in idx])
    lab = np.array([rows[i]["label"] for i in idx])
    masks = {f"source={s}": src == s for s in np.unique(src)}
    masks.update({f"{s}|{c}|label={l}": (src == s) & (cont == c) & (lab == l)
                  for s in np.unique(src) for c in np.unique(cont) for l in (0, 1)})
    return masks


def run_split_protocol(name, rows, X, y, M, split_fn, seeds, models, detail_seed=None, features=None,
                       reference_models=(), pairs=MAIN_PAIRS):
    """`features` maps a model to its own feature matrix (default X). `reference_models` are trained on the detail
    seed only, so that McNemar `pairs` can compare against them on the same test images."""
    features = features or {}
    detail_seed = detail_seed if detail_seed is not None else seeds[0]
    records, details = [], {}
    for seed in seeds:
        tr, te = split_fn(seed)
        preds = {}
        for model in list(models) + (list(reference_models) if seed == detail_seed else []):
            Xm = features.get(model, X)
            t0 = time.time()
            scores, info = fit_and_score(model, Xm[tr], y[tr], {"test": Xm[te]}, seed, M[tr], {"test": M[te]})
            rep = mt.binary_report(y[te], scores["test"])
            preds[model] = scores["test"]
            if model in models:
                records.append(dict(protocol=name, seed=seed, model=model, test="test", n_train=len(tr), **rep, **info))
            print(f"[{name}] seed {seed} {model:14s} acc {rep['accuracy']:.4f} auc {rep['auc']:.4f} ({time.time() - t0:.0f}s)"
                  + ("" if model in models else " (reference)"), flush=True)
        if seed == detail_seed:
            details = dict(seed=seed,
                           bootstrap_ci={m: mt.bootstrap_ci(y[te], preds[m]) for m in models},
                           subgroups={m: mt.subgroup_reports(y[te], preds[m], subgroup_masks(rows, te)) for m in models},
                           mcnemar={f"{a} vs {b}": mt.mcnemar(y[te], preds[a] >= 0.5, preds[b] >= 0.5)
                                    for a, b in pairs if a in preds and b in preds})
    summary = {}
    for model in models:
        reps = [r for r in records if r["model"] == model]
        summary[model] = mt.aggregate(reps)
    return dict(records=records, summary=summary, details=details)


def run_grouped_cv(name, rows, X, y, M, groups, seeds, models, features=None, reference_models=(), pairs=MAIN_PAIRS,
                   n_splits=5, keep_scores=False):
    """Source-disjoint StratifiedGroupKFold cross-validation: in every repetition each image is scored once by a model
    that never saw its source group, and metrics use the pooled out-of-fold scores. A single held-out fold is not used
    because CASIA's shared-source chains form one group with ~25% of the images; whichever fold receives it holds ~42%
    of the data with a different class ratio, so single-fold results change regime from seed to seed."""
    from sklearn.model_selection import StratifiedGroupKFold
    features = features or {}
    records, fold_records, details = [], [], {}
    everyone = np.arange(len(y))
    for seed in seeds:
        run = list(models) + (list(reference_models) if seed == seeds[0] else [])
        oof = {m: np.full(len(y), np.nan) for m in run}
        folds = []
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for k, (tr, te) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
            folds.append(dict(fold=k, n_train=len(tr), n_test=len(te), tampered_rate_test=float(y[te].mean())))
            for model in run:
                Xm = features.get(model, X)
                t0 = time.time()
                scores, info = fit_and_score(model, Xm[tr], y[tr], {"test": Xm[te]}, seed, M[tr], {"test": M[te]})
                oof[model][te] = scores["test"]
                rep = mt.binary_report(y[te], scores["test"])
                fold_records.append(dict(protocol=name, seed=seed, fold=k, model=model, n_train=len(tr), **rep, **info))
                print(f"[{name}] seed {seed} fold {k} (n_test {len(te)}) {model:20s} acc {rep['accuracy']:.4f} "
                      f"bal {rep['balanced_accuracy']:.4f} auc {rep['auc']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        for model in models:
            rep = mt.binary_report(y, oof[model])
            records.append(dict(protocol=name, seed=seed, model=model, test="pooled_out_of_fold", **rep))
            print(f"[{name}] seed {seed} pooled {model:20s} acc {rep['accuracy']:.4f} auc {rep['auc']:.4f}", flush=True)
        if seed == seeds[0]:
            first_scores = oof
            details = dict(seed=seed, folds=folds,
                           bootstrap_ci={m: mt.bootstrap_ci(y, oof[m]) for m in models},
                           subgroups={m: mt.subgroup_reports(y, oof[m], subgroup_masks(rows, everyone)) for m in models},
                           mcnemar={f"{a} vs {b}": mt.mcnemar(y, oof[a] >= 0.5, oof[b] >= 0.5)
                                    for a, b in pairs if a in oof and b in oof})
    summary = {m: mt.aggregate([r for r in records if r["model"] == m]) for m in models}
    # Fold-averaged metrics: each fold's model scores only its own test images, so fold-specific score offsets (which
    # follow the training-fold class rate) cannot distort AUC the way pooling out-of-fold scores does.
    fold_averaged = {}
    for m in models:
        per_rep = []
        for seed in seeds:
            fr = [r for r in fold_records if r["model"] == m and r["seed"] == seed]
            per_rep.append({k: float(np.nanmean([r[k] for r in fr])) for k in METRIC_KEYS})
        fold_averaged[m] = mt.aggregate(per_rep)
    result = dict(records=records, fold_records=fold_records, summary=summary, fold_averaged_summary=fold_averaged,
                  details=details)
    if keep_scores:
        result["scores"] = first_scores  # out-of-fold scores of the first repetition, kept in memory only
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "results")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--stages", nargs="+", default=["audit", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--deep-python", default=sys.executable, help="interpreter for src/deep_features.py (p9)")
    ap.add_argument("--deep-device", default="auto", choices=["auto", "cpu", "dml"])
    args = ap.parse_args()
    out, cache = args.out, args.out / "features"

    def todo(stage):
        return stage in args.stages and (args.force or not (out / f"{stage}.json").exists())

    rows = fd.build_index(args.data)
    fd.save_index(rows, out / "index.csv")
    paths = [r["path"] for r in rows]
    X, ok, sec = mf.extract(paths, "orig", cache=cache / "orig.npz")
    keep = np.where(ok & np.array([r["container"] != "UNREADABLE" for r in rows]))[0]
    rows, X, sec = [rows[i] for i in keep], X[keep], sec[keep]
    paths = [r["path"] for r in rows]
    y = np.array([r["label"] for r in rows])
    M, meta_names = fd.metadata_matrix(rows)
    groups = fd.assign_groups(rows)
    print(f"{len(rows)} images ({int(y.sum())} tampered), {len(set(groups))} source groups", flush=True)

    if todo("audit"):
        cnt = collections.Counter((r["source"], r["label"], r["container"]) for r in rows)
        quality = {}
        for s in ("casia", "columbia", "imd2020"):
            for lab in (0, 1):
                q = np.array([r["jpeg_quality"] for r in rows if r["source"] == s and r["label"] == lab and r["container"] == "JPEG"])
                px = np.array([r["width"] * r["height"] for r in rows if r["source"] == s and r["label"] == lab])
                quality[f"{s}|label={lab}"] = dict(
                    n_jpeg=int(q.size), quality_median=float(np.median(q)) if q.size else None,
                    quality_p10=float(np.quantile(q, 0.1)) if q.size else None,
                    quality_p90=float(np.quantile(q, 0.9)) if q.size else None,
                    megapixels_median=float(np.median(px) / 1e6) if px.size else None)
        save(dict(n_images=len(rows), n_tampered=int(y.sum()),
                  container_counts=[dict(source=s, label=l, container=c, n=n) for (s, l, c), n in sorted(cnt.items())],
                  per_source_class=quality), out / "audit.json")

    if todo("p1"):
        save(run_split_protocol("p1_random", rows, X, y, M, lambda s: stratified_split(y, s), args.seeds, MAIN_MODELS),
             out / "p1.json")

    if todo("p2"):
        save(run_grouped_cv("p2_source_disjoint", rows, X, y, M, groups, args.seeds, ["cnn_aligned", "hgb_markov", "hgb_metadata"],
                            pairs=[("cnn_aligned", "hgb_markov"), ("hgb_markov", "hgb_metadata")]), out / "p2.json")

    jpeg_only = np.array([i for i, r in enumerate(rows) if r["container"] == "JPEG"])
    if todo("p3"):
        sub = jpeg_only
        res = run_split_protocol("p3_jpeg_only", [rows[i] for i in sub], X[sub], y[sub], M[sub],
                                 lambda s: stratified_split(y[sub], s), args.seeds,
                                 ["cnn_aligned", "hgb_markov", "hgb_metadata"])
        res["n_images"], res["n_tampered"] = len(sub), int(y[sub].sum())
        save(res, out / "p3.json")

    if todo("p4"):
        Xh, okh, _ = mf.extract(paths, HISTORY_ERASED, cache=cache / "history_erased.npz")
        save(run_split_protocol("p4_history_erased", rows, Xh, y, M, lambda s: stratified_split(y, s), args.seeds,
                                ["cnn_aligned", "hgb_markov"]), out / "p4.json")

    if todo("p5"):
        seed = args.seeds[0]
        tr, te = stratified_split(y, seed)
        tests = {"orig": X[te]}
        for spec in DEGRADATIONS:
            Xd, _, _ = mf.extract([paths[i] for i in te], spec, cache=cache / f"test_seed{seed}_{spec}.npz")
            tests[spec] = Xd
        rng = np.random.default_rng(seed)
        aug_specs = [AUGMENT_POOL[k] for k in rng.integers(0, len(AUGMENT_POOL), len(tr))]
        Xaug, _, _ = mf.extract([paths[i] for i in tr], aug_specs, cache=cache / f"train_seed{seed}_augmented.npz")
        records = []
        for regime, (Xtr, ytr) in {"standard": (X[tr], y[tr]),
                                   "degradation_augmented": (np.concatenate([X[tr], Xaug]), np.concatenate([y[tr], y[tr]]))}.items():
            perm = np.random.default_rng(seed).permutation(len(ytr))
            for model in ["cnn_aligned", "hgb_markov"]:
                scores, info = fit_and_score(model, Xtr[perm], ytr[perm], tests, seed)
                for test_name, s in scores.items():
                    rep = mt.binary_report(y[te], s)
                    records.append(dict(protocol="p5_robustness", regime=regime, seed=seed, model=model, test=test_name, **rep, **info))
                    print(f"[p5] {regime:22s} {model:12s} {test_name:18s} acc {rep['accuracy']:.4f} auc {rep['auc']:.4f}", flush=True)
        save(dict(records=records, degradations=DEGRADATIONS, augment_pool=AUGMENT_POOL), out / "p5.json")

    src = np.array([r["source"] for r in rows])
    if todo("p6"):
        records = []
        for train_srcs, test_src in CROSS_DATASET:
            tr, te = np.where(np.isin(src, train_srcs))[0], np.where(src == test_src)[0]
            for model in ["cnn_aligned", "hgb_markov", "hgb_metadata"]:
                scores, info = fit_and_score(model, X[tr], y[tr], {"test": X[te]}, args.seeds[0], M[tr], {"test": M[te]})
                rep = mt.binary_report(y[te], scores["test"])
                records.append(dict(protocol="p6_cross_dataset", train="+".join(train_srcs), test=test_src, model=model, **rep, **info))
                print(f"[p6] {'+'.join(train_srcs):16s} -> {test_src:9s} {model:12s} acc {rep['accuracy']:.4f} auc {rep['auc']:.4f}", flush=True)
        save(dict(records=records), out / "p6.json")

    if todo("p7"):
        seed = args.seeds[0]
        tr, te = stratified_split(y, seed)
        per_ch = len(mf.TPM_NAMES) * 81
        col = np.arange(X.shape[1])
        ch_of, tpm_of = col // per_ch, (col % per_ch) // 81
        subsets = {"all": col, "Y": col[ch_of == 0], "Cr+Cb": col[ch_of > 0], "intra-block": col[tpm_of < 4],
                   "inter-block": col[tpm_of >= 4], "Y intra-block": col[(ch_of == 0) & (tpm_of < 4)]}
        records = []
        for name, cols in subsets.items():
            est = mm.fit_sklearn("hgb", X[tr][:, cols], y[tr], seed)
            rep = mt.binary_report(y[te], est.predict_proba(X[te][:, cols])[:, 1])
            records.append(dict(protocol="p7_ablation", features=name, n_features=len(cols), model="hgb_markov", **rep))
            print(f"[p7] {name:14s} ({len(cols)} dims) acc {rep['accuracy']:.4f} auc {rep['auc']:.4f}", flush=True)
        save(dict(records=records), out / "p7.json")

    if todo("p8"):
        import torch
        rng = np.random.default_rng(0)
        timing = {}
        for s in ("casia", "columbia", "imd2020"):
            idx = [i for i, r in enumerate(rows) if r["source"] == s]
            if not idx:
                continue
            pick = rng.choice(idx, size=min(60, len(idx)), replace=False)
            ts, mp = [], []
            for i in pick:
                bgr = mf.read_bgr(paths[i])
                t0 = time.perf_counter()
                mf.markov_dct_features(bgr)
                ts.append(time.perf_counter() - t0)
                mp.append(bgr.shape[0] * bgr.shape[1] / 1e6)
            timing[s] = dict(n=len(pick), feature_ms_median=1000 * float(np.median(ts)), megapixels_median=float(np.median(mp)))
        net = mm.MarkovCNN(layout="aligned").eval()
        xb = torch.from_numpy(X[:1]).float()
        with torch.no_grad():
            for _ in range(20):
                net(xb)
            t0 = time.perf_counter()
            for _ in range(500):
                net(xb)
        timing["cnn_inference_ms"] = 1000 * (time.perf_counter() - t0) / 500
        timing["cnn_parameters"] = mm.count_parameters(net)
        timing["cpu"] = __import__("platform").processor()
        timing["torch_threads"] = torch.get_num_threads()
        save(timing, out / "p8.json")
        print("[p8]", json.dumps(timing), flush=True)

    extraction = {}

    def deep(sel_paths, spec, name):
        d = deep_features(sel_paths, spec, cache / f"deep_{name}.npz", args.deep_python, args.deep_device)
        extraction[name] = json.loads(str(d["info"]))
        if not d["ok"].all():
            raise RuntimeError(f"frozen-ResNet18 inputs failed for {int((~d['ok']).sum())} images ({name})")
        return {"logreg_resnet_rgb": d["rgb"], "logreg_resnet_eladct": d["eladct"]}

    if todo("p9"):
        refs = ["hgb_markov", "hgb_metadata"]
        F = deep(paths, "orig", "orig")
        res = dict(models=DEEP_MODELS)
        res["p1"] = run_split_protocol("p1_random", rows, X, y, M, lambda s: stratified_split(y, s), args.seeds,
                                       DEEP_MODELS, features=F, reference_models=refs, pairs=DEEP_PAIRS)
        res["p2"] = run_grouped_cv("p2_source_disjoint", rows, X, y, M, groups, args.seeds, DEEP_MODELS, features=F,
                                   reference_models=refs, pairs=DEEP_PAIRS)
        sub = jpeg_only
        res["p3"] = run_split_protocol("p3_jpeg_only", [rows[i] for i in sub], X[sub], y[sub], M[sub],
                                       lambda s: stratified_split(y[sub], s), args.seeds, DEEP_MODELS,
                                       features={m: v[sub] for m, v in F.items()}, reference_models=refs, pairs=DEEP_PAIRS)
        Xh, _, _ = mf.extract(paths, HISTORY_ERASED, cache=cache / "history_erased.npz")
        Fh = deep(paths, HISTORY_ERASED, "history_erased")
        res["p4"] = run_split_protocol("p4_history_erased", rows, Xh, y, M, lambda s: stratified_split(y, s), args.seeds,
                                       DEEP_MODELS, features=Fh, reference_models=["hgb_markov"], pairs=DEEP_PAIRS)

        seed = args.seeds[0]
        tr, te = stratified_split(y, seed)
        tests = {m: {"orig": F[m][te]} for m in DEEP_MODELS}
        for spec in DEGRADATIONS:
            Fd = deep([paths[i] for i in te], spec, f"test_seed{seed}_{spec}")
            for m in DEEP_MODELS:
                tests[m][spec] = Fd[m]
        rng = np.random.default_rng(seed)
        aug_specs = [AUGMENT_POOL[k] for k in rng.integers(0, len(AUGMENT_POOL), len(tr))]
        Faug = deep([paths[i] for i in tr], aug_specs, f"train_seed{seed}_augmented")
        records = []
        for regime in ("standard", "degradation_augmented"):
            for m in DEEP_MODELS:
                Xtr = F[m][tr] if regime == "standard" else np.concatenate([F[m][tr], Faug[m]])
                ytr = y[tr] if regime == "standard" else np.concatenate([y[tr], y[tr]])
                scores, info = fit_and_score(m, Xtr, ytr, tests[m], seed)
                for test_name, s in scores.items():
                    rep = mt.binary_report(y[te], s)
                    records.append(dict(protocol="p5_robustness", regime=regime, seed=seed, model=m, test=test_name, **rep, **info))
                    print(f"[p9/p5] {regime:22s} {m:20s} {test_name:18s} acc {rep['accuracy']:.4f} auc {rep['auc']:.4f}", flush=True)
        res["p5"] = dict(records=records)

        records = []
        for train_srcs, test_src in CROSS_DATASET:
            tr, te = np.where(np.isin(src, train_srcs))[0], np.where(src == test_src)[0]
            for m in DEEP_MODELS:
                scores, info = fit_and_score(m, F[m][tr], y[tr], {"test": F[m][te]}, args.seeds[0])
                rep = mt.binary_report(y[te], scores["test"])
                records.append(dict(protocol="p6_cross_dataset", train="+".join(train_srcs), test=test_src, model=m, **rep, **info))
                print(f"[p9/p6] {'+'.join(train_srcs):16s} -> {test_src:9s} {m:20s} acc {rep['accuracy']:.4f} auc {rep['auc']:.4f}", flush=True)
        res["p6"] = dict(records=records)
        res["extraction"] = extraction
        save(res, out / "p9.json")

    if todo("p10"):
        # Where does the remaining signal come from? (a) a content-free source-prior rule under p1-p4; (b) within-source
        # AUC of models trained on the merged benchmark, with and without compression history, on pooled grouped-CV
        # scores so every IMD2020 image is scored; (c) train and test inside one dataset and container.
        from sklearn.model_selection import StratifiedGroupKFold
        src_codes = np.unique(src, return_inverse=True)[1].astype(float)[:, None]
        Xh, _, _ = mf.extract(paths, HISTORY_ERASED, cache=cache / "history_erased.npz")
        views = {"original": (X, deep(paths, "orig", "orig")),
                 "history_erased": (Xh, deep(paths, HISTORY_ERASED, "history_erased"))}
        res = dict(sources=sorted(set(src.tolist())), history_erased=HISTORY_ERASED)

        prior = {"source_prior": src_codes}
        res["source_prior"] = {
            "p1": run_split_protocol("p1_random", rows, X, y, M, lambda s: stratified_split(y, s), args.seeds,
                                     ["source_prior"], features=prior)["summary"]["source_prior"],
            "p2": run_grouped_cv("p2_source_disjoint", rows, X, y, M, groups, args.seeds, ["source_prior"],
                                 features=prior)["fold_averaged_summary"]["source_prior"],
            "p3": run_split_protocol("p3_jpeg_only", [rows[i] for i in jpeg_only], X[jpeg_only], y[jpeg_only],
                                     M[jpeg_only], lambda s: stratified_split(y[jpeg_only], s), args.seeds, ["source_prior"],
                                     features={"source_prior": src_codes[jpeg_only]})["summary"]["source_prior"],
        }
        res["source_prior"]["p4"] = res["source_prior"]["p1"]  # identical splits, and the rule never reads pixels

        def view_models(view):
            return P10_MODELS + (["hgb_metadata"] if view == "original" else [])

        seed = args.seeds[0]
        res["within_source"] = {}
        for view, (Xv, Fv) in views.items():
            cv = run_grouped_cv(f"p10_merged_{view}", rows, Xv, y, M, groups, [seed], view_models(view), features=Fv,
                                keep_scores=True)
            scores, table = cv.pop("scores"), {}
            # Same folds as run_grouped_cv; metrics are computed inside each test fold (pooled values kept for reference).
            folds = [te for _, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                     .split(np.zeros(len(y)), y, groups)]
            for m in view_models(view):
                table[m] = {}
                centred = np.empty_like(scores[m])
                for te in folds:
                    centred[te] = scores[m][te] - scores[m][te].mean()
                for s_name in res["sources"] + ["all"]:
                    msk = np.ones(len(y), bool) if s_name == "all" else src == s_name
                    per_fold = [mt.binary_report(y[te][msk[te]], scores[m][te][msk[te]]) for te in folds
                                if 0 < y[te][msk[te]].sum() < msk[te].sum()]
                    fold_auc = np.array([r["auc"] for r in per_fold])
                    fold_bal = np.array([r["balanced_accuracy"] for r in per_fold])
                    pooled = mt.binary_report(y[msk], scores[m][msk])
                    ci = mt.bootstrap_ci(y[msk], centred[msk], keys=("auc",))
                    table[m][s_name] = dict(n=pooled["n"], n_tampered=int(y[msk].sum()), n_folds=len(per_fold),
                                            auc_fold_mean=float(fold_auc.mean()), auc_fold_sd=float(fold_auc.std(ddof=1)),
                                            balanced_accuracy_fold_mean=float(fold_bal.mean()),
                                            balanced_accuracy_fold_sd=float(fold_bal.std(ddof=1)),
                                            auc_centred_pooled=float(mt.binary_report(y[msk], centred[msk])["auc"]),
                                            auc_centred_pooled_ci=ci["auc"], auc_pooled=pooled["auc"],
                                            balanced_accuracy_pooled=pooled["balanced_accuracy"], accuracy_pooled=pooled["accuracy"])
                    print(f"[p10 within-source {view}] {m:20s} {s_name:9s} fold auc {fold_auc.mean():.3f} ± "
                          f"{fold_auc.std(ddof=1):.3f} centred-pooled {table[m][s_name]['auc_centred_pooled']:.3f} "
                          f"pooled {pooled['auc']:.3f} fold bal {fold_bal.mean():.3f}", flush=True)
            res["within_source"][view] = dict(seed=seed, folds=cv["details"]["folds"], models=table)

        container = np.array([r["container"] for r in rows])
        subsets = {"casia_jpeg": np.where((src == "casia") & (container == "JPEG"))[0],
                   "imd2020": np.where(src == "imd2020")[0]}
        res["controls"] = {}
        for name, sub in subsets.items():
            rate = float(y[sub].mean())
            entry = dict(n_images=len(sub), n_tampered=int(y[sub].sum()), majority_rate=max(rate, 1 - rate), views={})
            for view, (Xv, Fv) in views.items():
                cv = run_grouped_cv(f"p10_{name}_{view}", [rows[i] for i in sub], Xv[sub], y[sub], M[sub], groups[sub],
                                    args.seeds, view_models(view), features={m: f[sub] for m, f in Fv.items()},
                                    pairs=[("hgb_markov", "logreg_resnet_eladct"), ("hgb_markov", "hgb_metadata")])
                entry["views"][view] = dict(summary=cv["summary"], fold_averaged_summary=cv["fold_averaged_summary"],
                                            fold_records=cv["fold_records"],
                                            details={k: cv["details"][k] for k in ("seed", "folds", "bootstrap_ci", "mcnemar")})
            res["controls"][name] = entry
        save(res, out / "p10.json")


if __name__ == "__main__":
    main()
