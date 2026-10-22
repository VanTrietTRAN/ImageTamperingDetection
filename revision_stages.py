"""Modelling stages of run_revision.py (see its docstring for the shared protocol).

Grid stage outputs one file per training condition (results_rev/grid_<condition>.json) so that conditions can run in
parallel processes (--grid-conds); 'orig' also holds the robustness (standard / degradation-augmented training) and
matched-content swap results, because those models are trained on the original files.
"""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import metrics as mt
import run_experiments as rx
import run_revision as rv

MARKOV_MODELS = ["cnn_aligned", "hgb_markov"]
DEEP_MODELS = ["logreg_resnet_rgb", "logreg_resnet_eladct"]
CNN_CONDS = ["orig", rv.HISTORY_ERASED]  # the CNN is trained on these grid conditions; GBT stands in elsewhere
N_BOOT = 1000


# ------------------------------------------------------------------------------------------------------------ views
class Views:
    """Lazily loaded feature matrices per (model, condition), aligned with the benchmark rows."""

    def __init__(self, rows, X, M, args):
        self.rows, self.args, self._m, self._d, self._p = rows, args, {"orig": (X, M)}, {}, {}
        self.src = np.array([r["source"] for r in rows])
        self.codes = np.unique(self.src, return_inverse=True)[1].astype(float)[:, None]
        self.ci = rv.casia_idx(rows)

    def markov(self, cond):
        if cond not in self._m:
            self._m[cond] = rv.markov_view(self.rows, cond)
        return self._m[cond]

    def deep(self, cond):
        if cond not in self._d:
            self._d[cond] = rv.deep_view(self.rows, cond, self.args)
        return self._d[cond]

    def pscc(self, cond):
        """Zero-shot PSCC-Net scores, or None when that condition was not scored."""
        if cond not in self._p:
            f = rv.FEAT / f"pscc_{cond}.npz"
            self._p[cond] = np.load(f)["score"].astype(np.float64) if f.exists() else None
        return self._p[cond]

    def matrix(self, model, cond):
        if model in MARKOV_MODELS or model == "logreg_markov":
            return self.markov(cond)[0]
        if model == "hgb_metadata":
            return self.markov(cond)[1]
        if model == "source_prior":
            return self.codes
        return self.deep(cond)[model]

    def swap_matrix(self, model, spec):
        if model in MARKOV_MODELS:
            return rv.swap_view(self.rows, spec, self.ci)[0]
        if model == "hgb_metadata":
            return rv.swap_view(self.rows, spec, self.ci)[1]
        return rv.deep_view(self.rows, spec, self.args, idx=self.ci, tag=f"swap_{spec}")[model]


def fit(model, Xtr, ytr, tests: dict, seed):
    """Train `model` on Xtr (its own feature matrix) and score every test matrix."""
    if model == "hgb_metadata":
        return rx.fit_and_score(model, np.zeros((len(ytr), 1)), ytr, {}, seed, Mtr=Xtr, Mtests=tests)[0]
    return rx.fit_and_score(model, Xtr, ytr, tests, seed)[0]


# ---------------------------------------------------------------------------------------------------------- metrics
def report(y, s, src=None):
    r = mt.binary_report(y, s)
    out = dict(n=r["n"], n_tampered=int(np.sum(y)), accuracy=r["accuracy"], balanced_accuracy=r["balanced_accuracy"],
               auc=r["auc"], sensitivity=r["recall_tampered"], specificity=r["recall_authentic"])
    if src is not None:
        for s_name in rv.SOURCES:
            m = src == s_name
            yy = y[m]
            if m.sum() and 0 < yy.sum() < m.sum():
                out[f"auc|{s_name}"] = float(roc_auc_score(yy, s[m]))
                out[f"bal|{s_name}"] = mt.binary_report(yy, s[m])["balanced_accuracy"]
    return out


def summarise(fold_records, keys=None):
    """Fold average within each repetition, then mean / SD / n over repetitions, for every numeric key."""
    by_rep = {}
    for r in fold_records:
        by_rep.setdefault(r["seed"], []).append(r)
    keys = keys or sorted({k for r in fold_records for k, v in r.items()
                           if isinstance(v, float) and k not in ("seed", "fold")})
    out = {}
    for k in keys:
        reps = [np.nanmean([r.get(k, np.nan) for r in rs]) for rs in by_rep.values()]
        reps = [v for v in reps if v == v]
        if reps:
            out[k] = dict(mean=float(np.mean(reps)), sd=float(np.std(reps, ddof=1)) if len(reps) > 1 else 0.0, n_reps=len(reps))
    return out


def group_bootstrap(y, s, groups, fold_of, metric, n_boot=N_BOOT, seed=0, mask=None):
    """95% CI of the fold-averaged `metric(y, s)` when source groups (not images) are resampled within each test fold.
    `fold_of` gives each image's test fold (-1: not scored); `mask` restricts the images (e.g. one dataset)."""
    rng = np.random.default_rng(seed)
    use = fold_of >= 0 if mask is None else (fold_of >= 0) & mask
    per_fold = []
    for k in np.unique(fold_of[use]):
        idx = np.where(use & (fold_of == k))[0]
        g = groups[idx]
        uniq, inv = np.unique(g, return_inverse=True)
        members = [idx[inv == j] for j in range(len(uniq))]
        per_fold.append(members)
    vals = []
    for _ in range(n_boot):
        fold_vals = []
        for members in per_fold:
            pick = rng.integers(0, len(members), len(members))
            idx = np.concatenate([members[j] for j in pick])
            if 0 < y[idx].sum() < len(idx):
                fold_vals.append(metric(y[idx], s[idx]))
        if fold_vals:
            vals.append(np.mean(fold_vals))
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def acc(y, s):
    return float(np.mean((s >= 0.5) == y))


def auc(y, s):
    return float(roc_auc_score(y, s))


def grouped_folds(y, groups, seed, idx=None):
    idx = np.arange(len(y)) if idx is None else idx
    sp = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return [(idx[tr], idx[te]) for tr, te in sp.split(np.zeros(len(idx)), y[idx], groups[idx])]


# ----------------------------------------------------------------------------------------------------------- stages
def stage_ladder(args, V, y, groups):
    """Cumulative ladder L0 random -> L1 source-disjoint -> L2 + JPEG-only -> L3 + history erasure (JPEG-only)."""
    rows, src = V.rows, V.src
    jpeg = np.array([i for i, r in enumerate(rows) if r["container"] == "JPEG"])
    models = MARKOV_MODELS + ["hgb_metadata"] + DEEP_MODELS + ["source_prior"]
    # L1 (source-disjoint, original files) and history erasure on all files are the 'orig' and 'resize0.5+jpeg75'
    # conditions of the intervention grid (same folds and seeds), so they are not recomputed here.
    steps = {"L0_random": ("orig", None, "random"), "L2_sd_jpeg_only": ("orig", jpeg, "grouped"),
             "L3_sd_jpeg_erased": (rv.HISTORY_ERASED, jpeg, "grouped")}
    res = dict(models=models + ["pscc_zero_shot"], steps={})
    for step, (cond, sub, kind) in steps.items():
        recs, ci = [], {}
        scores0 = {}
        fold0 = np.full(len(y), -1)
        for seed in args.seeds:
            if kind == "random":
                base = np.arange(len(y)) if sub is None else sub
                trl, tel = rx.stratified_split(y[base], seed)
                tr, te = base[trl], base[tel]
                splits = [(tr, te)]
            else:
                splits = grouped_folds(y, groups, seed, sub)
            for k, (tr, te) in enumerate(splits):
                for m in models:
                    Xm = V.matrix(m, cond)
                    t0 = time.time()
                    s = fit(m, Xm[tr], y[tr], {"t": Xm[te]}, seed)["t"]
                    recs.append(dict(seed=seed, fold=k, model=m, **report(y[te], s, src[te])))
                    if seed == args.seeds[0]:
                        scores0.setdefault(m, np.full(len(y), np.nan))[te] = s
                    print(f"[ladder {step}] seed {seed} fold {k} {m:22s} acc {recs[-1]['accuracy']:.4f} "
                          f"auc {recs[-1]['auc']:.4f} ({time.time() - t0:.0f}s)", flush=True)
                p = None  # see stage_pscc_eval
                if p is not None:
                    recs.append(dict(seed=seed, fold=k, model="pscc_zero_shot", **report(y[te], p[te], src[te])))
                    if seed == args.seeds[0]:
                        scores0.setdefault("pscc_zero_shot", np.full(len(y), np.nan))[te] = p[te]
                if seed == args.seeds[0]:
                    fold0[te] = k
        summary = {m: summarise([r for r in recs if r["model"] == m]) for m in res["models"]}
        if kind == "grouped":
            for m, s in scores0.items():
                ci[m] = dict(accuracy=group_bootstrap(y, s, groups, fold0, acc),
                             auc=group_bootstrap(y, s, groups, fold0, auc))
            if "hgb_metadata" in scores0 and "hgb_markov" in scores0:
                # paired accuracy difference, metadata minus GBT, on the same test images
                ci["metadata_minus_gbt_accuracy"] = group_bootstrap(
                    y, np.stack([scores0["hgb_metadata"], scores0["hgb_markov"]], 1), groups, fold0,
                    lambda yy, ss: float(np.mean((ss[:, 0] >= 0.5) == yy) - np.mean((ss[:, 1] >= 0.5) == yy)))
        res["steps"][step] = dict(condition=cond, subset="jpeg_only" if sub is not None else "all", split=kind,
                                  n_images=int(len(y) if sub is None else len(sub)),
                                  n_tampered=int(y.sum() if sub is None else y[sub].sum()),
                                  summary=summary, group_bootstrap_ci_seed0=ci, fold_records=recs)
        rv.save(res, rv.OUT / "ladder.json")


def stage_grid(args, V, y, groups, conds):
    src = V.src
    pos = np.full(len(y), -1)
    pos[V.ci] = np.arange(len(V.ci))
    rng_pool = rx.AUGMENT_POOL
    group = getattr(args, "grid_models", None) or "all"
    keep = {"markov": lambda m: m not in DEEP_MODELS, "deep": lambda m: m in DEEP_MODELS, "all": lambda m: True}[group]
    for cond in conds:
        out = rv.OUT / (f"grid_{cond}.json" if group != "deep" else f"grid_{cond}__deep.json")
        if out.exists() and not args.force:
            continue
        matched_models = [m for m in ["hgb_markov", "hgb_metadata"] + DEEP_MODELS
                          + (["cnn_aligned", "source_prior"] if cond in CNN_CONDS else [])
                          if keep(m)]
        rec = dict(matched=[], robustness=[], swap=[])
        boot = {}
        fold0 = np.full(len(y), -1)
        s0 = {}
        for seed in args.seeds:
            folds = grouped_folds(y, groups, seed)
            if cond == "orig":
                aug_choice = np.random.default_rng(seed).integers(0, len(rng_pool), len(y))
            for k, (tr, te) in enumerate(folds):
                if seed == args.seeds[0]:
                    fold0[te] = k
                # (a) matched: the intervention is applied to training and test images alike. On original files the
                # matched models are the 'standard' robustness models below, so only the remaining ones are fitted here.
                standard = [m for m in MARKOV_MODELS + ["hgb_metadata"] + DEEP_MODELS if keep(m)]
                for m in [m for m in matched_models if not (cond == "orig" and m in standard)]:
                    Xm = V.matrix(m, cond)
                    t0 = time.time()
                    s = fit(m, Xm[tr], y[tr], {"t": Xm[te]}, seed)["t"]
                    rec["matched"].append(dict(seed=seed, fold=k, model=m, **report(y[te], s, src[te])))
                    if seed == args.seeds[0]:
                        s0.setdefault(m, np.full(len(y), np.nan))[te] = s
                    print(f"[grid {cond}] seed {seed} fold {k} {m:22s} acc {rec['matched'][-1]['accuracy']:.4f} "
                          f"auc {rec['matched'][-1]['auc']:.4f} ({time.time() - t0:.0f}s)", flush=True)
                p = None  # PSCC-Net (zero-shot) is evaluated on the same folds by stage_pscc_eval
                if p is not None:
                    rec["matched"].append(dict(seed=seed, fold=k, model="pscc_zero_shot", **report(y[te], p[te], src[te])))
                    if seed == args.seeds[0]:
                        s0.setdefault("pscc_zero_shot", np.full(len(y), np.nan))[te] = p[te]
                if cond != "orig":
                    continue
                # (b) robustness: trained on original files (standard) or on originals plus one randomly degraded
                # copy of each training image (augmented), tested on every transformed version of the test fold.
                test_conds = ["orig"] + rv.GRID
                te_c = te[pos[te] >= 0]
                for regime in ("standard", "augmented"):
                    # Degradation-augmented training doubles the training set; it is evaluated with the GBT.
                    for m in (standard if regime == "standard" else [m for m in ["hgb_markov"] if keep(m)]):
                        Xtr = V.matrix(m, "orig")[tr]
                        ytr = y[tr]
                        if regime == "augmented":
                            Xa = np.empty_like(Xtr)
                            ch = aug_choice[tr]
                            for j, spec in enumerate(rng_pool):
                                sel = ch == j
                                Xa[sel] = V.matrix(m, spec)[tr[sel]]
                            Xtr, ytr = np.concatenate([Xtr, Xa]), np.concatenate([ytr, ytr])
                        tests = {c: V.matrix(m, c)[te] for c in test_conds}
                        if regime == "standard":
                            for spec in rv.SWAP:
                                tests[f"swap:{spec}"] = V.swap_matrix(m, spec)[pos[te_c]]
                        t0 = time.time()
                        sc = fit(m, Xtr, ytr, tests, seed)
                        if regime == "standard":
                            rec["matched"].append(dict(seed=seed, fold=k, model=m, **report(y[te], sc["orig"], src[te])))
                            if seed == args.seeds[0]:
                                s0.setdefault(m, np.full(len(y), np.nan))[te] = sc["orig"]
                        for c in test_conds:
                            rec["robustness"].append(dict(seed=seed, fold=k, model=m, regime=regime, test=c,
                                                          **report(y[te], sc[c], src[te])))
                        if regime == "standard":
                            # (c) matched-content swap: held-out CASIA images, identical pixels, controlled final encoding.
                            ref = sc["orig"][pos[te] >= 0]
                            yc = y[te_c]
                            jp = np.array([V.rows[i]["container"] == "JPEG" for i in te_c])
                            for spec in rv.SWAP:
                                s = sc[f"swap:{spec}"]
                                for lab in (0, 1):
                                    for name, msk in (("all", np.ones(len(yc), bool)), ("jpeg", jp)):
                                        mm = (yc == lab) & msk
                                        rec["swap"].append(dict(seed=seed, fold=k, model=m, encoding=spec, label=lab,
                                                                subset=name, n=int(mm.sum()),
                                                                flagged=float(np.mean(s[mm] >= 0.5)),
                                                                flagged_orig=float(np.mean(ref[mm] >= 0.5)),
                                                                mean_score=float(np.mean(s[mm]))))
                                rec["swap"].append(dict(seed=seed, fold=k, model=m, encoding=spec, label=-1, subset="all",
                                                        n=len(yc), auc=float(roc_auc_score(yc, s)),
                                                        auc_orig=float(roc_auc_score(yc, ref))))
                        print(f"[grid robustness] seed {seed} fold {k} {regime:9s} {m:22s} " +
                              " ".join(f"{c}:{np.mean((sc[c] >= 0.5) == y[te]):.3f}" for c in test_conds) +
                              f" ({time.time() - t0:.0f}s)", flush=True)
        res = dict(condition=cond, matched_models=matched_models, seeds=args.seeds)
        res["matched_summary"] = {m: summarise([r for r in rec["matched"] if r["model"] == m])
                                  for m in sorted({r["model"] for r in rec["matched"]})}
        ci = {}
        for m, s in s0.items():
            ci[m] = dict(auc=group_bootstrap(y, s, groups, fold0, auc), accuracy=group_bootstrap(y, s, groups, fold0, acc))
            for s_name in rv.SOURCES:
                ci[m][f"auc|{s_name}"] = group_bootstrap(y, s, groups, fold0, auc, mask=src == s_name)
        res["group_bootstrap_ci_seed0"] = ci
        if cond == "orig":
            res["robustness_summary"] = {
                f"{rg}|{m}|{c}": summarise([r for r in rec["robustness"] if r["regime"] == rg and r["model"] == m and r["test"] == c])
                for rg, m, c in sorted({(r["regime"], r["model"], r["test"]) for r in rec["robustness"]})}
            res["swap_summary"] = {}
            for m, spec, lab, sub in sorted({(r["model"], r["encoding"], r["label"], r["subset"]) for r in rec["swap"]}):
                rs = [r for r in rec["swap"] if (r["model"], r["encoding"], r["label"], r["subset"]) == (m, spec, lab, sub)]
                res["swap_summary"][f"{m}|{spec}|label={lab}|{sub}"] = summarise(rs)
        res["fold_records"] = rec
        rv.save(res, out)


def stage_pscc_eval(args, V, y, groups):
    """Zero-shot PSCC-Net on exactly the folds and subsets of the other stages (no training, so only the scored
    test folds change): ladder steps L0-L3, every scored grid condition, and each cross-dataset target."""
    src = V.src
    jpeg = np.array([i for i, r in enumerate(V.rows) if r["container"] == "JPEG"])
    res = dict(ladder={}, grid={}, cross={})

    def evaluate(score, splits_by_seed):
        recs, fold0 = [], np.full(len(y), -1)
        for seed, splits in splits_by_seed.items():
            for k, (_, te) in enumerate(splits):
                recs.append(dict(seed=seed, fold=k, **report(y[te], score[te], src[te])))
                if seed == args.seeds[0]:
                    fold0[te] = k
        ci = dict(auc=group_bootstrap(y, score, groups, fold0, auc), accuracy=group_bootstrap(y, score, groups, fold0, acc))
        for s_name in rv.SOURCES:
            ci[f"auc|{s_name}"] = group_bootstrap(y, score, groups, fold0, auc, mask=src == s_name)
        return dict(summary=summarise(recs), ci=ci)

    grouped_all = {s: grouped_folds(y, groups, s) for s in args.seeds}
    grouped_jpeg = {s: grouped_folds(y, groups, s, jpeg) for s in args.seeds}
    rnd = {}
    for s in args.seeds:
        tr, te = rx.stratified_split(y, s)
        rnd[s] = [(tr, te)]
    orig, erased = V.pscc("orig"), V.pscc(rv.HISTORY_ERASED)
    if orig is not None:
        res["ladder"]["L0_random"] = evaluate(orig, rnd)
        res["ladder"]["L1_source_disjoint"] = evaluate(orig, grouped_all)
        res["ladder"]["L2_sd_jpeg_only"] = evaluate(orig, grouped_jpeg)
        for train_srcs, test_src in rx.CROSS_DATASET:
            te = np.where(src == test_src)[0]
            r = report(y[te], orig[te])
            full = np.full(len(y), np.nan)
            full[te] = orig[te]
            fold_of = np.full(len(y), -1)
            fold_of[te] = 0
            res["cross"][f"{'+'.join(train_srcs)}->{test_src}"] = dict(
                **r, ci_auc=group_bootstrap(y, full, groups, fold_of, auc),
                ci_bal=group_bootstrap(y, full, groups, fold_of, lambda a, b: mt.binary_report(a, b)["balanced_accuracy"]))
    if erased is not None:
        res["ladder"]["L3_sd_jpeg_erased"] = evaluate(erased, grouped_jpeg)
    for cond in ["orig"] + rv.GRID:
        sc = V.pscc(cond)
        if sc is not None:
            res["grid"][cond] = evaluate(sc, grouped_all)
    rv.save(res, rv.OUT / "pscc_eval.json")


def stage_pscc_swap(args, V, y):
    """Matched-content swap for the zero-shot PSCC-Net on its fixed CASIA JPEG subset."""
    sub = np.array(json.loads((rv.OUT / "pscc_swap_subset.json").read_text())["rows"])
    ref = V.pscc("orig")
    if ref is None:
        return
    yy = y[sub]
    res = dict(n=len(sub), n_tampered=int(yy.sum()), orig=dict(auc=auc(yy, ref[sub])))
    for lab in (0, 1):
        res["orig"][f"flagged|label={lab}"] = float(np.mean(ref[sub][yy == lab] >= 0.5))
    for spec in rv.SWAP:
        f = rv.FEAT / f"pscc_swap_{spec}.npz"
        if not f.exists():
            continue
        s = np.load(f)["score"]
        res[spec] = dict(auc=auc(yy, s), **{f"flagged|label={lab}": float(np.mean(s[yy == lab] >= 0.5)) for lab in (0, 1)},
                         **{f"mean_score|label={lab}": float(np.mean(s[yy == lab])) for lab in (0, 1)})
    rv.save(res, rv.OUT / "pscc_swap.json")


def stage_cross(args, V, y, groups):
    src = V.src
    models = MARKOV_MODELS + ["hgb_metadata"] + DEEP_MODELS
    recs, ci = [], {}
    for train_srcs, test_src in rx.CROSS_DATASET:
        tr, te = np.where(np.isin(src, train_srcs))[0], np.where(src == test_src)[0]
        key = f"{'+'.join(train_srcs)}->{test_src}"
        info = dict(n_train=len(tr), n_train_tampered=int(y[tr].sum()), n_test=len(te), n_test_tampered=int(y[te].sum()))
        fold_of = np.full(len(y), -1)
        fold_of[te] = 0
        for m in models:
            for seed in args.seeds:
                Xm = V.matrix(m, "orig")
                s = fit(m, Xm[tr], y[tr], {"t": Xm[te]}, seed)["t"]
                recs.append(dict(direction=key, seed=seed, fold=0, model=m, **info, **report(y[te], s)))
                if seed == args.seeds[0]:
                    full = np.full(len(y), np.nan)
                    full[te] = s
                    ci[f"{key}|{m}"] = dict(auc=group_bootstrap(y, full, groups, fold_of, auc),
                                            balanced_accuracy=group_bootstrap(
                                                y, full, groups, fold_of, lambda a, b: mt.binary_report(a, b)["balanced_accuracy"]))
                print(f"[cross] {key:26s} {m:22s} seed {seed} acc {recs[-1]['accuracy']:.3f} bal "
                      f"{recs[-1]['balanced_accuracy']:.3f} auc {recs[-1]['auc']:.3f}", flush=True)
        p = None  # see stage_pscc_eval
        if p is not None:
            recs.append(dict(direction=key, seed=0, fold=0, model="pscc_zero_shot", **info, **report(y[te], p[te])))
            full = np.full(len(y), np.nan)
            full[te] = p[te]
            ci[f"{key}|pscc_zero_shot"] = dict(auc=group_bootstrap(y, full, groups, fold_of, auc))
    summary = {f"{d}|{m}": summarise([r for r in recs if r["direction"] == d and r["model"] == m])
               for d, m in sorted({(r["direction"], r["model"]) for r in recs})}
    counts = {r["direction"]: {k: r[k] for k in ("n_train", "n_train_tampered", "n_test", "n_test_tampered")} for r in recs}
    rv.save(dict(summary=summary, counts=counts, group_bootstrap_ci_seed0=ci, records=recs), rv.OUT / "cross.json")


def stage_splits(args, V, y, groups):
    """Split definitions for release: random 80/20 test membership and grouped fold of every image, per seed."""
    import csv
    path = rv.OUT / "splits.csv"
    cols = {}
    for seed in args.seeds:
        _, te = rx.stratified_split(y, seed)
        r = np.zeros(len(y), int)
        r[te] = 1
        cols[f"random_seed{seed}_is_test"] = r
        f = np.full(len(y), -1)
        for k, (_, te) in enumerate(grouped_folds(y, groups, seed)):
            f[te] = k
        cols[f"grouped_seed{seed}_fold"] = f
        jpeg = np.array([i for i, rr in enumerate(V.rows) if rr["container"] == "JPEG"])
        fj = np.full(len(y), -1)
        for k, (_, te) in enumerate(grouped_folds(y, groups, seed, jpeg)):
            fj[te] = k
        cols[f"grouped_jpeg_only_seed{seed}_fold"] = fj
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "label", "file", "container", "source_group"] + list(cols))
        for i, r in enumerate(V.rows):
            w.writerow([r["source"], r["label"], r["name"], r["container"], int(groups[i])] + [int(c[i]) for c in cols.values()])
    rv.save(dict(file=str(path), n=len(y), columns=list(cols)), rv.OUT / "splits.json")


def run(args, todo, rows, X, y, M, groups):
    V = Views(rows, X, M, args)
    if todo("splits"):
        stage_splits(args, V, y, groups)
    if "grid" in args.stages:
        stage_grid(args, V, y, groups, getattr(args, "grid_conds", None) or ["orig"] + rv.GRID)
    if todo("ladder"):
        stage_ladder(args, V, y, groups)
    if todo("cross"):
        stage_cross(args, V, y, groups)
    if todo("pscc_swap"):
        stage_pscc_swap(args, V, y)
    if todo("pscc_eval"):
        stage_pscc_eval(args, V, y, groups)
