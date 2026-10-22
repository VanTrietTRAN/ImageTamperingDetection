"""Is a pooled out-of-fold AUC below 0.5 in the p10 controls real anti-signal or a cross-fold artefact?

Refits the two fastest p10 models on the same grouped folds (seed 0) with history-erased features and reports the
pooled AUC (as in p10), the mean of per-fold AUCs, and the pooled AUC after subtracting each fold's mean score.
Per-fold AUC ignores fold-specific score offsets. Such offsets follow the training-fold class rate, which moves
opposite to the test-fold class rate when grouped folds differ in class balance, so pooling can push AUC below 0.5
even when there is no signal (Forman and Scholz, SIGKDD Explorations 2010).

    python p10_fold_check.py --data ../../datasets_forgery
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE))

import forensics_data as fd  # noqa: E402
import markov_features as mf  # noqa: E402
import run_experiments as rx  # noqa: E402


def auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def check(y, X, groups, model, masks=None, seed=0):
    """Pooled, centred-pooled and per-fold AUC; `masks` restricts the evaluation to parts of the data (e.g. a source)."""
    masks = masks or {"all": np.ones(len(y), bool)}
    oof, centred = np.full(len(y), np.nan), np.full(len(y), np.nan)
    per_fold, folds = {k: [] for k in masks}, []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups):
        s = rx.fit_and_score(model, X[tr], y[tr], {"t": X[te]}, seed)[0]["t"]
        oof[te], centred[te] = s, s - s.mean()
        folds.append(dict(n_test=len(te), train_rate=float(y[tr].mean()), test_rate=float(y[te].mean()),
                          mean_score=float(s.mean())))
        for k, m in masks.items():
            yt, st = y[te][m[te]], s[m[te]]
            if 0 < yt.sum() < len(yt):
                per_fold[k].append(auc(yt, st))
    out = {}
    for k, m in masks.items():
        pf = np.array(per_fold[k])
        out[k] = dict(pooled_auc=auc(y[m], oof[m]), centred_pooled_auc=auc(y[m], centred[m]), mean_fold_auc=float(pf.mean()),
                      std_fold_auc=float(pf.std(ddof=1)) if len(pf) > 1 else 0.0, n_folds=len(pf))
    rates, means = np.array([f["test_rate"] for f in folds]), np.array([f["mean_score"] for f in folds])
    return dict(results=out, folds=folds, corr_test_rate_vs_mean_score=float(np.corrcoef(rates, means)[0, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args()
    cache = args.out / "features"
    rows = fd.build_index(args.data)
    _, ok, _ = mf.extract([r["path"] for r in rows], "orig", cache=cache / "orig.npz")
    rows = [rows[i] for i in np.where(ok & np.array([r["container"] != "UNREADABLE" for r in rows]))[0]]
    Xh = mf.extract([r["path"] for r in rows], rx.HISTORY_ERASED, cache=cache / "history_erased.npz")[0]
    feats = {"hgb_markov": Xh, "logreg_resnet_eladct": np.load(cache / "deep_history_erased.npz")["eladct"]}
    y = np.array([r["label"] for r in rows])
    src = np.array([r["source"] for r in rows])
    container = np.array([r["container"] for r in rows])
    groups = fd.assign_groups(rows)

    res = dict(features=rx.HISTORY_ERASED, seed=0, controls={}, merged={})
    for name, m in {"casia_jpeg": (src == "casia") & (container == "JPEG"), "imd2020": src == "imd2020"}.items():
        idx = np.where(m)[0]
        for model, F in feats.items():
            t0 = time.time()
            r = check(y[idx], F[idx], groups[idx], model)
            res["controls"].setdefault(name, {})[model] = r
            a = r["results"]["all"]
            print(f"[control {name}] {model:22s} pooled {a['pooled_auc']:.3f} centred {a['centred_pooled_auc']:.3f} "
                  f"per-fold {a['mean_fold_auc']:.3f} ± {a['std_fold_auc']:.3f} "
                  f"corr(test rate, mean score) {r['corr_test_rate_vs_mean_score']:.2f} ({time.time() - t0:.0f}s)", flush=True)
    for model, F in feats.items():
        r = check(y, F, groups, model, masks={s: src == s for s in ("casia", "columbia", "imd2020")})
        res["merged"][model] = r
        for s, a in r["results"].items():
            print(f"[merged, within {s}] {model:22s} pooled {a['pooled_auc']:.3f} centred {a['centred_pooled_auc']:.3f} "
                  f"per-fold {a['mean_fold_auc']:.3f} ± {a['std_fold_auc']:.3f} ({a['n_folds']} folds)", flush=True)
    rx.save(res, args.out / "p10_fold_check.json")


if __name__ == "__main__":
    main()
