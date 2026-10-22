"""Metrics with uncertainty: point estimates, bootstrap confidence intervals, McNemar tests, subgroup breakdowns."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score


def binary_report(y, score, thr=0.5) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(score) >= thr).astype(int)
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())

    def prf(tp_, fp_, fn_):
        p = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        r = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    p1, r1, f1 = prf(tp, fp, fn)
    p0, r0, f0 = prf(tn, fn, fp)
    auc = roc_auc_score(y, score) if 0 < y.sum() < len(y) else float("nan")
    return dict(n=int(len(y)), accuracy=(tp + tn) / len(y), balanced_accuracy=(r0 + r1) / 2, auc=float(auc),
                precision_tampered=p1, recall_tampered=r1, f1_tampered=f1,
                precision_authentic=p0, recall_authentic=r0, f1_authentic=f0, f1_macro=(f0 + f1) / 2,
                tp=tp, tn=tn, fp=fp, fn=fn)


def bootstrap_ci(y, score, keys=("accuracy", "auc", "f1_macro"), n_boot=1000, seed=0, thr=0.5, alpha=0.05) -> dict:
    y, score = np.asarray(y), np.asarray(score)
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        rep = binary_report(y[idx], score[idx], thr)
        for k in keys:
            samples[k].append(rep[k])
    return {k: (float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))) for k, v in samples.items()}


def mcnemar(y, pred_a, pred_b) -> dict:
    """Exact McNemar test on paired predictions of two classifiers over the same test images."""
    y = np.asarray(y)
    a_ok, b_ok = np.asarray(pred_a) == y, np.asarray(pred_b) == y
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return dict(a_only_correct=b, b_only_correct=c, p_value=float(p))


def subgroup_reports(y, score, masks: dict, thr=0.5, min_n=20) -> dict:
    """Per-subgroup accuracy (and AUC when both classes are present)."""
    y, score = np.asarray(y), np.asarray(score)
    out = {}
    for name, m in masks.items():
        m = np.asarray(m, dtype=bool)
        if m.sum() < min_n:
            continue
        pred = (score[m] >= thr).astype(int)
        yy = y[m]
        out[name] = dict(n=int(m.sum()), n_tampered=int(yy.sum()), accuracy=float((pred == yy).mean()),
                         auc=float(roc_auc_score(yy, score[m])) if 0 < yy.sum() < len(yy) else float("nan"))
    return out


def aggregate(reports: list[dict], keys=("accuracy", "balanced_accuracy", "auc", "f1_macro", "f1_tampered")) -> dict:
    return {k: dict(mean=float(np.mean([r[k] for r in reports])), std=float(np.std([r[k] for r in reports], ddof=1))
                    if len(reports) > 1 else 0.0, n_runs=len(reports)) for k in keys}
