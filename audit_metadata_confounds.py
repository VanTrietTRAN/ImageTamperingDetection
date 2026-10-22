"""Metadata confounds that survive simple format control.

For each subset, a label is predicted from one non-forensic file property only (exact JPEG
quantisation-table signature, or image size): each signature is assigned the majority class of the
training folds, unseen signatures get the overall training majority. Accuracy is reported with
5-fold stratified cross-validation next to the majority-class rate, so it cannot be explained by
class imbalance alone.

    python audit_metadata_confounds.py --results results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold


def quantisation_signature(path: str) -> str:
    with Image.open(path) as im:
        q = getattr(im, "quantization", None) or {}
    return "-".join(str(sum(v)) for _, v in sorted(q.items())) or "none"


def signature_rule_cv(sig: pd.Series, y: pd.Series, seed: int = 0) -> float:
    correct = 0
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(sig, y):
        train = pd.DataFrame({"sig": sig.iloc[tr].values, "y": y.iloc[tr].values})
        rule = train.groupby("sig")["y"].agg(lambda s: int(s.mean() >= 0.5))
        fallback = int(train["y"].mean() >= 0.5)
        pred = sig.iloc[te].map(rule).fillna(fallback).astype(int)
        correct += int((pred.values == y.iloc[te].values).sum())
    return correct / len(y)


def describe(sub: pd.DataFrame, col: str) -> dict:
    tab = pd.crosstab(sub[col], sub["label"]).reindex(columns=[0, 1], fill_value=0)
    shared = tab[(tab[0] > 0) & (tab[1] > 0)].index
    return dict(
        n=len(sub), n_tampered=int(sub["label"].sum()), distinct_signatures=len(tab), shared_signatures=len(shared),
        authentic_with_shared_signature=float(sub.loc[sub.label == 0, col].isin(shared).mean()),
        tampered_with_shared_signature=float(sub.loc[sub.label == 1, col].isin(shared).mean()),
        rule_accuracy_5fold=signature_rule_cv(sub[col].reset_index(drop=True), sub["label"].reset_index(drop=True)),
        majority_class_rate=float(max(sub["label"].mean(), 1 - sub["label"].mean())),
        most_common=[dict(signature=str(k), authentic=int(r[0]), tampered=int(r[1]))
                     for k, r in tab.assign(total=tab.sum(axis=1)).sort_values("total", ascending=False).head(6).iterrows()],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    args = ap.parse_args()
    df = pd.read_csv(args.results / "index.csv")
    jpeg = df[df["container"] == "JPEG"].copy()
    jpeg["qtable"] = [quantisation_signature(p) for p in jpeg["path"]]
    df["size"] = df["width"].astype(str) + "x" + df["height"].astype(str)

    report = {
        "casia_jpeg_only_quantisation_table": describe(jpeg[jpeg.source == "casia"], "qtable"),
        "imd2020_jpeg_only_quantisation_table": describe(jpeg[jpeg.source == "imd2020"], "qtable"),
        "merged_jpeg_only_quantisation_table": describe(jpeg, "qtable"),
        "columbia_image_size": describe(df[df.source == "columbia"], "size"),
        "casia_image_size": describe(df[df.source == "casia"], "size"),
        "merged_container_format": describe(df, "container"),
    }
    out = args.results / "audit_confounds.json"
    out.write_text(json.dumps(report, indent=1))
    for name, r in report.items():
        print(f"{name:40s} n={r['n']:5d}  rule acc (5-fold) {100 * r['rule_accuracy_5fold']:.1f}%  "
              f"majority-class {100 * r['majority_class_rate']:.1f}%  shared signatures {r['shared_signatures']}/{r['distinct_signatures']}")
    print("written", out)


if __name__ == "__main__":
    main()
