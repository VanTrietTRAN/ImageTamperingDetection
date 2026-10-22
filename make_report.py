"""Turn results/*.json into paper tables (Markdown + LaTeX) and figures (PDF + PNG).

    python make_report.py --results results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MODEL_NAMES = {"cnn_keras": "CNN (original reshape)", "cnn_aligned": "CNN (TPM-aligned)", "hgb_markov": "GBT on Markov-DCT",
               "logreg_markov": "LogReg on Markov-DCT", "hgb_metadata": "GBT on file metadata",
               "logreg_resnet_rgb": "Frozen ResNet18 (ImageNet), RGB + LogReg",
               "logreg_resnet_eladct": "Frozen ResNet18 (ImageNet), ELA+DCT + LogReg",
               "source_prior": "Majority class of the source dataset (no pixels, no metadata)"}
SHORT_NAMES = {"cnn_aligned": "CNN Markov-DCT", "hgb_markov": "GBT Markov-DCT", "logreg_resnet_eladct": "ResNet18 ELA+DCT",
               "logreg_resnet_rgb": "ResNet18 RGB", "hgb_metadata": "GBT metadata"}
MODEL_STYLE = {"cnn_aligned": ("#2a78d6", "o"), "hgb_markov": ("#eb6834", "s"), "hgb_metadata": ("#1baf7a", "^"),
               "cnn_keras": ("#eda100", "D"), "logreg_markov": ("#e87ba4", "v"), "logreg_resnet_rgb": ("#9b8ec4", "P"),
               "logreg_resnet_eladct": ("#6d4c9f", "X")}
PROTOCOLS = [("p1", "Random split"), ("p2", "Source-disjoint"), ("p3", "JPEG-only"), ("p4", "History-erased")]
TEST_NAMES = {"orig": "Original", "jpeg95": "JPEG 95", "jpeg85": "JPEG 85", "jpeg75": "JPEG 75",
              "resize0.5": "Resize 0.5", "resize0.5+jpeg75": "Resize 0.5 + JPEG 75"}
SOURCE_NAMES = {"casia": "CASIA 2.0", "columbia": "Columbia", "imd2020": "IMD2020"}
VIEW_NAMES = {"original": "Original files", "history_erased": "History-erased"}
CONTROL_NAMES = {"casia_jpeg": "CASIA 2.0, JPEG files only", "imd2020": "IMD2020"}
INK_2, GRID, AXIS = "#52514e", "#e1e0d9", "#c3c2b7"
TEAM_RESNET = [("ResNet18 on RGB (team run, single split)", 0.72, 0.7876), ("ResNet18 on ELA+DCT (team run, single split)", 0.89, 0.9573)]


def load(results: Path, name: str):
    p = results / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def pm(stat, key="accuracy", scale=100, digits=1):
    s = stat.get(key) if stat else None
    return "-" if not s else f"{scale * s['mean']:.{digits}f} ± {scale * s['std']:.{digits}f}"


METRIC_KEYS = ("accuracy", "balanced_accuracy", "auc", "f1_macro", "f1_tampered")


def fold_averaged(block):
    """Grouped-CV block -> metrics averaged over test folds per repetition, then mean ± SD over repetitions.
    Returns None for single-split protocols. Pooling out-of-fold scores biases AUC when folds differ in class balance."""
    if not block or "fold_records" not in block:
        return None
    if "fold_averaged_summary" in block:
        return block["fold_averaged_summary"]
    out = {}
    for m in block["summary"]:
        recs = [r for r in block["fold_records"] if r["model"] == m]
        per_rep = [{k: float(np.nanmean([r[k] for r in recs if r["seed"] == s])) for k in METRIC_KEYS}
                   for s in sorted({r["seed"] for r in recs})]
        out[m] = {k: dict(mean=float(np.mean([p[k] for p in per_rep])),
                          std=float(np.std([p[k] for p in per_rep], ddof=1)) if len(per_rep) > 1 else 0.0,
                          n_runs=len(per_rep)) for k in METRIC_KEYS}
    return out


def ci_text(value, ci, scale=1, digits=3):
    return f"{scale * value:.{digits}f} [{scale * ci[0]:.{digits}f}, {scale * ci[1]:.{digits}f}]"


def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"


def tex_table(header, rows, caption, label):
    esc = lambda s: str(s).replace("±", r"$\pm$").replace("%", r"\%").replace("_", r"\_").replace("->", r"$\rightarrow$")
    return "\n".join([r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
                      r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{l" + "c" * (len(header) - 1) + "}", r"\toprule",
                      " & ".join(esc(h) for h in header) + r" \\", r"\midrule",
                      *[" & ".join(esc(c) for c in r) + r" \\" for r in rows], r"\bottomrule", r"\end{tabular}}",
                      r"\end{table}", ""])


def style_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(True, axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    args = ap.parse_args()
    res, out = args.results, args.results / "report"
    out.mkdir(parents=True, exist_ok=True)
    md, tex = ["# Results\n"], []

    audit = load(res, "audit")
    if audit:
        rows = [[c["source"], "tampered" if c["label"] else "authentic", c["container"], c["n"]] for c in audit["container_counts"]]
        md += ["## Benchmark composition\n", md_table(["Source", "Class", "Container", "Images"], rows)]
        q = [[k.replace("|", " / "), v["n_jpeg"], f"{v['quality_median']:.0f}" if v["quality_median"] is not None else "-",
              f"{v['megapixels_median']:.2f}" if v["megapixels_median"] is not None else "-"]
             for k, v in audit["per_source_class"].items()]
        md += [md_table(["Source / class", "JPEG files", "Median JPEG quality", "Median megapixels"], q)]
        tex.append(tex_table(["Source", "Class", "Container", "Images"], rows, "Composition of the merged benchmark.", "tab:audit"))

    confounds = load(res, "audit_confounds")
    if confounds:
        names = {"casia_jpeg_only_quantisation_table": "CASIA 2.0 JPEG files: quantisation tables",
                 "imd2020_jpeg_only_quantisation_table": "IMD2020 JPEG files: quantisation tables",
                 "merged_jpeg_only_quantisation_table": "All JPEG files: quantisation tables",
                 "columbia_image_size": "Columbia: image size", "casia_image_size": "CASIA 2.0: image size",
                 "merged_container_format": "All images: container format"}
        rows = [[names.get(k, k), v["n"], f"{100 * v['rule_accuracy_5fold']:.1f}", f"{100 * v['majority_class_rate']:.1f}"]
                for k, v in confounds.items()]
        header = ["File property used alone", "Images", "Rule acc (%)", "Majority class (%)"]
        md += ["## Single-attribute shortcut rules (5-fold CV; each value takes the majority label of the training folds)\n",
               md_table(header, rows)]
        tex.append(tex_table(header, rows, "Accuracy of predicting the label from one file property, without reading pixels.",
                             "tab:shortcuts"))

    proto = {k: load(res, k) for k, _ in PROTOCOLS}
    p9, p10 = load(res, "p9"), load(res, "p10")
    def protocol_summary(block):
        return (fold_averaged(block) or block["summary"]) if block else {}

    summ = {k: {**protocol_summary(proto[k]), **protocol_summary(p9[k] if p9 and k in p9 else None)} for k, _ in PROTOCOLS}
    models = [m for m in MODEL_NAMES if any(m in s for s in summ.values())]
    if models:
        header = ["Model"] + [f"{title} Acc (%)" for k, title in PROTOCOLS] + [f"{title} AUC" for k, title in PROTOCOLS]
        rows = [[MODEL_NAMES[m]] + [pm(summ[k].get(m)) for k, _ in PROTOCOLS]
                + [pm(summ[k].get(m), "auc", 1, 3) for k, _ in PROTOCOLS] for m in models]
        if p10:
            prior = p10["source_prior"]
            rows.append([MODEL_NAMES["source_prior"]] + [pm(prior.get(k)) for k, _ in PROTOCOLS]
                        + [pm(prior.get(k), "auc", 1, 3) for k, _ in PROTOCOLS])
        rows += [[name, f"{100 * acc:.1f}", "-", "-", "-", f"{auc:.3f}", "-", "-", "-"] for name, acc, auc in TEAM_RESNET]
        md += ["## Accuracy under increasingly strict protocols (mean ± std over seeds)\n", md_table(header, rows)]
        p2d = proto["p2"].get("details") if proto.get("p2") else None
        if p2d and "folds" in p2d:
            md += [f"Source-disjoint = grouped {len(p2d['folds'])}-fold cross-validation; metrics are averaged over the test "
                   f"folds, then mean ± std over repetitions. Fold test sizes for seed {p2d['seed']}: "
                   + ", ".join(f"{f['n_test']} ({100 * f['tampered_rate_test']:.0f}% tampered)" for f in p2d["folds"]) + ".\n"]
        tex.append(tex_table(header, rows, "Detection performance under increasingly strict evaluation protocols.", "tab:protocols"))

    p1 = proto.get("p1")
    if p1 and p1.get("details"):
        d = p1["details"]
        keys = sorted({k for sg in d["subgroups"].values() for k in sg if "|" in k})
        rows = [[k.replace("|", " / ")] + [f"{100 * d['subgroups'][m][k]['accuracy']:.1f} (n={d['subgroups'][m][k]['n']})" if k in d["subgroups"][m] else "-"
                       for m in d["subgroups"]] for k in keys]
        md += [f"## Random split, seed {d['seed']}: accuracy by source, container and class\n",
               md_table(["Subgroup"] + [MODEL_NAMES[m] for m in d["subgroups"]], rows)]
        md += ["## Random split: 95% bootstrap CIs and McNemar tests\n",
               md_table(["Model", "Accuracy CI", "AUC CI"], [[MODEL_NAMES[m], f"{100 * v['accuracy'][0]:.1f}–{100 * v['accuracy'][1]:.1f}",
                                                              f"{v['auc'][0]:.3f}–{v['auc'][1]:.3f}"] for m, v in d["bootstrap_ci"].items()]),
               md_table(["Comparison", "A-only correct", "B-only correct", "p-value"],
                        [[k, v["a_only_correct"], v["b_only_correct"], f"{v['p_value']:.2e}"] for k, v in d["mcnemar"].items()])]

    if p9:
        rows = [[title, c, v["a_only_correct"], v["b_only_correct"], f"{v['p_value']:.2e}"]
                for k, title in PROTOCOLS if k in p9 and p9[k].get("details") for c, v in p9[k]["details"]["mcnemar"].items()]
        md += ["## Frozen ResNet18 baselines: McNemar tests on the first seed\n",
               md_table(["Protocol", "Comparison", "A-only correct", "B-only correct", "p-value"], rows)]
        rows = [[name, v["device"].rstrip("\x00"), v["n_images"], f"{v['wall_seconds'] / 60:.1f}", f"{3 * v['n_images'] / v['model_seconds']:.0f}"]
                for name, v in p9.get("extraction", {}).items()]
        md += ["## Frozen ResNet18 embedding cost (3 forward passes per image)\n",
               md_table(["Job", "Device", "Images", "Wall time (min)", "ResNet18 images/s"], rows)]

    if p10:
        srcs = p10["sources"]
        rows = []
        for view, w in p10["within_source"].items():
            for m, t in w["models"].items():
                rows.append([VIEW_NAMES[view], MODEL_NAMES[m]]
                            + [f"{t[s]['auc_fold_mean']:.3f} ± {t[s]['auc_fold_sd']:.3f}" for s in srcs]
                            + [f"{100 * t['imd2020']['balanced_accuracy_fold_mean']:.1f} ± "
                               f"{100 * t['imd2020']['balanced_accuracy_fold_sd']:.1f}"])
        first = next(iter(next(iter(p10["within_source"].values()))["models"].values()))
        header = ["Features", "Model"] + [f"AUC within {SOURCE_NAMES[s]}" for s in srcs] + ["IMD2020 balanced acc (%)"]
        md += ["## Detection within each source dataset (trained on the merged benchmark; grouped 5-fold CV, first seed; "
               "mean ± SD over the 5 test folds)\n", md_table(header, rows),
               "Images per source: " + ", ".join(f"{SOURCE_NAMES[s]} {first[s]['n']} ({first[s]['n_tampered']} tampered)" for s in srcs)
               + f". History erasure = {p10['history_erased']}. Metrics are computed inside each test fold: pooling scores across "
               "folds mixes fold-specific score offsets and can push AUC below 0.5 when there is no signal.\n"]
        tex.append(tex_table(header, rows, "AUC within each source dataset for models trained on the merged benchmark, with the "
                             "original files and after history erasure (mean $\\pm$ SD over five source-disjoint test folds).",
                             "tab:within"))

        rows, tests = [], []
        for name, c in p10["controls"].items():
            for view, v in c["views"].items():
                for m, st in v["fold_averaged_summary"].items():
                    rows.append([f"{CONTROL_NAMES[name]} (n={c['n_images']}, majority {100 * c['majority_rate']:.1f}%)",
                                 VIEW_NAMES[view], MODEL_NAMES[m], pm(st), pm(st, "balanced_accuracy"), pm(st, "auc", 1, 3)])
                tests += [[CONTROL_NAMES[name], VIEW_NAMES[view], comp, t["a_only_correct"], t["b_only_correct"], f"{t['p_value']:.2e}"]
                          for comp, t in v["details"]["mcnemar"].items()]
        header = ["Subset", "Features", "Model", "Acc (%)", "Balanced acc (%)", "AUC"]
        md += ["## Train and test inside one dataset and container (grouped 5-fold CV; metrics averaged over test folds, "
               "mean ± std over 5 repetitions)\n",
               md_table(header, rows),
               md_table(["Subset", "Features", "Comparison (first seed, pooled)", "A-only correct", "B-only correct", "p-value"], tests)]
        tex.append(tex_table(header, rows, "Training and testing inside a single dataset and container, with and without "
                             "compression history.", "tab:controls"))

    p5 = load(res, "p5")
    if p5 and p9 and "p5" in p9:
        p5["records"] = p5["records"] + p9["p5"]["records"]
    if p5:
        tests = ["orig"] + p5["degradations"]
        combos = sorted({(r["model"], r["regime"]) for r in p5["records"]})
        rows = []
        for m, reg in combos:
            got = {r["test"]: r for r in p5["records"] if r["model"] == m and r["regime"] == reg}
            rows.append([f"{MODEL_NAMES[m]} / {reg}"] + [f"{100 * got[t]['accuracy']:.1f} / {got[t]['auc']:.3f}" for t in tests])
        md += ["## Robustness to messaging-app processing (accuracy % / AUC)\n", md_table(["Model / training"] + [TEST_NAMES[t] for t in tests], rows)]
        tex.append(tex_table(["Model / training"] + [TEST_NAMES[t] for t in tests], rows, "Robustness to recompression and resizing.", "tab:robustness"))

    p6 = load(res, "p6")
    if p6 and p9 and "p6" in p9:
        p6["records"] = p6["records"] + p9["p6"]["records"]
    if p6:
        rows = [[f"{r['train']} -> {r['test']}", MODEL_NAMES[r["model"]], f"{100 * r['accuracy']:.1f}", f"{r['auc']:.3f}",
                 f"{100 * r['balanced_accuracy']:.1f}"] for r in p6["records"]]
        md += ["## Cross-dataset generalisation\n", md_table(["Train -> test", "Model", "Acc (%)", "AUC", "Balanced acc (%)"], rows)]
        tex.append(tex_table(["Train -> test", "Model", "Acc (%)", "AUC", "Balanced acc (%)"], rows, "Cross-dataset generalisation.", "tab:cross"))

    p7 = load(res, "p7")
    if p7:
        md += ["## Feature ablation (GBT, random split)\n",
               md_table(["Features", "Dims", "Acc (%)", "AUC"], [[r["features"], r["n_features"], f"{100 * r['accuracy']:.1f}", f"{r['auc']:.3f}"] for r in p7["records"]])]

    p8 = load(res, "p8")
    if p8:
        rows = [[s, f"{v['megapixels_median']:.2f}", f"{v['feature_ms_median']:.0f}"] for s, v in p8.items() if isinstance(v, dict)]
        md += ["## CPU cost\n", md_table(["Source", "Median megapixels", "Feature extraction (ms/image)"], rows),
               f"CNN inference: {p8['cnn_inference_ms']:.2f} ms/image, {p8['cnn_parameters']:,} parameters, CPU {p8['cpu']}, {p8['torch_threads']} threads.\n"]

    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    (out / "tables.tex").write_text("\n".join(tex), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})

    if models:
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        xs = np.arange(len(PROTOCOLS))
        for m in [m for m in ("cnn_aligned", "hgb_markov", "hgb_metadata", "logreg_resnet_eladct") if m in models]:
            pts = [(i, summ[k][m]["accuracy"]) for i, (k, _) in enumerate(PROTOCOLS) if m in summ[k]]
            if not pts:
                continue
            color, marker = MODEL_STYLE[m]
            ax.errorbar([p[0] for p in pts], [100 * p[1]["mean"] for p in pts], yerr=[100 * p[1]["std"] for p in pts],
                        color=color, marker=marker, markersize=5, linewidth=1.5, capsize=2, label=MODEL_NAMES[m])
        ax.axhline(50, color=AXIS, linewidth=0.8, linestyle=":")
        ax.set_xticks(xs, [t for _, t in PROTOCOLS])
        style_axes(ax, "Evaluation protocol (stricter to the right)", "Accuracy (%)")
        ax.legend(fontsize=6.5, frameon=False, loc="lower left")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out / f"fig_protocols.{ext}", dpi=300, facecolor="white")
        plt.close(fig)

    if p5:
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        tests = ["orig"] + p5["degradations"]
        for m in ("cnn_aligned", "hgb_markov"):
            for reg, ls in (("standard", "-"), ("degradation_augmented", "--")):
                got = {r["test"]: r for r in p5["records"] if r["model"] == m and r["regime"] == reg}
                if not got:
                    continue
                color, marker = MODEL_STYLE[m]
                ax.plot(range(len(tests)), [got[t]["auc"] for t in tests], color=color, marker=marker, markersize=4,
                        linewidth=1.5, linestyle=ls, label=f"{MODEL_NAMES[m]} ({'augmented' if reg != 'standard' else 'standard'})")
        short = {"orig": "Original", "jpeg95": "Q95", "jpeg85": "Q85", "jpeg75": "Q75", "resize0.5": "×0.5",
                 "resize0.5+jpeg75": "×0.5+Q75"}
        ax.set_xticks(range(len(tests)), [short.get(t, t) for t in tests])
        style_axes(ax, "Test-time processing", "AUC")
        fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=6, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.84))
        for ext in ("pdf", "png"):
            fig.savefig(out / f"fig_robustness.{ext}", dpi=300, facecolor="white")
        plt.close(fig)

    if p10:
        # Dot plot, mean ± SD over test folds: the AUC axis does not start at zero, so bars would exaggerate differences.
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        srcs = p10["sources"]
        shown = [m for m in ("cnn_aligned", "hgb_markov", "logreg_resnet_eladct") if m in p10["within_source"]["original"]["models"]]
        series = [(m, v) for m in shown for v in ("original", "history_erased")]
        step = 0.8 / len(series)
        for j, (m, view) in enumerate(series):
            t = p10["within_source"][view]["models"][m]
            x = np.arange(len(srcs)) - 0.4 + step * (j + 0.5)
            y = np.array([t[s]["auc_fold_mean"] for s in srcs])
            err = np.array([t[s]["auc_fold_sd"] for s in srcs])
            color, marker = MODEL_STYLE[m]
            ax.errorbar(x, y, yerr=err, fmt=marker, markersize=4, color=color, ecolor=color, elinewidth=0.8, capsize=1.5,
                        markerfacecolor=color if view == "original" else "white", markeredgewidth=1,
                        label=f"{SHORT_NAMES[m]}, {'original' if view == 'original' else 'history-erased'}")
        ax.axhline(0.5, color=INK_2, linewidth=0.8, linestyle=":")
        ax.text(-0.45, 0.492, "chance", color=INK_2, fontsize=6, ha="left", va="top")
        ax.set_xticks(range(len(srcs)), [SOURCE_NAMES[s] for s in srcs])
        ax.set_ylim(0.3, 1.02)
        style_axes(ax, "Test images from", "AUC within dataset")
        fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=5.5, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.8))
        for ext in ("pdf", "png"):
            fig.savefig(out / f"fig_within_source.{ext}", dpi=300, facecolor="white")
        plt.close(fig)
    print("report written to", out)


if __name__ == "__main__":
    main()
