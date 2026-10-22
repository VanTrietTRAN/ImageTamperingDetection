"""Numbers, figures and table rows of the camera-ready revision, from results_rev/*.json.

    python make_revision_report.py [--res results_rev] [--out results_rev/report]

Writes report/numbers.json (every value quoted in the paper), report/summary.md (readable digest), and the figures
fig_interventions.png/.pdf and fig_robustness.png/.pdf.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GRID_ORDER = ["orig", "jpeg95", "jpeg75", "resize0.75", "resize0.75+jpeg95", "resize0.75+jpeg75",
              "resize0.5", "resize0.5+jpeg95", "resize0.5+jpeg75"]
GRID_LABEL = {"orig": "orig.", "jpeg95": "Q95", "jpeg85": "Q85", "jpeg75": "Q75", "resize0.75": "0.75×",
              "resize0.75+jpeg95": "0.75×\n+Q95", "resize0.75+jpeg75": "0.75×\n+Q75", "resize0.5": "0.5×",
              "resize0.5+jpeg95": "0.5×\n+Q95", "resize0.5+jpeg75": "0.5×\n+Q75"}
NAMES = {"cnn_aligned": "CNN, Markov-DCT", "hgb_markov": "GBT, Markov-DCT", "hgb_metadata": "GBT, file metadata",
         "logreg_resnet_rgb": "ResNet18 RGB", "logreg_resnet_eladct": "ResNet18 ELA+DCT",
         "pscc_zero_shot": "PSCC-Net (zero-shot)", "source_prior": "Source prior"}
COLORS = {"cnn_aligned": "#2a6fdb", "hgb_markov": "#e8590c", "hgb_metadata": "#2b8a3e", "logreg_resnet_rgb": "#9c6ade",
          "logreg_resnet_eladct": "#5f3dc4", "pscc_zero_shot": "#c2255c", "source_prior": "#868e96"}
MARKERS = {"cnn_aligned": "o", "hgb_markov": "s", "hgb_metadata": "D", "logreg_resnet_rgb": "P",
           "logreg_resnet_eladct": "X", "pscc_zero_shot": "^", "source_prior": "v"}


def load(res, name):
    p = res / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def ms(st, key, scale=100, d=1):
    if not st or key not in st:
        return None
    return dict(mean=round(scale * st[key]["mean"], d + 2), sd=round(scale * st[key]["sd"], d + 2))


def fmt(v, d=1):
    return "–" if v is None else f"{v['mean']:.{d}f} ± {v['sd']:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=Path, default=HERE / "results_rev")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    res, out = args.res, args.out or args.res / "report"
    out.mkdir(parents=True, exist_ok=True)
    N, md = {}, []

    sig = load(res, "signatures")
    if sig:
        N["signatures"] = {k: dict(n=v["source_disjoint"]["n"], rule=round(100 * v["source_disjoint"]["accuracy"], 1),
                                   rule_stratified=round(100 * v["stratified"]["accuracy"], 1),
                                   bal=round(100 * v["source_disjoint"]["balanced_accuracy"], 1),
                                   majority=round(100 * v["source_disjoint"]["majority_rate"], 1))
                           for k, v in sig["rules"].items()}
        N["subsampling_crosstab"] = sig["crosstab"]
        N["swap_tables"] = sig["chosen"]
        md += ["## Signature rules (source-disjoint 5-fold)", "| rule | n | acc | bal | majority |", "|---|---|---|---|---|"]
        md += [f"| {k} | {v['n']} | {v['rule']} | {v['bal']} | {v['majority']} |" for k, v in N["signatures"].items()]

    lad = load(res, "ladder")
    if lad:
        N["ladder"] = {}
        md += ["", "## Ladder (balanced accuracy / accuracy / AUC, mean ± SD over 5 repetitions)"]
        for step, s in lad["steps"].items():
            N["ladder"][step] = dict(n=s["n_images"], n_tampered=s["n_tampered"], models={}, ci=s["group_bootstrap_ci_seed0"])
            md.append(f"### {step} (n={s['n_images']}, tampered={s['n_tampered']})")
            for m, st in s["summary"].items():
                if not st:
                    continue
                row = {k: ms(st, k) for k in ("accuracy", "balanced_accuracy", "auc", "sensitivity", "specificity",
                                              "auc|casia", "auc|columbia", "auc|imd2020")}
                N["ladder"][step]["models"][m] = row
                md.append(f"- {m}: acc {fmt(row['accuracy'])}, bal {fmt(row['balanced_accuracy'])}, auc {fmt(row['auc'])}, "
                          f"sens {fmt(row['sensitivity'])}, spec {fmt(row['specificity'])}, AUC casia/col/imd "
                          f"{fmt(row['auc|casia'])} / {fmt(row['auc|columbia'])} / {fmt(row['auc|imd2020'])}")

    grid = {}
    for c in GRID_ORDER + ["jpeg85"]:
        g, gd = load(res, f"grid_{c}"), load(res, f"grid_{c}__deep")
        if g and gd:  # merge the ResNet18 pass into the Markov/metadata pass
            for k in ("matched_summary", "group_bootstrap_ci_seed0", "robustness_summary", "swap_summary"):
                if k in gd:
                    g.setdefault(k, {}).update(gd[k])
        g = g or gd
        if g:
            grid[c] = g
    if grid:
        N["grid"] = {}
        md += ["", "## Matched interventions (train and test transformed), per-dataset AUC"]
        for c, g in grid.items():
            N["grid"][c] = {}
            for m, st in g["matched_summary"].items():
                N["grid"][c][m] = {k: ms(st, k, 1, 3) for k in ("auc", "auc|casia", "auc|columbia", "auc|imd2020")}
                N["grid"][c][m].update({k: ms(st, k) for k in ("accuracy", "balanced_accuracy", "bal|imd2020", "bal|casia")})
                N["grid"][c][m]["ci"] = g["group_bootstrap_ci_seed0"].get(m)
                r = N["grid"][c][m]
                md.append(f"- {c:18s} {m:22s} AUC all {fmt(r['auc'], 3)} casia {fmt(r['auc|casia'], 3)} columbia "
                          f"{fmt(r['auc|columbia'], 3)} imd {fmt(r['auc|imd2020'], 3)} | bal {fmt(r['balanced_accuracy'])}")
        g0 = grid.get("orig")
        if g0 and "robustness_summary" in g0:
            N["robustness"] = {k: {kk: ms(v, kk) for kk in ("accuracy", "balanced_accuracy", "auc", "auc|imd2020", "auc|casia")}
                               for k, v in g0["robustness_summary"].items()}
            md += ["", "## Robustness (trained on originals, tested on transformed), balanced accuracy / AUC"]
            md += [f"- {k}: acc {fmt(v['accuracy'])} bal {fmt(v['balanced_accuracy'])} auc {fmt(v['auc'])}"
                   for k, v in N["robustness"].items()]
            N["swap"] = {k: {kk: ms(v, kk) for kk in ("flagged", "flagged_orig", "mean_score", "auc", "auc_orig")}
                         for k, v in g0["swap_summary"].items()}
            md += ["", "## Matched-content swap (held-out CASIA; % flagged as tampered)"]
            md += [f"- {k}: flagged {fmt(v['flagged'])} (orig {fmt(v['flagged_orig'])}) auc {fmt(v['auc'])}"
                   for k, v in N["swap"].items()]
    ps = load(res, "pscc_swap")
    if ps:
        N["pscc_swap"] = ps
        md += ["", "## PSCC-Net swap", "```", json.dumps(ps, indent=1), "```"]

    cross = load(res, "cross")
    if cross:
        N["cross"] = dict(counts=cross["counts"], ci=cross["group_bootstrap_ci_seed0"],
                          models={k: {kk: ms(v, kk) for kk in ("accuracy", "balanced_accuracy", "auc", "sensitivity", "specificity")}
                                  for k, v in cross["summary"].items()})
        md += ["", "## Cross-dataset (5 seeds)", "```", json.dumps(cross["counts"], indent=1), "```"]
        md += [f"- {k}: acc {fmt(v['accuracy'])} bal {fmt(v['balanced_accuracy'])} sens {fmt(v['sensitivity'])} "
               f"spec {fmt(v['specificity'])} auc {fmt(v['auc'])}" for k, v in N["cross"]["models"].items()]

    # PSCC-Net (zero-shot) on the same folds, merged into the grid, ladder and cross-dataset numbers.
    pe = load(res, "pscc_eval")
    keys = ("accuracy", "balanced_accuracy", "auc", "sensitivity", "specificity", "auc|casia", "auc|columbia", "auc|imd2020",
            "bal|imd2020", "bal|casia")
    if pe:
        for c, v in pe["grid"].items():
            row = {k: ms(v["summary"], k, 1, 3) if k.startswith("auc") else ms(v["summary"], k) for k in keys}
            row["ci"] = v["ci"]
            N.setdefault("grid", {}).setdefault(c, {})["pscc_zero_shot"] = row
        N["pscc_ladder"] = {st: {k: ms(v["summary"], k) for k in keys} for st, v in pe["ladder"].items()}
        N["pscc_cross"] = pe["cross"]
    # Table III: cumulative ladder. L1 is the 'orig' grid condition (same folds and seeds).
    if lad and "grid" in N and "orig" in N["grid"]:
        table = {}
        for step in ("L0_random", "L1_source_disjoint", "L2_sd_jpeg_only", "L3_sd_jpeg_erased"):
            if step == "L1_source_disjoint":
                g0 = grid["orig"]["matched_summary"]
                rows = {m: {k: ms(st, k) for k in keys} for m, st in g0.items()}
            else:
                rows = N["ladder"].get(step, {}).get("models", {})
            if "pscc_ladder" in N and step in N["pscc_ladder"]:
                rows = dict(rows, pscc_zero_shot=N["pscc_ladder"][step])
            table[step] = rows
        N["table3"] = table
    (out / "numbers.json").write_text(json.dumps(N, indent=1))
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    if grid:
        fig_interventions(N, out)
    if "robustness" in N:
        fig_robustness(N, out)
    print("written", out)


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e9ecef", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.5)


def fig_interventions(N, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = [c for c in GRID_ORDER if c in N["grid"]]
    models = [m for m in ["hgb_markov", "hgb_metadata", "logreg_resnet_eladct", "logreg_resnet_rgb", "pscc_zero_shot"]
              if any(m in N["grid"][c] for c in conds)]
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.5), sharex=True)
    x = np.arange(len(conds))
    off = np.linspace(-0.27, 0.27, len(models))
    for ax, ds, title in zip(axes, ["casia", "imd2020"], ["CASIA 2.0", "IMD2020"]):
        for j, m in enumerate(models):
            xs, ys, es = [], [], []
            for i, c in enumerate(conds):
                v = N["grid"][c].get(m, {}).get(f"auc|{ds}")
                if v:
                    xs.append(i + off[j])
                    ys.append(v["mean"])
                    es.append(v["sd"])
            ax.errorbar(xs, ys, yerr=es, fmt=MARKERS[m], ms=3.2, color=COLORS[m], lw=0.8, capsize=1.2,
                        label=NAMES[m], mfc=COLORS[m] if m != "hgb_metadata" else "white")
        ax.axhline(0.5, color="#495057", lw=0.7, ls=":")
        for b in (2.5, 5.5):
            ax.axvline(b, color="#ced4da", lw=0.6)
        ax.set_ylim(0.35, 1.02)
        ax.set_ylabel(f"AUC within {title}", fontsize=7)
        _style(ax)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([GRID_LABEL[c] for c in conds], fontsize=6)
    axes[0].text(1, 1.04, "scale 1", ha="center", fontsize=6.5, color="#495057", transform=axes[0].get_xaxis_transform())
    axes[0].text(4, 1.04, "scale 0.75", ha="center", fontsize=6.5, color="#495057", transform=axes[0].get_xaxis_transform())
    axes[0].text(7, 1.04, "scale 0.5", ha="center", fontsize=6.5, color="#495057", transform=axes[0].get_xaxis_transform())
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, fontsize=5.8, frameon=False, bbox_to_anchor=(0.53, 1.0),
               handletextpad=0.3, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_interventions.{ext}", dpi=300)
    plt.close(fig)


def fig_robustness(N, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tests = ["orig", "jpeg95", "jpeg85", "jpeg75", "resize0.75", "resize0.5", "resize0.5+jpeg95", "resize0.5+jpeg75"]
    unseen = {"resize0.75", "resize0.5", "resize0.5+jpeg95"}
    fig, ax = plt.subplots(figsize=(3.45, 2.15))
    x = np.arange(len(tests))
    series = [("cnn_aligned", "standard", -0.1), ("hgb_markov", "standard", 0.0), ("hgb_markov", "augmented", 0.1)]
    for m, rg, dx in series:
        v = [N["robustness"].get(f"{rg}|{m}|{t}", {}).get("balanced_accuracy") for t in tests]
        if not any(v):
            continue
        ys = [a["mean"] if a else np.nan for a in v]
        es = [a["sd"] if a else 0 for a in v]
        ax.errorbar(x + dx, ys, yerr=es, color=COLORS[m], ls="--" if rg == "augmented" else "-", marker=MARKERS[m],
                    ms=3, lw=0.9, capsize=1.2, mfc="white" if rg == "augmented" else COLORS[m],
                    label=f"{NAMES[m]}, {'degradation-augmented' if rg == 'augmented' else 'standard'} training")
    ax.axhline(50, color="#495057", lw=0.7, ls=":")
    for i, t in enumerate(tests):
        if t in unseen:
            ax.axvspan(i - 0.45, i + 0.45, color="#eef0f2", zorder=0)
    ax.text(5, 96, "not in aug. pool", ha="center", fontsize=5.8, color="#495057")
    ax.set_xticks(x)
    ax.set_xticklabels([GRID_LABEL[t] for t in tests], fontsize=6)
    ax.set_ylabel("Balanced accuracy (%)", fontsize=7)
    ax.set_xlabel("Test-time processing (training on original files)", fontsize=6.5)
    ax.set_ylim(40, 100)
    _style(ax)
    ax.legend(fontsize=5.6, ncol=1, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.93), handletextpad=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_robustness.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
