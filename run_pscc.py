"""Zero-shot PSCC-Net scores (src/pscc_scores.py) for the conditions used in the revision.

    python run_pscc.py --data <datasets_forgery> --pscc <PSCC-Net-main> --python <torch-directml python> [--device dml]

Caches: results_rev/features/pscc_<condition>.npz (scores aligned with the benchmark rows, or with the swap subset
listed in results_rev/pscc_swap_subset.json).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_revision as rv  # noqa: E402

FULL = ["orig", "resize0.5+jpeg75", "resize0.5", "jpeg75"]
SWAP_N = {0: 1000, 1: 500}  # CASIA JPEG images per label in the swap subset


def swap_subset(rows):
    """Fixed random subset of CASIA JPEG files (seed 0) for the matched-content test of PSCC-Net."""
    path = rv.OUT / "pscc_swap_subset.json"
    if path.exists():
        return np.array(json.loads(path.read_text())["rows"])
    rng = np.random.default_rng(0)
    pick = []
    for lab, k in SWAP_N.items():
        cand = [i for i, r in enumerate(rows) if r["source"] == "casia" and r["container"] == "JPEG" and r["label"] == lab]
        pick += sorted(rng.choice(cand, size=k, replace=False).tolist())
    path.write_text(json.dumps(dict(rows=pick, names=[rows[i]["name"] for i in pick])))
    return np.array(pick)


def run(paths, spec, cache, args):
    if cache.exists():
        return
    jobs = cache.with_suffix(".jobs.json")
    jobs.write_text(json.dumps(dict(paths=paths, spec=spec)), encoding="utf-8")
    subprocess.run([args.python, str(HERE / "src" / "pscc_scores.py"), "--jobs", str(jobs), "--cache", str(cache),
                    "--pscc", str(args.pscc), "--device", args.device], check=True)
    jobs.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--pscc", type=Path, required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    rows, *_ = rv.load_benchmark(args.data)
    paths = [r["path"] for r in rows]
    rv.FEAT.mkdir(parents=True, exist_ok=True)
    order = [("full", "orig"), ("full", "resize0.5+jpeg75")] + [("swap", s) for s in rv.SWAP] + \
            [("full", "resize0.5"), ("full", "jpeg75")]
    sub = swap_subset(rows)
    for kind, spec in order:
        if kind == "full":
            run(paths, spec, rv.FEAT / f"pscc_{spec}.npz", args)
        else:
            run([paths[i] for i in sub], spec, rv.FEAT / f"pscc_swap_{spec}.npz", args)
        print(f"[pscc] done {kind} {spec}", flush=True)
    # The orig condition also scores the swap subset, so the swap test has its unmodified reference.


if __name__ == "__main__":
    main()
