"""Assemble the public release of code, split definitions and per-image file signatures (reviewer request R5).

    python make_release.py            ->  ../release_TSE2026/  and  ../release_TSE2026.zip

Contents: all experiment code, the audited pipeline's three original notebooks, the benchmark index with
dataset-relative paths, split definitions for every repetition (random split and source-disjoint folds), JPEG encoding
signatures, and README.md with the dataset sources and the commands that regenerate every table and figure.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DST = HERE.parent / "release_TSE2026"
NOTEBOOKS = HERE.parent / "Nhóm 8 (bản gốc)" / "Cuối kỳ" / "Code"
CODE = ["run_experiments.py", "run_revision.py", "revision_stages.py", "run_pscc.py", "audit_metadata_confounds.py",
        "p10_fold_check.py", "make_report.py", "make_revision_report.py", "make_release.py", "requirements.txt"]

README = """# Shortcut audit of a merged CASIA 2.0 + Columbia + IMD2020 tampering-detection benchmark

Code and split definitions for the TSE 2026 paper "Is 94% Tampering-Detection Accuracy Real? Auditing Shortcuts in a
Merged Image-Forensics Benchmark for Emergency Operations Centers".

## Audited pipeline
`audited_pipeline/` holds the three notebooks of the audited pipeline, an unpublished university course project that
merges the Kaggle copies of the three datasets: Markov-DCT features + CNN (`cnn-markov-dct.ipynb`, reported 94%),
ResNet18 on RGB (`cnn-tampering-baseline-rgb.ipynb`, 72%), ResNet18 on ELA + global DCT (`ela-dct.ipynb`, 89%).

## Data (not redistributed)
| Dataset | Kaggle copy used by the pipeline | Folders |
|---|---|---|
| CASIA 2.0 | `divg07/casia-20-image-tampering-detection-dataset` | `casia2/CASIA2/{Au,Tp}` |
| Columbia uncompressed | `shriya0/columbia` | `columbia/.../4cam_auth`, `4cam_splc` |
| IMD2020 (real-life part) | `saikotahmed/imd2020` | `imd2020/IMD2020/{Authentic,img}` |

Files with extensions .jpg/.jpeg/.png/.tif/.tiff are used, as in the pipeline (54 CASIA BMP files are excluded):
15,347 images, 8,034 authentic, 7,313 tampered.

## Split definitions
* `data/index.csv`: every image with dataset, label, dataset-relative path, real container, size, file bytes,
  estimated JPEG quality and source-group id (connected components of shared source identifiers; 9,267 groups).
* `data/splits.csv`: for seeds 0-4, membership of the random 80/20 stratified test split, the source-disjoint fold
  (StratifiedGroupKFold, 5 folds, shuffle, random_state = seed) over all images, and the fold over JPEG files only.
* `data/jpeg_signatures.csv`: exact quantisation-table hash and chroma subsampling of every JPEG file.
* `data/casia_signatures.json`: the two CASIA quantisation tables used by the matched-content encoding test.

## Results
`results/numbers.json` holds every number quoted in the paper; `results/summary.md` is a readable digest.
`results/revision/*.json` are the raw outputs of `run_revision.py` (per-fold records for every repetition);
`results/original_protocols/*.json` those of `run_experiments.py`.

## Reproduce
```
python run_experiments.py --data <datasets> --out results                 # original protocol ladder (p1-p10)
python run_revision.py --data <datasets> --stages signatures features deep \\
       --deep-python <python with torch-directml> --deep-device dml       # interventions, embeddings
python run_pscc.py --data <datasets> --pscc <PSCC-Net-main> --python <python with torch>   # zero-shot PSCC-Net
python run_revision.py --data <datasets> --stages splits grid ladder cross pscc_swap
python make_revision_report.py                                             # tables and figures of the paper
```
PSCC-Net code and weights: https://github.com/proteus1991/PSCC-Net (not redistributed).
CPU: AMD Ryzen 5 5600H, 16 GB RAM; embeddings and PSCC-Net on an AMD Radeon RX 5500M through DirectML.
"""


def main():
    if DST.exists():
        shutil.rmtree(DST)
    (DST / "src").mkdir(parents=True)
    (DST / "data").mkdir()
    (DST / "audited_pipeline").mkdir()
    for f in CODE:
        if (HERE / f).exists():
            shutil.copy(HERE / f, DST / f)
    rev = HERE / "results_rev"
    for f in (HERE / "src").glob("*.py"):
        shutil.copy(f, DST / "src" / f.name)
    for f in NOTEBOOKS.glob("*.ipynb"):  # code only: outputs (result plots, demo photos) are stripped
        nb = json.loads(f.read_text(encoding="utf-8"))
        for c in nb.get("cells", []):
            if c.get("cell_type") == "code":
                c["outputs"], c["execution_count"] = [], None
        (DST / "audited_pipeline" / f.name).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    (DST / "results" / "original_protocols").mkdir(parents=True)
    (DST / "results" / "revision").mkdir(parents=True)
    for f in (HERE / "results").glob("*.json"):
        shutil.copy(f, DST / "results" / "original_protocols" / f.name)
    for f in rev.glob("*.json"):
        shutil.copy(f, DST / "results" / "revision" / f.name)
    sp = DST / "results" / "revision" / "splits.json"  # drop the absolute local path
    if sp.exists():
        d = json.loads(sp.read_text())
        d["file"] = "data/splits.csv"
        sp.write_text(json.dumps(d, indent=1))
    for f in ("numbers.json", "summary.md"):
        shutil.copy(rev / "report" / f, DST / "results" / f)
    for f in ("splits.csv", "jpeg_signatures.csv", "casia_signatures.json", "pscc_swap_subset.json"):
        if (rev / f).exists():
            shutil.copy(rev / f, DST / "data" / f)
    with open(HERE / "results" / "index.csv", encoding="utf-8") as fi, \
            open(DST / "data" / "index.csv", "w", newline="", encoding="utf-8") as fo:
        rd = csv.DictReader(fi)
        w = csv.DictWriter(fo, fieldnames=rd.fieldnames)
        w.writeheader()
        for r in rd:
            p = r["path"].replace("\\", "/")
            r["path"] = p.split("datasets_forgery/")[-1]
            w.writerow(r)
    (DST / "README.md").write_text(README, encoding="utf-8")
    shutil.make_archive(str(DST), "zip", DST)
    print("written", DST, "and", DST.with_suffix(".zip"))


if __name__ == "__main__":
    main()
