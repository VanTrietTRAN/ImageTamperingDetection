# Shortcut audit of a merged CASIA 2.0 + Columbia + IMD2020 tampering-detection benchmark

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
python run_revision.py --data <datasets> --stages signatures features deep \
       --deep-python <python with torch-directml> --deep-device dml       # interventions, embeddings
python run_pscc.py --data <datasets> --pscc <PSCC-Net-main> --python <python with torch>   # zero-shot PSCC-Net
python run_revision.py --data <datasets> --stages splits grid ladder cross pscc_swap
python make_revision_report.py                                             # tables and figures of the paper
```
PSCC-Net code and weights: https://github.com/proteus1991/PSCC-Net (not redistributed).
CPU: AMD Ryzen 5 5600H, 16 GB RAM; embeddings and PSCC-Net on an AMD Radeon RX 5500M through DirectML.

## License
Code and the derived files in `data/` and `results/` are released under the MIT License (see `LICENSE`).
The notebooks in `audited_pipeline/`, the dataset images, and PSCC-Net are not covered; see `LICENSE` for details.
