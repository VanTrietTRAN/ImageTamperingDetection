"""Zero-shot image-level scores of PSCC-Net (Liu et al., IEEE TCSVT 2022), a purpose-built manipulation detector.

The released weights were trained on synthetic splicing/copy-move/removal data built from MS COCO, KCMI, VISION,
Dresden and USCISI-CMFD, never on CASIA 2.0, Columbia or IMD2020, so the detector is applied without any training on
the benchmark. Images larger than 512 x 512 pixels (only in IMD2020) do not fit in GPU memory; they are scored tile by tile and
the image score is the maximum tile score, so no resampling alters their traces. Only the HRNet backbone and the image-level detection head are run (the localisation head is not
needed); the score is the softmax probability of the 'forged' class, as in the official test.py.

    python src/pscc_scores.py --jobs jobs.json --cache out.npz --pscc <PSCC-Net-main folder> [--device auto|cpu|dml]
where jobs.json is {"paths": [...], "spec": "orig" | transform spec of markov_features.apply_transform}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from markov_features import apply_transform, read_bgr  # noqa: E402

TILE = 512  # images above TILE x TILE pixels are scored on non-overlapping TILE x TILE tiles (max over tiles), not resized


def load_pscc(root: Path, device):
    import torch
    if not hasattr(np, "int"):
        np.int = int  # the released HRNet code still uses the removed NumPy alias
    sys.path.insert(0, str(root))
    from models.detection_head import DetectionHead
    from models.seg_hrnet import get_seg_model
    from models.seg_hrnet_config import get_hrnet_cfg
    from utils.config import get_pscc_args

    fe = get_seg_model(get_hrnet_cfg())
    head = DetectionHead(get_pscc_args())
    for net, name in ((fe, "HRNet"), (head, "DetectionHead")):
        state = torch.load(root / "checkpoint" / f"{name}_checkpoint" / f"{name}.pth", map_location="cpu")
        net.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
        net.eval().to(device)
    return fe, head


def get_device(name):
    import torch
    if name in ("auto", "dml"):
        try:
            import torch_directml
            return torch_directml.device()
        except ImportError:
            if name == "dml":
                raise
    return torch.device("cpu")


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--pscc", type=Path, required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "dml"])
    args = ap.parse_args()
    job = json.loads(args.jobs.read_text(encoding="utf-8"))
    paths, spec = job["paths"], job["spec"]
    dev = get_device(args.device)
    fe, head = load_pscc(args.pscc, dev)
    n = len(paths)
    score = np.full(n, np.nan, dtype=np.float32)
    score_mean = np.full(n, np.nan, dtype=np.float32)
    n_tiles = np.zeros(n, dtype=np.int16)
    t0 = time.time()

    def score_one(i, device):
        x = apply_transform(read_bgr(paths[i]), spec)
        rgb = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        h, w = rgb.shape[:2]
        if h * w <= TILE * TILE:
            tiles = [rgb]
        else:
            ys = sorted({min(t, max(h - TILE, 0)) for t in range(0, h, TILE)})
            xs = sorted({min(t, max(w - TILE, 0)) for t in range(0, w, TILE)})
            tiles = [rgb[a:a + TILE, b:b + TILE] for a in ys for b in xs]
        s_tiles = []
        for tile in tiles:
            t = torch.from_numpy(np.ascontiguousarray(tile.transpose(2, 0, 1)))[None].to(device)
            s_tiles.append(float(torch.softmax(head(fe(t)), dim=1)[0, 1].cpu()))
        score[i], score_mean[i], n_tiles[i] = max(s_tiles), float(np.mean(s_tiles)), len(tiles)

    with torch.no_grad():
        for i in range(n):
            try:
                score_one(i, dev)
            except Exception as e:  # noqa: BLE001
                print(f"  failed on {dev}: {paths[i]}: {str(e)[:80]}", flush=True)
            if (i + 1) % 500 == 0 or i + 1 == n:
                print(f"  [pscc {spec}] {i + 1}/{n} images, {time.time() - t0:.0f} s", flush=True)
        retry = np.where(np.isnan(score))[0]
        if len(retry):  # GPU out of memory: same weights on the CPU
            fe.to("cpu")
            head.to("cpu")
            for i in retry:
                try:
                    score_one(i, torch.device("cpu"))
                except Exception as e:  # noqa: BLE001
                    print(f"  failed on cpu: {paths[i]}: {str(e)[:80]}", flush=True)
            print(f"  [pscc {spec}] retried {len(retry)} on cpu, {int(np.isnan(score).sum())} still failing", flush=True)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.cache, score=score, score_mean=score_mean, ok=~np.isnan(score), n_tiles=n_tiles,
                        info=np.array(json.dumps(dict(device=str(dev), seconds=time.time() - t0, n=n))))


if __name__ == "__main__":
    main()
