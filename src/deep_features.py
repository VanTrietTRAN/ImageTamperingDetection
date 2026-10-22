"""Frozen ImageNet ResNet18 embeddings of the group's two ResNet18 inputs (no fine-tuning).

The group trained ResNet18 from scratch on (a) RGB resized to 224x224 and (b) a 4-channel ELA(3) + global-DCT(1)
stack, which needs a GPU for every protocol and seed. Here both inputs are built as in the evaluation transforms of
`cnn-tampering-baseline-rgb.ipynb` and `ela-dct.ipynb`, passed once through an ImageNet-pretrained ResNet18 with
frozen weights, and the 512-d global-average-pooled activations are kept; run_experiments.py trains a logistic
regression on top. ImageNet weights expect three channels, so the ELA map and the DCT map (replicated to three
channels) are embedded separately and concatenated (1024-d). All inputs use ImageNet normalisation.

Runs on the CPU or on any DirectX 12 GPU through torch-directml:
    python src/deep_features.py --jobs jobs.json --cache out.npz [--device auto|cpu|dml] [--batch 128]
where jobs.json is {"paths": [...], "spec": "orig" | [one transform per path]} (transforms as in markov_features).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from scipy.fftpack import dct

sys.path.insert(0, str(Path(__file__).resolve().parent))
from markov_features import apply_transform, read_bgr  # noqa: E402

IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
KINDS = {"rgb": 512, "eladct": 1024}
CHUNK = 1024  # bounds the queue of prepared images when the workers outpace the GPU


def ela_map(rgb: np.ndarray, quality: int = 90) -> np.ndarray:
    """`convert_to_ela_image` of ela-dct.ipynb, starting from decoded pixels instead of a file path."""
    image = Image.fromarray(rgb)
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    ela = ImageChops.difference(image, Image.open(buf).copy())
    max_diff = max(ex[1] for ex in ela.getextrema()) or 1
    return np.array(ImageEnhance.Brightness(ela).enhance(255.0 / max_diff))


def dct_map(rgb: np.ndarray) -> np.ndarray:
    """`compute_dct_channel` of ela-dct.ipynb: magnitude of the global 2-D DCT of the grey image, min-max to uint8."""
    gray = np.float32(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    d = np.abs(dct(dct(gray.T, norm="ortho").T, norm="ortho"))
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    return (d * 255).astype(np.uint8)


def _init_worker():
    cv2.setNumThreads(1)


def _prepare(job):
    path, spec = job
    try:
        bgr = read_bgr(path)
        if bgr is None:
            return None
        rgb = cv2.cvtColor(apply_transform(bgr, spec), cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        # RGB notebook: transforms.Resize((224, 224)) on a PIL image (bilinear).
        rgb_in = np.asarray(Image.fromarray(rgb).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
        # ELA-DCT notebook: both maps computed at full resolution, then cv2.resize to 224x224.
        ela_in = cv2.resize(ela_map(rgb), (IMG_SIZE, IMG_SIZE)).astype(np.uint8)
        dct_in = cv2.resize(dct_map(rgb), (IMG_SIZE, IMG_SIZE)).astype(np.uint8)
        return rgb_in, ela_in, np.repeat(dct_in[:, :, None], 3, axis=2), time.perf_counter() - t0
    except Exception:
        return None


def get_device(name: str = "auto"):
    import torch
    if name in ("auto", "dml"):
        try:
            import torch_directml
            return torch_directml.device(), torch_directml.device_name(torch_directml.default_device()).rstrip("\x00")
        except ImportError:
            if name == "dml":
                raise
    return torch.device("cpu"), platform.processor()


def frozen_resnet18(device):
    import torch
    import torchvision
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    for p in net.parameters():
        p.requires_grad_(False)
    return net.eval().to(device)


def embed(net, images_u8: list[np.ndarray], device) -> np.ndarray:
    import torch
    x = (np.stack(images_u8).astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    with torch.no_grad():
        return net(torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).to(device)).cpu().numpy()


def extract(paths, spec: str | list[str] = "orig", cache: Path | None = None, device: str = "auto",
            batch: int = 128, workers: int | None = None) -> dict:
    """Embeddings for every path: {'rgb': (n, 512), 'eladct': (n, 1024), 'ok', 'sec' (input building), 'info'}."""
    if cache is not None and Path(cache).exists():
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    n = len(paths)
    specs = [spec] * n if isinstance(spec, str) else list(spec)
    if len(specs) != n:
        raise ValueError("one transform per image is required")
    label = spec if isinstance(spec, str) else "per-image transforms"
    out = {k: np.zeros((n, dim), dtype=np.float32) for k, dim in KINDS.items()}
    ok = np.zeros(n, dtype=bool)
    sec = np.full(n, np.nan, dtype=np.float32)
    dev, dev_name = get_device(device)
    net = frozen_resnet18(dev)
    idx, rgb_b, ela_b, dct_b = [], [], [], []
    model_seconds = 0.0

    def flush():
        nonlocal model_seconds
        if not idx:
            return
        t0 = time.perf_counter()
        out["rgb"][idx] = embed(net, rgb_b, dev)
        out["eladct"][idx] = np.concatenate([embed(net, ela_b, dev), embed(net, dct_b, dev)], axis=1)
        model_seconds += time.perf_counter() - t0
        ok[idx] = True
        for buf in (idx, rgb_b, ela_b, dct_b):
            buf.clear()

    jobs = [(str(p), s) for p, s in zip(paths, specs)]
    t_start = time.time()
    workers = workers or int(os.environ.get("DEEP_WORKERS", 0)) or max(1, min(6, (os.cpu_count() or 2) - 1))
    with Pool(workers, initializer=_init_worker) as pool:
        for start in range(0, n, CHUNK):
            for i, r in enumerate(pool.imap(_prepare, jobs[start:start + CHUNK], chunksize=4), start=start):
                if r is not None:
                    idx.append(i)
                    rgb_b.append(r[0])
                    ela_b.append(r[1])
                    dct_b.append(r[2])
                    sec[i] = r[3]
                if len(idx) == batch:
                    flush()
            flush()
            print(f"  [deep {label}] {min(start + CHUNK, n)}/{n} images, {time.time() - t_start:.0f} s "
                  f"(ResNet18 {model_seconds:.0f} s on {dev_name})", flush=True)
    info = dict(device=dev_name, n_images=n, n_failed=int((~ok).sum()), batch=batch, wall_seconds=time.time() - t_start,
                model_seconds=model_seconds, forward_passes_per_image=3)
    res = dict(out, ok=ok, sec=sec, info=np.array(json.dumps(info)))
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "dml"])
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    job = json.loads(args.jobs.read_text(encoding="utf-8"))
    extract(job["paths"], job["spec"], cache=args.cache, device=args.device, batch=args.batch)


if __name__ == "__main__":
    main()
