import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scipy import linalg

# Script may be run with cwd=SwinGAN/ or Restormer/; repo root must be on path for `models.*`.
_VSLITE_ROOT = Path(__file__).resolve().parents[1]
_THIS_DIR = Path(__file__).resolve().parent

# Ensure repo root is importable as a top-level package root (for `models.*`).
if str(_VSLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_VSLITE_ROOT))

# When this file is executed as a script (`python util/wsi_tile_metrics.py`),
# sys.path[0] is the util/ directory which contains `html.py` and can shadow
# stdlib `html` (breaking torchvision's `from html.parser import HTMLParser`).
try:
    # Drop util/ from sys.path entirely, then re-add repo root at front.
    _cleaned = []
    for p in sys.path:
        try:
            if p and Path(p).resolve() == _THIS_DIR:
                continue
        except Exception:
            pass
        _cleaned.append(p)
    sys.path[:] = _cleaned
    if not sys.path or sys.path[0] != str(_VSLITE_ROOT):
        sys.path.insert(0, str(_VSLITE_ROOT))
except Exception:
    pass

Image.MAX_IMAGE_PIXELS = None


_PATCH_RE = re.compile(r"^(?P<slide>.+)_y(?P<y>\d+)_x(?P<x>\d+)\.npy$")


@dataclass(frozen=True)
class Running:
    n: int
    psnr_sum: float
    psnr_sq_sum: float
    ssim_sum: float
    ssim_sq_sum: float


@dataclass
class _ActStats:
    """Online mean/cov accumulator for FID without storing activations."""

    n: int = 0
    sum: np.ndarray | None = None  # (d,)
    sum_outer: np.ndarray | None = None  # (d, d)

    def update(self, acts: np.ndarray) -> None:
        # acts: (b, d) float64
        if acts.ndim != 2 or acts.shape[0] == 0:
            return
        acts = acts.astype(np.float64, copy=False)
        if self.sum is None:
            d = int(acts.shape[1])
            self.sum = np.zeros((d,), dtype=np.float64)
            self.sum_outer = np.zeros((d, d), dtype=np.float64)
        self.n += int(acts.shape[0])
        self.sum += acts.sum(axis=0)
        # (d,d) += X^T X
        self.sum_outer += acts.T @ acts

    def mean_cov(self) -> tuple[np.ndarray, np.ndarray]:
        if self.sum is None or self.sum_outer is None or self.n < 2:
            raise RuntimeError(f"Not enough activations to compute covariance (n={self.n}).")
        mu = self.sum / float(self.n)
        # Sample covariance to match np.cov(..., bias=False)
        cov = (self.sum_outer - float(self.n) * np.outer(mu, mu)) / float(self.n - 1)
        return mu, cov


def _psnr_01(gt: np.ndarray, pr: np.ndarray) -> float:
    mse = float(np.mean((gt - pr) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def _ssim_01(gt: np.ndarray, pr: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        l1 = float(np.mean(np.abs(gt - pr)))
        return float(max(0.0, 1.0 - l1))
    return float(structural_similarity(gt, pr, channel_axis=2, data_range=1.0))


def _list_he_patches(he_dir: str) -> list[str]:
    out = []
    for fn in os.listdir(he_dir):
        if fn.endswith(".npy"):
            out.append(os.path.join(he_dir, fn))
    out.sort()
    return out


def _wsi_pred_path(pred_wsi_dir: str, slide: str, suffix: str) -> str:
    return os.path.join(pred_wsi_dir, f"{slide}_{suffix}.png")


def _crop_wsi(img: Image.Image, x: int, y: int, size: int) -> np.ndarray:
    patch = img.crop((x, y, x + size, y + size)).convert("RGB")
    arr = np.asarray(patch).astype(np.float32) / 255.0
    return arr


def _mean_cov(acts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return acts.mean(axis=0), np.cov(acts, rowvar=False)


def _frechet(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def main():
    ap = argparse.ArgumentParser(description="Compute tile-level PSNR/SSIM/FID by cropping WSI predictions.")
    ap.add_argument("--pred_wsi_dir", required=True, type=str, help="Directory with <slide>_<suffix>.png predictions")
    ap.add_argument("--he_dir", required=True, type=str, help="Directory with HE patch .npy (uint8 HWC)")
    ap.add_argument("--suffix", type=str, default="13ch", help="WSI output suffix used by inference scripts")
    ap.add_argument("--patch_size", type=int, default=1024)
    ap.add_argument("--max_items", type=int, default=2000)
    ap.add_argument("--fid_batch_size", type=int, default=32)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    he_paths = _list_he_patches(args.he_dir)
    if not he_paths:
        raise FileNotFoundError(f"No .npy patches found in {args.he_dir}")

    max_items = min(len(he_paths), int(args.max_items))
    he_paths = he_paths[:max_items]

    # Cache open WSI images per slide
    wsi_cache: dict[str, Image.Image] = {}

    psnrs: list[float] = []
    ssims: list[float] = []

    # FID: stream activations (do NOT store per-patch tensors in RAM).
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    # Use the same ConvNeXt-Tiny TCGA/H&E feature space as patch eval FID.
    from util.i2i_metrics import _ConvNeXtTinyFID

    fid_net = _ConvNeXtTinyFID().to(device)
    fid_net.eval()
    fid_bs = int(args.fid_batch_size)
    _real_buf: list[torch.Tensor] = []
    _pred_buf: list[torch.Tensor] = []
    real_stats = _ActStats()
    pred_stats = _ActStats()

    def _flush_fid_buffers() -> None:
        if not _real_buf:
            return
        with torch.no_grad():
            x_real = torch.stack(_real_buf, dim=0).to(device=device, dtype=torch.float32)
            x_pred = torch.stack(_pred_buf, dim=0).to(device=device, dtype=torch.float32)
            a_real = fid_net(x_real).detach().cpu().numpy().reshape(x_real.size(0), -1)
            a_pred = fid_net(x_pred).detach().cpu().numpy().reshape(x_pred.size(0), -1)
        real_stats.update(a_real)
        pred_stats.update(a_pred)
        _real_buf.clear()
        _pred_buf.clear()

    for idx, he_path in enumerate(he_paths, start=1):
        m = _PATCH_RE.match(os.path.basename(he_path))
        if not m:
            continue
        slide = m.group("slide")
        y = int(m.group("y"))
        x = int(m.group("x"))

        pred_path = _wsi_pred_path(args.pred_wsi_dir, slide, args.suffix)
        if slide not in wsi_cache:
            if not os.path.isfile(pred_path):
                continue
            wsi_cache[slide] = Image.open(pred_path)

        img = wsi_cache[slide]
        gt = np.load(he_path).astype(np.float32) / 255.0  # HWC 0..1
        pr = _crop_wsi(img, x=x, y=y, size=int(args.patch_size))  # HWC 0..1

        if gt.shape != pr.shape:
            # Skip if crop runs off bounds or mismatch
            continue

        psnrs.append(_psnr_01(gt, pr))
        ssims.append(_ssim_01(gt, pr))

        # CHW tensors for inception (kept briefly; flushed in batches)
        _real_buf.append(torch.from_numpy(gt).permute(2, 0, 1))
        _pred_buf.append(torch.from_numpy(pr).permute(2, 0, 1))
        if len(_real_buf) >= fid_bs:
            _flush_fid_buffers()

        if idx % 200 == 0:
            ps = float(np.mean(psnrs)) if psnrs else float("nan")
            ss = float(np.mean(ssims)) if ssims else float("nan")
            print(f"Processed {idx}/{max_items} patches | PSNR: {ps:.3f} SSIM: {ss:.3f}")

    n = len(psnrs)
    if n < 2:
        raise RuntimeError(f"Not enough matched patches to compute metrics (n={n}).")

    # FID (final flush + compute from online stats)
    _flush_fid_buffers()
    mu1, s1 = real_stats.mean_cov()
    mu2, s2 = pred_stats.mean_cov()
    fid = _frechet(mu1, s1, mu2, s2)

    psnr_mean = float(np.mean(psnrs))
    psnr_std = float(np.std(psnrs))
    ssim_mean = float(np.mean(ssims))
    ssim_std = float(np.std(ssims))

    print(
        "MODE: EVAL | "
        + " ".join(
            [
                f"PSNR_test: {psnr_mean:.3f}",
                f"SSIM_test: {ssim_mean:.3f}",
                f"FID_test: {fid:.3f}",
                f"N: {n}",
            ]
        )
    )


if __name__ == "__main__":
    main()

