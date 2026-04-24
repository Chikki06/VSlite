import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset

from models.inception import InceptionV3


@dataclass(frozen=True)
class MetricsResult:
    n: int
    psnr_mean: float | None
    psnr_std: float | None
    ssim_mean: float | None
    ssim_std: float | None
    fid: float | None


@dataclass
class _ActStats:
    """Online mean/cov accumulator for FID without storing activations."""

    n: int = 0
    sum: np.ndarray | None = None  # (d,)
    sum_outer: np.ndarray | None = None  # (d, d)

    def update(self, acts: np.ndarray) -> None:
        if acts.ndim != 2 or acts.shape[0] == 0:
            return
        acts = acts.astype(np.float64, copy=False)
        if self.sum is None:
            d = int(acts.shape[1])
            self.sum = np.zeros((d,), dtype=np.float64)
            self.sum_outer = np.zeros((d, d), dtype=np.float64)
        self.n += int(acts.shape[0])
        self.sum += acts.sum(axis=0)
        self.sum_outer += acts.T @ acts

    def mean_cov(self) -> tuple[np.ndarray, np.ndarray]:
        if self.sum is None or self.sum_outer is None or self.n < 2:
            raise RuntimeError(f"Not enough activations to compute covariance (n={self.n}).")
        mu = self.sum / float(self.n)
        cov = (self.sum_outer - float(self.n) * np.outer(mu, mu)) / float(self.n - 1)
        return mu, cov


def _list_images(d: str) -> list[str]:
    if not d or not os.path.isdir(d):
        return []
    out = []
    for fn in os.listdir(d):
        if fn.lower().endswith((".png", ".jpg", ".jpeg")):
            out.append(os.path.join(d, fn))
    out.sort()
    return out


def _basename_no_ext(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def _match_pairs(target_dir: str, pred_dir: str) -> list[tuple[str, str, str]]:
    """Return list of (key, target_path, pred_path) pairs by basename."""
    t_files = _list_images(target_dir)
    p_files = _list_images(pred_dir)
    t_map = {_basename_no_ext(p): p for p in t_files}
    p_map = {_basename_no_ext(p): p for p in p_files}
    keys = sorted(set(t_map.keys()) & set(p_map.keys()))
    return [(k, t_map[k], p_map[k]) for k in keys]


def _read_rgb_01(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def _psnr_01(gt: np.ndarray, pr: np.ndarray) -> float:
    mse = float(np.mean((gt - pr) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def _ssim_01(gt: np.ndarray, pr: np.ndarray) -> float:
    # Use skimage SSIM if available; otherwise fall back to a stub.
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        l1 = float(np.mean(np.abs(gt - pr)))
        return float(max(0.0, 1.0 - l1))
    return float(structural_similarity(gt, pr, channel_axis=2, data_range=1.0))


class _ImageTensorDataset(Dataset):
    def __init__(self, paths: list[str]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        arr = _read_rgb_01(self.paths[idx])
        # HWC -> CHW
        t = torch.from_numpy(arr).permute(2, 0, 1)
        return t


def _to_rgb01_from_minus1_1(t: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor in [-1,1] (or [0,1]) to float32 RGB in [0,1], CHW.
    Accepts CHW or BCHW; returns BCHW.
    """
    if t.ndim == 3:
        t = t.unsqueeze(0)
    if t.ndim != 4:
        raise ValueError(f"Expected CHW or BCHW tensor, got {tuple(t.shape)}")
    t = t.detach().float()
    # If likely in [-1,1], map to [0,1]
    if float(t.min().item()) < 0.0:
        t = (t + 1.0) * 0.5
    return t.clamp(0.0, 1.0)


@torch.no_grad()
def compute_metrics_streaming(
    *,
    pairs: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: str = "cuda",
    fid_max_items: int | None = None,
    fid_batch_size: int = 16,
    compute_fid: bool = True,
    compute_ssim: bool = True,
) -> MetricsResult:
    """
    Streaming metrics for large patch sets without writing PNGs for every patch.

    pairs yields (target, pred) tensors as CHW in either [-1,1] or [0,1].
    """
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

    psnrs: list[float] = []
    ssims: list[float] = []

    # FID accumulators
    fid_val: float | None = None
    fid_n = 0
    stats_t = _ActStats()
    stats_p = _ActStats()
    model = None
    buf_t: list[torch.Tensor] = []
    buf_p: list[torch.Tensor] = []

    if compute_fid:
        model = InceptionV3(inception_weights="fid_inception").to(dev)
        model.eval()

    n = 0
    for gt_t, pr_t in pairs:
        if fid_max_items is not None and n >= int(fid_max_items):
            break

        gt01 = _to_rgb01_from_minus1_1(gt_t).cpu()
        pr01 = _to_rgb01_from_minus1_1(pr_t).cpu()

        # PSNR/SSIM on CPU numpy (keeps behavior consistent with compute_metrics_from_dirs)
        gt = gt01[0].permute(1, 2, 0).numpy()
        pr = pr01[0].permute(1, 2, 0).numpy()
        psnrs.append(_psnr_01(gt, pr))
        if compute_ssim:
            ssims.append(_ssim_01(gt, pr))

        if compute_fid and model is not None:
            buf_t.append(gt01[0])
            buf_p.append(pr01[0])
            if len(buf_t) >= int(fid_batch_size):
                bt = torch.stack(buf_t, dim=0).to(dev, dtype=torch.float32)
                bp = torch.stack(buf_p, dim=0).to(dev, dtype=torch.float32)
                at = model(bt)[0].detach().cpu().numpy().reshape(bt.size(0), -1)
                ap = model(bp)[0].detach().cpu().numpy().reshape(bp.size(0), -1)
                stats_t.update(at)
                stats_p.update(ap)
                fid_n += int(bt.size(0))
                buf_t.clear()
                buf_p.clear()

        n += 1

    # Flush remainder
    if compute_fid and model is not None and buf_t:
        bt = torch.stack(buf_t, dim=0).to(dev, dtype=torch.float32)
        bp = torch.stack(buf_p, dim=0).to(dev, dtype=torch.float32)
        at = model(bt)[0].detach().cpu().numpy().reshape(bt.size(0), -1)
        ap = model(bp)[0].detach().cpu().numpy().reshape(bp.size(0), -1)
        stats_t.update(at)
        stats_p.update(ap)
        fid_n += int(bt.size(0))

    if compute_fid and fid_n >= 2:
        try:
            mu1, s1 = stats_t.mean_cov()
            mu2, s2 = stats_p.mean_cov()
            fid_val = _frechet(mu1, s1, mu2, s2)
        except Exception:
            fid_val = None

    return MetricsResult(
        n=int(n),
        psnr_mean=float(np.mean(psnrs)) if psnrs else None,
        psnr_std=float(np.std(psnrs)) if psnrs else None,
        ssim_mean=float(np.mean(ssims)) if ssims else None,
        ssim_std=float(np.std(ssims)) if ssims else None,
        fid=fid_val,
    )


@torch.no_grad()
def _accumulate_inception_stats(
    *,
    paths: list[str],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not paths:
        raise RuntimeError("No image paths provided for FID.")
    ds = _ImageTensorDataset(paths)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = InceptionV3(inception_weights="fid_inception").to(device)
    model.eval()

    stats = _ActStats()
    for x in dl:
        x = x.to(device=device, dtype=torch.float32)
        pred = model(x)[0]
        acts = pred.detach().cpu().numpy().reshape(pred.size(0), -1)
        stats.update(acts)
    return stats.mean_cov()


def _frechet(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape
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


def compute_metrics_from_dirs(
    *,
    target_dir: str,
    pred_dir: str,
    device: str = "cuda",
    fid_max_items: int | None = None,
    fid_batch_size: int = 16,
) -> MetricsResult:
    pairs = _match_pairs(target_dir, pred_dir)
    if not pairs:
        return MetricsResult(n=0, psnr_mean=None, psnr_std=None, ssim_mean=None, ssim_std=None, fid=None)

    psnrs: list[float] = []
    ssims: list[float] = []
    for _k, t_path, p_path in pairs:
        gt = _read_rgb_01(t_path)
        pr = _read_rgb_01(p_path)
        psnrs.append(_psnr_01(gt, pr))
        ssims.append(_ssim_01(gt, pr))

    # FID uses distributions; it does not require per-image alignment, but we keep the same paired subset.
    fid_val: float | None = None
    try:
        n_fid = len(pairs) if fid_max_items is None else min(len(pairs), int(fid_max_items))
        if n_fid >= 2:
            dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
            t_paths = [tp for _k, tp, _pp in pairs[:n_fid]]
            p_paths = [pp for _k, _tp, pp in pairs[:n_fid]]
            mu1, s1 = _accumulate_inception_stats(paths=t_paths, device=dev, batch_size=fid_batch_size)
            mu2, s2 = _accumulate_inception_stats(paths=p_paths, device=dev, batch_size=fid_batch_size)
            fid_val = _frechet(mu1, s1, mu2, s2)
    except Exception:
        fid_val = None

    return MetricsResult(
        n=len(pairs),
        psnr_mean=float(np.mean(psnrs)),
        psnr_std=float(np.std(psnrs)),
        ssim_mean=float(np.mean(ssims)),
        ssim_std=float(np.std(ssims)),
        fid=fid_val,
    )

