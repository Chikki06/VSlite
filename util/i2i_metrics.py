import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset

import torch.nn as nn
import torch.nn.functional as F


def _default_convnext_fid_ckpt() -> str:
    # Canonical HE-trained ConvNeXt feature checkpoint used in this repo (see diffusion configs).
    # NOTE: the filename is not ".pt" in this workspace; it's the raw checkpoint name.
    return "/projects/illinois/eng/bioe/rxb/akshat/VSlite/tcga_tl_checkpoints/convnext_t_dp_spv_supcon_1e-5"


class _ConvNeXtTinyFID(nn.Module):
    """
    ConvNeXt-Tiny feature extractor for Frechet distance on H&E realism.

    - Uses timm convnext_tiny backbone.
    - Loads TCGA/H&E feature checkpoint when available (recommended).
    - Produces a (B, 768) feature vector via global average pooling.
    """

    def __init__(self, *, ckpt_path: str | None = None):
        super().__init__()
        try:
            import timm  # local import: timm is an optional dependency in some envs
        except Exception as e:
            raise RuntimeError(
                "ConvNeXt FID requires `timm` to be installed in the environment."
            ) from e

        self.backbone = timm.create_model("timm/convnext_tiny.in12k_ft_in1k")
        # Ensure classifier head isn't used (we only need features).
        if hasattr(self.backbone, "head") and hasattr(self.backbone.head, "fc"):
            self.backbone.head.fc = nn.Identity()

        ckpt = (ckpt_path or "").strip() if ckpt_path is not None else ""
        if not ckpt:
            ckpt = os.environ.get("VSLITE_CONVNEXT_FID_CKPT", "").strip() or _default_convnext_fid_ckpt()

        # Allow passing a directory: pick the expected file if present, else any single file in it.
        if os.path.isdir(ckpt):
            expected = os.path.join(ckpt, os.path.basename(_default_convnext_fid_ckpt()))
            if os.path.isfile(expected):
                ckpt = expected
            else:
                files = [os.path.join(ckpt, f) for f in os.listdir(ckpt) if os.path.isfile(os.path.join(ckpt, f))]
                files.sort()
                if len(files) == 1:
                    ckpt = files[0]

        if not os.path.isfile(ckpt):
            raise FileNotFoundError(
                f"ConvNeXt FID checkpoint not found: {ckpt}. "
                "Set env var VSLITE_CONVNEXT_FID_CKPT to a valid .pt/.pth file."
            )

        state = torch.load(ckpt, map_location="cpu")
        # Accept common wrappers.
        if isinstance(state, dict):
            for k in ("state_dict", "model", "net", "weights"):
                if k in state and isinstance(state[k], dict):
                    state = state[k]
                    break
        # Strip prefixes if present (e.g., 'module.').
        if isinstance(state, dict):
            cleaned = {}
            for k, v in state.items():
                kk = k[7:] if k.startswith("module.") else k
                cleaned[kk] = v
            state = cleaned
        if not isinstance(state, dict):
            raise RuntimeError(f"Unexpected checkpoint format for ConvNeXt FID: {type(state)}")

        self.backbone.load_state_dict(state, strict=False)

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        # x01: BCHW float in [0,1]
        if x01.ndim != 4 or x01.size(1) != 3:
            raise ValueError(f"Expected BCHW with 3 channels, got {tuple(x01.shape)}")
        # ConvNeXt TCGA feature encoder is trained with ImageNet-style normalization at 224x224.
        if x01.shape[-2:] != (224, 224):
            x01 = F.interpolate(x01, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x01 - self.mean) / self.std

        # timm convnext forward_features returns BCHW.
        feats = self.backbone.forward_features(x)  # type: ignore[attr-defined]
        if feats.ndim == 4:
            feats = feats.mean(dim=(2, 3))
        return feats


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
        model = _ConvNeXtTinyFID().to(dev)
        model.eval()

    n = 0
    fid_pairs_seen = 0
    for gt_t, pr_t in pairs:
        gt01 = _to_rgb01_from_minus1_1(gt_t).cpu()
        pr01 = _to_rgb01_from_minus1_1(pr_t).cpu()

        # PSNR/SSIM on CPU numpy (keeps behavior consistent with compute_metrics_from_dirs)
        gt = gt01[0].permute(1, 2, 0).numpy()
        pr = pr01[0].permute(1, 2, 0).numpy()
        psnrs.append(_psnr_01(gt, pr))
        if compute_ssim:
            ssims.append(_ssim_01(gt, pr))

        # FID can be capped independently of PSNR/SSIM.
        # If fid_max_items is set, we still compute PSNR/SSIM on all pairs,
        # but only feed the first fid_max_items pairs into Inception stats.
        if (
            compute_fid
            and model is not None
            and (fid_max_items is None or fid_pairs_seen < int(fid_max_items))
        ):
            buf_t.append(gt01[0])
            buf_p.append(pr01[0])
            if len(buf_t) >= int(fid_batch_size):
                bt = torch.stack(buf_t, dim=0).to(dev, dtype=torch.float32)
                bp = torch.stack(buf_p, dim=0).to(dev, dtype=torch.float32)
                at = model(bt).detach().cpu().numpy().reshape(bt.size(0), -1)
                ap = model(bp).detach().cpu().numpy().reshape(bp.size(0), -1)
                stats_t.update(at)
                stats_p.update(ap)
                fid_n += int(bt.size(0))
                fid_pairs_seen += int(bt.size(0))
                buf_t.clear()
                buf_p.clear()

        n += 1

    # Flush remainder
    if (
        compute_fid
        and model is not None
        and buf_t
        and (fid_max_items is None or fid_pairs_seen < int(fid_max_items))
    ):
        if fid_max_items is not None:
            # Truncate any overflow (possible when fid_max_items < fid_batch_size).
            remaining = int(fid_max_items) - int(fid_pairs_seen)
            if remaining <= 0:
                buf_t = []
                buf_p = []
            else:
                buf_t = buf_t[:remaining]
                buf_p = buf_p[:remaining]
        bt = torch.stack(buf_t, dim=0).to(dev, dtype=torch.float32)
        bp = torch.stack(buf_p, dim=0).to(dev, dtype=torch.float32)
        at = model(bt).detach().cpu().numpy().reshape(bt.size(0), -1)
        ap = model(bp).detach().cpu().numpy().reshape(bp.size(0), -1)
        stats_t.update(at)
        stats_p.update(ap)
        fid_n += int(bt.size(0))
        fid_pairs_seen += int(bt.size(0))

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

    model = _ConvNeXtTinyFID().to(device)
    model.eval()

    stats = _ActStats()
    for x in dl:
        x = x.to(device=device, dtype=torch.float32)
        pred = model(x)
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

