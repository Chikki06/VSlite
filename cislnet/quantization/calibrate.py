"""Calibration dataloader and activation statistics for PTQ."""

from __future__ import annotations

from argparse import Namespace
from typing import Dict, Iterator, Optional, Tuple

import torch

from data import create_dataloader


def build_calibration_dataloader(
    opt: Namespace,
    *,
    num_patches: int = 512,
    batch_size: int = 4,
    num_workers: int = 8,
    image_size: int = 1024,
) -> torch.utils.data.DataLoader:
    """Train-split patches without augmentation for calibration."""
    calib_opt = Namespace(**vars(opt))
    calib_opt.isTrain = False
    calib_opt.phase = "train"
    calib_opt.serial_batches = True
    calib_opt.no_flip = True
    calib_opt.no_flip_vert = True
    calib_opt.no_rotate = True
    calib_opt.batch_size = max(1, int(batch_size))
    calib_opt.nThreads = int(num_workers)
    calib_opt.num_test = float("inf")
    # CISLNet trains/tests at 1024; opt.pkl may omit image_size.
    calib_opt.image_size = int(getattr(opt, "image_size", None) or image_size)
    if not getattr(calib_opt, "dataroot", ""):
        raise ValueError("opt.dataroot is required for calibration (set DATAROOT in job env)")
    return create_dataloader(calib_opt)


def iter_calibration_batches(
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int,
) -> Iterator[dict]:
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        yield {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def collect_activation_scales(
    model: torch.nn.Module,
    calib_loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    max_batches: int,
    percentile: float = 100.0,
) -> Dict[str, float]:
    """Per-module input activation magnitude from calibration forwards."""
    scales: Dict[str, float] = {}
    hooks = []

    def _make_hook(name: str):
        def _hook(_module, inputs, _outputs):
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            flat = x.detach().abs().reshape(-1)
            if percentile >= 100.0:
                val = float(flat.max().cpu())
            else:
                q = max(0.0, min(100.0, percentile)) / 100.0
                val = float(torch.quantile(flat.float(), q).cpu())
            scales[name] = max(scales.get(name, 0.0), val)

        return _hook

    for name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight is not None:
            hooks.append(module.register_forward_hook(_make_hook(name)))

    model.eval()
    with torch.no_grad():
        for batch in iter_calibration_batches(calib_loader, device, max_batches):
            x = batch["label"]
            model(x)

    for h in hooks:
        h.remove()
    return scales


def calibration_batch_count(num_patches: int, batch_size: int) -> int:
    return max(1, (int(num_patches) + int(batch_size) - 1) // int(batch_size))
