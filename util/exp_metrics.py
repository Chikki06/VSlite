"""Experiment-only metrics: torchmetrics VGG-LPIPS + synchronized GPU latency."""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn


def build_lpips_vgg(device: torch.device | str = "cuda") -> nn.Module:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    mod = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
    return mod.to(device).eval()


@torch.no_grad()
def lpips_vgg_pair(
    lpips_mod: nn.Module,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute VGG-LPIPS. Inputs in [-1,1] or [0,1] BCHW; normalize=True expects [0,1]."""
    def _to01(t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        if t.min().item() < 0.0:
            t = (t + 1.0) * 0.5
        return t.clamp(0.0, 1.0)

    return lpips_mod(_to01(pred), _to01(target))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def measure_latency_ms(
    forward_fn,
    *,
    device: torch.device,
    warmup: int = 10,
    timed: int = 100,
) -> dict[str, Any]:
    """Warmup ``warmup`` iters, then mean/std over ``timed`` iters with cuda sync."""
    for _ in range(max(0, int(warmup))):
        _sync(device)
        forward_fn()
        _sync(device)

    times_ms: list[float] = []
    for _ in range(max(0, int(timed))):
        _sync(device)
        t0 = time.perf_counter()
        forward_fn()
        _sync(device)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    if not times_ms:
        return {
            "latency_ms_mean": None,
            "latency_ms_std": None,
            "latency_warmup": int(warmup),
            "latency_timed_n": 0,
        }
    t = torch.tensor(times_ms, dtype=torch.float64)
    return {
        "latency_ms_mean": float(t.mean().item()),
        "latency_ms_std": float(t.std(unbiased=False).item()),
        "latency_warmup": int(warmup),
        "latency_timed_n": int(len(times_ms)),
    }
