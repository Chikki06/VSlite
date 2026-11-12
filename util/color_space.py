"""sRGB <-> linear RGB for H&E / VIS tensors."""

from __future__ import annotations

import torch


def _to_01(t: torch.Tensor, *, minus1_1: bool) -> torch.Tensor:
    """Map to [0, 1]. When ``minus1_1``, always treat input as [-1, 1] (do not infer)."""
    if minus1_1:
        return ((t + 1.0) / 2.0).clamp(0.0, 1.0)
    return t.clamp(0.0, 1.0)


def _from_01(t: torch.Tensor, minus1_1: bool) -> torch.Tensor:
    if minus1_1:
        return t.clamp(0.0, 1.0) * 2.0 - 1.0
    return t.clamp(0.0, 1.0)


def srgb_to_linear(t: torch.Tensor, *, minus1_1: bool = True) -> torch.Tensor:
    """Convert RGB tensor (CHW or BCHW) from sRGB to linear light."""
    x = _to_01(t.float(), minus1_1=minus1_1)
    linear = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return _from_01(linear, minus1_1)


def linear_to_srgb(
    t: torch.Tensor, *, minus1_1: bool = True, eps: float = 1e-6
) -> torch.Tensor:
    """Convert RGB tensor from linear to sRGB.

    ``x ** (1/2.4)`` has an unbounded derivative at 0. Without a floor, backprop
    through this (C1: G output → D/NCE heads) yields NaN grads on near-black
    pixels within a few hundred iters. Clamp only the power branch.
    """
    x = _to_01(t.float(), minus1_1=minus1_1)
    lo = x * 12.92
    hi = 1.055 * (x.clamp_min(eps) ** (1.0 / 2.4)) - 0.055
    srgb = torch.where(x <= 0.0031308, lo, hi)
    return _from_01(srgb, minus1_1)


def apply_he_color_space(t: torch.Tensor, space: str, *, minus1_1: bool = True) -> torch.Tensor:
    """Apply color space transform for training targets (default passthrough for srgb)."""
    if str(space).lower() == "linear":
        return srgb_to_linear(t, minus1_1=minus1_1)
    return t
