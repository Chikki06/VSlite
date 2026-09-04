"""sRGB <-> linear RGB for H&E / VIS tensors."""

from __future__ import annotations

import torch


def _to_01(t: torch.Tensor) -> torch.Tensor:
    if t.min().item() < 0.0:
        return (t + 1.0) / 2.0
    return t.clamp(0.0, 1.0)


def _from_01(t: torch.Tensor, minus1_1: bool) -> torch.Tensor:
    if minus1_1:
        return t.clamp(0.0, 1.0) * 2.0 - 1.0
    return t.clamp(0.0, 1.0)


def srgb_to_linear(t: torch.Tensor, *, minus1_1: bool = True) -> torch.Tensor:
    """Convert RGB tensor (CHW or BCHW) from sRGB to linear light."""
    x = _to_01(t.float())
    linear = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return _from_01(linear, minus1_1)


def linear_to_srgb(t: torch.Tensor, *, minus1_1: bool = True) -> torch.Tensor:
    """Convert RGB tensor from linear to sRGB."""
    x = _to_01(t.float())
    srgb = torch.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)
    return _from_01(srgb, minus1_1)


def apply_he_color_space(t: torch.Tensor, space: str, *, minus1_1: bool = True) -> torch.Tensor:
    """Apply color space transform for training targets (default passthrough for srgb)."""
    if str(space).lower() == "linear":
        return srgb_to_linear(t, minus1_1=minus1_1)
    return t
