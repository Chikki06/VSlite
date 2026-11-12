"""Numerical stability for sRGB <-> linear converts used by C1."""

from __future__ import annotations

import torch

from util.color_space import linear_to_srgb, srgb_to_linear


def test_linear_to_srgb_black_pixel_grads_are_finite():
    """Regression: unbounded d/dx x**(1/2.4) at 0 caused C1 train NaNs."""
    t = torch.full((1, 3, 4, 4), -1.0, requires_grad=True)  # pure black in [-1,1]
    y = linear_to_srgb(t, minus1_1=True)
    y.pow(2).mean().backward()
    assert t.grad is not None
    assert torch.isfinite(t.grad).all(), t.grad


def test_minus1_1_flag_honored_when_batch_min_is_nonnegative():
    """Do not infer range from min(); all-nonnegative [-1,1] batches must still scale."""
    # Mid-gray in [-1,1] is 0.0; old heuristic treated this as already [0,1].
    x = torch.zeros(1, 3, 2, 2)
    lin = srgb_to_linear(x, minus1_1=True)
    # 0.5 sRGB → linear ≈ 0.214; in [-1,1] ≈ -0.572
    assert float(lin.mean()) < -0.4
    back = linear_to_srgb(lin, minus1_1=True)
    assert torch.allclose(back, x, atol=1e-5)


def test_linear_to_srgb_roundtrip_on_ramp():
    x = torch.linspace(-1.0, 1.0, 16).view(1, 1, 1, 16).repeat(1, 3, 1, 1)
    back = linear_to_srgb(srgb_to_linear(x, minus1_1=True), minus1_1=True)
    assert torch.allclose(back, x, atol=1e-5)
