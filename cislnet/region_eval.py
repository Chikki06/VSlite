"""Tiled CISLNet inference over a fixed region canvas (e.g. 2048 from 512/1024 models)."""

from __future__ import annotations

import torch
import torch.nn as nn


def validate_region_tiling(region_size: int, tile_size: int) -> int:
    """Return grid factor (region_size // tile_size); raise if not evenly divisible."""
    region_size = int(region_size)
    tile_size = int(tile_size)
    if region_size <= 0 or tile_size <= 0:
        raise ValueError(f"region_size and tile_size must be positive, got {region_size}, {tile_size}")
    if region_size % tile_size != 0:
        raise ValueError(
            f"region_size={region_size} must be a multiple of model tile_size={tile_size}"
        )
    return region_size // tile_size


@torch.no_grad()
def tiled_generator_forward(
    net_g: nn.Module,
    x: torch.Tensor,
    tile_size: int,
) -> torch.Tensor:
    """
    Run generator on a CHW region, tiling when x spatial size exceeds tile_size.

    Args:
        net_g: AttUnetGenerator (or compatible fully-convolutional generator).
        x: (C, H, W) input tensor on the target device.
        tile_size: model training patch size.

    Returns:
        (3, H, W) prediction in the same value range as net_g output (typically tanh).
    """
    if x.ndim != 3:
        raise ValueError(f"Expected CHW input, got {tuple(x.shape)}")
    c, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    tile_size = int(tile_size)
    validate_region_tiling(h, tile_size)
    if w != h:
        raise ValueError(f"Region eval expects square regions, got {h}x{w}")
    if h == tile_size:
        return net_g(x.unsqueeze(0))[0]

    grid = h // tile_size
    out_c = 3
    out = torch.zeros(out_c, h, w, device=x.device, dtype=x.dtype)
    for ih in range(grid):
        for iw in range(grid):
            y0 = ih * tile_size
            x0 = iw * tile_size
            tile = x[:, y0 : y0 + tile_size, x0 : x0 + tile_size].unsqueeze(0)
            pred = net_g(tile)[0]
            out[:, y0 : y0 + tile_size, x0 : x0 + tile_size] = pred
    return out
