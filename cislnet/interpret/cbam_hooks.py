"""Forward hooks for CBAM spatial / feature-channel attention in AttUnetGenerator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class CBAMCapture:
    """Attention tensors captured during one forward pass."""

    channel_weights: Dict[str, torch.Tensor] = field(default_factory=dict)
    spatial_maps: Dict[str, torch.Tensor] = field(default_factory=dict)
    hook_handles: List = field(default_factory=list)

    def clear(self) -> None:
        self.channel_weights.clear()
        self.spatial_maps.clear()

    def remove(self) -> None:
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()


def _channel_hook(name: str, capture: CBAMCapture):
    def hook(module, inputs, output):
        x = inputs[0]
        avg_values = module.avg_pool(x)
        max_values = module.max_pool(x)
        logits = module.mlp(avg_values) + module.mlp(max_values)
        weights = torch.sigmoid(logits).squeeze(-1).squeeze(-1)  # (B, C)
        capture.channel_weights[name] = weights.detach().cpu()

    return hook


def _spatial_hook(name: str, capture: CBAMCapture):
    def hook(module, inputs, output):
        x = inputs[0]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        stacked = torch.cat([avg_out, max_out], dim=1)
        spatial = torch.sigmoid(module.bn(module.conv(stacked)))
        capture.spatial_maps[name] = spatial.detach().cpu()

    return hook


def register_cbam_hooks(net_g: nn.Module) -> CBAMCapture:
    """
    Register hooks on cbam_0..cbam_4 channel + spatial attention submodules.

    Returns a CBAMCapture that is populated on each net_g forward.
    """
    capture = CBAMCapture()
    for i in range(5):
        cbam = getattr(net_g, f"cbam_{i}", None)
        if cbam is None:
            continue
        name = f"cbam_{i}"
        if hasattr(cbam, "channel_att"):
            h = cbam.channel_att.register_forward_hook(_channel_hook(name, capture))
            capture.hook_handles.append(h)
        if hasattr(cbam, "spatial_att"):
            h = cbam.spatial_att.register_forward_hook(_spatial_hook(name, capture))
            capture.hook_handles.append(h)
    return capture


def upsample_spatial_map(spatial: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Upsample (B,1,H,W) spatial map to target (H,W)."""
    if spatial.ndim == 3:
        spatial = spatial.unsqueeze(0)
    if spatial.ndim != 4 or spatial.shape[1] != 1:
        raise ValueError(f"Expected (B,1,H,W) spatial map, got {tuple(spatial.shape)}")
    out = torch.nn.functional.interpolate(
        spatial.float(),
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )
    return out[0, 0]
