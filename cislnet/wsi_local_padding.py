"""Local padding for patch-by-patch WSI inference.

Implements border-context padding from adjacent, previously generated tiles instead
of zero-padding at tile seams. Adapted from Infinite_Texture_GANs:

  https://github.com/ai4netzero/Infinite_Texture_GANs
  https://arxiv.org/abs/2309.02340

For fixed-size models (e.g. SwinGAN), padding is applied in-place on the tile
tensor (same H×W) by overwriting border strips before each forward pass.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F

OuterPaddingMode = Literal["replicate", "constant"]


@dataclass(frozen=True)
class LocalPaddingConfig:
    enabled: bool
    padding_size: int = 1
    outer_padding: OuterPaddingMode = "replicate"

    def __post_init__(self) -> None:
        if self.padding_size < 0:
            raise ValueError(f"padding_size must be >= 0, got {self.padding_size}")
        if self.outer_padding not in ("replicate", "constant"):
            raise ValueError(f"outer_padding must be 'replicate' or 'constant', got {self.outer_padding!r}")


def add_local_padding_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register flags shared by all WSI inference entrypoints."""
    g = parser.add_argument_group(
        "local padding (Infinite_Texture_GANs)",
        "Patch-by-patch border context from previously generated tiles.",
    )
    g.add_argument(
        "--local_padding",
        action="store_true",
        help="Enable local padding at tile seams (forces sequential tile order, batch_size=1). "
        "Also enabled when WSI_LOCAL_PADDING=1.",
    )
    g.add_argument(
        "--local_padding_size",
        type=int,
        default=1,
        help="Border strip width in pixels (default: 1, matches ITG conv padding_size).",
    )
    g.add_argument(
        "--local_padding_outer",
        type=str,
        default="replicate",
        choices=["replicate", "constant"],
        help="Padding for slide/WSI outer edges when no neighbor exists (ITG outer_padding).",
    )


def resolve_local_padding_config(
    *,
    local_padding: bool = False,
    local_padding_size: int = 1,
    local_padding_outer: str = "replicate",
) -> LocalPaddingConfig:
    env_on = os.environ.get("WSI_LOCAL_PADDING", "").strip().lower() in ("1", "true", "yes", "on")
    enabled = bool(local_padding) or env_on
    outer: OuterPaddingMode = "constant" if str(local_padding_outer).strip().lower() == "constant" else "replicate"
    return LocalPaddingConfig(
        enabled=enabled,
        padding_size=int(local_padding_size),
        outer_padding=outer,
    )


def _as_bchw(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if x.dim() == 3:
        return x.unsqueeze(0), True
    if x.dim() == 4:
        return x, False
    raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(x.shape)}")


def _from_bchw(x: torch.Tensor, was_3d: bool) -> torch.Tensor:
    return x.squeeze(0) if was_3d else x


def _fill_constant_border(t: torch.Tensor, ps: int, *, fill_value: float) -> torch.Tensor:
    if ps <= 0:
        return t
    out = t.clone()
    out[..., :ps, :] = fill_value
    out[..., -ps:, :] = fill_value
    out[..., :, :ps] = fill_value
    out[..., :, -ps:] = fill_value
    return out


def _fill_replicate_border(t: torch.Tensor, ps: int) -> torch.Tensor:
    if ps <= 0:
        return t
    out = t.clone()
    h, w = int(out.shape[-2]), int(out.shape[-1])
    if h <= 1 or w <= 1:
        return _fill_constant_border(out, ps, fill_value=0.0)
    out[..., :ps, :] = out[..., ps : ps + 1, :].expand_as(out[..., :ps, :])
    out[..., -ps:, :] = out[..., -ps - 1 : -ps, :].expand_as(out[..., -ps:, :])
    out[..., :, :ps] = out[..., :, ps : ps + 1].expand_as(out[..., :, :ps])
    out[..., :, -ps:] = out[..., :, -ps - 1 : -ps].expand_as(out[..., :, -ps:])
    return out


def _output_to_vis_input(y_bchw: torch.Tensor) -> torch.Tensor:
    """Map generator output in [-1, 1] to VIS input range [0, 1]."""
    return y_bchw.clamp(-1.0, 1.0).mul(0.5).add(0.5)


class WSILocalPaddingState:
    """Row-major state: border strips from the previous tile / previous row."""

    def __init__(self, config: LocalPaddingConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._right_vis: Optional[torch.Tensor] = None  # (1, 3, H, ps)
        self._right_ir: Optional[torch.Tensor] = None  # (1, n_ir, H, ps)
        self._bottom_vis: Optional[torch.Tensor] = None  # (1, 3, ps, W)
        self._bottom_ir: Optional[torch.Tensor] = None  # (1, n_ir, ps, W)

    @property
    def padding_size(self) -> int:
        return int(self.config.padding_size)

    def apply_to_input(
        self,
        x: torch.Tensor,
        *,
        ii: int,
        jj: int,
        vis_offset: int,
    ) -> torch.Tensor:
        """Overwrite border pixels on a full tile before forward (fixed tile size)."""
        if not self.config.enabled or self.padding_size <= 0:
            return x

        x4, was_3d = _as_bchw(x)
        ps = self.padding_size
        c = int(x4.shape[1])
        vis_off = int(vis_offset)
        n_vis = 3
        out = x4.clone()

        if jj > 0 and self._right_vis is not None and vis_off + n_vis <= c:
            strip = self._right_vis.to(device=out.device, dtype=out.dtype)
            if strip.shape[-2] == out.shape[-2] and strip.shape[-1] == ps:
                out[:, vis_off : vis_off + n_vis, :, :ps] = strip[:, :n_vis, :, :ps]

        if jj > 0 and self._right_ir is not None and vis_off > 0:
            strip = self._right_ir.to(device=out.device, dtype=out.dtype)
            if strip.shape[-2] == out.shape[-2] and strip.shape[-1] == ps:
                out[:, :vis_off, :, :ps] = strip[:, :vis_off, :, :ps]

        if ii > 0 and self._bottom_vis is not None and vis_off + n_vis <= c:
            strip = self._bottom_vis.to(device=out.device, dtype=out.dtype)
            if strip.shape[-1] == out.shape[-1] and strip.shape[-2] == ps:
                out[:, vis_off : vis_off + n_vis, :ps, :] = strip[:, :n_vis, :ps, :]

        if ii > 0 and self._bottom_ir is not None and vis_off > 0:
            strip = self._bottom_ir.to(device=out.device, dtype=out.dtype)
            if strip.shape[-1] == out.shape[-1] and strip.shape[-2] == ps:
                out[:, :vis_off, :ps, :] = strip[:, :vis_off, :ps, :]

        if jj == 0 and ii == 0:
            if self.config.outer_padding == "replicate":
                out = _fill_replicate_border(out, ps)
            else:
                fill = 0.0
                out = _fill_constant_border(out, ps, fill_value=fill)
        elif jj == 0:
            if self.config.outer_padding == "replicate":
                out[..., :ps, :] = out[..., ps : ps + 1, :].expand_as(out[..., :ps, :])
            else:
                out[..., :ps, :] = 0.0
        elif ii == 0:
            if self.config.outer_padding == "replicate":
                out[..., :, :ps] = out[..., :, ps : ps + 1].expand_as(out[..., :, :ps])
            else:
                out[..., :, :ps] = 0.0

        return _from_bchw(out, was_3d)

    def update_after_tile(
        self,
        *,
        x_input: torch.Tensor,
        y_output: torch.Tensor,
        vis_offset: int,
        is_last_col: bool,
        is_last_row: bool,
    ) -> None:
        """Store right/bottom strips for the next tile in row-major order."""
        if not self.config.enabled or self.padding_size <= 0:
            return

        ps = self.padding_size
        x4, _ = _as_bchw(x_input)
        y4, _ = _as_bchw(y_output)
        vis_off = int(vis_offset)
        h, w = int(y4.shape[-2]), int(y4.shape[-1])
        if h < ps or w < ps:
            return

        if not is_last_col:
            self._right_vis = _output_to_vis_input(y4[:, :3, :, -ps:]).detach().cpu()
            if vis_off > 0:
                self._right_ir = x4[:, :vis_off, :, -ps:].detach().cpu()
            else:
                self._right_ir = None
        else:
            self._right_vis = None
            self._right_ir = None

        if not is_last_row:
            self._bottom_vis = _output_to_vis_input(y4[:, :3, -ps:, :]).detach().cpu()
            if vis_off > 0:
                self._bottom_ir = x4[:, :vis_off, -ps:, :].detach().cpu()
            else:
                self._bottom_ir = None
        else:
            self._bottom_vis = None
            self._bottom_ir = None

    def on_new_row(self) -> None:
        """Clear horizontal carry when starting a new row (jj == 0)."""
        self._right_vis = None
        self._right_ir = None


def effective_wsi_batch_size(requested: int, lp: LocalPaddingConfig) -> int:
    if lp.enabled and int(requested) != 1:
        print(
            f"[WSI] local_padding enabled: forcing batch_size=1 (requested {requested}).",
            flush=True,
        )
        return 1
    return max(1, int(requested))
