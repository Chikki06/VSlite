"""Unified image-saving utility for all VSlite training and test scripts.

Layout produced by ``save_unified_sample``:

    +----+----+----+----+
    | R0 | R1 | R4 | R5 |     <- top half  (8 real patches)
    +----+----+----+----+        region 1: reals[0..3] (TL TR BL BR)
    | R2 | R3 | R6 | R7 |        region 2: reals[4..7]
    +----+----+----+----+
    | P0 | P1 | P4 | P5 |     <- bottom half (8 predicted patches)
    +----+----+----+----+        same region splits as reals
    | P2 | P3 | P6 | P7 |
    +----+----+----+----+

Each cell is rendered at the patch's native resolution (no resizing).
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image


def tensor_to_png_array(
    tensor: torch.Tensor,
    *,
    prefer_last_three: bool = False,
) -> np.ndarray:
    """Convert a CHW/BCHW tensor in [-1,1] or [0,1] to an HxWxC uint8 array."""
    tensor = tensor.detach().float().cpu()
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW or BCHW tensor, got shape {tuple(tensor.shape)}")
    if tensor.shape[0] > 3:
        tensor = tensor[-3:] if prefer_last_three else tensor[:3]
    if tensor.min().item() < 0.0:
        tensor = (tensor + 1.0) / 2.0
    tensor = tensor.clamp(0.0, 1.0)
    array = (tensor * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    if array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    return array


def _pad_or_pick(items: Sequence[torch.Tensor], n: int) -> list[Optional[torch.Tensor]]:
    out: list[Optional[torch.Tensor]] = []
    for i in range(n):
        out.append(items[i] if i < len(items) else None)
    return out


def _to_image(
    tensor: Optional[torch.Tensor],
    fallback_size: tuple[int, int],
    *,
    prefer_last_three: bool = False,
    bg: int = 0,
) -> Image.Image:
    if tensor is None:
        return Image.new("RGB", fallback_size, color=(bg, bg, bg))
    arr = tensor_to_png_array(tensor, prefer_last_three=prefer_last_three)
    im = Image.fromarray(arr)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def _max_cell_size(images: Sequence[Image.Image]) -> tuple[int, int]:
    w = max((im.size[0] for im in images), default=0)
    h = max((im.size[1] for im in images), default=0)
    return (max(1, w), max(1, h))


def save_unified_sample(
    out_dir: str,
    name: str,
    reals: Sequence[torch.Tensor],
    preds: Sequence[torch.Tensor],
    *,
    pad: int = 4,
    bg: int = 0,
    input_mode: Optional[str] = None,
) -> str:
    """Save the 4-col x 4-row composite to ``<out_dir>/<name>.png``.

    Parameters
    ----------
    out_dir : str
        Directory to write into (created if missing).
    name : str
        Filename stem (no extension).
    reals : sequence of tensors
        Up to 8 real/target patches (CHW or 1xCHW), in [-1,1] or [0,1].
    preds : sequence of tensors
        Up to 8 predicted patches with the same convention as ``reals``.
    pad : int
        Padding (in pixels) between cells.
    bg : int
        Greyscale background value (0..255).
    input_mode : Optional[str]
        Reserved for parity with multi-channel inputs (e.g. "irvis13").
        Currently ignored for real/pred (they are RGB).

    Returns
    -------
    str
        The path of the written PNG.
    """
    os.makedirs(out_dir, exist_ok=True)

    reals8 = _pad_or_pick(list(reals), 8)
    preds8 = _pad_or_pick(list(preds), 8)

    real_imgs = [_to_image(t, (1, 1), bg=bg) for t in reals8]
    pred_imgs = [_to_image(t, (1, 1), bg=bg) for t in preds8]
    cw, ch = _max_cell_size([im for im in real_imgs + pred_imgs])

    def _resize_cells(images: list[Image.Image]) -> list[Image.Image]:
        out = []
        for im in images:
            if im.size != (cw, ch):
                resized = Image.new("RGB", (cw, ch), color=(bg, bg, bg))
                resized.paste(im, (0, 0))
                out.append(resized)
            else:
                out.append(im)
        return out

    real_imgs = _resize_cells(real_imgs)
    pred_imgs = _resize_cells(pred_imgs)

    cols = 4
    rows = 4
    out_w = pad + cols * (cw + pad)
    out_h = pad + rows * (ch + pad)
    canvas = Image.new("RGB", (out_w, out_h), color=(bg, bg, bg))

    def _paste_region(region: list[Image.Image], row0: int, col0: int) -> None:
        slots = [(row0, col0), (row0, col0 + 1), (row0 + 1, col0), (row0 + 1, col0 + 1)]
        for im, (r, c) in zip(region, slots):
            x0 = pad + c * (cw + pad)
            y0 = pad + r * (ch + pad)
            canvas.paste(im, (x0, y0))

    _paste_region(real_imgs[0:4], row0=0, col0=0)
    _paste_region(real_imgs[4:8], row0=0, col0=2)
    _paste_region(pred_imgs[0:4], row0=2, col0=0)
    _paste_region(pred_imgs[4:8], row0=2, col0=2)

    out_path = os.path.join(out_dir, f"{name}.png")
    canvas.save(out_path)
    return out_path


def split_batch_to_list(tensor: torch.Tensor) -> list[torch.Tensor]:
    """Split a BxCxHxW tensor into a list of CxHxW tensors."""
    if tensor.ndim == 3:
        return [tensor]
    if tensor.ndim != 4:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(tensor.shape)}")
    return [tensor[i] for i in range(tensor.shape[0])]
