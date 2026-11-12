"""Unified image-saving utility for all VSlite training and test scripts.

Layout produced by ``save_unified_sample`` — one PNG per pair:

    +------+------+
    | pred | real |     <- left = generation, right = ground truth
    +------+------+

Each call with ``N`` reals/preds writes ``N`` files named
``{name}_p{00..N-1}.png``. Patches keep native resolution (no resize).
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


def _match_size(im: Image.Image, size: tuple[int, int], *, bg: int = 0) -> Image.Image:
    if im.size == size:
        return im
    canvas = Image.new("RGB", size, color=(bg, bg, bg))
    canvas.paste(im, (0, 0))
    return canvas


def save_unified_sample(
    out_dir: str,
    name: str,
    reals: Sequence[torch.Tensor],
    preds: Sequence[torch.Tensor],
    *,
    pad: int = 4,
    bg: int = 0,
    input_mode: Optional[str] = None,
) -> list[str]:
    """Save one left=pred / right=real PNG per pair under ``<out_dir>``.

    Parameters
    ----------
    out_dir : str
        Directory to write into (created if missing).
    name : str
        Filename stem (no extension). Pair ``i`` is written as
        ``{name}_p{i:02d}.png``.
    reals : sequence of tensors
        Real/target patches (CHW or 1xCHW), in [-1,1] or [0,1].
    preds : sequence of tensors
        Predicted patches with the same convention as ``reals``.
    pad : int
        Padding (in pixels) between the two tiles.
    bg : int
        Greyscale background value (0..255).
    input_mode : Optional[str]
        Reserved for parity with multi-channel inputs (e.g. "irvis13").
        Currently ignored for real/pred (they are RGB).

    Returns
    -------
    list[str]
        Paths of the written PNGs (one per pair).
    """
    del input_mode  # reserved
    os.makedirs(out_dir, exist_ok=True)

    n = max(len(reals), len(preds))
    if n == 0:
        return []

    paths: list[str] = []
    for i in range(n):
        real_t = reals[i] if i < len(reals) else None
        pred_t = preds[i] if i < len(preds) else None
        pred_im = _to_image(pred_t, (1, 1), bg=bg)
        real_im = _to_image(real_t, (1, 1), bg=bg)
        cw = max(pred_im.size[0], real_im.size[0], 1)
        ch = max(pred_im.size[1], real_im.size[1], 1)
        pred_im = _match_size(pred_im, (cw, ch), bg=bg)
        real_im = _match_size(real_im, (cw, ch), bg=bg)

        out_w = pad + 2 * (cw + pad)
        out_h = pad + (ch + pad)
        canvas = Image.new("RGB", (out_w, out_h), color=(bg, bg, bg))
        canvas.paste(pred_im, (pad, pad))
        canvas.paste(real_im, (pad + cw + pad, pad))

        out_path = os.path.join(out_dir, f"{name}_p{i:02d}.png")
        canvas.save(out_path)
        paths.append(out_path)
    return paths


def split_batch_to_list(tensor: torch.Tensor) -> list[torch.Tensor]:
    """Split a BxCxHxW tensor into a list of CxHxW tensors."""
    if tensor.ndim == 3:
        return [tensor]
    if tensor.ndim != 4:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(tensor.shape)}")
    return [tensor[i] for i in range(tensor.shape[0])]
