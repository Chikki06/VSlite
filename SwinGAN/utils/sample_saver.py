from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PIL import Image
import torch


def _tensor_to_png_array(tensor: torch.Tensor, *, prefer_last_three: bool = False) -> np.ndarray:
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
    return array


def save_sample_pngs(
    out_dir: str,
    name: str,
    x: torch.Tensor,
    y: torch.Tensor,
    yhat: torch.Tensor,
    input_mode: str,
):
    os.makedirs(out_dir, exist_ok=True)
    input_preview = _tensor_to_png_array(x, prefer_last_three=(input_mode == "irvis13"))
    target_preview = _tensor_to_png_array(y)
    pred_preview = _tensor_to_png_array(yhat)
    Image.fromarray(input_preview).save(os.path.join(out_dir, f"{name}_input.png"))
    Image.fromarray(target_preview).save(os.path.join(out_dir, f"{name}_target.png"))
    Image.fromarray(pred_preview).save(os.path.join(out_dir, f"{name}_pred.png"))


def save_sample_palette_png(
    out_dir: str,
    name: str,
    x: torch.Tensor,
    y: torch.Tensor,
    yhat: torch.Tensor,
    input_mode: str,
    *,
    pad: int = 8,
    bg: int = 0,
    filename_suffix: str = "palette",
    max_size: Optional[int] = None,
):
    """
    Save a single side-by-side palette image: [input | pred | target].

    - Writes: {out_dir}/{name}_{filename_suffix}.png
    - Uses the same CHW->HWC conversion as save_sample_pngs.
    """
    os.makedirs(out_dir, exist_ok=True)
    parts = [
        _tensor_to_png_array(x, prefer_last_three=(input_mode == "irvis13")),
        _tensor_to_png_array(yhat),
        _tensor_to_png_array(y),
    ]

    pil_parts = []
    for arr in parts:
        im = Image.fromarray(arr)
        if max_size is not None:
            w, h = im.size
            scale = min(max_size / max(1, w), max_size / max(1, h), 1.0)
            if scale < 1.0:
                im = im.resize((int(round(w * scale)), int(round(h * scale))), Image.BILINEAR)
        if im.mode != "RGB":
            im = im.convert("RGB")
        pil_parts.append(im)

    heights = [im.size[1] for im in pil_parts]
    widths = [im.size[0] for im in pil_parts]
    out_h = max(heights)
    out_w = sum(widths) + pad * (len(pil_parts) + 1)

    canvas = Image.new("RGB", (out_w, out_h + 2 * pad), color=(bg, bg, bg))
    x0 = pad
    y0 = pad
    for im in pil_parts:
        canvas.paste(im, (x0, y0))
        x0 += im.size[0] + pad

    canvas.save(os.path.join(out_dir, f"{name}_{filename_suffix}.png"))


def save_palette5_pred_target_png(
    out_dir: str,
    name: str,
    items: list[tuple[str, torch.Tensor, torch.Tensor]],
    *,
    pad: int = 8,
    bg: int = 0,
    filename_suffix: str = "palette5",
    max_size: Optional[int] = None,
):
    """
    Save a 5-row palette image with 2 columns: [pred | target].

    items: list of (sample_name, target_tensor_CHW, pred_tensor_CHW)
      - target/pred are in [-1,1] or [0,1]
      - only the first 5 are used
    Writes: {out_dir}/{name}_{filename_suffix}.png
    """
    os.makedirs(out_dir, exist_ok=True)
    if not items:
        return
    items = items[:5]

    rows: list[tuple[Image.Image, Image.Image]] = []
    for _sname, y, yhat in items:
        pr = Image.fromarray(_tensor_to_png_array(yhat))
        tg = Image.fromarray(_tensor_to_png_array(y))
        if max_size is not None:
            for im in (pr, tg):
                w, h = im.size
                scale = min(max_size / max(1, w), max_size / max(1, h), 1.0)
                if scale < 1.0:
                    im.thumbnail((int(round(w * scale)), int(round(h * scale))), Image.BILINEAR)
        if pr.mode != "RGB":
            pr = pr.convert("RGB")
        if tg.mode != "RGB":
            tg = tg.convert("RGB")
        rows.append((pr, tg))

    # Normalize row widths/heights
    row_heights = [max(pr.size[1], tg.size[1]) for pr, tg in rows]
    col_widths = [
        max(pr.size[0] for pr, _tg in rows),
        max(tg.size[0] for _pr, tg in rows),
    ]

    out_w = pad * 3 + col_widths[0] + col_widths[1]
    out_h = pad + sum(h + pad for h in row_heights)
    canvas = Image.new("RGB", (out_w, out_h), color=(bg, bg, bg))

    y0 = pad
    for (pr, tg), h in zip(rows, row_heights):
        x0 = pad
        canvas.paste(pr, (x0, y0))
        x0 += col_widths[0] + pad
        canvas.paste(tg, (x0, y0))
        y0 += h + pad

    canvas.save(os.path.join(out_dir, f"{name}_{filename_suffix}.png"))

