"""Whole-slide PNG inference using Restormer i2i checkpoints.

Uses the lazy WSI loader + tile-windowed read helpers from
`SwinGAN.infer_wsi_png` so the WSI is never fully materialized in host RAM
or GPU memory; only one tile-sized tensor crosses device boundaries at a
time. See the docstring in `SwinGAN/infer_wsi_png.py` for the rationale.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

_THIS_DIR = Path(__file__).resolve().parent
_VSLITE_ROOT = _THIS_DIR.parent
if str(_VSLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_VSLITE_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
# Ensure SwinGAN modules import cleanly even when running from Restormer/.
_SWINGAN_DIR = _VSLITE_ROOT / "SwinGAN"
if str(_SWINGAN_DIR) not in sys.path:
    sys.path.insert(0, str(_SWINGAN_DIR))

from SwinGAN.infer_wsi_png import (  # noqa: E402
    FolderData,
    VSLiteFolderLoader,
    output_geometry,
    read_ir_tile_to_device,
    read_vis_tile_to_device,
)
from train_i2i_restormer import build_model  # noqa: E402


def load_config(path: str) -> SimpleNamespace:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return SimpleNamespace(**cfg)


def build_restorer(config_path: str, ckpt_path: str, model_type: int, device: torch.device):
    cfg = load_config(config_path)
    in_chans = 13 if int(model_type) == 1 else 10
    model = build_model(cfg, in_chans=in_chans).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg


def infer_folder(
    model,
    data: FolderData,
    model_type: int,
    tile_size: int,
    device: torch.device,
) -> np.ndarray:
    r, c = output_geometry(data, model_type)

    rn = int(math.ceil(r / tile_size))
    cn = int(math.ceil(c / tile_size))

    out = np.zeros((rn * tile_size + 1, cn * tile_size + 1, 3), dtype=np.uint8)
    I = list(range(0, rn * tile_size + 1, tile_size))
    J = list(range(0, cn * tile_size + 1, tile_size))

    with torch.no_grad():
        for ii in range(len(I) - 1):
            for jj in range(len(J) - 1):
                ir_patch = read_ir_tile_to_device(
                    data,
                    y0_tgt=I[ii],
                    y1_tgt=I[ii + 1],
                    x0_tgt=J[jj],
                    x1_tgt=J[jj + 1],
                    tile_size=tile_size,
                    device=device,
                )

                if int(model_type) == 1:
                    vis_patch = read_vis_tile_to_device(
                        data,
                        y0_tgt=I[ii],
                        y1_tgt=I[ii + 1],
                        x0_tgt=J[jj],
                        x1_tgt=J[jj + 1],
                        tile_size=tile_size,
                        device=device,
                    )
                    x = torch.cat([ir_patch, vis_patch], dim=1)
                else:
                    vis_patch = None
                    x = ir_patch

                y_hat = torch.tanh(model(x))
                y_np = (y_hat[0].clamp(-1.0, 1.0).cpu().numpy() * 0.5) + 0.5
                y_u8 = np.clip(y_np * 255.0, 0, 255).astype(np.uint8)
                out[I[ii] : I[ii + 1], J[jj] : J[jj + 1], :] = np.transpose(y_u8, (1, 2, 0))

                del ir_patch, x, y_hat, y_np, y_u8
                if vis_patch is not None:
                    del vis_patch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            gc.collect()

    out = out[:r, :c, :]
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Whole-slide PNG inference using Restormer i2i checkpoints")
    p.add_argument("--ckpt", required=True, type=str, help="Path to Restormer checkpoint (.pth)")
    p.add_argument(
        "--config",
        default="/projects/illinois/eng/bioe/rxb/akshat/VSlite/Restormer/config_i2i_restormer.yaml",
        type=str,
        help="Restormer config used to build the model",
    )
    p.add_argument("--model", type=int, default=0, choices=[0, 1], help="0: 10ch IR, 1: 13ch IR+VIS")
    p.add_argument("--folder", type=str, default=None, help="Optional single folder path to process")
    p.add_argument("--parent_folder", type=str, default=None, help="Parent folder containing sample folders")
    p.add_argument("--output_dir", type=str, default=None, help="Output directory for PNG files")
    p.add_argument("--tile_size", type=int, default=None, help="Patch size for model inference")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = build_restorer(args.config, args.ckpt, args.model, device)

    tile_size = int(args.tile_size if args.tile_size is not None else getattr(cfg, "img_size", 256))

    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "wsi_png")
    else:
        output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.folder:
        folder_path = os.path.abspath(args.folder)
        parent = os.path.dirname(folder_path)
        folder_names = [os.path.basename(folder_path)]
    elif args.parent_folder:
        parent = os.path.abspath(args.parent_folder)
        folder_names = [
            d for d in sorted(os.listdir(parent)) if os.path.isdir(os.path.join(parent, d)) and not d.endswith("_VIS")
        ]
    else:
        raise ValueError("Provide --folder or --parent_folder")

    for folder_name in folder_names:
        print(f"Processing {folder_name} ...", flush=True)
        loader = VSLiteFolderLoader(parent, folder_name, args.model)
        data = loader.load()
        pred = infer_folder(model, data, args.model, tile_size=tile_size, device=device)

        suffix = "13ch" if int(args.model) == 1 else "10ch"
        out_path = os.path.join(output_dir, f"{folder_name}_{suffix}.png")
        Image.fromarray(pred).save(out_path)
        print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
