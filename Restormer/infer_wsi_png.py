"""Whole-slide PNG inference using Restormer i2i checkpoints.

Uses the lazy WSI loader + batched tile inference from `SwinGAN.infer_wsi_png`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

_THIS_DIR = Path(__file__).resolve().parent
_VSLITE_ROOT = _THIS_DIR.parent
if str(_VSLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_VSLITE_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
_SWINGAN_DIR = _VSLITE_ROOT / "SwinGAN"
if str(_SWINGAN_DIR) not in sys.path:
    sys.path.insert(0, str(_SWINGAN_DIR))

from SwinGAN.infer_wsi_png import (  # noqa: E402
    VSLiteFolderLoader,
    infer_folder,
    resolve_wsi_batch_size,
)
from cislnet.wsi_local_padding import (  # noqa: E402
    add_local_padding_cli_args,
    resolve_local_padding_config,
)
from train_i2i_restormer import build_model  # noqa: E402


def load_config(path: str):
    from types import SimpleNamespace

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Whole-slide PNG inference using Restormer i2i checkpoints")
    p.add_argument("--ckpt", required=True, type=str, help="Path to Restormer checkpoint (.pth)")
    p.add_argument(
        "--config",
        default="/projects/illinois/eng/bioe/rxb/akshat/VSlite/Restormer/config_restormer.yaml",
        type=str,
        help="Restormer config used to build the model",
    )
    p.add_argument("--model", type=int, default=0, choices=[0, 1], help="0: 10ch IR, 1: 13ch IR+VIS")
    p.add_argument("--folder", type=str, default=None, help="Optional single folder path to process")
    p.add_argument("--parent_folder", type=str, default=None, help="Parent folder containing sample folders")
    p.add_argument("--output_dir", type=str, default=None, help="Output directory for PNG files")
    p.add_argument("--tile_size", type=int, default=None, help="Patch size for model inference")
    p.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="GPU micro-batch size in tiles (default: WSI_BATCH_SIZE env, else tile-size heuristic)",
    )
    add_local_padding_cli_args(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = build_restorer(args.config, args.ckpt, args.model, device)

    tile_size = int(args.tile_size if args.tile_size is not None else getattr(cfg, "img_size", 256))
    lp_cfg = resolve_local_padding_config(
        local_padding=bool(getattr(args, "local_padding", False)),
        local_padding_size=int(getattr(args, "local_padding_size", 1)),
        local_padding_outer=str(getattr(args, "local_padding_outer", "replicate")),
    )
    batch_size = resolve_wsi_batch_size(args.batch_size, tile_size)
    cfg_img_size = int(getattr(cfg, "img_size", tile_size))
    if tile_size != cfg_img_size:
        print(
            f"Warning: tile_size={tile_size} differs from config img_size={cfg_img_size}; "
            "model was built for config img_size.",
            flush=True,
        )
    print(
        f"Restormer WSI: tile_size={tile_size} batch_size={batch_size} "
        f"config_img_size={cfg_img_size} model_type={args.model} ckpt={args.ckpt}"
        + (f" local_padding=1 ps={lp_cfg.padding_size}" if lp_cfg.enabled else ""),
        flush=True,
    )

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
        pred = infer_folder(
            model,
            data,
            args.model,
            tile_size=tile_size,
            device=device,
            batch_size=batch_size,
            local_padding=lp_cfg,
        )

        suffix = "13ch" if int(args.model) == 1 else "10ch"
        out_path = os.path.join(output_dir, f"{folder_name}_{suffix}.png")
        Image.fromarray(pred).save(out_path)
        print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
