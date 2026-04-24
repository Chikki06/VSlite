import argparse
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml

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

from SwinGAN.infer_wsi_png import FolderData, VSLiteFolderLoader

from NAFNet.model_utils import build_model


def load_config(path: str) -> SimpleNamespace:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return SimpleNamespace(**cfg)


def build_nafnet_restorer(config_path: str, ckpt_path: str, model_type: int, device: torch.device):
    cfg = load_config(config_path)
    in_chans = 13 if int(model_type) == 1 else 10
    model = build_model(cfg, in_chans=in_chans).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg


def _prep_vis(vis: torch.Tensor, vis_pixel_size: float) -> torch.Tensor:
    target_pixel_size = 0.69
    scale = float(vis_pixel_size) / target_pixel_size
    _, h, w = vis.shape
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    vis_resized = F.interpolate(vis.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
    return vis_resized.clamp(0.0, 1.0)


def infer_folder(model, data: FolderData, model_type: int, tile_size: int, device: torch.device) -> np.ndarray:
    scaling_factor = 0.69 / 2.0

    ir = data.ir.to(device=device, dtype=torch.float32)
    if model_type == 1:
        if data.vis is None or data.vis_pixel_size is None:
            raise RuntimeError("13ch mode requires VIS data")
        vis = _prep_vis(data.vis.to(device=device, dtype=torch.float32), data.vis_pixel_size)
        _, r, c = vis.shape
    else:
        _, h_ir, w_ir = ir.shape
        ratio = 2.0 / 0.69
        r = int(h_ir * ratio)
        c = int(w_ir * ratio)
        vis = None

    rn = int(math.ceil(r / tile_size))
    cn = int(math.ceil(c / tile_size))

    out = np.zeros((3, rn * tile_size + 1, cn * tile_size + 1), dtype=np.uint8)
    I = list(range(0, rn * tile_size + 1, tile_size))
    J = list(range(0, cn * tile_size + 1, tile_size))

    with torch.no_grad():
        for ii in range(len(I) - 1):
            for jj in range(len(J) - 1):
                ir_patch = ir[
                    :,
                    int(I[ii] * scaling_factor) : int(I[ii + 1] * scaling_factor),
                    int(J[jj] * scaling_factor) : int(J[jj + 1] * scaling_factor),
                ]
                target_lr = int(tile_size * scaling_factor)
                pad_rows = max(0, target_lr - ir_patch.shape[1])
                pad_cols = max(0, target_lr - ir_patch.shape[2])
                ir_patch = F.pad(ir_patch.unsqueeze(0), (0, pad_cols, 0, pad_rows), mode="constant", value=0.0)
                ir_patch = F.interpolate(ir_patch, size=(tile_size, tile_size), mode="bicubic", align_corners=True)

                if model_type == 1:
                    vis_patch = vis[:, int(I[ii]) : int(I[ii + 1]), int(J[jj]) : int(J[jj + 1])]
                    v_pad_rows = max(0, tile_size - vis_patch.shape[1])
                    v_pad_cols = max(0, tile_size - vis_patch.shape[2])
                    vis_patch = F.pad(vis_patch.unsqueeze(0), (0, v_pad_cols, 0, v_pad_rows), mode="constant", value=0.0)
                    x = torch.cat([ir_patch, vis_patch], dim=1)
                else:
                    x = ir_patch

                y_hat = torch.tanh(model(x))
                y_np = (y_hat[0].clamp(-1.0, 1.0).cpu().numpy() * 0.5) + 0.5
                y_u8 = np.clip(y_np * 255.0, 0, 255).astype(np.uint8)
                out[:, I[ii] : I[ii + 1], J[jj] : J[jj + 1]] = y_u8

    out = out[:, :r, :c]
    out = np.transpose(out, (1, 2, 0))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Whole-slide PNG inference using NAFNet i2i checkpoints")
    p.add_argument("--ckpt", required=True, type=str, help="Path to NAFNet checkpoint (.pth)")
    p.add_argument(
        "--config",
        default="/projects/illinois/eng/bioe/rxb/akshat/VSlite/NAFNet/config_i2i_nafnet_base.yaml",
        type=str,
        help="NAFNet config used to build the model",
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
    model, cfg = build_nafnet_restorer(args.config, args.ckpt, args.model, device)

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
        print(f"Processing {folder_name} ...")
        loader = VSLiteFolderLoader(parent, folder_name, args.model)
        data = loader.load()
        pred = infer_folder(model, data, args.model, tile_size=tile_size, device=device)

        suffix = "13ch" if int(args.model) == 1 else "10ch"
        out_path = os.path.join(output_dir, f"{folder_name}_{suffix}.png")
        Image.fromarray(pred).save(out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
