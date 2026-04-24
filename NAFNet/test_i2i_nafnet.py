"""Infer first, then compute PSNR/SSIM/FID on saved PNGs."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_VSLITE_ROOT = _THIS_DIR.parent
if str(_VSLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_VSLITE_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from SwinGAN.utils.dataset_vslite_i2i import VSLiteI2IDataset
from SwinGAN.utils.sample_saver import save_palette5_pred_target_png
from util.i2i_metrics import compute_metrics_streaming

from NAFNet.model_utils import build_model, build_split_fracs, load_config


def _get_env(name: str, default=None):
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def _tensor_to_png_array(tensor: torch.Tensor, prefer_last_three: bool = False) -> np.ndarray:
    tensor = tensor.detach().float().cpu()
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(tensor.shape)}")
    if tensor.shape[0] > 3:
        tensor = tensor[-3:] if prefer_last_three else tensor[:3]
    if tensor.min().item() < 0.0:
        tensor = (tensor + 1.0) / 2.0
    tensor = tensor.clamp(0.0, 1.0)
    arr = (tensor * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config_i2i_nafnet_base.yaml")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--num_samples", type=int, default=20, help="Number of PNG triplets to save")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--fid_max_items", type=int, default=None)
    ap.add_argument("--fid_batch_size", type=int, default=16)
    ap.add_argument("--save_every", type=int, default=0, help="If >0, save standardized PNG triplets every N patches.")
    ap.add_argument("--save_palette_every", type=int, default=0, help="If >0, save a palette preview every N patches.")
    args = ap.parse_args()

    cfg, _cfg_dict = load_config(args.config)
    cfg.dataroot = _get_env("NAFNET_DATAROOT", getattr(cfg, "dataroot", ""))
    cfg.output_dir = _get_env("NAFNET_OUTPUT_DIR", getattr(cfg, "output_dir", "OUTPUT_PATH"))
    cfg.input_mode = _get_env("NAFNET_INPUT_MODE", getattr(cfg, "input_mode", "ir10"))
    cfg.gpu_id = _get_env("NAFNET_GPU_ID", getattr(cfg, "gpu_id", "0"))

    if not cfg.dataroot:
        raise ValueError("Config 'dataroot' is empty. Set NAFNET_DATAROOT or edit config.")
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    out_dir = args.out_dir or os.path.join(cfg.output_dir, f"test_latest_{args.split}")
    os.makedirs(out_dir, exist_ok=True)
    try:
        shutil.copyfile(args.config, os.path.join(out_dir, "config.yaml"))
    except Exception:
        pass

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.input_mode == "ir10":
        in_chans = 10
    elif cfg.input_mode == "irvis13":
        in_chans = 13
    else:
        raise ValueError(f"Unsupported input_mode: {cfg.input_mode}")

    split_fracs = build_split_fracs(cfg)
    ds = VSLiteI2IDataset(
        cfg.dataroot,
        input_mode=cfg.input_mode,
        num_ir_channels=int(getattr(cfg, "num_ir_channels", 10)),
        img_size=int(getattr(cfg, "img_size", 256)),
        he_scale=str(getattr(cfg, "he_scale", "minus1_1")),
        split=args.split,
        split_seed=int(getattr(cfg, "split_seed", 1337)),
        split_fracs=split_fracs,
        max_items=getattr(cfg, "max_items", None),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    model = build_model(cfg, in_chans=in_chans).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    print(f"Running inference on {len(ds)} {args.split} samples...")

    def _save_standard_dirs(root: str, name: str, x: torch.Tensor, y: torch.Tensor, yhat: torch.Tensor, input_mode: str):
        from PIL import Image

        in_dir = os.path.join(root, "input")
        tg_dir = os.path.join(root, "target")
        pr_dir = os.path.join(root, "pred")
        os.makedirs(in_dir, exist_ok=True)
        os.makedirs(tg_dir, exist_ok=True)
        os.makedirs(pr_dir, exist_ok=True)
        Image.fromarray(_tensor_to_png_array(x, prefer_last_three=(input_mode == "irvis13"))).save(os.path.join(in_dir, f"{name}.png"))
        Image.fromarray(_tensor_to_png_array(y)).save(os.path.join(tg_dir, f"{name}.png"))
        Image.fromarray(_tensor_to_png_array(yhat)).save(os.path.join(pr_dir, f"{name}.png"))

    def _pairs():
        saved = 0
        buf5: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        for i, batch in enumerate(tqdm(loader, desc="Infer", unit="img")):
            x = batch["x"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device, dtype=torch.float32)
            with torch.no_grad():
                yhat = torch.tanh(model(x))
            name = batch["name"][0]
            buf5.append((name, y[0].detach().cpu(), yhat[0].detach().cpu()))
            if len(buf5) > 5:
                buf5.pop(0)

            if saved < args.num_samples:
                if len(buf5) == 5:
                    save_palette5_pred_target_png(out_dir, f"{name}_{saved:03d}", list(buf5), filename_suffix="palette5")
                    saved += 5

            if args.save_palette_every and (i % int(args.save_palette_every) == 0) and len(buf5) == 5:
                save_palette5_pred_target_png(out_dir, f"{name}_every{i}", list(buf5), filename_suffix="palette5_every")
            if args.save_every and (i % int(args.save_every) == 0) and len(buf5) == 5:
                save_palette5_pred_target_png(out_dir, f"{name}_save{i}", list(buf5), filename_suffix="palette5_save")

            yield (y[0].detach(), yhat[0].detach())

    res = compute_metrics_streaming(
        pairs=_pairs(),
        device="cuda" if torch.cuda.is_available() else "cpu",
        fid_max_items=args.fid_max_items,
        fid_batch_size=int(args.fid_batch_size),
    )
    metrics = {
        "n": res.n,
        "psnr_mean": res.psnr_mean,
        "psnr_std": res.psnr_std,
        "ssim_mean": res.ssim_mean,
        "ssim_std": res.ssim_std,
        "fid": res.fid,
        "ckpt": args.ckpt,
        "split": args.split,
    }
    print(
        "MODE: EVAL | "
        + " ".join(
            [
                f"PSNR_{args.split}: {metrics['psnr_mean']:.3f}" if metrics["psnr_mean"] is not None else f"PSNR_{args.split}: nan",
                f"SSIM_{args.split}: {metrics['ssim_mean']:.3f}" if metrics["ssim_mean"] is not None else f"SSIM_{args.split}: nan",
                f"FID_{args.split}: {metrics['fid']:.3f}" if metrics["fid"] is not None else f"FID_{args.split}: nan",
            ]
        )
    )
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "metrics.yaml"), "w") as f:
        yaml.safe_dump(metrics, f)


if __name__ == "__main__":
    main()
