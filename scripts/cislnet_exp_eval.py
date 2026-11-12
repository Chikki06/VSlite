#!/usr/bin/env python3
"""CISLNet experiment eval: standard metrics + VGG-LPIPS + GPU latency → tracking.csv.

Does not replace scripts/vs_infer_and_metrics.py; experiment jobs call this instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_THIS_DIR = Path(__file__).resolve().parent
_VSLITE_ROOT = _THIS_DIR.parent
if str(_VSLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_VSLITE_ROOT))

from SwinGAN.utils.sample_saver import save_palette5_pred_target_png
from data import create_dataloader
from models import create_model
from options.test_options import TestOptions
from util.exp_metrics import build_lpips_vgg, lpips_vgg_pair, measure_latency_ms
from util.i2i_metrics import compute_metrics_streaming
from cislnet.region_eval import tiled_generator_forward, validate_region_tiling

TRACKING_CSV = _VSLITE_ROOT / "results" / "cislnet" / "experiments" / "tracking.csv"
TRACKING_FIELDS = [
    "run",
    "group",
    "channels",
    "attention",
    "backbone",
    "decoder_conv",
    "color",
    "epoch",
    "psnr",
    "ssim",
    "fid",
    "lpips_convnext_tcga",
    "lpips_vgg",
    "latency_ms_mean",
    "latency_ms_std",
    "ckpt",
    "notes",
]


def _tensor_to_png_array(tensor: torch.Tensor, prefer_last_three: bool = False) -> np.ndarray:
    t = tensor.detach().float().cpu()
    if t.ndim == 4:
        t = t[0]
    if t.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got {tuple(t.shape)}")
    if t.shape[0] > 3:
        t = t[-3:] if prefer_last_three else t[:3]
    if t.min().item() < 0.0:
        t = (t + 1.0) * 0.5
    t = t.clamp(0.0, 1.0)
    return (t * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()


def _to_display_srgb(t: torch.Tensor, he_color_space: str) -> torch.Tensor:
    if str(he_color_space).lower() != "linear":
        return t
    from util.color_space import linear_to_srgb

    return linear_to_srgb(t, minus1_1=True)


def append_tracking_row(row: dict) -> None:
    TRACKING_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not TRACKING_CSV.is_file()
    with open(TRACKING_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRACKING_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in TRACKING_FIELDS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument("--checkpoints_dir", type=str, required=True)
    ap.add_argument("--dataroot", type=str, required=True)
    ap.add_argument("--dataset_mode", type=str, default="VS")
    ap.add_argument("--model", type=str, default="contrastive_pix2pix")
    ap.add_argument("--which_epoch", type=str, default="latest")
    ap.add_argument("--results_dir", type=str, default="./results")
    ap.add_argument("--gpu_ids", type=str, default="0")
    ap.add_argument("--split", type=str, default="test", choices=["train", "test"])
    ap.add_argument("--num_test", type=int, default=0)
    ap.add_argument("--input_nc", type=int, default=13)
    ap.add_argument("--num_ir_channels", type=int, default=10)
    ap.add_argument("--ir_channel_indices", type=str, default="")
    ap.add_argument("--fid_max_items", type=int, default=None)
    ap.add_argument("--fid_batch_size", type=int, default=16)
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--save_triplets", action="store_true")
    ap.add_argument("--region_eval_size", type=int, default=0)
    ap.add_argument("--save_region_triplets", action="store_true")
    ap.add_argument("--test_list_path", type=str, default="")
    ap.add_argument("--val_list_path", type=str, default="")
    ap.add_argument("--test_includes_val", type=int, default=None)
    ap.add_argument("--results_name", type=str, default="")
    # Tracking metadata
    ap.add_argument("--track_run", type=str, default="")
    ap.add_argument("--track_group", type=str, default="exp")
    ap.add_argument("--track_notes", type=str, default="")
    ap.add_argument("--latency_warmup", type=int, default=10)
    ap.add_argument("--latency_timed", type=int, default=100)
    ap.add_argument("--skip_full_metrics", action="store_true", help="Latency + VGG only (debug).")
    args = ap.parse_args()

    num_ir_channels = int(args.num_ir_channels)
    ir_idx = str(args.ir_channel_indices or "").strip()
    if ir_idx:
        idx = [int(x.strip()) for x in ir_idx.split(",") if x.strip() != ""]
        num_ir_channels = len(idx)

    num_test_for_parser = int(args.num_test) if args.num_test > 0 else 10**9
    argv = [
        "cislnet_exp_eval.py",
        "--name", args.name,
        "--checkpoints_dir", args.checkpoints_dir,
        "--dataroot", args.dataroot,
        "--dataset_mode", args.dataset_mode,
        "--model", args.model,
        "--which_epoch", args.which_epoch,
        "--results_dir", args.results_dir,
        "--gpu_ids", args.gpu_ids,
        "--num_test", str(num_test_for_parser),
        "--input_nc", str(args.input_nc),
        "--num_ir_channels", str(num_ir_channels),
        "--load_from_opt_file",
        "--use_wandb", "false",
    ]
    if ir_idx:
        argv += ["--ir_channel_indices", ir_idx]
    sys.argv = argv
    opt = TestOptions().parse()
    opt.isTrain = False
    opt.phase = str(args.split)
    opt.serial_batches = True
    opt.no_flip = True
    opt.no_flip_vert = True
    opt.no_rotate = True
    opt.num_test = float("inf") if args.num_test <= 0 else int(args.num_test)
    if args.test_includes_val is None:
        opt.test_includes_val = bool(args.split == "test")
    else:
        opt.test_includes_val = bool(int(args.test_includes_val))
    if str(args.test_list_path or "").strip():
        opt.test_list_path = str(args.test_list_path).strip()
    if str(args.val_list_path or "").strip():
        opt.val_list_path = str(args.val_list_path).strip()

    infer_bs = int(os.environ.get("VSLITE_INFER_BS", "16"))
    opt.batch_size = max(1, infer_bs)

    model_tile_size = int(getattr(opt, "image_size", 1024))
    region_eval_size = int(args.region_eval_size or 0)
    use_region_eval = region_eval_size > 0
    if use_region_eval:
        validate_region_tiling(region_eval_size, model_tile_size)
        opt.image_size = region_eval_size

    dataset = create_dataloader(opt)
    model = create_model(opt)
    model.eval()
    net_g = getattr(model, "netG", None)

    results_leaf = str(args.results_name or "").strip() or str(opt.name)
    run_name = results_leaf if str(opt.which_epoch) == "latest" else f"{results_leaf}_{opt.which_epoch}"
    patch_eval_leaf = "patch_eval_test" if args.split == "test" else "patch_eval_train"
    out_dir = os.path.join(opt.results_dir, "test", run_name, patch_eval_leaf)
    input_dir = os.path.join(out_dir, "input")
    target_dir = os.path.join(out_dir, "target")
    pred_dir = os.path.join(out_dir, "pred")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    he_space = str(getattr(opt, "he_color_space", "srgb"))
    save_triplets = bool(args.save_triplets or (use_region_eval and args.save_region_triplets))

    # --- Latency (batch=1, fixed spatial size from training opt) ---
    latency = {"latency_ms_mean": None, "latency_ms_std": None}
    if net_g is not None:
        spatial = int(getattr(opt, "image_size", 1024))
        if use_region_eval:
            spatial = int(model_tile_size)
        dummy = torch.randn(1, int(args.input_nc), spatial, spatial, device=device)

        def _fwd():
            with torch.no_grad():
                _ = net_g(dummy)

        latency = measure_latency_ms(
            _fwd,
            device=device if isinstance(device, torch.device) else torch.device(device),
            warmup=int(args.latency_warmup),
            timed=int(args.latency_timed),
        )

    # --- Streaming metrics + VGG LPIPS ---
    lpips_mod = build_lpips_vgg(device)
    vgg_vals: list[float] = []

    def _pairs():
        n_seen = 0
        buf5: list = []
        for data_i in dataset:
            if n_seen >= opt.num_test:
                break
            data_i = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data_i.items()}
            if use_region_eval:
                bs = int(data_i["label"].shape[0])
                generated_list = []
                for b in range(bs):
                    generated_list.append(
                        tiled_generator_forward(net_g, data_i["label"][b], model_tile_size)
                    )
                generated = torch.stack(generated_list, dim=0)
            else:
                with torch.no_grad():
                    generated = model(data_i, mode="inference")
            bs = int(generated.shape[0])
            for b in range(bs):
                if n_seen >= opt.num_test:
                    break
                name = os.path.splitext(os.path.basename(data_i["path"][b]))[0]
                x = data_i["label"][b]
                y = data_i["image"][b]
                yhat = generated[b]
                y_m = _to_display_srgb(y, he_space)
                yhat_m = _to_display_srgb(yhat, he_space)
                try:
                    v = float(lpips_vgg_pair(lpips_mod, yhat_m.unsqueeze(0), y_m.unsqueeze(0)).mean().item())
                    vgg_vals.append(v)
                except Exception as e:
                    print(f"[warn] lpips_vgg failed on {name}: {e}")

                buf5.append((name, y_m.detach().cpu(), yhat_m.detach().cpu()))
                if len(buf5) > 5:
                    buf5.pop(0)
                if args.save_every and (n_seen % int(args.save_every) == 0) and len(buf5) == 5:
                    save_palette5_pred_target_png(
                        out_dir, f"{name}_save{n_seen}", list(buf5), filename_suffix="palette5_save"
                    )
                if save_triplets:
                    Image.fromarray(_tensor_to_png_array(x)).save(os.path.join(input_dir, f"{name}.png"))
                    Image.fromarray(_tensor_to_png_array(y_m)).save(os.path.join(target_dir, f"{name}.png"))
                    Image.fromarray(_tensor_to_png_array(yhat_m)).save(os.path.join(pred_dir, f"{name}.png"))
                n_seen += 1
                if not args.skip_full_metrics:
                    yield (y_m.detach(), yhat_m.detach())

    if args.skip_full_metrics:
        # Drain pairs for VGG only (limited by num_test / latency already done).
        for _ in _pairs():
            pass
        res_n = len(vgg_vals)
        metrics = {
            "n": res_n,
            "psnr_mean": None,
            "psnr_std": None,
            "ssim_mean": None,
            "ssim_std": None,
            "fid": None,
            "lpips_convnext_tcga_mean": None,
            "lpips_convnext_tcga_std": None,
        }
    else:
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
            "lpips_convnext_tcga_mean": res.lpips_convnext_tcga_mean,
            "lpips_convnext_tcga_std": res.lpips_convnext_tcga_std,
        }

    lpips_vgg_mean = float(np.mean(vgg_vals)) if vgg_vals else None
    lpips_vgg_std = float(np.std(vgg_vals)) if vgg_vals else None
    metrics.update(
        {
            "pred_dir": pred_dir,
            "target_dir": target_dir,
            "split": str(args.split),
            "lpips_vgg_mean": lpips_vgg_mean,
            "lpips_vgg_std": lpips_vgg_std,
            "latency_ms_mean": latency.get("latency_ms_mean"),
            "latency_ms_std": latency.get("latency_ms_std"),
            "latency_warmup": latency.get("latency_warmup"),
            "latency_timed_n": latency.get("latency_timed_n"),
            "he_color_space": he_space,
        }
    )

    print(
        "MODE: EXP_EVAL | "
        + " ".join(
            [
                f"PSNR: {metrics['psnr_mean']:.3f}" if metrics["psnr_mean"] is not None else "PSNR: nan",
                f"SSIM: {metrics['ssim_mean']:.3f}" if metrics["ssim_mean"] is not None else "SSIM: nan",
                f"FID: {metrics['fid']:.3f}" if metrics["fid"] is not None else "FID: nan",
                (
                    f"LPIPS_cn: {metrics['lpips_convnext_tcga_mean']:.4f}"
                    if metrics["lpips_convnext_tcga_mean"] is not None
                    else "LPIPS_cn: nan"
                ),
                f"LPIPS_vgg: {lpips_vgg_mean:.4f}" if lpips_vgg_mean is not None else "LPIPS_vgg: nan",
                (
                    f"latency_ms: {metrics['latency_ms_mean']:.2f}"
                    if metrics["latency_ms_mean"] is not None
                    else "latency_ms: nan"
                ),
            ]
        )
    )

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    try:
        import yaml

        with open(os.path.join(out_dir, "metrics.yaml"), "w") as f:
            yaml.safe_dump(metrics, f)
    except Exception:
        pass

    track_run = str(args.track_run or "").strip() or results_leaf
    append_tracking_row(
        {
            "run": track_run,
            "group": args.track_group,
            "channels": int(args.input_nc),
            "attention": str(getattr(opt, "attention_type", "cbam")),
            "backbone": str(getattr(opt, "encoder_backbone", "resnet50")),
            "decoder_conv": str(getattr(opt, "decoder_conv", "standard")),
            "color": he_space,
            "epoch": str(args.which_epoch),
            "psnr": metrics.get("psnr_mean"),
            "ssim": metrics.get("ssim_mean"),
            "fid": metrics.get("fid"),
            "lpips_convnext_tcga": metrics.get("lpips_convnext_tcga_mean"),
            "lpips_vgg": lpips_vgg_mean,
            "latency_ms_mean": metrics.get("latency_ms_mean"),
            "latency_ms_std": metrics.get("latency_ms_std"),
            "ckpt": os.path.join(args.checkpoints_dir, args.name, f"{args.which_epoch}_net_G.pth"),
            "notes": args.track_notes,
        }
    )
    print(f"appended tracking row → {TRACKING_CSV}")


if __name__ == "__main__":
    main()
