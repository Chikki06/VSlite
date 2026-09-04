#!/usr/bin/env python3
"""Visualize CISLNet CBAM attention and input-channel importance."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_THIS_DIR = Path(__file__).resolve().parent
_VSLITE_ROOT = _THIS_DIR.parent
if str(_VSLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_VSLITE_ROOT))

from data import create_dataloader
from models import create_model
from options.test_options import TestOptions
from cislnet.interpret.cbam_hooks import register_cbam_hooks, upsample_spatial_map


def _tensor_to_rgb_uint8(tensor: torch.Tensor, prefer_last_three: bool = False) -> np.ndarray:
    t = tensor.detach().float().cpu()
    if t.ndim == 4:
        t = t[0]
    if t.shape[0] > 3:
        t = t[-3:] if prefer_last_three else t[:3]
    if t.min().item() < 0.0:
        t = (t + 1.0) * 0.5
    t = t.clamp(0.0, 1.0)
    return (t * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()


def _channel_labels(input_nc: int, num_ir: int) -> list[str]:
    labels = [f"IR_{i}" for i in range(num_ir)]
    if input_nc > num_ir:
        labels.extend(["VIS_R", "VIS_G", "VIS_B"][: input_nc - num_ir])
    return labels[:input_nc]


def _overlay_heatmap(rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = heat.astype(np.float32)
    if heat.max() > heat.min():
        heat = (heat - heat.min()) / (heat.max() - heat.min())
    else:
        heat = np.zeros_like(heat)
    cmap = np.stack(
        [
            heat,
            0.25 * (1.0 - np.abs(heat - 0.5) * 2.0),
            1.0 - heat,
        ],
        axis=-1,
    )
    base = rgb.astype(np.float32) / 255.0
    out = (1.0 - alpha) * base + alpha * cmap
    return (out.clip(0.0, 1.0) * 255.0).round().astype(np.uint8)


@torch.no_grad()
def _occlusion_importance(
    net_g: torch.nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    channel_labels: list[str],
) -> dict:
    x = x.unsqueeze(0)
    target = target.unsqueeze(0)
    baseline = net_g(x)
    full_l1 = F.l1_loss(baseline, target).item()
    per_channel = {}
    for c, label in enumerate(channel_labels):
        x_occ = x.clone()
        x_occ[:, c] = 0.0
        pred = net_g(x_occ)
        delta = F.l1_loss(pred, baseline).item()
        pred_l1 = F.l1_loss(pred, target).item()
        per_channel[label] = {
            "delta_l1_vs_full_pred": float(delta),
            "l1_vs_target": float(pred_l1),
            "relative_degradation": float(delta / max(full_l1, 1e-8)),
        }
    ranked = sorted(per_channel.items(), key=lambda kv: kv[1]["delta_l1_vs_full_pred"], reverse=True)
    return {
        "full_pred_l1_vs_target": float(full_l1),
        "per_channel": per_channel,
        "ranked_by_delta_l1": [k for k, _ in ranked],
    }


def main():
    ap = argparse.ArgumentParser(description="CISLNet CBAM + input-channel interpretability")
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument("--checkpoints_dir", type=str, required=True)
    ap.add_argument("--dataroot", type=str, required=True)
    ap.add_argument("--results_dir", type=str, default=None, help="Defaults to checkpoints_dir")
    ap.add_argument("--which_epoch", type=str, default="latest")
    ap.add_argument("--gpu_ids", type=str, default="0")
    ap.add_argument("--input_nc", type=int, default=13)
    ap.add_argument("--num_ir_channels", type=int, default=10)
    ap.add_argument("--ir_channel_indices", type=str, default="")
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--split", type=str, default="test", choices=["train", "test", "val"])
    ap.add_argument("--methods", type=str, default="cbam,occlusion", help="Comma list: cbam,occlusion")
    ap.add_argument("--out_subdir", type=str, default="cbam_vis")
    args = ap.parse_args()

    num_ir = int(args.num_ir_channels)
    ir_idx = str(args.ir_channel_indices or "").strip()
    if ir_idx:
        num_ir = len([x for x in ir_idx.split(",") if x.strip()])

    argv = [
        "visualize_cislnet_cbam.py",
        "--name",
        args.name,
        "--checkpoints_dir",
        args.checkpoints_dir,
        "--dataroot",
        args.dataroot,
        "--which_epoch",
        args.which_epoch,
        "--gpu_ids",
        args.gpu_ids,
        "--input_nc",
        str(args.input_nc),
        "--num_ir_channels",
        str(num_ir),
        "--load_from_opt_file",
        "--use_wandb",
        "false",
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
    opt.num_test = int(args.num_samples)
    opt.batch_size = 1

    results_dir = args.results_dir or args.checkpoints_dir
    run_name = opt.name if str(opt.which_epoch) == "latest" else f"{opt.name}_{opt.which_epoch}"
    out_root = Path(results_dir) / "test" / run_name / args.out_subdir
    spatial_dir = out_root / "spatial_overlays"
    out_root.mkdir(parents=True, exist_ok=True)
    spatial_dir.mkdir(parents=True, exist_ok=True)

    methods = {m.strip().lower() for m in args.methods.split(",") if m.strip()}
    channel_labels = _channel_labels(int(args.input_nc), num_ir)

    dataset = create_dataloader(opt)
    model = create_model(opt)
    model.eval()
    net_g = model.netG
    device = next(net_g.parameters()).device

    capture = register_cbam_hooks(net_g) if "cbam" in methods else None
    summary = {"run": run_name, "methods": sorted(methods), "samples": []}

    n_done = 0
    for data_i in dataset:
        if n_done >= args.num_samples:
            break
        data_i = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data_i.items()}
        x = data_i["label"][0]
        y = data_i["image"][0]
        name = os.path.splitext(os.path.basename(data_i["path"][0]))[0]

        if capture is not None:
            capture.clear()
        pred = net_g(x.unsqueeze(0))[0]

        sample_rec = {"name": name}
        pred_rgb = _tensor_to_rgb_uint8(pred)
        target_rgb = _tensor_to_rgb_uint8(y)
        Image.fromarray(pred_rgb).save(out_root / f"{name}_pred.png")
        Image.fromarray(target_rgb).save(out_root / f"{name}_target.png")

        if "cbam" in methods and capture is not None:
            cbam_rec = {}
            h, w = int(pred.shape[1]), int(pred.shape[2])
            for layer_name, spatial in capture.spatial_maps.items():
                smap = upsample_spatial_map(spatial, (h, w)).numpy()
                overlay = _overlay_heatmap(pred_rgb, smap)
                Image.fromarray(overlay).save(spatial_dir / f"{name}_{layer_name}_spatial.png")
                cbam_rec[layer_name] = {
                    "spatial_mean": float(smap.mean()),
                    "spatial_max": float(smap.max()),
                }
            for layer_name, ch_w in capture.channel_weights.items():
                weights = ch_w[0].numpy()
                top_idx = np.argsort(weights)[::-1][:10].tolist()
                cbam_rec.setdefault(layer_name, {})["top_feature_channels"] = [
                    {"index": int(i), "weight": float(weights[i])} for i in top_idx
                ]
            sample_rec["cbam"] = cbam_rec
            sample_rec["cbam_note"] = (
                "CBAM channel weights index ResNet feature channels, not IR/VIS acquisition bands."
            )

        if "occlusion" in methods:
            occ = _occlusion_importance(net_g, x, y, channel_labels)
            sample_rec["input_channel_occlusion"] = occ

        summary["samples"].append(sample_rec)
        n_done += 1

    if capture is not None:
        capture.remove()

    agg_occlusion = {}
    if "occlusion" in methods:
        for label in channel_labels:
            deltas = [
                s["input_channel_occlusion"]["per_channel"][label]["delta_l1_vs_full_pred"]
                for s in summary["samples"]
                if "input_channel_occlusion" in s
            ]
            if deltas:
                agg_occlusion[label] = {
                    "mean_delta_l1_vs_full_pred": float(np.mean(deltas)),
                    "std_delta_l1_vs_full_pred": float(np.std(deltas)),
                }
        summary["aggregate_input_channel_occlusion"] = dict(
            sorted(
                agg_occlusion.items(),
                key=lambda kv: kv[1]["mean_delta_l1_vs_full_pred"],
                reverse=True,
            )
        )

    importance_path = out_root / "importance.json"
    with open(importance_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote interpretability artifacts to {out_root}")
    print(f"importance.json: {importance_path}")


if __name__ == "__main__":
    main()
