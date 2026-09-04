#!/usr/bin/env python3
"""Visual side-by-side comparison of fp32 vs quantized CISLNet outputs."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

_THIS_DIR = Path(__file__).resolve().parent
_VSLITE_ROOT = _THIS_DIR.parent
if str(_VSLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_VSLITE_ROOT))

from data import create_dataloader
from options.test_options import TestOptions
from cislnet.quantization.build import build_generator_from_checkpoint
from cislnet.quantization.inference import run_generator
from cislnet.quantization.save_load import load_quantized_generator


def _tensor_to_rgb_uint8(tensor: torch.Tensor) -> np.ndarray:
    t = tensor.detach().float().cpu()
    if t.ndim == 4:
        t = t[0]
    if t.shape[0] > 3:
        t = t[:3]
    if t.min().item() < 0.0:
        t = (t + 1.0) * 0.5
    t = t.clamp(0.0, 1.0)
    return (t * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()


def _label_bar(width: int, text: str, height: int = 28) -> Image.Image:
    bar = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(bar)
    draw.text((8, 6), text, fill=(240, 240, 240))
    return bar


def _make_grid_row(images: list[np.ndarray], labels: list[str], thumb: int) -> Image.Image:
    cols = []
    for arr, label in zip(images, labels):
        im = Image.fromarray(arr).resize((thumb, thumb), Image.BILINEAR)
        w, h = im.size
        col = Image.new("RGB", (w, h + 28))
        col.paste(_label_bar(w, label), (0, 0))
        col.paste(im, (0, 28))
        cols.append(col)
    total_w = sum(c.size[0] for c in cols)
    max_h = max(c.size[1] for c in cols)
    row = Image.new("RGB", (total_w, max_h))
    x = 0
    for c in cols:
        row.paste(c, (x, 0))
        x += c.size[0]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_ckpt", type=str, required=True)
    ap.add_argument("--opt_dir", type=str, required=True)
    ap.add_argument("--quant_root", type=str, required=True, help="e.g. results/cislnet/quant/c13")
    ap.add_argument("--dataroot", type=str, required=True)
    ap.add_argument("--run", type=str, default="c13")
    ap.add_argument("--num_patches", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--thumb_size", type=int, default=256, help="Thumbnail px per column")
    ap.add_argument("--gpu_ids", type=str, default="0")
    ap.add_argument("--input_nc", type=int, default=13)
    ap.add_argument("--num_ir_channels", type=int, default=10)
    ap.add_argument("--precisions", type=str, default="int32,fp16,fp8,int4,int8")
    args = ap.parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip() != ""]
    device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    precisions = [p.strip().lower() for p in args.precisions.split(",") if p.strip()]
    models = {}

    int32_path = os.path.join(args.quant_root, "int32", "net_G_int32.pt")
    if os.path.isfile(int32_path):
        net_int32, meta, _ = load_quantized_generator(
            int32_path,
            args.opt_dir,
            gpu_ids=gpu_ids,
            source_ckpt=args.source_ckpt,
        )
        models["int32"] = (net_int32.to(device), meta.get("precision", "int32"))
    else:
        net_fp32, _opt = build_generator_from_checkpoint(args.source_ckpt, args.opt_dir, gpu_ids=gpu_ids)
        models["int32"] = (net_fp32.to(device), "int32")

    for prec in precisions:
        if prec in ("fp32", "int32"):
            continue
        qpath = os.path.join(args.quant_root, prec, f"net_G_{prec}.pt")
        if not os.path.isfile(qpath) and prec == "int4":
            legacy = os.path.join(args.quant_root, "fp4", "net_G_fp4.pt")
            if os.path.isfile(legacy):
                qpath = legacy
        if not os.path.isfile(qpath):
            print(f"Skipping missing {prec}: {qpath}")
            continue
        net_q, meta, _ = load_quantized_generator(
            qpath,
            args.opt_dir,
            gpu_ids=gpu_ids,
            source_ckpt=args.source_ckpt,
        )
        models[prec] = (net_q.to(device), meta.get("precision", prec))

    argv = [
        "quant_visual_compare.py",
        "--name",
        args.run,
        "--checkpoints_dir",
        args.opt_dir,
        "--dataroot",
        args.dataroot,
        "--dataset_mode",
        "VS",
        "--model",
        "contrastive_pix2pix",
        "--which_epoch",
        "latest",
        "--results_dir",
        "./results",
        "--gpu_ids",
        args.gpu_ids,
        "--num_test",
        str(max(args.num_patches * 4, args.num_patches)),
        "--input_nc",
        str(args.input_nc),
        "--num_ir_channels",
        str(args.num_ir_channels),
        "--load_from_opt_file",
        "--use_wandb",
        "false",
    ]
    sys.argv = argv
    eval_opt = TestOptions().parse()
    eval_opt.isTrain = False
    eval_opt.phase = "test"
    eval_opt.serial_batches = True
    eval_opt.no_flip = True
    eval_opt.no_flip_vert = True
    eval_opt.no_rotate = True
    eval_opt.test_includes_val = True
    eval_opt.batch_size = 1
    eval_opt.num_test = args.num_patches * 8

    dataset = create_dataloader(eval_opt)
    indices = list(range(min(len(dataset), args.num_patches * 8)))
    random.seed(args.seed)
    random.shuffle(indices)
    indices = sorted(indices[: args.num_patches])

    out_dir = os.path.join(args.quant_root, "visual_compare", "grids")
    os.makedirs(out_dir, exist_ok=True)

    row_images = []
    display_precs = [p for p in ["int32", "fp16", "fp8", "int4", "int8"] if p in models]
    col_labels = ["input"] + display_precs + ["target"]

    for idx in indices:
        data_i = dataset[idx]
        x = data_i["label"].unsqueeze(0).to(device)
        y = data_i["image"]
        name = os.path.splitext(os.path.basename(data_i["path"]))[0]

        panels = [_tensor_to_rgb_uint8(x[0])]
        with torch.no_grad():
            for prec in display_precs:
                net, ptag = models[prec]
                pred = run_generator(net, x, precision=ptag)
                panels.append(_tensor_to_rgb_uint8(pred[0]))
        panels.append(_tensor_to_rgb_uint8(y))

        grid = _make_grid_row(panels, col_labels, args.thumb_size)
        out_path = os.path.join(out_dir, f"patch_{name}.png")
        grid.save(out_path)
        row_images.append(np.array(grid))

    if row_images:
        # Summary montage: stack up to 4 rows per strip
        montage_parts = []
        strip = []
        for i, arr in enumerate(row_images):
            strip.append(Image.fromarray(arr))
            if len(strip) == 4 or i == len(row_images) - 1:
                w = max(im.size[0] for im in strip)
                h = sum(im.size[1] for im in strip)
                canvas = Image.new("RGB", (w, h), (0, 0, 0))
                yoff = 0
                for im in strip:
                    canvas.paste(im, (0, yoff))
                    yoff += im.size[1]
                montage_parts.append(canvas)
                strip = []
        montage = montage_parts[0] if len(montage_parts) == 1 else montage_parts[0]
        montage_path = os.path.join(args.quant_root, "visual_compare", "summary_montage.png")
        montage.save(montage_path)
        print(f"Wrote {len(indices)} grids to {out_dir}")
        print(f"Summary: {montage_path}")


if __name__ == "__main__":
    main()
