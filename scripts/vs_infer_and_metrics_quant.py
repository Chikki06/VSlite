#!/usr/bin/env python3
"""Patch eval + PSNR/SSIM/FID for quantized CISLNet generators."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image

_THIS_DIR = Path(__file__).resolve().parent
_VSLITE_ROOT = _THIS_DIR.parent
if str(_VSLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_VSLITE_ROOT))

from data import create_dataloader
from options.test_options import TestOptions
from util.i2i_metrics import compute_metrics_streaming
from cislnet.quantization.inference import run_generator
from cislnet.quantization.save_load import load_quantized_generator
from cislnet.quantization.schemes import normalize_precision


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timing_summary(times_sec: List[float]) -> Dict[str, Any]:
    if not times_sec:
        return {"n": 0, "mean": None, "std": None, "p50": None, "p95": None}
    t = torch.tensor(times_sec, dtype=torch.float64)
    return {
        "n": int(t.numel()),
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()) if t.numel() > 1 else 0.0,
        "p50": float(t.quantile(0.5).item()),
        "p95": float(t.quantile(0.95).item()),
    }


def _tensor_to_png_array(tensor: torch.Tensor, prefer_last_three: bool = False):
    t = tensor.detach().float().cpu()
    if t.ndim == 4:
        t = t[0]
    if t.shape[0] > 3:
        t = t[-3:] if prefer_last_three else t[:3]
    if t.min().item() < 0.0:
        t = (t + 1.0) * 0.5
    t = t.clamp(0.0, 1.0)
    return (t * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant_ckpt", type=str, required=True, help="Path to net_G_{precision}.pt")
    ap.add_argument("--opt_dir", type=str, required=True, help="Dir with opt.pkl (e.g. results/cislnet/c13)")
    ap.add_argument("--source_ckpt", type=str, default="", help="Optional fp32 ckpt for architecture init")
    ap.add_argument("--precision", type=str, default="", choices=["", "fp16", "fp8", "int4", "int8", "int32", "fp4", "fp32"])
    ap.add_argument("--dataroot", type=str, required=True)
    ap.add_argument("--results_dir", type=str, default="./results")
    ap.add_argument("--run", type=str, default="c13", help="Run tag for output layout")
    ap.add_argument("--gpu_ids", type=str, default="0")
    ap.add_argument("--split", type=str, default="test", choices=["train", "test"])
    ap.add_argument("--num_test", type=int, default=0, help="0 = full split")
    ap.add_argument("--input_nc", type=int, default=13)
    ap.add_argument("--num_ir_channels", type=int, default=10)
    ap.add_argument("--ir_channel_indices", type=str, default="")
    ap.add_argument("--fid_max_items", type=int, default=None)
    ap.add_argument("--fid_batch_size", type=int, default=16)
    ap.add_argument("--save_triplets", action="store_true")
    args = ap.parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip() != "" and int(x.strip()) >= 0]
    if not torch.cuda.is_available():
        gpu_ids = []
    if gpu_ids and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    source_ckpt = args.source_ckpt or None
    netG, meta, opt = load_quantized_generator(
        args.quant_ckpt,
        args.opt_dir,
        gpu_ids=gpu_ids,
        source_ckpt=source_ckpt,
    )
    precision = normalize_precision(args.precision or meta.get("precision", "int32"))

    num_ir = int(args.num_ir_channels)
    ir_idx = str(args.ir_channel_indices or "").strip()
    if ir_idx:
        num_ir = len([x for x in ir_idx.split(",") if x.strip()])

    num_test_for_parser = int(args.num_test) if args.num_test > 0 else 10**9
    argv = [
        "vs_infer_and_metrics_quant.py",
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
        args.results_dir,
        "--gpu_ids",
        args.gpu_ids,
        "--num_test",
        str(num_test_for_parser),
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
    eval_opt = TestOptions().parse()
    eval_opt.isTrain = False
    eval_opt.phase = str(args.split)
    eval_opt.serial_batches = True
    eval_opt.no_flip = True
    eval_opt.no_flip_vert = True
    eval_opt.no_rotate = True
    eval_opt.num_test = float("inf") if args.num_test <= 0 else int(args.num_test)
    eval_opt.test_includes_val = bool(args.split == "test")
    infer_bs = int(os.environ.get("VSLITE_INFER_BS", "16"))
    eval_opt.batch_size = max(1, infer_bs)

    dataset = create_dataloader(eval_opt)
    device = next(netG.parameters()).device

    prec_tag = str(precision).lower()
    patch_eval_leaf = "patch_eval_test" if args.split == "test" else "patch_eval_train"
    out_dir = os.path.join(args.results_dir, "quant", args.run, "test", prec_tag, patch_eval_leaf)
    input_dir = os.path.join(out_dir, "input")
    target_dir = os.path.join(out_dir, "target")
    pred_dir = os.path.join(out_dir, "pred")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    warmup_batches = int(os.environ.get("VSLITE_INFER_WARMUP_BATCHES", "2"))
    batch_times_sec: List[float] = []
    per_patch_times_sec: List[float] = []

    def _run_forward(data_i: dict) -> torch.Tensor:
        _sync_device(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = run_generator(netG, data_i["label"], precision=precision)
        _sync_device(device)
        batch_times_sec.append(time.perf_counter() - t0)
        return out

    data_iter = iter(dataset)
    for _ in range(max(0, warmup_batches)):
        try:
            warm = next(data_iter)
        except StopIteration:
            break
        warm = {k: v.to(device) if torch.is_tensor(v) else v for k, v in warm.items()}
        with torch.no_grad():
            run_generator(netG, warm["label"], precision=precision)
        _sync_device(device)

    def _pairs():
        nonlocal data_iter
        n_seen = 0
        while True:
            if n_seen >= eval_opt.num_test:
                break
            try:
                data_i = next(data_iter)
            except StopIteration:
                break
            data_i = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data_i.items()}
            generated = _run_forward(data_i)
            bs = int(generated.shape[0])
            patch_dt = batch_times_sec[-1] / max(bs, 1)
            for b in range(bs):
                if n_seen >= eval_opt.num_test:
                    break
                name = os.path.basename(data_i["path"][b])
                name = os.path.splitext(name)[0]
                x = data_i["label"][b]
                y = data_i["image"][b]
                yhat = generated[b]
                per_patch_times_sec.append(patch_dt)

                if args.save_triplets:
                    Image.fromarray(_tensor_to_png_array(x)).save(os.path.join(input_dir, f"{name}.png"))
                    Image.fromarray(_tensor_to_png_array(y)).save(os.path.join(target_dir, f"{name}.png"))
                    Image.fromarray(_tensor_to_png_array(yhat)).save(os.path.join(pred_dir, f"{name}.png"))

                n_seen += 1
                yield (y.detach(), yhat.detach())

    res = compute_metrics_streaming(
        pairs=_pairs(),
        device="cuda" if torch.cuda.is_available() else "cpu",
        fid_max_items=args.fid_max_items,
        fid_batch_size=int(args.fid_batch_size),
    )

    per_patch_summary = _timing_summary(per_patch_times_sec)
    per_batch_summary = _timing_summary(batch_times_sec)
    metrics = {
        "n": res.n,
        "psnr_mean": res.psnr_mean,
        "psnr_std": res.psnr_std,
        "ssim_mean": res.ssim_mean,
        "ssim_std": res.ssim_std,
        "fid": res.fid,
        "lpips_convnext_tcga_mean": res.lpips_convnext_tcga_mean,
        "lpips_convnext_tcga_std": res.lpips_convnext_tcga_std,
        "precision": prec_tag,
        "quant_ckpt": args.quant_ckpt,
        "scheme": meta.get("scheme"),
        "pred_dir": pred_dir,
        "target_dir": target_dir,
        "split": str(args.split),
        "inference_batch_size": int(eval_opt.batch_size),
        "inference_warmup_batches": warmup_batches,
        "inference_time_per_patch_mean_sec": per_patch_summary["mean"],
        "inference_time_per_patch_std_sec": per_patch_summary["std"],
        "inference_time_per_patch_p50_sec": per_patch_summary["p50"],
        "inference_time_per_patch_p95_sec": per_patch_summary["p95"],
        "inference_time_per_patch_n": per_patch_summary["n"],
        "inference_time_per_sample_mean_sec": per_batch_summary["mean"],
        "inference_time_per_sample_std_sec": per_batch_summary["std"],
        "inference_time_per_sample_p50_sec": per_batch_summary["p50"],
        "inference_time_per_sample_p95_sec": per_batch_summary["p95"],
        "inference_time_per_sample_n": per_batch_summary["n"],
    }
    timing_detail = {
        "precision": prec_tag,
        "split": str(args.split),
        "inference_batch_size": int(eval_opt.batch_size),
        "per_patch": per_patch_summary,
        "per_sample": per_batch_summary,
        "per_patch_sec": per_patch_times_sec,
        "per_sample_sec": batch_times_sec,
    }
    print(
        "MODE: EVAL_QUANT | "
        + " ".join(
            [
                f"precision: {prec_tag}",
                f"PSNR: {metrics['psnr_mean']:.3f}" if metrics["psnr_mean"] is not None else "PSNR: nan",
                f"SSIM: {metrics['ssim_mean']:.3f}" if metrics["ssim_mean"] is not None else "SSIM: nan",
                f"FID: {metrics['fid']:.3f}" if metrics["fid"] is not None else "FID: nan",
                (
                    f"LPIPS_convnext_tcga_test: {metrics['lpips_convnext_tcga_mean']:.3f}"
                    if metrics["lpips_convnext_tcga_mean"] is not None
                    else "LPIPS_convnext_tcga_test: nan"
                ),
                (
                    f"patch_infer_ms: {metrics['inference_time_per_patch_mean_sec'] * 1000:.2f}"
                    if metrics["inference_time_per_patch_mean_sec"] is not None
                    else "patch_infer_ms: nan"
                ),
                (
                    f"sample_infer_ms: {metrics['inference_time_per_sample_mean_sec'] * 1000:.2f}"
                    if metrics["inference_time_per_sample_mean_sec"] is not None
                    else "sample_infer_ms: nan"
                ),
            ]
        )
    )
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "inference_timing.json"), "w") as f:
        json.dump(timing_detail, f, indent=2)
    try:
        import yaml

        with open(os.path.join(out_dir, "metrics.yaml"), "w") as f:
            yaml.safe_dump(metrics, f)
    except Exception:
        pass


if __name__ == "__main__":
    main()
