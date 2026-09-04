#!/usr/bin/env python3
"""Post-training quantization for CISLNet (AttUnetGenerator)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_VSLITE_ROOT = _THIS_DIR.parent
if str(_VSLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_VSLITE_ROOT))

from cislnet.quantization.build import build_generator_from_checkpoint
from cislnet.quantization.calibrate import build_calibration_dataloader
from cislnet.quantization.config import QuantAblationConfig
from cislnet.quantization.save_load import save_quantized_generator
from cislnet.quantization.schemes import apply_precision, normalize_precision

INTEGER_PRECS = frozenset({"int4", "int8"})
FLOAT_PRECS = frozenset({"fp8"})
CALIBRATED_PRECS = INTEGER_PRECS | FLOAT_PRECS


def main():
    ap = argparse.ArgumentParser(description="Quantize CISLNet generator checkpoint.")
    ap.add_argument("--source_ckpt", type=str, required=True, help="Path to fp32 latest_net_G.pth")
    ap.add_argument("--opt_dir", type=str, required=True, help="Checkpoint dir containing opt.pkl")
    ap.add_argument(
        "--precision",
        type=str,
        required=True,
        choices=["fp16", "fp8", "int4", "int8", "int32", "fp4", "fp32"],
        help="fp4/fp32 are legacy aliases for int4/int32",
    )
    ap.add_argument("--out_dir", type=str, required=True, help="Output dir for quantized artifact")
    ap.add_argument("--dataroot", type=str, default="", help="Calibration dataroot (required for int4/int8/fp8)")
    ap.add_argument("--calib_num_patches", type=int, default=512)
    ap.add_argument("--calib_batch_size", type=int, default=4)
    ap.add_argument("--fp8_static", action="store_true", help="Try torchao static fp8 (Conv2d may fail)")
    ap.add_argument(
        "--fp8_native",
        type=lambda x: str(x).lower() not in ("0", "false", "no"),
        default=None,
        help="Native e4m3fn modules: 1/on (default from QUANT_FP8_NATIVE) or 0/off",
    )
    ap.add_argument("--use_act_scale", action="store_true", help="Merge activation scales into weight quant")
    ap.add_argument("--calib_percentile", type=float, default=None, help="Activation/weight clip percentile")
    ap.add_argument("--skip_cbam", action="store_true", help="Skip quantizing CBAM layers")
    ap.add_argument("--fpga_export", action="store_true", help="Write fpga_manifest.json for int8/int4")
    ap.add_argument("--gpu_ids", type=str, default="0")
    args = ap.parse_args()

    prec = normalize_precision(args.precision)
    config = QuantAblationConfig.from_env()
    config.calib_num_patches = int(args.calib_num_patches)
    config.calib_batch_size = int(args.calib_batch_size)
    if args.use_act_scale:
        config.use_act_scale = True
    if args.skip_cbam:
        config.skip_cbam = True
    if args.calib_percentile is not None:
        config.calib_percentile = float(args.calib_percentile)
    if args.fp8_native is not None:
        config.fp8_native = bool(args.fp8_native)
    if args.fpga_export:
        config.fpga_export = True
        config.extra["fpga_out_dir"] = str(Path(args.out_dir) / "fpga_export")

    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip() != ""]
    if not gpu_ids:
        gpu_ids = []
    device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids and torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    netG, opt = build_generator_from_checkpoint(args.source_ckpt, args.opt_dir, gpu_ids=gpu_ids)
    netG = netG.to(device)

    calib_loader = None
    if prec in CALIBRATED_PRECS:
        dataroot = args.dataroot or getattr(opt, "dataroot", "")
        if not dataroot:
            raise ValueError(f"--dataroot is required for {prec} calibration")
        opt.dataroot = dataroot
        calib_loader = build_calibration_dataloader(
            opt,
            num_patches=config.calib_num_patches,
            batch_size=config.calib_batch_size,
        )

    fp8_static = args.fp8_static or prec == "fp8"

    meta = apply_precision(
        netG,
        prec,
        calib_loader=calib_loader,
        device=device,
        calib_num_patches=config.calib_num_patches,
        calib_batch_size=config.calib_batch_size,
        fp8_static=fp8_static,
        config=config,
    )

    paths = save_quantized_generator(
        netG,
        args.out_dir,
        prec,
        meta,
        source_ckpt=args.source_ckpt,
        opt_dir=args.opt_dir,
    )
    config.save_json(str(Path(args.out_dir) / "quant_ablation.json"))
    print(f"Saved quantized {prec} artifact to {paths['weights']}")
    print(f"Metadata: {paths['meta']}")
    print(f"Scheme: {meta.get('scheme')} storage: {meta.get('storage')}")


if __name__ == "__main__":
    main()
