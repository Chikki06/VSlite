#!/usr/bin/env python3
"""
Quantization-aware fine-tuning for CISLNet AttUnetGenerator at 1024px.

Fine-tunes fp32 checkpoint with fake-quant on netG; exports packed int4/int8.
Modular ablations via QUANT_* / QAT_* env (see jobs/_lib/QUANT_ABLATIONS.md).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
_VSLITE_ROOT = _THIS_DIR.parent
if str(_VSLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_VSLITE_ROOT))

import data
import util.util as util
from options.train_options import TrainOptions
from cislnet.quantization.build import build_generator_from_checkpoint
from cislnet.quantization.calibrate import build_calibration_dataloader, collect_activation_scales, calibration_batch_count
from cislnet.quantization.config import QuantAblationConfig
from cislnet.quantization.qat.config import QATTrainConfig
from cislnet.quantization.qat.export import export_qat_generator
from cislnet.quantization.qat.prepare import prepare_qat_generator
from cislnet.quantization.save_load import save_quantized_generator
from cislnet.quantization.schemes import normalize_precision
from util.batch_device import move_to_device


def _build_dataloaders(cfg: QATTrainConfig, opt):
    opt.dataroot = cfg.dataroot
    opt.image_size = cfg.patch_size
    opt.batch_size = cfg.batch_size
    opt.isTrain = True
    opt.phase = "train"
    opt.serial_batches = False
    opt.no_flip = bool(os.environ.get("NO_AUG", "0") == "1")
    opt.no_flip_vert = opt.no_flip
    opt.no_rotate = opt.no_flip
    train_loader = data.create_dataloader(opt)
    val_loader = data.create_dataloader(
        util.copyconf(opt, phase="val", serial_batches=True, isTrain=False, batch_size=1)
    )
    return train_loader, val_loader


def _eval_l1(netG, loader, device, max_batches: int = 50) -> float:
    netG.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break
            batch = move_to_device(batch, device)
            pred = netG(batch["label"])
            y = batch["image"]
            l1 = float(torch.mean(torch.abs(pred - y)).item())
            bs = int(y.shape[0])
            tot += l1 * bs
            n += bs
    netG.train()
    return tot / max(1, n)


def main():
    cfg = QATTrainConfig.from_env()
    if not cfg.dataroot:
        raise ValueError("DATAROOT is required for QAT")

    q = cfg.quant
    gpu_ids = [int(x) for x in cfg.gpu_ids.split(",") if x.strip() != ""]
    device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids and torch.cuda.is_available() else "cpu")

    netG, opt = build_generator_from_checkpoint(cfg.source_ckpt, cfg.opt_dir, gpu_ids=gpu_ids)
    netG = netG.to(device)

    calib_loader = build_calibration_dataloader(
        opt,
        num_patches=min(256, q.calib_num_patches),
        batch_size=q.calib_batch_size,
        image_size=cfg.patch_size,
    )
    max_batches = calibration_batch_count(min(256, q.calib_num_patches), q.calib_batch_size)
    act_scales = collect_activation_scales(
        netG, calib_loader, device, max_batches=max_batches, percentile=q.calib_percentile
    )

    netG = prepare_qat_generator(netG, q, activation_scales=act_scales)
    netG.train()

    train_loader, val_loader = _build_dataloaders(cfg, opt)
    optimizer = torch.optim.AdamW(netG.parameters(), lr=cfg.quant.qat_lr, weight_decay=0.0)

    os.makedirs(cfg.output_dir, exist_ok=True)
    q.save_json(os.path.join(cfg.output_dir, "quant_ablation.json"))
    best_val = float("inf")
    prec = normalize_precision("int4" if q.bits == 4 else "int8")

    print(
        f"QAT bits={q.bits} epochs={q.qat_epochs} lr={q.qat_lr} "
        f"use_act_scale={q.use_act_scale} skip_cbam={q.skip_cbam} patch={cfg.patch_size}"
    )

    for epoch in range(1, q.qat_epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred = netG(batch["label"])
            loss = F.l1_loss(pred, batch["image"])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        val_l1 = _eval_l1(netG, val_loader, device)
        print(f"epoch {epoch}/{q.qat_epochs} train_l1={epoch_loss / max(1, n_batches):.4f} val_l1={val_l1:.4f}")
        if val_l1 < best_val:
            best_val = val_l1
            meta = export_qat_generator(
                netG, q, source_ckpt=cfg.source_ckpt, opt_dir=cfg.opt_dir
            )
            save_quantized_generator(
                netG,
                cfg.output_dir,
                prec,
                meta,
                source_ckpt=cfg.source_ckpt,
                opt_dir=cfg.opt_dir,
            )

    print(f"QAT complete. Best val L1={best_val:.4f}. Artifacts -> {cfg.output_dir}")


if __name__ == "__main__":
    main()
