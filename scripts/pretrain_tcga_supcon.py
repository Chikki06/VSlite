#!/usr/bin/env python
"""Supervised-contrastive pretraining of TCGA-UT encoders (ConvNeXt-T / Swin-T).

Retrains the repo's TCGA perceptual-loss encoder lineage
(``tcga_tl_checkpoints/convnext_t_dp_spv_supcon_1e-5``) on the full TCGA-UT
dataset. ConvNeXt defaults below replace the collapsed SGD peak-LR=1.0 recipe
(see ``enc_convnext_t_supcon/loss_log.txt`` epochs 41-45 locking at ln(1023)).

Recipes:
  convnext_t — AdamW peak LR 5e-4 (fixed, no batch scaling), wd 0.05 with
      LN/BN/bias/LayerScale gamma exempt, 5-epoch linear warmup then cosine
      to 0, temperature 0.1, projection head Linear(768)->SyncBN->GELU->Linear(128)
      then L2-normalize, 2 views/sample (Simaugment). Collapse kill switch aborts
      if loss stays at ln(2N-1) for 2 epochs after epoch 5.
  swin_t — transformer-adapted SupCon (CTransPath/SRCL; MoCo-v3 tiny-Swin):
      AdamW LR 1.5e-4 (fixed), wd 0.05, 40 warmup epochs, cosine to 0,
      temperature 0.1, same head/views.

Checkpoints (results/tcga_encoder/<run>/):
  ckpt_latest.pth  full resume state   tcga_encoder.pt  bare encoder state_dict
  ckpt_epoch{N}.pth periodic snapshots  opt.txt / loss_log.txt / iter.txt
TensorBoard mirror: results/unified/tcga_encoder/<run>/

Usage (via jobs/tcga_encoder/train/train_supcon.sh):
  python scripts/pretrain_tcga_supcon.py --arch convnext_t \
      --manifest dataset/tcga_ut/manifest.csv --extract_dir dataset/tcga_ut/extracted \
      --run_name enc_convnext_t_supcon --epochs 350 --batch_size 512
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.tcga_ut import (  # noqa: E402
    CollapseKillSwitch,
    SupConLoss,
    TCGAUTDataset,
    TwoCropTransform,
    build_backbone,
    build_eval_transforms,
    build_supcon_projection_head,
    build_supcon_transforms,
    knn_accuracy,
    load_run_state,
    read_manifest,
    save_run_state,
    seed_everything,
    single_collate,
    split_decay_param_groups,
    supcon_collate,
    warmup_cosine_lambda,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default=os.environ.get("ARCH", "convnext_t"),
                   choices=["convnext_t", "swin_t"])
    p.add_argument("--optimizer", default=None, choices=["sgd", "adamw"],
                   help="default: adamw for both arches")
    p.add_argument("--base_lr", type=float, default=None,
                   help="convnext_t adamw: 5e-4; swin_t adamw: 1.5e-4; "
                        "sgd: 0.5 scaled by bs/256")
    p.add_argument("--weight_decay", type=float, default=None,
                   help="adamw: 0.05; sgd: 1e-4 (LN/BN/bias/gamma always wd=0)")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("--head_hidden", type=int, default=768)
    p.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", 100)))
    p.add_argument("--warmup_epochs", type=int, default=None,
                   help="convnext_t: 5; swin_t: 40; sgd: 10 when bs>256 else 0")
    p.add_argument("--batch_size", type=int, default=int(os.environ.get("BATCH_SIZE", 512)))
    p.add_argument("--img_size", type=int, default=int(os.environ.get("IMG_SIZE", 256)))
    p.add_argument("--grad_ckpt", default=os.environ.get("GRAD_CKPT", "auto"),
                   choices=["auto", "on", "off"],
                   help="auto: checkpoint swin_t (both two-view graphs must coexist "
                        "until the single SupCon backward; without ckpt that is "
                        ">80 GB of activations at bs 512/256px -> OOM on A100-80)")
    p.add_argument("--manifest", default=os.environ.get("TCGA_UT_MANIFEST",
                   "dataset/tcga_ut/manifest.csv"))
    p.add_argument("--extract_dir", default=os.environ.get("TCGA_UT_EXTRACT",
                   "dataset/tcga_ut/extracted"))
    p.add_argument("--res_filter", default=os.environ.get("TCGA_UT_RES_FILTER", ""),
                   help="comma list of resolution levels; empty = all")
    p.add_argument("--run_name", default=None)
    p.add_argument("--results_root", default="results/tcga_encoder")
    p.add_argument("--save_epoch_freq", type=int, default=int(os.environ.get("SAVE_EPOCH_FREQ", 5)))
    p.add_argument("--num_workers", type=int, default=int(os.environ.get("NUM_WORKERS", 8)))
    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 0)))
    p.add_argument("--print_freq", type=int, default=50)
    p.add_argument("--knn_freq", type=int, default=int(os.environ.get("KNN_FREQ", 10)),
                   help="epochs between kNN sanity evals (0 = off)")
    p.add_argument("--knn_train_bank", type=int, default=65536)
    p.add_argument("--knn_val_max", type=int, default=8192)
    p.add_argument("--limit_train", type=int, default=0, help="debug: cap train items")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()

    a.optimizer = a.optimizer or "adamw"
    if a.optimizer == "sgd":
        a.base_lr = a.base_lr if a.base_lr is not None else 0.5
        a.weight_decay = a.weight_decay if a.weight_decay is not None else 1e-4
        if a.warmup_epochs is None:
            a.warmup_epochs = 10 if a.batch_size > 256 else 0
    elif a.arch == "convnext_t":
        a.base_lr = a.base_lr if a.base_lr is not None else 5e-4
        a.weight_decay = a.weight_decay if a.weight_decay is not None else 0.05
        a.warmup_epochs = 5 if a.warmup_epochs is None else a.warmup_epochs
    else:
        a.base_lr = a.base_lr if a.base_lr is not None else 1.5e-4
        a.weight_decay = a.weight_decay if a.weight_decay is not None else 0.05
        a.warmup_epochs = 40 if a.warmup_epochs is None else a.warmup_epochs
    if a.run_name is None:
        a.run_name = f"enc_{a.arch}_supcon"
    if a.smoke:
        a.limit_train = a.limit_train or 2048
        a.epochs = min(a.epochs, 2)
        a.batch_size = min(a.batch_size, 64)
        a.num_workers = min(a.num_workers, 4)
    return a


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    run_dir = REPO_ROOT / args.results_root / args.run_name
    tb_dir = REPO_ROOT / "results/unified/tcga_encoder" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "opt.txt").write_text(json.dumps(vars(args), indent=2, default=str))

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(tb_dir))

    res_filter = None
    if args.res_filter.strip():
        res_filter = {int(s) for s in args.res_filter.split(",") if s.strip()}
    train_rows, label_map = read_manifest(args.manifest, splits=("train",),
                                          res_filter=res_filter,
                                          limit=args.limit_train or None)
    val_rows, _ = read_manifest(args.manifest, splits=("val",),
                                res_filter=res_filter,
                                limit=None if not args.limit_train else 1024)
    print(f"[{args.run_name}] train={len(train_rows)} val={len(val_rows)} "
          f"labels={len(label_map)} device={device} arch={args.arch} opt={args.optimizer}")

    ds_train = TCGAUTDataset(train_rows, args.extract_dir, TwoCropTransform(
        build_supcon_transforms(args.img_size)))
    ds_val = TCGAUTDataset(val_rows, args.extract_dir, build_eval_transforms(args.img_size))
    from torch.utils.data import DataLoader
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True,
                          collate_fn=supcon_collate, drop_last=True,
                          persistent_workers=args.num_workers > 0, prefetch_factor=2)
    dl_val = DataLoader(ds_val, batch_size=256, shuffle=False, num_workers=8,
                        pin_memory=True, collate_fn=single_collate, persistent_workers=True)

    backbone, feat_dim = build_backbone(args.arch, args.img_size, pretrained=True)
    backbone.to(device)
    if args.arch.startswith("convnext"):
        backbone = backbone.to(memory_format=torch.channels_last)
    grad_ckpt = {"on": True, "off": False,
                 "auto": args.arch.startswith("swin")}[args.grad_ckpt]
    if grad_ckpt:
        backbone.set_grad_checkpointing(True)
        print(f"[{args.run_name}] gradient checkpointing ON "
              f"(peak-activation savings for {args.arch})", flush=True)
    head = build_supcon_projection_head(
        feat_dim, hidden=args.head_hidden, out_dim=args.head_dim)
    head.to(device)

    param_groups = split_decay_param_groups(
        [backbone, head], weight_decay=args.weight_decay)
    use_sgd = args.optimizer == "sgd"
    if use_sgd:
        # SupCon paper batch scaling: LR = base_lr * bs / 256
        lr = args.base_lr * args.batch_size / 256.0
        # wd lives on param_groups (LN/BN/bias/gamma already 0)
        optimizer = torch.optim.SGD(param_groups, lr=lr, momentum=0.9)
    else:
        lr = args.base_lr  # fixed peak LR; no batch scaling
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999))
    print(f"effective LR: {lr:.6g} (base {args.base_lr} bs {args.batch_size})")

    loss_fn = SupConLoss(temperature=args.temperature).to(device)
    kill_switch = CollapseKillSwitch(args.batch_size, min_epoch=5, patience=2)

    start_epoch, best_metric = 0, float("inf")
    state = load_run_state(run_dir, backbone, head, optimizer)
    if state is not None:
        start_epoch = state["epoch"] + 1
        best_metric = state.get("best_metric", float("inf"))
        print(f"resumed from ckpt_latest.pth at epoch {start_epoch}")

    log_fh = open(run_dir / "loss_log.txt", "a")
    global_step = start_epoch * max(1, len(dl_train))

    def log_scalars(epoch, lr_now, loss_avg, knn_acc=None):
        writer.add_scalar("train/loss", loss_avg, epoch)
        writer.add_scalar("train/lr", lr_now, epoch)
        if knn_acc is not None:
            writer.add_scalar("val/knn5_acc", knn_acc, epoch)
        log_fh.write(f"epoch {epoch} lr {lr_now:.6g} supcon_loss {loss_avg:.5f}"
                     + (f" knn5 {knn_acc:.4f}" if knn_acc is not None else "") + "\n")
        log_fh.flush()

    @torch.no_grad()
    def knn_eval(epoch: int) -> float | None:
        if not val_rows:
            return None
        backbone.eval(); head.eval()

        def embed(dl, cap):
            feats, labs = [], []
            n = 0
            for imgs, labels, _ in dl:
                if n >= cap:
                    break
                imgs = imgs.to(device, non_blocking=True)
                if args.arch.startswith("convnext"):
                    imgs = imgs.to(memory_format=torch.channels_last)
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    z = head(backbone(imgs)).float()
                feats.append(F.normalize(z, dim=1).cpu())
                labs.append(labels.cpu())
                n += imgs.shape[0]
            return torch.cat(feats), torch.cat(labs)

        f_val, y_val = embed(dl_val, args.knn_val_max)
        f_val, y_val = f_val[:args.knn_val_max].to(device), y_val[:args.knn_val_max].to(device)
        bank_ds = TCGAUTDataset(train_rows[: args.knn_train_bank], args.extract_dir,
                                build_eval_transforms(args.img_size))
        bank_dl = DataLoader(bank_ds, batch_size=256, shuffle=False, num_workers=4,
                             collate_fn=single_collate)
        f_bank, y_bank = embed(bank_dl, args.knn_train_bank)
        f_bank, y_bank = f_bank.to(device), y_bank.to(device)
        acc = knn_accuracy(f_bank, y_bank, f_val, y_val, k=5)
        backbone.train(); head.train()
        del f_bank, f_val
        torch.cuda.empty_cache()
        return acc

    backbone.train(); head.train()
    steps_per_epoch = len(dl_train)
    for epoch in range(start_epoch, args.epochs):
        # per-epoch schedule (official SupContrast granularity)
        lam = warmup_cosine_lambda(
            epoch, warmup_steps=args.warmup_epochs, total_steps=args.epochs,
            base=1.0, min_factor=0.01 if use_sgd else 0.0,
            warmup_start_factor=0.01 if use_sgd else 0.1)
        for gi, g in enumerate(optimizer.param_groups):
            g["lr"] = lr * lam
        lr_now = lr * lam

        t0 = time.time(); running, nb = 0.0, 0
        for imgs1, imgs2, labels, _ in dl_train:
            imgs1 = imgs1.to(device, non_blocking=True)
            imgs2 = imgs2.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if args.arch.startswith("convnext"):
                imgs1 = imgs1.to(memory_format=torch.channels_last)
                imgs2 = imgs2.to(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                z1 = head(backbone(imgs1))
                z2 = head(backbone(imgs2))
                z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
                lab2 = torch.cat([labels, labels], dim=0)
                loss = loss_fn(z, lab2)
            loss.backward()
            optimizer.step()
            running += loss.item(); nb += 1
            if nb % args.print_freq == 0:
                ips = args.batch_size * nb / max(1e-6, time.time() - t0)
                mem = f" peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" \
                    if device.type == "cuda" else ""
                print(f"ep {epoch} it {nb}/{steps_per_epoch} loss {running/nb:.4f} "
                      f"lr {lr_now:.5g} {ips:.0f} img/s{mem}", flush=True)
        loss_avg = running / max(1, nb)
        writer.add_scalar("train/loss_iter", loss_avg, global_step)
        global_step += steps_per_epoch

        knn_acc = None
        if args.knn_freq > 0 and (epoch + 1) % args.knn_freq == 0:
            try:
                knn_acc = knn_eval(epoch)
            except Exception as e:  # never lose a training epoch over eval
                print(f"WARN knn eval failed: {e}", flush=True)
        log_scalars(epoch + 1, lr_now, loss_avg, knn_acc)
        kill_switch.update(epoch, loss_avg)
        if knn_acc is not None and (best_metric == float("inf") or knn_acc > best_metric):
            best_metric = knn_acc
            torch.save({"epoch": epoch, "backbone": backbone.state_dict(),
                        "extras": head.state_dict(), "meta": vars(args)},
                       run_dir / "ckpt_best.pth")
        save_run_state(run_dir, backbone, head, optimizer, epoch + 1, best_metric,
                       vars(args), args.save_epoch_freq)

    print(f"done: {args.epochs} epochs, best knn {best_metric}")
    writer.close(); log_fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
