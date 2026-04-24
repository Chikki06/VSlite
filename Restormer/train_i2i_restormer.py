import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

_THIS_DIR = Path(__file__).resolve().parent
_VSLITE_ROOT = _THIS_DIR.parent
if str(_VSLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_VSLITE_ROOT))

from SwinGAN.utils.dataset_vslite_i2i import SplitFractions, VSLiteI2IDataset
from SwinGAN.utils.sample_saver import save_sample_palette_png, save_sample_pngs
from basicsr.models.archs.restormer_arch import Restormer


logger = logging.getLogger(__name__)


def _get_env(name: str, default=None):
    v = os.environ.get(name)
    if v is not None and v != "":
        return v
    return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _get_env(name, None)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_int_list(values: Sequence[int]) -> list[int]:
    return [int(v) for v in values]


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return SimpleNamespace(**cfg), cfg


def build_split_fracs(cfg):
    fr = getattr(cfg, "split_fracs", [0.9, 0.05, 0.05])
    return SplitFractions(train=float(fr[0]), val=float(fr[1]), test=float(fr[2]))


def build_model(cfg, in_chans: int) -> Restormer:
    dual_pixel_task = bool(getattr(cfg, "dual_pixel_task", in_chans != 3))
    return Restormer(
        inp_channels=in_chans,
        out_channels=3,
        dim=int(getattr(cfg, "dim", 48)),
        num_blocks=_as_int_list(getattr(cfg, "num_blocks", [4, 6, 6, 8])),
        num_refinement_blocks=int(getattr(cfg, "num_refinement_blocks", 4)),
        heads=_as_int_list(getattr(cfg, "heads", [1, 2, 4, 8])),
        ffn_expansion_factor=float(getattr(cfg, "ffn_expansion_factor", 2.66)),
        bias=bool(getattr(cfg, "bias", False)),
        LayerNorm_type=str(getattr(cfg, "LayerNorm_type", "WithBias")),
        dual_pixel_task=dual_pixel_task,
    )


@torch.inference_mode()
def validate(model, loader, device, l1_loss):
    if len(loader) == 0:
        return float("nan")

    model.eval()
    tot_l1 = 0.0
    n = 0
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        y = batch["y"].to(device=device, dtype=torch.float32)
        y_hat = torch.tanh(model(x))
        tot_l1 += float(l1_loss(y_hat, y).item()) * x.shape[0]
        n += x.shape[0]
    return tot_l1 / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config_i2i_restormer.yaml")
    args_cli = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg, cfg_dict = load_config(args_cli.config)

    cfg.dataroot = _get_env("RESTORMER_DATAROOT", getattr(cfg, "dataroot", ""))
    cfg.output_dir = _get_env("RESTORMER_OUTPUT_DIR", getattr(cfg, "output_dir", "OUTPUT_PATH"))
    cfg.tcga_feature_ckpt = _get_env("RESTORMER_TCGA_CKPT", getattr(cfg, "tcga_feature_ckpt", ""))
    cfg.input_mode = _get_env("RESTORMER_INPUT_MODE", getattr(cfg, "input_mode", "ir10"))
    cfg.gpu_id = _get_env("RESTORMER_GPU_ID", getattr(cfg, "gpu_id", "0"))
    cfg.batch_size = int(_get_env("RESTORMER_BATCH_SIZE", getattr(cfg, "batch_size", 4)))
    cfg.train_num_workers = int(_get_env("RESTORMER_TRAIN_NUM_WORKERS", getattr(cfg, "train_num_workers", 4)))
    cfg.val_num_workers = int(_get_env("RESTORMER_VAL_NUM_WORKERS", getattr(cfg, "val_num_workers", 2)))

    # Perceptual loss defaults (match pix2pix path unless explicitly overridden).
    cfg.vgg_normalize_input = bool(getattr(cfg, "vgg_normalize_input", True))
    cfg.vgg_loss_include_input = bool(getattr(cfg, "vgg_loss_include_input", False))

    resume_ckpt_env = _get_env("RESTORMER_RESUME_CKPT", "")
    auto_resume = _env_bool("RESTORMER_AUTO_RESUME", True)
    print_freq = int(_get_env("RESTORMER_PRINT_FREQ", getattr(cfg, "print_freq", 100)))
    use_amp = _env_bool("RESTORMER_USE_AMP", bool(getattr(cfg, "use_amp", False)))

    if not getattr(cfg, "dataroot", None) or cfg.dataroot == "":
        raise ValueError(f"Config 'dataroot' is empty (currently {cfg.dataroot}). Set RESTORMER_DATAROOT or edit config.")

    os.makedirs(cfg.output_dir, exist_ok=True)
    try:
        shutil.copyfile(args_cli.config, os.path.join(cfg.output_dir, "config.yaml"))
    except Exception as e:
        logger.warning("Could not copy config to output dir: %s", e)

    if str(getattr(cfg, "device", "cuda")) == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
    else:
        device = torch.device("cpu")

    if cfg.input_mode == "ir10":
        in_chans = 10
    elif cfg.input_mode == "irvis13":
        in_chans = 13
    else:
        raise ValueError(f"Unsupported input_mode: {cfg.input_mode}")

    split_fracs = build_split_fracs(cfg)
    use_split = bool(getattr(cfg, "split", None))
    train_ds = VSLiteI2IDataset(
        cfg.dataroot,
        input_mode=cfg.input_mode,
        num_ir_channels=int(getattr(cfg, "num_ir_channels", 10)),
        img_size=int(getattr(cfg, "img_size", 256)),
        he_scale=str(getattr(cfg, "he_scale", "minus1_1")),
        split="train" if use_split else None,
        split_seed=int(getattr(cfg, "split_seed", 1337)),
        split_fracs=split_fracs,
        max_items=getattr(cfg, "max_items", None),
    )
    val_ds = VSLiteI2IDataset(
        cfg.dataroot,
        input_mode=cfg.input_mode,
        num_ir_channels=int(getattr(cfg, "num_ir_channels", 10)),
        img_size=int(getattr(cfg, "img_size", 256)),
        he_scale=str(getattr(cfg, "he_scale", "minus1_1")),
        split="val" if use_split else None,
        split_seed=int(getattr(cfg, "split_seed", 1337)),
        split_fracs=split_fracs,
        max_items=getattr(cfg, "max_items", None),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.train_num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=int(cfg.train_num_workers) > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.val_num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=int(cfg.val_num_workers) > 0,
    )

    model = build_model(cfg, in_chans=in_chans).to(device)
    l1 = torch.nn.L1Loss()

    lr = float(getattr(cfg, "lr", 2e-4))
    beta1 = float(getattr(cfg, "beta1", 0.9))
    beta2 = float(getattr(cfg, "beta2", 0.999))
    weight_decay = float(getattr(cfg, "weight_decay", 0.0))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )

    amp_enabled = use_amp and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    writer = SummaryWriter(log_dir=os.path.join(cfg.output_dir, "tensorboard"))

    total_epochs = int(getattr(cfg, "epochs_n", 50))
    save_epoch_freq = int(getattr(cfg, "save_epoch_freq", 5))
    lambda_l1 = float(getattr(cfg, "lambda_l1", 1.0))
    lambda_perceptual = float(getattr(cfg, "lambda_perceptual", 0.0))
    if lambda_perceptual > 0:
        from SwinGAN.loss_i2i import ConvNeXtFeatureExtractor
        perceptual_loss_fn = ConvNeXtFeatureExtractor(opt=cfg).to(device)
    else:
        perceptual_loss_fn = None
    grad_clip = float(getattr(cfg, "grad_clip", 0.0))

    start_epoch = 0
    global_step = 0

    resume_ckpt = ""
    if resume_ckpt_env:
        resume_ckpt = resume_ckpt_env
    elif auto_resume:
        latest = os.path.join(cfg.output_dir, "ckpt_latest.pth")
        if os.path.isfile(latest):
            resume_ckpt = latest

    if resume_ckpt:
        if not os.path.isfile(resume_ckpt):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and amp_enabled:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", start_epoch * max(1, len(train_loader))))
        logger.info(
            "Resuming from %s at epoch %d (next epoch: %d), global_step=%d",
            resume_ckpt,
            start_epoch,
            start_epoch + 1,
            global_step,
        )

    sample_dir = os.path.join(cfg.output_dir, "samples")

    for epoch in range(start_epoch, total_epochs):
        model.train()
        epoch_start_time = time.time()
        last_iter_time = time.time()
        last_preview = None

        for batch_idx, batch in enumerate(train_loader):
            x = batch["x"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                y_hat = torch.tanh(model(x))
                loss_l1 = l1(y_hat, y) * lambda_l1
                
                loss_total = loss_l1
                
                if perceptual_loss_fn is not None:
                    # Standardized perceptual loss: pass tensors in [-1, 1]
                    loss_perc = perceptual_loss_fn(y_hat, y) * lambda_perceptual
                    loss_total = loss_total + loss_perc

            scaler.scale(loss_total).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            writer.add_scalar("train/l1", float(loss_l1.item()), global_step)
            if perceptual_loss_fn is not None:
                writer.add_scalar("train/perc", float(loss_perc.item()), global_step)
            last_preview = (x[:1].detach(), y[:1].detach(), y_hat[:1].detach())

            current_time = time.time()
            batch_size = x.shape[0]
            time_per_img = (current_time - last_iter_time) / max(1, batch_size)
            last_iter_time = current_time

            global_step += 1
            total_img_steps = global_step * batch_size
            if print_freq > 0 and (total_img_steps % print_freq) < batch_size:
                lr_now = float(optimizer.param_groups[0]["lr"])
                log_str = "(epoch: {}, iters: {}, time: {:.3f}, lr: {:.6f}) L1: {:.4f}".format(
                    epoch + 1, total_img_steps, time_per_img, lr_now, float(loss_l1.item())
                )
                if perceptual_loss_fn is not None:
                    log_str += " Perc: {:.4f}".format(float(loss_perc.item()))
                logger.info(log_str)

        epoch_sec = int(time.time() - epoch_start_time)
        val_l1 = validate(model, val_loader, device, l1)
        writer.add_scalar("val/l1", float(val_l1) if not np.isnan(val_l1) else 0.0, epoch)
        logger.info(
            "End of epoch %d / %d \t Time Taken: %d sec \t val_l1: %s",
            epoch + 1,
            total_epochs,
            epoch_sec,
            "nan" if np.isnan(val_l1) else f"{val_l1:.6f}",
        )

        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg_dict,
            "input_mode": cfg.input_mode,
        }
        if amp_enabled:
            ckpt["scaler"] = scaler.state_dict()

        torch.save(ckpt, os.path.join(cfg.output_dir, "ckpt_latest.pth"))
        if (epoch + 1) % save_epoch_freq == 0 or (epoch + 1) == total_epochs:
            torch.save(ckpt, os.path.join(cfg.output_dir, f"ckpt_epoch{epoch + 1}.pth"))
            logger.info("Saved checkpoint epoch %d", epoch + 1)

        if last_preview is not None:
            preview_name = f"epoch{epoch + 1:03d}"
            save_sample_pngs(
                sample_dir,
                preview_name,
                last_preview[0],
                last_preview[1],
                last_preview[2],
                cfg.input_mode,
            )
            save_sample_palette_png(
                sample_dir,
                preview_name,
                last_preview[0],
                last_preview[1],
                last_preview[2],
                cfg.input_mode,
            )
            logger.info("Saved training preview images: %s", preview_name)

    writer.close()


if __name__ == "__main__":
    main()
