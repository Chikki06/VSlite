import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

_THIS_DIR = Path(__file__).resolve().parent
_VSLITE_ROOT = _THIS_DIR.parent
if str(_VSLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_VSLITE_ROOT))

from SwinGAN.utils.dataset_vslite_i2i import SplitFractions, VSLiteI2IDataset

from NAFNet.model_utils import build_model, build_split_fracs, load_config
from util.atomic_io import atomic_torch_save
from util.fixed_preview import gather_fixed_preview_batch
from util.run_logger import RunLogger
from util.sample_grid import save_unified_sample, split_batch_to_list
from util.save_schedule import make_iter_schedule
from util.unified_tb import build_unified_writer
from util.eval_caps import eval_max_items_train, eval_max_items_val, eval_max_items_test


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


def _subset_indices(n: int, max_items: int, seed: int) -> list[int]:
    if max_items <= 0 or n <= 0:
        return []
    m = min(n, int(max_items))
    step = max(1, n // m)
    start = int(seed) % max(1, step)
    return list(range(start, n, step))[:m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config_i2i_nafnet_base.yaml")
    args_cli = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg, cfg_dict = load_config(args_cli.config)

    # Perceptual loss defaults (match pix2pix path unless explicitly overridden).
    cfg.vgg_normalize_input = bool(getattr(cfg, "vgg_normalize_input", True))
    cfg.vgg_loss_include_input = bool(getattr(cfg, "vgg_loss_include_input", False))

    cfg.dataroot = _get_env("NAFNET_DATAROOT", getattr(cfg, "dataroot", ""))
    cfg.output_dir = _get_env("NAFNET_OUTPUT_DIR", getattr(cfg, "output_dir", "OUTPUT_PATH"))
    cfg.tcga_feature_ckpt = _get_env("NAFNET_TCGA_CKPT", getattr(cfg, "tcga_feature_ckpt", ""))
    cfg.input_mode = _get_env("NAFNET_INPUT_MODE", getattr(cfg, "input_mode", "ir10"))
    cfg.gpu_id = _get_env("NAFNET_GPU_ID", getattr(cfg, "gpu_id", "0"))
    cfg.batch_size = int(_get_env("NAFNET_BATCH_SIZE", getattr(cfg, "batch_size", 4)))
    cfg.train_num_workers = int(_get_env("NAFNET_TRAIN_NUM_WORKERS", getattr(cfg, "train_num_workers", 4)))
    cfg.val_num_workers = int(_get_env("NAFNET_VAL_NUM_WORKERS", getattr(cfg, "val_num_workers", 2)))

    resume_ckpt_env = _get_env("NAFNET_RESUME_CKPT", "")
    auto_resume = _env_bool("NAFNET_AUTO_RESUME", True)
    print_freq = int(_get_env("NAFNET_PRINT_FREQ", getattr(cfg, "print_freq", 100)))
    use_amp = _env_bool("NAFNET_USE_AMP", bool(getattr(cfg, "use_amp", False)))

    if not getattr(cfg, "dataroot", None) or cfg.dataroot == "":
        raise ValueError(f"Config 'dataroot' is empty (currently {cfg.dataroot}). Set NAFNET_DATAROOT or edit config.")

    lambda_perceptual = float(getattr(cfg, "lambda_perceptual", 0.0))
    if lambda_perceptual > 0 and (not getattr(cfg, "tcga_feature_ckpt", "") or str(cfg.tcga_feature_ckpt).strip() == ""):
        raise ValueError("lambda_perceptual > 0 but tcga_feature_ckpt is empty. Set NAFNET_TCGA_CKPT or edit config.")

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

    split_fracs: SplitFractions = build_split_fracs(cfg)
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
    run_name = os.path.basename(os.path.abspath(cfg.output_dir))
    u_writer, _u_tb = build_unified_writer(model="nafnet_i2i", run=run_name)
    run_logger = RunLogger(cfg.output_dir, name=run_name)
    run_logger.dump_opts(cfg_dict)

    total_epochs = int(getattr(cfg, "epochs_n", 50))
    save_epoch_freq = int(getattr(cfg, "save_epoch_freq", 5))
    lambda_l1 = float(getattr(cfg, "lambda_l1", 1.0))
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
    os.makedirs(sample_dir, exist_ok=True)
    eval_seed = int(getattr(cfg, "split_seed", 1337))
    eval_train_max = int(eval_max_items_train(getattr(cfg, "eval_max_items_train", 512)))
    eval_val_max = int(eval_max_items_val(getattr(cfg, "eval_max_items_val", 512)))
    eval_test_max = int(eval_max_items_test(getattr(cfg, "eval_max_items_test", 512)))

    test_ds = VSLiteI2IDataset(
        cfg.dataroot,
        input_mode=cfg.input_mode,
        num_ir_channels=int(getattr(cfg, "num_ir_channels", 10)),
        img_size=int(getattr(cfg, "img_size", 256)),
        he_scale=str(getattr(cfg, "he_scale", "minus1_1")),
        split="test" if use_split else None,
        split_seed=int(getattr(cfg, "split_seed", 1337)),
        split_fracs=split_fracs,
        max_items=getattr(cfg, "max_items", None),
        indices=_subset_indices(len(train_ds), eval_test_max, eval_seed) if int(eval_test_max) > 0 else None,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.val_num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=int(cfg.val_num_workers) > 0,
    )

    train_eval_ds = VSLiteI2IDataset(
        cfg.dataroot,
        input_mode=cfg.input_mode,
        num_ir_channels=int(getattr(cfg, "num_ir_channels", 10)),
        img_size=int(getattr(cfg, "img_size", 256)),
        he_scale=str(getattr(cfg, "he_scale", "minus1_1")),
        split="train" if use_split else None,
        split_seed=int(getattr(cfg, "split_seed", 1337)),
        split_fracs=split_fracs,
        max_items=getattr(cfg, "max_items", None),
        indices=_subset_indices(len(train_ds), eval_train_max, eval_seed),
    )
    val_eval_ds = None
    if int(eval_val_max) > 0:
        val_eval_ds = VSLiteI2IDataset(
            cfg.dataroot,
            input_mode=cfg.input_mode,
            num_ir_channels=int(getattr(cfg, "num_ir_channels", 10)),
            img_size=int(getattr(cfg, "img_size", 256)),
            he_scale=str(getattr(cfg, "he_scale", "minus1_1")),
            split="val" if use_split else None,
            split_seed=int(getattr(cfg, "split_seed", 1337)),
            split_fracs=split_fracs,
            max_items=getattr(cfg, "max_items", None),
            indices=_subset_indices(len(val_ds), int(eval_val_max), eval_seed),
        )
    val_eval_loader = (
        DataLoader(
            val_eval_ds,
            batch_size=int(cfg.batch_size),
            shuffle=False,
            num_workers=int(cfg.val_num_workers),
            pin_memory=(device.type == "cuda"),
            persistent_workers=int(cfg.val_num_workers) > 0,
        )
        if val_eval_ds is not None
        else val_loader
    )
    train_eval_loader = DataLoader(
        train_eval_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.val_num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=int(cfg.val_num_workers) > 0,
    )

    fixed_preview = gather_fixed_preview_batch(val_loader, n=8)
    total_iters = max(1, int(total_epochs) * int(len(train_loader)))
    should_save_image = make_iter_schedule(total_iters, target_saves=50)

    @torch.no_grad()
    def _save_unified_preview(epoch_idx: int, step_idx: int) -> None:
        if fixed_preview is None:
            return
        was_train = model.training
        model.eval()
        x_fp = fixed_preview["x"].to(device=device, dtype=torch.float32)
        y_fp = fixed_preview["y"].to(device=device, dtype=torch.float32)
        y_hat_fp = torch.tanh(model(x_fp))
        reals = split_batch_to_list(y_fp.detach().cpu())[:8]
        preds = split_batch_to_list(y_hat_fp.detach().cpu())[:8]
        name = f"epoch{int(epoch_idx) + 1:03d}_step{int(step_idx):09d}"
        save_unified_sample(sample_dir, name, reals, preds, input_mode=cfg.input_mode)
        if was_train:
            model.train()

    for epoch in range(start_epoch, total_epochs):
        model.train()
        epoch_start_time = time.time()
        last_iter_time = time.time()

        for _batch_idx, batch in enumerate(train_loader):
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
                losses_for_log = {"L1": float(loss_l1.item())}
                if perceptual_loss_fn is not None:
                    log_str += " Perc: {:.4f}".format(float(loss_perc.item()))
                    losses_for_log["Perc"] = float(loss_perc.item())
                logger.info(log_str)
                run_logger.append_loss(
                    epoch=epoch + 1,
                    step=total_img_steps,
                    losses=losses_for_log,
                    lr=lr_now,
                    time_per_iter=time_per_img,
                )

            if should_save_image(global_step - 1):
                _save_unified_preview(epoch, global_step)
                run_logger.write_iter(epoch + 1, global_step)

        epoch_sec = int(time.time() - epoch_start_time)
        train_l1 = validate(model, train_eval_loader, device, l1)
        val_l1 = validate(model, val_eval_loader, device, l1)
        test_l1 = validate(model, test_loader, device, l1)
        writer.add_scalar("val/l1", float(val_l1) if not np.isnan(val_l1) else 0.0, epoch)
        u_writer.add_scalar("l1/train", float(train_l1) if not np.isnan(train_l1) else 0.0, epoch)
        u_writer.add_scalar("l1/val", float(val_l1) if not np.isnan(val_l1) else 0.0, epoch)
        u_writer.add_scalar("l1/test", float(test_l1) if not np.isnan(test_l1) else 0.0, epoch)
        u_writer.add_scalar("meta/epoch_sec", float(epoch_sec), epoch)
        logger.info(
            "End of epoch %d / %d \t Time Taken: %d sec \t train_l1: %s \t val_l1: %s \t test_l1: %s",
            epoch + 1,
            total_epochs,
            epoch_sec,
            "nan" if np.isnan(train_l1) else f"{train_l1:.6f}",
            "nan" if np.isnan(val_l1) else f"{val_l1:.6f}",
            "nan" if np.isnan(test_l1) else f"{test_l1:.6f}",
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

        atomic_torch_save(torch.save, ckpt, os.path.join(cfg.output_dir, "ckpt_latest.pth"))
        if (epoch + 1) % save_epoch_freq == 0 or (epoch + 1) == total_epochs:
            atomic_torch_save(torch.save, ckpt, os.path.join(cfg.output_dir, f"ckpt_epoch{epoch + 1}.pth"))
            logger.info("Saved checkpoint epoch %d", epoch + 1)

        # End-of-epoch composite save (always, even if the iter schedule didn't fire).
        _save_unified_preview(epoch, global_step)
        run_logger.write_iter(epoch + 1, global_step)

    writer.close()
    u_writer.close()


if __name__ == "__main__":
    main()
