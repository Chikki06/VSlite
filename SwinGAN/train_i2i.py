import argparse
import logging
import os
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

from Networks.discriminator_model import PatchGAN
from Networks.vision_transformer import ProjectedSwinUnet
from loss_i2i import GANLoss, ConvNeXtFeatureExtractor
from utils.dataset_vslite_i2i import SplitFractions, VSLiteI2IDataset


logger = logging.getLogger(__name__)


def _get_env(name: str, default=None):
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def _env_bool(name: str, default: bool) -> bool:
    raw = _get_env(name, None)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_config(path: str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return SimpleNamespace(**cfg), cfg


def build_split_fracs(cfg):
    fr = getattr(cfg, "split_fracs", [0.9, 0.05, 0.05])
    return SplitFractions(train=float(fr[0]), val=float(fr[1]), test=float(fr[2]))


def _tensor_to_png_array(tensor: torch.Tensor, prefer_last_three: bool = False) -> np.ndarray:
    tensor = tensor.detach().float().cpu()
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW or BCHW tensor, got shape {tuple(tensor.shape)}")
    if tensor.shape[0] > 3:
        tensor = tensor[-3:] if prefer_last_three else tensor[:3]
    if tensor.min().item() < 0.0:
        tensor = (tensor + 1.0) / 2.0
    tensor = tensor.clamp(0.0, 1.0)
    array = (tensor * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    if array.shape[2] == 1:
        array = array[:, :, 0]
    return array


def save_sample_pngs(out_dir: str, name: str, x: torch.Tensor, y: torch.Tensor, yhat: torch.Tensor, input_mode: str):
    os.makedirs(out_dir, exist_ok=True)
    input_preview = _tensor_to_png_array(x, prefer_last_three=(input_mode == "irvis13"))
    target_preview = _tensor_to_png_array(y)
    pred_preview = _tensor_to_png_array(yhat)
    Image.fromarray(input_preview).save(os.path.join(out_dir, f"{name}_input.png"))
    Image.fromarray(target_preview).save(os.path.join(out_dir, f"{name}_target.png"))
    Image.fromarray(pred_preview).save(os.path.join(out_dir, f"{name}_pred.png"))


@torch.no_grad()
def validate(G, loader, device, l1_loss):
    G.eval()
    tot_l1 = 0.0
    n = 0
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        y = batch["y"].to(device=device, dtype=torch.float32)
        y_hat = torch.tanh(G(x))  # keep outputs in [-1,1]
        tot_l1 += float(l1_loss(y_hat, y).item()) * x.shape[0]
        n += x.shape[0]
    return tot_l1 / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config_i2i_baseline.yaml")
    args_cli = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg, cfg_dict = load_config(args_cli.config)

    # Env overrides (batch scripts)
    cfg.dataroot = _get_env("SWINGAN_DATAROOT", getattr(cfg, "dataroot", ""))
    cfg.output_dir = _get_env("SWINGAN_OUTPUT_DIR", getattr(cfg, "output_dir", "OUTPUT_PATH"))
    cfg.pretrained_swin_ckpt = _get_env(
        "SWINGAN_PRETRAIN_CKPT", getattr(cfg, "pretrained_swin_ckpt", "")
    )
    cfg.input_mode = _get_env("SWINGAN_INPUT_MODE", getattr(cfg, "input_mode", "ir10"))
    cfg.gpu_id = _get_env("SWINGAN_GPU_ID", getattr(cfg, "gpu_id", "0"))
    cfg.batch_size = int(_get_env("SWINGAN_BATCH_SIZE", getattr(cfg, "batch_size", 4)))
    cfg.train_num_workers = int(
        _get_env("SWINGAN_TRAIN_NUM_WORKERS", getattr(cfg, "train_num_workers", 4))
    )
    cfg.val_num_workers = int(
        _get_env("SWINGAN_VAL_NUM_WORKERS", getattr(cfg, "val_num_workers", 2))
    )
    resume_ckpt_env = _get_env("SWINGAN_RESUME_CKPT", "")
    auto_resume = _env_bool("SWINGAN_AUTO_RESUME", True)
    print_freq = int(_get_env("SWINGAN_PRINT_FREQ", getattr(cfg, "print_freq", 100)))

    if not cfg.dataroot:
        raise ValueError("Config 'dataroot' is empty. Set SWINGAN_DATAROOT or edit config.")

    os.makedirs(cfg.output_dir, exist_ok=True)
    try:
        shutil.copyfile(args_cli.config, os.path.join(cfg.output_dir, "config.yaml"))
    except Exception as e:
        logger.warning("Could not copy config to output dir: %s", e)

    # device / gpu
    if getattr(cfg, "device", "cuda") == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
    else:
        device = torch.device("cpu")

    split_fracs = build_split_fracs(cfg)

    train_ds = VSLiteI2IDataset(
        cfg.dataroot,
        input_mode=cfg.input_mode,
        num_ir_channels=int(getattr(cfg, "num_ir_channels", 10)),
        img_size=int(getattr(cfg, "img_size", 256)),
        he_scale=str(getattr(cfg, "he_scale", "minus1_1")),
        split="train" if getattr(cfg, "split", None) else None,
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
        split="val" if getattr(cfg, "split", None) else None,
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

    in_chans = 10 if cfg.input_mode == "ir10" else 13
    use_cbam = bool(getattr(cfg, "use_cbam", False))
    use_perc = bool(getattr(cfg, "use_perceptual", False))
    G = ProjectedSwinUnet(cfg, in_chans=in_chans, out_chans=3, use_cbam=use_cbam).to(device)
    resume_ckpt = ""
    if resume_ckpt_env:
        resume_ckpt = resume_ckpt_env
    elif auto_resume:
        latest = os.path.join(cfg.output_dir, "ckpt_latest.pth")
        if os.path.isfile(latest):
            resume_ckpt = latest

    if resume_ckpt:
        logger.info("Resume checkpoint detected; skipping pretrained backbone initialization.")
    elif cfg.pretrained_swin_ckpt:
        if not os.path.exists(cfg.pretrained_swin_ckpt):
            logger.warning(
                "Pretrained ckpt not found (%s). Continuing from scratch. "
                "Set SWINGAN_PRETRAIN_CKPT to a valid path to enable pretrained init.",
                cfg.pretrained_swin_ckpt,
            )
        else:
            G.load_pretrained_backbone(cfg.pretrained_swin_ckpt)
    else:
        logger.warning("No pretrained_swin_ckpt provided; training from scratch.")

    D = PatchGAN(input_nc=3, crop_center=int(getattr(cfg, "crop_center", 128))).to(device)

    optG = torch.optim.Adam(G.parameters(), lr=float(cfg.lr_g), betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=float(cfg.lr_d), betas=(0.5, 0.999))

    gan = GANLoss()
    l1 = torch.nn.L1Loss()
    perceptual = ConvNeXtFeatureExtractor(opt=cfg).to(device) if use_perc else None

    writer = SummaryWriter(log_dir=os.path.join(cfg.output_dir, "tensorboard"))

    use_tqdm = tqdm is not None and sys.stderr.isatty()
    bs = int(cfg.batch_size)
    total_epochs = int(cfg.epochs_n)
    start_epoch = 0
    global_step = 0

    if resume_ckpt:
        if not os.path.isfile(resume_ckpt):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device)
        G.load_state_dict(ckpt["G_state"], strict=True)
        D.load_state_dict(ckpt["D_state"], strict=True)
        if "optG" in ckpt:
            optG.load_state_dict(ckpt["optG"])
        if "optD" in ckpt:
            optD.load_state_dict(ckpt["optD"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", start_epoch * len(train_loader)))
        logger.info(
            "Resuming from %s at epoch %d (next epoch: %d), global_step=%d",
            resume_ckpt,
            start_epoch,
            start_epoch + 1,
            global_step,
        )

    sample_dir = os.path.join(cfg.output_dir, "samples")

    for epoch in range(start_epoch, total_epochs):
        G.train()
        D.train()

        epoch_start_time = time.time()
        last_iter_time = time.time()
        last_preview = None
        loader_iter = train_loader
        if use_tqdm:
            loader_iter = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{total_epochs}",
                unit="batch",
                mininterval=2.0,
            )

        for batch_idx, batch in enumerate(loader_iter):
            x = batch["x"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device, dtype=torch.float32)

            # --- Train D ---
            with torch.no_grad():
                y_hat = torch.tanh(G(x))
            pred_real = D(y)
            pred_fake = D(y_hat.detach())
            d_loss = gan.d_loss(pred_real, pred_fake) * float(getattr(cfg, "lambda_gan", 1.0))
            optD.zero_grad()
            d_loss.backward()
            optD.step()

            # --- Train G ---
            y_hat = torch.tanh(G(x))
            pred_fake_for_g = D(y_hat)
            g_gan = gan.g_loss(pred_fake_for_g) * float(getattr(cfg, "lambda_gan", 1.0))
            g_l1 = l1(y_hat, y) * float(getattr(cfg, "lambda_l1", 100.0))
            if perceptual is not None:
                # ConvNeXt expects 0..1 range; map [-1,1] -> [0,1]
                y01 = (y + 1.0) / 2.0
                yhat01 = (y_hat + 1.0) / 2.0
                g_perc = perceptual(yhat01, y01) * float(getattr(cfg, "lambda_perceptual", 10.0))
            else:
                g_perc = torch.tensor(0.0, device=device)
            g_loss = g_gan + g_l1 + g_perc

            optG.zero_grad()
            g_loss.backward()
            optG.step()

            writer.add_scalar("train/d_loss", float(d_loss.item()), global_step)
            writer.add_scalar("train/g_loss", float(g_loss.item()), global_step)
            writer.add_scalar("train/g_l1", float(g_l1.item()), global_step)
            writer.add_scalar("train/g_gan", float(g_gan.item()), global_step)
            writer.add_scalar("train/g_perc", float(g_perc.item()), global_step)
            last_preview = (x[:1].detach(), y[:1].detach(), y_hat[:1].detach())

            if use_tqdm:
                loader_iter.set_postfix(
                    {"d": float(d_loss.item()), "g": float(g_loss.item())}
                )

            current_time = time.time()
            time_per_img = (current_time - last_iter_time) / max(1, bs)
            last_iter_time = current_time
            global_step += 1

            total_img_steps = global_step * bs
            if (
                not use_tqdm
                and print_freq > 0
                and (total_img_steps % print_freq) < bs
            ):
                epoch_iters = (batch_idx + 1) * bs
                lr_g = float(optG.param_groups[0]["lr"])
                logger.info(
                    "(epoch: %d, iters: %d, time: %.3f, lr: %.5f) "
                    "D: %.3f G: %.3f G_L1: %.3f G_GAN: %.3f G_perc: %.3f",
                    epoch + 1,
                    epoch_iters,
                    time_per_img,
                    lr_g,
                    float(d_loss.item()),
                    float(g_loss.item()),
                    float(g_l1.item()),
                    float(g_gan.item()),
                    float(g_perc.item()),
                )

        epoch_sec = int(time.time() - epoch_start_time)
        val_l1 = validate(G, val_loader, device, l1)
        writer.add_scalar("val/l1", float(val_l1), epoch)
        logger.info(
            "End of epoch %d / %d \t Time Taken: %d sec \t val_l1: %.6f",
            epoch + 1,
            total_epochs,
            epoch_sec,
            float(val_l1),
        )

        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "G_state": G.state_dict(),
            "D_state": D.state_dict(),
            "optG": optG.state_dict(),
            "optD": optD.state_dict(),
            "config": cfg_dict,
        }
        # Keep both a per-epoch checkpoint and a stable filename so downstream jobs can
        # reference a predictable path (e.g., $SWINGAN_OUTPUT_DIR/ckpt_latest.pth).
        torch.save(ckpt, os.path.join(cfg.output_dir, f"ckpt_epoch{epoch+1}.pth"))
        torch.save(ckpt, os.path.join(cfg.output_dir, "ckpt_latest.pth"))
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
            logger.info("Saved training preview images: %s", preview_name)

    writer.close()


if __name__ == "__main__":
    main()

