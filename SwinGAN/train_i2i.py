import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

_THIS_DIR = Path(__file__).resolve().parent
_VSLITE_ROOT = _THIS_DIR.parent
if str(_VSLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_VSLITE_ROOT))

from Networks.vision_transformer import ProjectedSwinUnet
from loss_i2i import GANLoss, ConvNeXtFeatureExtractor
from utils.dataset_vslite_i2i import SplitFractions, VSLiteI2IDataset
from util.atomic_io import atomic_torch_save
from util.fixed_preview import gather_fixed_preview_batch
from util.run_logger import RunLogger, should_log_epoch_samples
from util.sample_grid import save_unified_sample, split_batch_to_list
from util.save_schedule import make_iter_schedule
from util.unified_tb import build_unified_writer
from util.eval_caps import eval_max_items_train, eval_max_items_val, eval_max_items_test
from models.networks.discriminator import NLayerDiscriminator


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


def _dataloader_kwargs(num_workers: int, device: torch.device) -> dict:
    """Pin memory + worker settings tuned to reduce **host RAM** spikes on HPC.

    Default PyTorch ``prefetch_factor=2`` with many workers multiplies resident
    memory (each worker queues full batches of large 512² tensors). A separate
    preview pass must not use the long-lived val ``DataLoader`` or it spawns
    val workers for the entire training epoch alongside train workers.
    """
    nw = max(0, int(num_workers))
    kwargs: dict = {"pin_memory": device.type == "cuda"}
    if nw > 0:
        kwargs["num_workers"] = nw
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 1
    return kwargs


def build_discriminator(cfg):
    disc_variant = str(getattr(cfg, "discriminator_variant", "default")).strip().lower()
    if disc_variant not in {"default", "patchd"}:
        raise ValueError(
            f"Unsupported discriminator_variant='{disc_variant}'. "
            "Expected one of: default, patchd"
        )
    # Use the baseline pix2pix discriminator implementation (PatchGAN).
    # Configure it to return a single logits tensor (not intermediate feature lists),
    # so downstream GANLoss keeps working unchanged.
    dopt = SimpleNamespace(
        ndf=int(getattr(cfg, "ndf", 64)),
        n_layers_D=int(getattr(cfg, "n_layers_D", 4)),
        norm_D=str(getattr(cfg, "norm_D", "spectralinstance")),
        no_ganFeat_loss=True,
        # Baseline option schema fields (not used for D input nc in this repo).
        label_nc=0,
        output_nc=3,
        contain_dontcare_label=False,
        no_instance=True,
    )
    return NLayerDiscriminator(dopt)


def _subset_indices(n: int, max_items: int, seed: int) -> list[int]:
    if max_items <= 0 or n <= 0:
        return []
    m = min(n, int(max_items))
    # Deterministic pseudo-random subset without importing extra RNG state.
    # Spread picks across the dataset to reduce ordering bias.
    step = max(1, n // m)
    start = int(seed) % max(1, step)
    idx = list(range(start, n, step))[:m]
    return idx


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

    # Perceptual loss defaults (match pix2pix path unless explicitly overridden).
    cfg.vgg_normalize_input = bool(getattr(cfg, "vgg_normalize_input", True))
    cfg.vgg_loss_include_input = bool(getattr(cfg, "vgg_loss_include_input", False))

    # Env overrides (batch scripts)
    cfg.dataroot = _get_env("SWINGAN_DATAROOT", getattr(cfg, "dataroot", ""))
    cfg.output_dir = _get_env("SWINGAN_OUTPUT_DIR", getattr(cfg, "output_dir", "OUTPUT_PATH"))
    cfg.pretrained_swin_ckpt = _get_env(
        "SWINGAN_PRETRAIN_CKPT", getattr(cfg, "pretrained_swin_ckpt", "")
    )
    cfg.input_mode = _get_env("SWINGAN_INPUT_MODE", getattr(cfg, "input_mode", "ir10"))
    cfg.gpu_id = _get_env("SWINGAN_GPU_ID", getattr(cfg, "gpu_id", "0"))
    cfg.batch_size = int(_get_env("SWINGAN_BATCH_SIZE", getattr(cfg, "batch_size", 4)))
    cfg.discriminator_variant = _get_env(
        "SWINGAN_DISC_VARIANT",
        getattr(cfg, "discriminator_variant", "default"),
    )
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
        drop_last=True,
        **_dataloader_kwargs(int(cfg.train_num_workers), device),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        **_dataloader_kwargs(int(cfg.val_num_workers), device),
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

    D = build_discriminator(cfg).to(device)

    optG = torch.optim.Adam(G.parameters(), lr=float(cfg.lr_g), betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=float(cfg.lr_d), betas=(0.5, 0.999))

    gan = GANLoss()
    l1 = torch.nn.L1Loss()
    perceptual = ConvNeXtFeatureExtractor(opt=cfg).to(device) if use_perc else None

    writer = SummaryWriter(log_dir=os.path.join(cfg.output_dir, "tensorboard"))
    run_logger = RunLogger(str(cfg.output_dir), name=os.path.basename(os.path.abspath(cfg.output_dir)))
    run_logger.dump_opts(cfg_dict)
    # Unified TensorBoard under repo-root results/unified/.
    run_name = os.path.basename(os.path.abspath(cfg.output_dir))
    u_writer, u_tb = build_unified_writer(model="swingan_i2i", run=run_name)

    use_tqdm = tqdm is not None and sys.stderr.isatty()
    total_epochs = int(cfg.epochs_n)
    save_epoch_freq = int(getattr(cfg, "save_epoch_freq", 1))
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
    os.makedirs(sample_dir, exist_ok=True)
    # Eval subset sizes (overhead control). Defaults are conservative.
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
        split="test" if getattr(cfg, "split", None) else None,
        split_seed=int(getattr(cfg, "split_seed", 1337)),
        split_fracs=split_fracs,
        max_items=getattr(cfg, "max_items", None),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        **_dataloader_kwargs(int(cfg.val_num_workers), device),
    )

    # Deterministic eval subset for train (and optionally val/test if you want to cap them).
    train_eval_ds = VSLiteI2IDataset(
        cfg.dataroot,
        input_mode=cfg.input_mode,
        num_ir_channels=int(getattr(cfg, "num_ir_channels", 10)),
        img_size=int(getattr(cfg, "img_size", 256)),
        he_scale=str(getattr(cfg, "he_scale", "minus1_1")),
        split="train" if getattr(cfg, "split", None) else None,
        split_seed=int(getattr(cfg, "split_seed", 1337)),
        split_fracs=split_fracs,
        max_items=getattr(cfg, "max_items", None),
        indices=_subset_indices(len(train_ds), eval_train_max, eval_seed),
    )
    train_eval_loader = DataLoader(
        train_eval_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        **_dataloader_kwargs(int(cfg.val_num_workers), device),
    )

    # 8 fixed preview samples (deterministic) for the unified composite saver.
    # Use a dedicated single-process loader so we do not spawn ``val_loader``
    # workers here; keeping val workers alive for all of training alongside train
    # workers roughly doubles DataLoader RSS on large-patch runs (Slurm OOM).
    _preview_bs = min(int(cfg.batch_size), 8) if int(cfg.batch_size) > 0 else 1
    preview_loader = DataLoader(
        val_ds,
        batch_size=max(1, _preview_bs),
        shuffle=False,
        pin_memory=False,
    )
    fixed_preview = gather_fixed_preview_batch(preview_loader, n=8)
    total_iters = max(1, int(total_epochs) * int(len(train_loader)))
    should_save_image = make_iter_schedule(total_iters, target_saves=50)

    @torch.no_grad()
    def _save_unified_preview(epoch_idx: int, step_idx: int) -> None:
        if fixed_preview is None:
            return
        was_train = G.training
        G.eval()
        x_fp = fixed_preview["x"].to(device=device, dtype=torch.float32)
        y_fp = fixed_preview["y"].to(device=device, dtype=torch.float32)
        y_hat_fp = torch.tanh(G(x_fp))
        reals = split_batch_to_list(y_fp.detach().cpu())[:8]
        preds = split_batch_to_list(y_hat_fp.detach().cpu())[:8]
        name = f"epoch{int(epoch_idx) + 1:03d}_step{int(step_idx):09d}"
        save_unified_sample(sample_dir, name, reals, preds, input_mode=cfg.input_mode)
        if was_train:
            G.train()

    for epoch in range(start_epoch, total_epochs):
        G.train()
        D.train()

        epoch_start_time = time.time()
        last_iter_time = time.time()
        epoch_samples = 0
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
            bs_step = int(x.shape[0])

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
                # Standardized perceptual loss: pass tensors in [-1, 1] (normalization handled inside).
                g_perc = perceptual(y_hat, y) * float(getattr(cfg, "lambda_perceptual", 10.0))
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

            if use_tqdm:
                loader_iter.set_postfix(
                    {"d": float(d_loss.item()), "g": float(g_loss.item())}
                )

            current_time = time.time()
            time_per_img = (current_time - last_iter_time) / max(1, bs_step)
            last_iter_time = current_time
            global_step += 1
            epoch_samples += bs_step

            if not use_tqdm and print_freq > 0 and should_log_epoch_samples(
                epoch_samples, bs_step, int(print_freq)
            ):
                lr_g = float(optG.param_groups[0]["lr"])
                logger.info(
                    "(epoch: %d, iters: %d, time: %.3f, lr: %.5f) "
                    "D: %.3f G: %.3f G_L1: %.3f G_GAN: %.3f G_perc: %.3f",
                    epoch + 1,
                    epoch_samples,
                    time_per_img,
                    lr_g,
                    float(d_loss.item()),
                    float(g_loss.item()),
                    float(g_l1.item()),
                    float(g_gan.item()),
                    float(g_perc.item()),
                )
                run_logger.append_loss(
                    epoch=epoch + 1,
                    step=int(epoch_samples),
                    losses={
                        "D": float(d_loss.item()),
                        "G": float(g_loss.item()),
                        "G_L1": float(g_l1.item()),
                        "G_GAN": float(g_gan.item()),
                        "G_perc": float(g_perc.item()),
                    },
                    lr=lr_g,
                    time_per_iter=time_per_img,
                )

            if should_save_image(global_step - 1):
                _save_unified_preview(epoch, global_step)
                run_logger.write_iter(epoch + 1, global_step)

        epoch_sec = int(time.time() - epoch_start_time)
        train_l1 = validate(G, train_eval_loader, device, l1)
        val_l1 = validate(G, val_loader, device, l1)
        test_l1 = validate(G, test_loader, device, l1)

        writer.add_scalar("val/l1", float(val_l1), epoch)
        # Unified: log all three splits under consistent tags.
        u_writer.add_scalar("l1/train", float(train_l1), epoch)
        u_writer.add_scalar("l1/val", float(val_l1), epoch)
        u_writer.add_scalar("l1/test", float(test_l1), epoch)
        u_writer.add_scalar("meta/epoch_sec", float(epoch_sec), epoch)
        logger.info(
            "End of epoch %d / %d \t Time Taken: %d sec \t train_l1: %.6f \t val_l1: %.6f \t test_l1: %.6f",
            epoch + 1,
            total_epochs,
            epoch_sec,
            float(train_l1),
            float(val_l1),
            float(test_l1),
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
        # Keep a stable filename so downstream jobs can reference a predictable path
        # (e.g., $SWINGAN_OUTPUT_DIR/ckpt_latest.pth).
        atomic_torch_save(torch.save, ckpt, os.path.join(cfg.output_dir, "ckpt_latest.pth"))
        if (epoch + 1) % save_epoch_freq == 0 or (epoch + 1) == total_epochs:
            atomic_torch_save(torch.save, ckpt, os.path.join(cfg.output_dir, f"ckpt_epoch{epoch+1}.pth"))
            logger.info("Saved checkpoint epoch %d", epoch + 1)

        # End-of-epoch composite save (always, even if the iter schedule didn't fire).
        _save_unified_preview(epoch, global_step)
        run_logger.write_iter(epoch + 1, global_step)

    writer.close()
    u_writer.close()


if __name__ == "__main__":
    main()

