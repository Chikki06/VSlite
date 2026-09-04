"""Apply standard ablation environment variables to i2i YAML configs."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any


def _env(name: str, default: Any = None) -> Any:
    v = os.environ.get(name)
    if v is not None and str(v).strip() != "":
        return v
    return default


def apply_i2i_env(cfg: SimpleNamespace) -> SimpleNamespace:
    """Override cfg fields from NUM_CHANNELS, PATCH_SIZE, BATCH_SIZE, EPOCHS, etc."""
    ps = _env("PATCH_SIZE", _env("IMG_SIZE", None))
    if ps is not None:
        cfg.img_size = int(ps)

    bs = _env("BATCH_SIZE", None)
    if bs is not None:
        cfg.batch_size = int(bs)

    accum = _env("GRAD_ACCUM_STEPS", None)
    if accum is not None:
        cfg.grad_accum_steps = int(accum)

    ep = _env("EPOCHS", None)
    if ep is not None:
        cfg.epochs_n = int(ep)

    wd = _env("WEIGHT_DECAY", None)
    if wd is not None:
        cfg.weight_decay = float(wd)

    lr = _env("LR", None)
    if lr is not None:
        cfg.lr = float(lr)

    lp = _env("LAMBDA_L1", None)
    if lp is not None:
        cfg.lambda_l1 = float(lp)

    lperc = _env("LAMBDA_PERC", None)
    if lperc is not None:
        cfg.lambda_perceptual = float(lperc)

    nc = _env("NUM_CHANNELS", None)
    if nc is not None:
        n = int(nc)
        if n == 10:
            cfg.input_mode = "ir10"
        elif n == 13:
            cfg.input_mode = "irvis13"
        elif n == 3:
            cfg.input_mode = "ir3"
        else:
            cfg.input_mode = getattr(cfg, "input_mode", "ir10")
        cfg.num_channels = n

    cfg.convnext_normalize_input = bool(
        getattr(cfg, "convnext_normalize_input", getattr(cfg, "vgg_normalize_input", True))
    )
    cfg.convnext_loss_include_input = bool(
        getattr(
            cfg,
            "convnext_loss_include_input",
            getattr(cfg, "vgg_loss_include_input", False),
        )
    )
    # Legacy aliases for ConvNeXtFeatureExtractor
    cfg.vgg_normalize_input = cfg.convnext_normalize_input
    cfg.vgg_loss_include_input = cfg.convnext_loss_include_input

    hcs = _env("HE_COLOR_SPACE", None)
    if hcs is not None:
        cfg.he_color_space = str(hcs).lower()

    drop = _env("DROPOUT", None)
    if drop is not None:
        cfg.dropout = float(drop)

    no_aug = _env("NO_AUG", None)
    if no_aug is not None and str(no_aug).strip().lower() in {"1", "true", "yes", "on"}:
        cfg.augment = False
    elif _env("AUG_STRENGTH", None) is not None:
        cfg.augment = True

    ema = _env("EMA_DECAY", None)
    if ema is not None:
        cfg.ema_decay = float(ema)

    opt_name = _env("OPTIMIZER", None)
    if opt_name is not None:
        cfg.optimizer = str(opt_name).lower()

    lr_sched = _env("LR_SCHEDULE", None)
    if lr_sched is not None:
        cfg.lr_schedule = str(lr_sched).lower()

    warmup = _env("WARMUP_EPOCHS", None)
    if warmup is not None:
        cfg.warmup_epochs = int(warmup)

    lgan = _env("LAMBDA_GAN", None)
    if lgan is not None:
        cfg.lambda_gan = float(lgan)

    no_gan = _env("NO_GAN", None)
    if no_gan is not None and str(no_gan).strip().lower() in {"1", "true", "yes", "on"}:
        cfg.no_gan = True

    return cfg
