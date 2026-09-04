"""Shared hooks for BBDM/I2SB diffusion training (unified TB, val L1, early stop)."""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F

from util.early_stopping import EarlyStopping
from util.tb_logger import UnifiedTBLogger


def env_int(name: str, default: int = 0) -> int:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return int(default)
    return int(v)


def env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return float(default)
    return float(v)


@torch.no_grad()
def batch_l1(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean L1 between prediction and target in [-1, 1] (or compatible) range."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    return float(F.l1_loss(pred, target, reduction="mean").item())


def make_diffusion_tb(model: str, run: str, *, run_dir: Optional[str] = None) -> UnifiedTBLogger:
    return UnifiedTBLogger(model, run, run_dir=run_dir)


def make_diffusion_early_stop() -> EarlyStopping:
    return EarlyStopping(
        patience=env_int("EARLY_STOP_PATIENCE", 0),
        min_delta=env_float("EARLY_STOP_MIN_DELTA", 1e-4),
    )
