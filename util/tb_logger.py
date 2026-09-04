"""Unified TensorBoard logger with preview image slider support."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from PIL import Image

from util.unified_tb import build_unified_writer, resolve_unified_tb
from cislnet.registry import canonical_model_tag


def _mirror_train_enabled() -> bool:
    v = os.environ.get("UNIFIED_TB_MIRROR_TRAIN", "1").strip().lower()
    return v not in {"0", "false", "no", "off"}


class UnifiedTBLogger:
    """Wraps per-run and unified SummaryWriters with consistent tags."""

    PREVIEW_GRID_TAG = "preview/fixed_val/grid"

    def __init__(
        self,
        model: str,
        run: str,
        *,
        run_dir: Optional[Union[str, Path]] = None,
        create_per_run: bool = True,
    ) -> None:
        self.model = canonical_model_tag(model)
        self.run = str(run)
        tag = self.model
        self._u_writer, self._u_meta = build_unified_writer(model=tag, run=self.run)
        self._per_run_writer = None
        if create_per_run and run_dir:
            tb_dir = Path(run_dir) / "tensorboard"
            tb_dir.mkdir(parents=True, exist_ok=True)
            from torch.utils.tensorboard import SummaryWriter

            self._per_run_writer = SummaryWriter(log_dir=str(tb_dir))

    @property
    def unified_log_dir(self) -> Path:
        return self._u_meta.log_dir

    def log_meta(self, cfg: Any, *, epoch: int = 0) -> None:
        """Log static run metadata once (epoch 0)."""
        nc = int(getattr(cfg, "num_channels", 0))
        ps = int(getattr(cfg, "patch_size", getattr(cfg, "image_size", getattr(cfg, "img_size", 0))))
        bs = int(getattr(cfg, "batch_size", 0))
        accum = int(getattr(cfg, "grad_accum_steps", 1))
        self.log_scalar("meta/num_channels", nc, epoch)
        self.log_scalar("meta/patch_size", ps, epoch)
        self.log_scalar("meta/batch_size_effective", bs * max(1, accum), epoch)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        v = float(value)
        if v != v or abs(v) == float("inf"):
            return
        self._u_writer.add_scalar(tag, v, int(step))
        if self._per_run_writer is not None:
            self._per_run_writer.add_scalar(tag, v, int(step))

    def log_train_scalars(self, losses: dict[str, float], step: int) -> None:
        if not _mirror_train_enabled():
            return
        for k, v in losses.items():
            self.log_scalar(f"train/{k}", float(v), step)

    def log_epoch_l1(self, train_l1: float, val_l1: float, test_l1: float, epoch: int) -> None:
        self.log_scalar("l1/train", train_l1, epoch)
        self.log_scalar("l1/val", val_l1, epoch)
        self.log_scalar("l1/test", test_l1, epoch)

    def log_epoch_lpips(
        self, train_lpips: float, val_lpips: float, test_lpips: float, epoch: int
    ) -> None:
        self.log_scalar("lpips_convnext_tcga/train", train_lpips, epoch)
        self.log_scalar("lpips_convnext_tcga/val", val_lpips, epoch)
        self.log_scalar("lpips_convnext_tcga/test", test_lpips, epoch)

    def log_preview_png(self, png_path: Union[str, Path], epoch: int) -> None:
        """Log composite PNG for TensorBoard image slider (step = epoch)."""
        im = Image.open(png_path).convert("RGB")
        arr = np.array(im)
        chw = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        self._u_writer.add_image(self.PREVIEW_GRID_TAG, chw, int(epoch))
        if self._per_run_writer is not None:
            self._per_run_writer.add_image(self.PREVIEW_GRID_TAG, chw, int(epoch))

    def log_early_stop(self, best_val: float, stopped_epoch: int) -> None:
        self.log_scalar("early_stop/best_val_l1", best_val, stopped_epoch)

    def close(self) -> None:
        self._u_writer.close()
        if self._per_run_writer is not None:
            self._per_run_writer.close()


def unified_log_dir(model: str, run: str) -> Path:
    return resolve_unified_tb(canonical_model_tag(model), run).log_dir
