from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from data.VS_dataset import VSDataset


@dataclass(frozen=True)
class VSBatch:
    x_cond: torch.Tensor  # (C,H,W), float32
    x_he: torch.Tensor  # (3,H,W), float32 in [-1,1]
    patch_id: str
    path: str


def _basename_no_ext(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def _to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    """
    Convert an input tensor that is likely in [0,1] or [0,5] to [-1,1].
    - IR is clipped to [0,5] in `VS_dataset`.
    - VIS is scaled by /256 in `VS_dataset`, so approximately [0,1].
    We normalize per-tensor with conservative heuristics (no dataset stats required).
    """
    x = x.detach().float()
    # If clearly already in [-1,1], keep.
    if float(x.min().item()) < -0.05:
        return x.clamp(-1.0, 1.0)

    xmax = float(x.max().item())
    if xmax <= 1.5:
        y = x.clamp(0.0, 1.0) * 2.0 - 1.0
        return y
    # Assume IR-like scale up to ~5.
    scale = 5.0 if xmax <= 6.0 else xmax
    y = (x / scale).clamp(0.0, 1.0) * 2.0 - 1.0
    return y


class VSliteVSDiffusionDataset(Dataset):
    """
    Wrap VSlite's `VSDataset` so diffusion models can consume:
      - x_cond: 10ch or 13ch conditioning tensor (normalized to [-1,1])
      - x_he:   3ch target tensor in [-1,1]
      - patch_id: stable key for saving/alignment
    """

    def __init__(
        self,
        *,
        dataroot: str,
        split: str,
        input_nc: int,
        num_ir_channels: int = 10,
        image_size: int | None = None,
        max_dataset_size: int = 10**18,
        test_includes_val: bool = False,
    ) -> None:
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")
        if input_nc not in {num_ir_channels, num_ir_channels + 3}:
            raise ValueError(
                f"input_nc must be {num_ir_channels} (IR-only) or {num_ir_channels + 3} (IR+VIS), got {input_nc}"
            )

        # Build a lightweight opt namespace matching what `VSDataset.initialize` reads.
        class _Opt:
            pass

        opt = _Opt()
        opt.dataroot = dataroot
        opt.max_dataset_size = max_dataset_size
        opt.input_nc = input_nc
        opt.num_ir_channels = num_ir_channels
        # Ensure VS_dataset applies the same train/val/test split logic.
        opt.phase = split
        opt.test_includes_val = bool(test_includes_val)
        # Forward to VS_dataset so it can resize patches appropriately.
        if image_size is not None:
            opt.image_size = int(image_size)

        self._base = VSDataset()
        self._base.initialize(opt)
        self._split = split
        self._input_nc = int(input_nc)
        self._num_ir_channels = int(num_ir_channels)

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> VSBatch:
        ex: dict[str, Any] = self._base[idx]
        x_cond = ex["input"]
        x_he = ex["target"]
        path = str(ex.get("path") or ex.get("name") or "")
        patch_id = _basename_no_ext(path) if path else f"idx{idx}"

        x_cond = _to_minus1_1(x_cond)
        # x_he is already normalized to [-1,1] in `VS_dataset`.
        x_he = x_he.detach().float().clamp(-1.0, 1.0)

        return VSBatch(x_cond=x_cond, x_he=x_he, patch_id=patch_id, path=path)


def collate_vs_batch(batch: list[VSBatch]) -> dict[str, Any]:
    x_cond = torch.stack([b.x_cond for b in batch], dim=0)
    x_he = torch.stack([b.x_he for b in batch], dim=0)
    patch_id = [b.patch_id for b in batch]
    path = [b.path for b in batch]
    return {"x_cond": x_cond, "x_he": x_he, "patch_id": patch_id, "path": path}

