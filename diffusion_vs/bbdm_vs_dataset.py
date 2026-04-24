from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from diffusion_vs.data import VSliteVSDiffusionDataset
from diffusion_vs.i2sb_vs import CondToRGB


@dataclass(frozen=True)
class _BBDMExample:
    x: torch.Tensor
    x_name: str
    x_cond: torch.Tensor
    x_cond_name: str


class VSliteAlignedNPYDataset(Dataset):
    """
    A BBDM-style aligned dataset backed by VSlite's `.npy` VS patches.

    Returns: ((x, name), (x_cond, name))
      - x: H&E RGB in [-1,1]
      - x_cond: learned 3ch projection of 10/13ch VS input, in [-1,1]
    """

    def __init__(self, dataset_config: Any, stage: str = "train") -> None:
        super().__init__()
        if stage not in {"train", "val", "test"}:
            raise ValueError(f"stage must be train/val/test, got {stage!r}")

        # BBDM configs use stage=val; VSlite uses test split for validation.
        split = "train" if stage == "train" else "test"
        self.image_size = int(getattr(dataset_config, "image_size", 256))
        self.to_normal = bool(getattr(dataset_config, "to_normal", True))

        self._base = VSliteVSDiffusionDataset(
            dataroot=str(getattr(dataset_config, "dataroot")),
            split=split,
            input_nc=int(getattr(dataset_config, "input_nc")),
            num_ir_channels=int(getattr(dataset_config, "num_ir_channels", 10)),
        )
        self._cond_to_rgb = CondToRGB(int(getattr(dataset_config, "input_nc"))).eval()

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int):
        b = self._base[idx]
        x = b.x_he
        x_cond = self._cond_to_rgb(b.x_cond.unsqueeze(0)).squeeze(0)

        if self.image_size and (x.shape[-1] != self.image_size or x.shape[-2] != self.image_size):
            x = F.interpolate(x.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False).squeeze(0)
            x_cond = F.interpolate(
                x_cond.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
            ).squeeze(0)

        x_name = b.patch_id
        x_cond_name = b.patch_id
        return (x, x_name), (x_cond, x_cond_name)


def register_vslite_dataset(vslite_root: Path) -> None:
    """
    Register `vslite_vs_npy` dataset into BBDM's registry.
    Must be called before BBDM runner builds datasets.
    """
    bbdm_root = vslite_root / "third_party" / "BBDM"
    if str(bbdm_root) not in sys.path:
        sys.path.insert(0, str(bbdm_root))

    from Register import Registers  # type: ignore

    # If already registered, no-op.
    if "vslite_vs_npy" in Registers.datasets.keys():
        return

    Registers.datasets.register_with_name("vslite_vs_npy")(VSliteAlignedNPYDataset)

