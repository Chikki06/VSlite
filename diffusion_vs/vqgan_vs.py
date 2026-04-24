from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from diffusion_vs.data import VSliteVSDiffusionDataset


@dataclass(frozen=True)
class VQGANBatch:
    x_he: torch.Tensor  # (3,H,W) in [-1,1]
    patch_id: str


class VQGANHEDataset(Dataset):
    """
    Dataset for training/testing a VQGAN autoencoder on H&E targets.
    """

    def __init__(self, *, dataroot: str, split: str, input_nc: int, num_ir_channels: int = 10):
        super().__init__()
        self.base = VSliteVSDiffusionDataset(
            dataroot=dataroot,
            split=split,
            input_nc=input_nc,
            num_ir_channels=num_ir_channels,
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> VQGANBatch:
        b = self.base[idx]
        return VQGANBatch(x_he=b.x_he, patch_id=b.patch_id)


def import_third_party_vqgan(vslite_root: Path):
    tp = vslite_root / "third_party" / "VQGAN-pytorch"
    if str(tp) not in sys.path:
        sys.path.insert(0, str(tp))
    from discriminator import Discriminator  # type: ignore
    from vqgan import VQGAN  # type: ignore

    return VQGAN, Discriminator

