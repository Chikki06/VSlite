from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from diffusion_vs.data import VSliteVSDiffusionDataset


@dataclass(frozen=True)
class VQGANBatch:
    x_he: torch.Tensor  # (3,H,W) in [-1,1]
    patch_id: str


@dataclass(frozen=True)
class VQGANCollatedBatch:
    x_he: torch.Tensor  # (B,3,H,W)
    patch_id: tuple[str, ...]


def collate_vqgan_batch(batch: Sequence[VQGANBatch]) -> VQGANCollatedBatch:
    x_he = torch.stack([b.x_he for b in batch], dim=0)
    return VQGANCollatedBatch(x_he=x_he, patch_id=tuple(b.patch_id for b in batch))


class VQGANHEDataset(Dataset):
    """
    Dataset for training/testing a VQGAN autoencoder on H&E targets.
    """

    def __init__(
        self,
        *,
        dataroot: str,
        split: str,
        input_nc: int,
        num_ir_channels: int = 10,
        image_size: int | None = None,
        test_includes_val: bool = False,
    ):
        super().__init__()
        self.base = VSliteVSDiffusionDataset(
            dataroot=dataroot,
            split=split,
            input_nc=input_nc,
            num_ir_channels=num_ir_channels,
            image_size=image_size,
            test_includes_val=bool(test_includes_val),
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


def build_vqgan_discriminator(vslite_root: Path, args, *, variant: str):
    """Build the VQGAN discriminator variant.

    Use the baseline pix2pix discriminator implementation (PatchGAN) so the
    discriminator matches the baseline training stack exactly.

    Note: `variant` is kept for stable naming, but currently both variants map
    to the same architecture.
    """
    disc_variant = str(variant).strip().lower()
    if disc_variant not in {"default", "patchd"}:
        raise ValueError(
            f"Unsupported discriminator_variant='{disc_variant}'. "
            "Expected one of: default, patchd"
        )
    # Import here to keep this module lightweight for other codepaths.
    from types import SimpleNamespace

    from models.networks.discriminator import NLayerDiscriminator

    dopt = SimpleNamespace(
        ndf=int(getattr(args, "ndf", 64)),
        n_layers_D=int(getattr(args, "n_layers_D", 4)),
        norm_D=str(getattr(args, "norm_D", "spectralinstance")),
        no_ganFeat_loss=True,
        label_nc=0,
        output_nc=3,
        contain_dontcare_label=False,
        no_instance=True,
    )
    return NLayerDiscriminator(dopt)

