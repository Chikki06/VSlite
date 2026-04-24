from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


class CondToRGB(nn.Module):
    """
    Map a 10/13-channel VS conditioning tensor to 3 channels for I2SB's conditional ADM backbone.
    """

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x_cond: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(x_cond))


@dataclass(frozen=True)
class I2SBVSPaths:
    repo_root: Path
    third_party_i2sb: Path
    ckpt_dir: Path


def resolve_i2sb_paths(vslite_root: str | os.PathLike[str]) -> I2SBVSPaths:
    repo_root = Path(vslite_root).resolve()
    third_party_i2sb = repo_root / "third_party" / "I2SB"
    ckpt_dir = third_party_i2sb / "data"
    return I2SBVSPaths(
        repo_root=repo_root,
        third_party_i2sb=third_party_i2sb,
        ckpt_dir=ckpt_dir,
    )

