from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class VQGANPytorchAdapter(nn.Module):
    """
    Adapter to make `third_party/VQGAN-pytorch` behave like BBDM's `VQModel`.

    BBDM latent model expects:
      - `encoder(x)` -> latent tensor
      - `quant_conv(latent)` (optional)
      - `quantize(latent)` -> (quant_latent, loss, info)
      - `decode(quant_latent)` -> reconstructed image
    """

    def __init__(self, *, vslite_root: Path, ckpt_path: str, device: torch.device) -> None:
        super().__init__()
        tp = vslite_root / "third_party" / "VQGAN-pytorch"
        if str(tp) not in sys.path:
            sys.path.insert(0, str(tp))

        from vqgan import VQGAN  # type: ignore

        # Build args namespace expected by that VQGAN implementation.
        class _Args:
            pass

        a = _Args()
        a.latent_dim = 256
        a.image_size = 256
        a.num_codebook_vectors = 1024
        a.beta = 0.25
        a.image_channels = 3
        a.device = str(device)

        self.vq = VQGAN(a).to(device=device)  # type: ignore
        self.vq.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        self.vq.eval()
        for p in self.vq.parameters():
            p.requires_grad = False

        # Expose the expected attributes
        self.encoder = self.vq.encoder
        self.decoder = self.vq.decoder
        self.quant_conv = self.vq.quant_conv
        self.post_quant_conv = self.vq.post_quant_conv
        self.codebook = self.vq.codebook

    @torch.no_grad()
    def quantize(self, z: torch.Tensor):
        # The VQGAN-pytorch codebook API: codebook(z) returns (z_q, indices, loss)
        z_q, indices, q_loss = self.codebook(z)
        info = {"indices": indices}
        return z_q, q_loss, info

    @torch.no_grad()
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.vq.decode(z_q)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        # For convenience; mirrors BBDM's usage in sample_vqgan.
        return self.vq(x)

