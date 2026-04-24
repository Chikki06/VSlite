from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .nafnet_core import NAFBackbone


@dataclass(frozen=True)
class NAFNetI2IConfig:
    width: int = 32
    middle_blk_num: int = 1
    enc_blk_nums: tuple[int, ...] = (1, 1, 1, 28)
    dec_blk_nums: tuple[int, ...] = (1, 1, 1, 1)


class NAFNetI2I(nn.Module):
    """
    VSlite i2i wrapper around NAFNet-style backbone.

    - Base (b): NAFBackbone(in=10/13, out=3)
    - Attention (a): optional 1x1 projection to 3ch + CBAM(3), then NAFBackbone(in=3, out=3)
    """

    def __init__(
        self,
        *,
        raw_in_chans: int,
        out_chans: int = 3,
        use_attention_stem: bool,
        backbone_cfg: NAFNetI2IConfig,
    ):
        super().__init__()
        self.raw_in_chans = int(raw_in_chans)
        self.out_chans = int(out_chans)
        self.use_attention_stem = bool(use_attention_stem)

        if self.use_attention_stem:
            self.proj = nn.Conv2d(self.raw_in_chans, 3, kernel_size=1, stride=1, padding=0, bias=True)
            from SwinGAN.Networks.cbam import CBAM  # type: ignore

            self.cbam = CBAM(3)
            backbone_in = 3
        else:
            self.proj = nn.Identity()
            self.cbam = None
            backbone_in = self.raw_in_chans

        self.backbone = NAFBackbone(
            in_channels=backbone_in,
            out_channels=self.out_chans,
            width=int(backbone_cfg.width),
            middle_blk_num=int(backbone_cfg.middle_blk_num),
            enc_blk_nums=list(backbone_cfg.enc_blk_nums),
            dec_blk_nums=list(backbone_cfg.dec_blk_nums),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        if self.cbam is not None:
            z = self.cbam(z)
        return self.backbone(z)


def build_nafnet_i2i(*, cfg, raw_in_chans: int) -> NAFNetI2I:
    use_attention_stem = bool(getattr(cfg, "use_cbam", False))
    backbone_cfg = NAFNetI2IConfig(
        width=int(getattr(cfg, "width", 32)),
        middle_blk_num=int(getattr(cfg, "middle_blk_num", 1)),
        enc_blk_nums=tuple(int(x) for x in getattr(cfg, "enc_blk_nums", [1, 1, 1, 28])),
        dec_blk_nums=tuple(int(x) for x in getattr(cfg, "dec_blk_nums", [1, 1, 1, 1])),
    )
    return NAFNetI2I(
        raw_in_chans=int(raw_in_chans),
        out_chans=int(getattr(cfg, "out_chans", 3)),
        use_attention_stem=use_attention_stem,
        backbone_cfg=backbone_cfg,
    )
