from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from models.networks.architecture import ConvNeXtFeatureExtractor

from diffusion_vs.bbdm_vqgan_pytorch import VQGANPytorchAdapter


class ConvNeXtCondStage(nn.Module):
    """
    Convert a 3ch conditioning image into cross-attention context tokens.
    """

    def __init__(self, *, context_dim: int, vslite_root: Path, tcga_feature_ckpt: str, freeze: bool = True):
        super().__init__()

        class _Opt:
            pass

        opt = _Opt()
        opt.tcga_feature_ckpt = tcga_feature_ckpt
        opt.convnext_normalize_input = True
        opt.convnext_resize_input = True
        opt.vgg_normalize_input = True  # legacy alias for ConvNeXtFeatureExtractor
        opt.vgg_resize_input = True
        self.backbone = ConvNeXtFeatureExtractor(opt=opt, pretrained=False, freeze_weights=freeze)
        # Take last stage output channels and project to context_dim.
        # convnext_tiny stages[-1] output is (B,768,H',W') for timm convnext_tiny.
        self.proj = nn.Conv2d(768, context_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)[-1]  # (B,C,H,W)
        ctx = self.proj(feats)  # (B,context_dim,H,W)
        b, d, h, w = ctx.shape
        # tokens: (B, N, D)
        tokens = ctx.flatten(2).transpose(1, 2).contiguous()
        return tokens


def build_latent_bbdm_model_class(vslite_root: Path):
    """
    Return a LatentBrownianBridgeModel subclass supporting:
      - condition_key = 'convnext'
      - vqgan_impl = 'vqgan_pytorch' (load VQGAN-pytorch checkpoint)
    """
    bbdm_root = vslite_root / "third_party" / "BBDM"
    if str(bbdm_root) not in sys.path:
        sys.path.insert(0, str(bbdm_root))

    from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel  # type: ignore
    from model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler  # type: ignore
    from model.VQGAN.vqgan import VQModel  # type: ignore

    class LatentBrownianBridgeModelVS(BrownianBridgeModel):  # type: ignore
        def __init__(self, model_config: Any):
            super().__init__(model_config)
            self.model_config = model_config

            vqgan_impl = getattr(model_config.VQGAN.params, "vqgan_impl", "bbdm_vqmodel")
            if vqgan_impl == "vqgan_pytorch":
                ckpt = str(getattr(model_config.VQGAN.params, "vqgan_pytorch_ckpt"))
                device = next(self.denoise_fn.parameters()).device
                self.vqgan = VQGANPytorchAdapter(vslite_root=vslite_root, ckpt_path=ckpt, device=device).eval()
            else:
                self.vqgan = VQModel(**vars(model_config.VQGAN.params)).eval()

            # freeze VQ
            for p in self.vqgan.parameters():
                p.requires_grad = False

            # Condition Stage Model
            if self.condition_key == "nocond":
                self.cond_stage_model = None
            elif self.condition_key == "first_stage":
                self.cond_stage_model = self.vqgan
            elif self.condition_key == "SpatialRescaler":
                self.cond_stage_model = SpatialRescaler(**vars(model_config.CondStageParams))
            elif self.condition_key == "convnext":
                context_dim = int(getattr(model_config.BB.params.UNetParams, "context_dim") or 256)
                ckpt_path = str(getattr(model_config.CondStageParams, "tcga_feature_ckpt"))
                self.cond_stage_model = ConvNeXtCondStage(
                    context_dim=context_dim,
                    vslite_root=vslite_root,
                    tcga_feature_ckpt=ckpt_path,
                    freeze=True,
                )
            else:
                raise NotImplementedError(f"Unsupported condition_key={self.condition_key!r}")

        def get_ema_net(self):
            return self

        def get_parameters(self):
            # Optimize UNet and optional cond stage (convnext/spatialrescaler).
            if self.cond_stage_model is not None and self.condition_key in {"SpatialRescaler", "convnext"}:
                return list(self.denoise_fn.parameters()) + list(self.cond_stage_model.parameters())
            return self.denoise_fn.parameters()

        def forward(self, x, x_cond, context=None):
            with torch.no_grad():
                x_latent = self.encode(x, cond=False)
                x_cond_latent = self.encode(x_cond, cond=True)
            context = self.get_cond_stage_context(x_cond)
            return super().forward(x_latent.detach(), x_cond_latent.detach(), context)

        def get_cond_stage_context(self, x_cond):
            if self.cond_stage_model is None:
                return None
            ctx = self.cond_stage_model(x_cond)
            if self.condition_key == "first_stage":
                ctx = ctx.detach()
            return ctx

        @torch.no_grad()
        def encode(self, x, cond=True, normalize=None):
            normalize = self.model_config.normalize_latent if normalize is None else normalize
            model = self.vqgan
            x_latent = model.encoder(x)
            if not self.model_config.latent_before_quant_conv:
                x_latent = model.quant_conv(x_latent)
            if normalize:
                if cond:
                    x_latent = (x_latent - self.cond_latent_mean) / self.cond_latent_std
                else:
                    x_latent = (x_latent - self.ori_latent_mean) / self.ori_latent_std
            return x_latent

        @torch.no_grad()
        def decode(self, x_latent, cond=True, normalize=None):
            normalize = self.model_config.normalize_latent if normalize is None else normalize
            if normalize:
                if cond:
                    x_latent = x_latent * self.cond_latent_std + self.cond_latent_mean
                else:
                    x_latent = x_latent * self.ori_latent_std + self.ori_latent_mean
            model = self.vqgan
            if self.model_config.latent_before_quant_conv:
                x_latent = model.quant_conv(x_latent)
            x_latent_quant, _loss, _info = model.quantize(x_latent)
            out = model.decode(x_latent_quant)
            return out

    return LatentBrownianBridgeModelVS

