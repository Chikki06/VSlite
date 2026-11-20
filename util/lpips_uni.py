"""Zhang lin0 perceptual distance on the frozen UNI v1 (MahmoodLab/UNI) encoder.

This is a **metric only** — a test-time companion to ``lpips_convnext_tcga``
(:mod:`util.lpips_convnext`), built on the same UNI backbone used by
``--netF uni`` feature extraction. Train and test must call the same
implementation (this module).

Quantity name: ``lpips_uni``.

Recipe (Zhang et al. LPIPS lin0, equal-weight stages):
  * Backbone: ``models.networks.architecture.UNIFeatureExtractor``
    (MahmoodLab/UNI ViT-L/16; gated checkpoint resolved via
    ``--uni_feature_ckpt`` / ``VSLITE_UNI_CKPT`` /
    ``foundation_models/UNI/pytorch_model.bin``).
  * Frozen, eval, ``torch.no_grad`` (forward is always no-grad).
  * Inputs in ``[-1, 1]``; ``[0, 1]`` mapping + ImageNet mean/std is applied
    inside the extractor (same as netF).
  * Feature maps: transformer blocks 2/5/8/11 patch tokens reshaped to
    ``(B, 1024, H', W')`` (``H' = size / 16``).
  * Per stage map: channel-wise unit-norm at each spatial location, squared
    difference, spatial mean (then mean over channels).
  * Mean over the 4 blocks with equal weights (no learned 1x1 NetLin layers).

Note: this is not comparable to official AlexNet/VGG LPIPS, ConvNeXt-TCGA
LPIPS, or Inception-FID.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from util.lpips_convnext import stage_lin0_distance


class LPIPSUNI(nn.Module):
    """Frozen UNI Zhang-lin0 distance. Forward is always no-grad."""

    def __init__(
        self,
        ckpt_path: str | None = None,
        input_size: int | None = None,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        from models.networks.architecture import UNIFeatureExtractor

        self.backbone = UNIFeatureExtractor(
            opt=None,
            pretrained=True,
            freeze_weights=True,
            uni_input_size=input_size if input_size is not None else 1024,
            uni_ckpt_path=ckpt_path,
            grad_checkpoint=grad_checkpoint,
        )
        self.backbone.eval()
        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):  # noqa: ARG002
        # Metric network: stay in eval even if a parent module is set to train.
        return super().train(False)

    @property
    def ckpt_path(self) -> str | None:
        return self.backbone.ckpt_path

    @property
    def input_size(self) -> int:
        return self.backbone.input_size

    @torch.no_grad()
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """``pred`` / ``target``: BCHW RGB in ``[-1, 1]``. Returns ``(B,)`` distances."""
        if pred.ndim == 3:
            pred = pred.unsqueeze(0)
        if target.ndim == 3:
            target = target.unsqueeze(0)
        if pred.shape != target.shape:
            raise ValueError(
                f"lpips_uni expected matching pred/target shapes, "
                f"got {tuple(pred.shape)} vs {tuple(target.shape)}"
            )
        if pred.size(1) != 3:
            raise ValueError(
                f"lpips_uni expected 3-channel RGB, got {tuple(pred.shape)}"
            )
        pred = pred.float()
        target = target.float()
        feats_p = self.backbone(pred)
        feats_t = self.backbone(target)
        if len(feats_p) != len(feats_t) or len(feats_p) != len(self.backbone.take_blocks):
            raise RuntimeError(
                f"Expected {len(self.backbone.take_blocks)} UNI block maps, got "
                f"{len(feats_p)} and {len(feats_t)}"
            )
        dists = [stage_lin0_distance(a, b) for a, b in zip(feats_p, feats_t)]
        return torch.stack(dists, dim=0).mean(dim=0)


def build_lpips_uni(
    *,
    ckpt_path: str | None = None,
    input_size: int | None = None,
    device: torch.device | str = "cpu",
) -> LPIPSUNI:
    """Construct one frozen extractor (call once per run; do not reload each epoch)."""
    mod = LPIPSUNI(ckpt_path=ckpt_path, input_size=input_size)
    mod.to(device)
    mod.eval()
    return mod


def resolve_lpips_uni_ckpt(ckpt_path: str | None = None) -> str | None:
    """Resolve UNI weights for the test metric; return ``None`` when unavailable.

    Unlike the training encoder (hard requirement), the metric is optional:
    a missing checkpoint disables ``lpips_uni`` instead of failing the job.
    """
    try:
        from models.networks.architecture import resolve_uni_feature_ckpt

        return resolve_uni_feature_ckpt(ckpt_path)
    except FileNotFoundError:
        return None
