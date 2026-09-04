"""Zhang lin0 perceptual distance on the frozen ConvNeXt-Tiny TCGA encoder.

This is a **metric only** (no training loss, no NetLin layers, no official `lpips`
package, no AlexNet/VGG). Train and test must call the same implementation.

Quantity name: ``lpips_convnext_tcga``.

Recipe (Zhang et al. LPIPS lin0, equal-weight stages):
  * Backbone: ``models.networks.architecture.ConvNeXtFeatureExtractor``
    (``timm/convnext_tiny.in12k_ft_in1k`` + TCGA checkpoint).
  * Frozen, eval, ``torch.no_grad``.
  * Inputs in ``[-1, 1]``; ImageNet mean/std is applied inside the extractor
    (same as perceptual / netF).
  * Per stage map: channel-wise unit-norm at each spatial location, squared
    difference, spatial mean (then mean over channels so stages with more
    channels do not dominate).
  * Mean over stages 1–5 with equal weights (no learned 1x1 NetLin layers).
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CKPT_NAME = "convnext_t_dp_spv_supcon_1e-5"
_DEFAULT_CKPT = _REPO_ROOT / "tcga_tl_checkpoints" / _DEFAULT_CKPT_NAME


def resolve_tcga_feature_ckpt(ckpt_path: str | None = None) -> str:
    """Resolve TCGA ConvNeXt weights from CLI, ``VSLITE_CONVNEXT_FID_CKPT``, or default."""
    candidates: list[str] = []
    if ckpt_path is not None and str(ckpt_path).strip():
        candidates.append(str(ckpt_path).strip())
    env = os.environ.get("VSLITE_CONVNEXT_FID_CKPT", "").strip()
    if env:
        candidates.append(env)
    candidates.append(str(_DEFAULT_CKPT))

    seen: set[str] = set()
    tried: list[str] = []
    for raw in candidates:
        if raw in seen:
            continue
        seen.add(raw)
        for p in (Path(raw), _REPO_ROOT / raw):
            tried.append(str(p))
            if p.is_dir():
                expected = p / _DEFAULT_CKPT_NAME
                if expected.is_file():
                    return str(expected)
                files = sorted(x for x in p.iterdir() if x.is_file())
                if len(files) == 1:
                    return str(files[0])
                continue
            if p.is_file():
                return str(p)

    raise FileNotFoundError(
        "TCGA ConvNeXt checkpoint not found for lpips_convnext_tcga. "
        "Set --tcga_feature_ckpt or env VSLITE_CONVNEXT_FID_CKPT. "
        f"Tried: {tried}"
    )


def _unit_norm_channels(feat: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Zhang LPIPS: L2-normalize the channel vector at each spatial location."""
    denom = torch.sqrt(torch.sum(feat * feat, dim=1, keepdim=True) + eps)
    return feat / denom


def stage_lin0_distance(feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
    """Per-stage lin0 distance: channel unit-norm, squared diff, spatial+channel mean.

    Returns shape ``(B,)``.
    """
    a = _unit_norm_channels(feat_a)
    b = _unit_norm_channels(feat_b)
    # Equal channel contribution (no NetLin). Spatial mean of squared diffs,
    # then mean over channels so deeper stages (more C) do not dominate.
    return ((a - b) ** 2).mean(dim=(1, 2, 3))


class LPIPSConvNeXtTCGA(nn.Module):
    """Frozen ConvNeXt-Tiny TCGA Zhang-lin0 distance. Forward is always no-grad."""

    def __init__(self, ckpt_path: str | None = None):
        super().__init__()
        from models.networks.architecture import ConvNeXtFeatureExtractor

        resolved = resolve_tcga_feature_ckpt(ckpt_path)
        opt = SimpleNamespace(
            tcga_feature_ckpt=resolved,
            vgg_normalize_input=True,
            convnext_normalize_input=True,
        )
        self.backbone = ConvNeXtFeatureExtractor(
            opt=opt, pretrained=True, freeze_weights=True
        )
        self.backbone.eval()
        for p in self.parameters():
            p.requires_grad = False
        self.ckpt_path = resolved

    def train(self, mode: bool = True):  # noqa: ARG002
        # Metric network: stay in eval even if a parent module is set to train.
        return super().train(False)

    @torch.no_grad()
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """``pred`` / ``target``: BCHW RGB in ``[-1, 1]``. Returns ``(B,)`` distances."""
        if pred.ndim == 3:
            pred = pred.unsqueeze(0)
        if target.ndim == 3:
            target = target.unsqueeze(0)
        if pred.shape != target.shape:
            raise ValueError(
                f"lpips_convnext_tcga expected matching pred/target shapes, "
                f"got {tuple(pred.shape)} vs {tuple(target.shape)}"
            )
        if pred.size(1) != 3:
            raise ValueError(
                f"lpips_convnext_tcga expected 3-channel RGB, got {tuple(pred.shape)}"
            )
        pred = pred.float()
        target = target.float()
        feats_p = self.backbone(pred)
        feats_t = self.backbone(target)
        if len(feats_p) != 5 or len(feats_t) != 5:
            raise RuntimeError(
                f"Expected 5 ConvNeXt stage maps, got {len(feats_p)} and {len(feats_t)}"
            )
        dists = [stage_lin0_distance(a, b) for a, b in zip(feats_p, feats_t)]
        return torch.stack(dists, dim=0).mean(dim=0)


def build_lpips_convnext_tcga(
    *,
    ckpt_path: str | None = None,
    device: torch.device | str = "cpu",
) -> LPIPSConvNeXtTCGA:
    """Construct one frozen extractor (call once per run; do not reload each epoch)."""
    mod = LPIPSConvNeXtTCGA(ckpt_path=ckpt_path)
    mod.to(device)
    mod.eval()
    return mod
