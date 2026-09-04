"""Fake-quant modules with straight-through estimator (STE) for QAT."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cislnet.quantization.config import QuantAblationConfig


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.round(x) - x).detach()


def fake_quant_symmetric(
    x: torch.Tensor,
    scale: torch.Tensor,
    qmax: float,
    *,
    enabled: bool = True,
) -> torch.Tensor:
    if not enabled:
        return x
    scale = scale.clamp(min=1e-8)
    q = _ste_round(x / scale).clamp(-qmax, qmax)
    return q * scale


def compute_weight_scale(
    w: torch.Tensor,
    qmax: float,
    *,
    act_scale: Optional[float] = None,
    percentile: float = 100.0,
) -> torch.Tensor:
    flat = w.detach().float().reshape(w.shape[0], -1)
    if percentile >= 100.0:
        row_max = flat.abs().amax(dim=1)
    else:
        q = max(0.0, min(100.0, percentile)) / 100.0
        row_max = torch.quantile(flat.abs(), q, dim=1)
    row_max = row_max.clamp(min=1e-8)
    if act_scale is not None and act_scale > 0:
        row_max = torch.maximum(row_max, torch.tensor(act_scale, device=w.device, dtype=row_max.dtype))
    return (row_max / qmax).to(dtype=w.dtype)


class _FakeQuantMixin:
    config: QuantAblationConfig
    qmax: float
    module_name: str
    weight_scale: torch.Tensor
    act_scale: torch.Tensor

    def _maybe_fake_quant_act(self, x: torch.Tensor) -> torch.Tensor:
        if not self.config.qat_fake_quant_activation and not self.config.act_quant_at_inference:
            return x
        enabled = self.training or self.config.act_quant_at_inference
        return fake_quant_symmetric(x, self.act_scale, self.qmax, enabled=enabled)

    def _maybe_fake_quant_weight(self, w: torch.Tensor) -> torch.Tensor:
        if not self.config.qat_fake_quant_weight and not self.training:
            return w
        enabled = self.training or self.config.qat_fake_quant_weight
        scale = self.weight_scale.view(-1, *([1] * (w.dim() - 1)))
        return fake_quant_symmetric(w, scale, self.qmax, enabled=enabled)


class FakeQuantConv2d(_FakeQuantMixin, nn.Module):
    def __init__(
        self,
        conv: nn.Conv2d,
        config: QuantAblationConfig,
        *,
        qmax: float,
        module_name: str = "",
        act_scale: Optional[float] = None,
    ):
        super().__init__()
        self.conv = conv
        self.config = config
        self.qmax = qmax
        self.module_name = module_name
        ws = compute_weight_scale(
            conv.weight,
            qmax,
            act_scale=act_scale if config.use_act_scale else None,
            percentile=config.calib_percentile,
        )
        self.register_buffer("weight_scale", ws)
        init_act = float(act_scale) if act_scale and config.use_act_scale else 1.0
        self.register_buffer("act_scale", torch.tensor(init_act, dtype=conv.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._maybe_fake_quant_act(x)
        w = self._maybe_fake_quant_weight(self.conv.weight)
        return self.conv._conv_forward(x, w, self.conv.bias)


class FakeQuantConvTranspose2d(_FakeQuantMixin, nn.Module):
    def __init__(
        self,
        conv: nn.ConvTranspose2d,
        config: QuantAblationConfig,
        *,
        qmax: float,
        module_name: str = "",
        act_scale: Optional[float] = None,
    ):
        super().__init__()
        self.conv = conv
        self.config = config
        self.qmax = qmax
        self.module_name = module_name
        ws = compute_weight_scale(
            conv.weight,
            qmax,
            act_scale=act_scale if config.use_act_scale else None,
            percentile=config.calib_percentile,
        )
        self.register_buffer("weight_scale", ws)
        init_act = float(act_scale) if act_scale and config.use_act_scale else 1.0
        self.register_buffer("act_scale", torch.tensor(init_act, dtype=conv.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._maybe_fake_quant_act(x)
        w = self._maybe_fake_quant_weight(self.conv.weight)
        return F.conv_transpose2d(
            x,
            w,
            self.conv.bias,
            self.conv.stride,
            self.conv.padding,
            self.conv.output_padding,
            self.conv.groups,
            self.conv.dilation,
        )


class FakeQuantLinear(_FakeQuantMixin, nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        config: QuantAblationConfig,
        *,
        qmax: float,
        module_name: str = "",
        act_scale: Optional[float] = None,
    ):
        super().__init__()
        self.linear = linear
        self.config = config
        self.qmax = qmax
        self.module_name = module_name
        ws = compute_weight_scale(
            linear.weight,
            qmax,
            act_scale=act_scale if config.use_act_scale else None,
            percentile=config.calib_percentile,
        )
        self.register_buffer("weight_scale", ws)
        init_act = float(act_scale) if act_scale and config.use_act_scale else 1.0
        self.register_buffer("act_scale", torch.tensor(init_act, dtype=linear.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._maybe_fake_quant_act(x)
        w = self._maybe_fake_quant_weight(self.linear.weight)
        return F.linear(x, w, self.linear.bias)
