"""Apply / remove fake-quant wrappers on AttUnetGenerator."""

from __future__ import annotations

from typing import Dict, Optional

import torch.nn as nn

from cislnet.quantization.config import QuantAblationConfig
from cislnet.quantization.qat.fake_quant import FakeQuantConv2d, FakeQuantConvTranspose2d, FakeQuantLinear


def _qmax(config: QuantAblationConfig) -> float:
    return 7.0 if int(config.bits) == 4 else 127.0


def _replace_module(root: nn.Module, name: str, new_mod: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)


def prepare_qat_generator(
    model: nn.Module,
    config: QuantAblationConfig,
    *,
    activation_scales: Optional[Dict[str, float]] = None,
) -> nn.Module:
    """Swap Conv/Linear layers with fake-quant wrappers (in-place module tree surgery)."""
    activation_scales = activation_scales or {}
    qmax = _qmax(config)
    to_wrap: list[tuple[str, nn.Module, Optional[float]]] = []

    for name, module in model.named_modules():
        if config.should_skip_layer(name):
            continue
        if any(name.endswith(p) for p in config.weight_only_layers):
            act = None
        else:
            act = activation_scales.get(name)
            if not config.use_act_scale:
                act = None

        if isinstance(module, nn.Conv2d):
            to_wrap.append((name, module, act))
        elif isinstance(module, nn.ConvTranspose2d):
            to_wrap.append((name, module, act))
        elif isinstance(module, nn.Linear) and module.weight.dim() == 2:
            to_wrap.append((name, module, act))

    for name, module, act in to_wrap:
        if isinstance(module, (FakeQuantConv2d, FakeQuantConvTranspose2d, FakeQuantLinear)):
            continue
        if isinstance(module, nn.Conv2d):
            wrapped = FakeQuantConv2d(module, config, qmax=qmax, module_name=name, act_scale=act)
        elif isinstance(module, nn.ConvTranspose2d):
            wrapped = FakeQuantConvTranspose2d(module, config, qmax=qmax, module_name=name, act_scale=act)
        else:
            wrapped = FakeQuantLinear(module, config, qmax=qmax, module_name=name, act_scale=act)
        _replace_module(model, name, wrapped)

    model.train()
    return model


def enable_fake_quant(model: nn.Module) -> None:
    model.train()


def disable_fake_quant(model: nn.Module) -> None:
    """Eval mode + disable activation fake-quant for export / metrics."""
    model.eval()
