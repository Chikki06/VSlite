"""Native FP8 (e4m3fn) modules for H100/H200 — weights stored as float8, not fp32 bake."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

FP8_MAX = 448.0
FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)


def fp8_available() -> bool:
    return FP8_DTYPE is not None and torch.cuda.is_available()


def _is_sm89_plus() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) >= (8, 9)


def compute_fp8_weight_scale(
    w: torch.Tensor,
    *,
    act_scale: Optional[float] = None,
) -> torch.Tensor:
    row_max = w.detach().float().abs().amax(dim=tuple(range(1, w.dim())), keepdim=False).clamp(min=1e-8)
    if act_scale is not None and act_scale > 0:
        row_max = torch.maximum(row_max, torch.tensor(act_scale, device=w.device, dtype=row_max.dtype))
    return (row_max / FP8_MAX).to(dtype=w.dtype)


class NativeFP8Conv2d(nn.Module):
    """Conv2d with e4m3fn weight storage; compute in fp16/bf16 with dequantized weights."""

    def __init__(
        self,
        conv: nn.Conv2d,
        *,
        weight_scale: Optional[torch.Tensor] = None,
        store_native: bool = True,
    ):
        super().__init__()
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size

        scale = weight_scale if weight_scale is not None else compute_fp8_weight_scale(conv.weight)
        self.register_buffer("weight_scale", scale.view(-1, 1, 1, 1))

        if store_native and FP8_DTYPE is not None:
            q = (conv.weight.data / self.weight_scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
            self.register_buffer("weight_fp8", q)
            self.store_native = True
        else:
            self.register_parameter("weight_fp8", None)  # type: ignore
            self.register_buffer("_weight_fp32", conv.weight.data.clone())
            self.store_native = False

        if conv.bias is not None:
            self.register_buffer("bias", conv.bias.data.clone())
        else:
            self.bias = None

    def _effective_weight(self, compute_dtype: torch.dtype) -> torch.Tensor:
        if self.store_native and self.weight_fp8 is not None:
            w = self.weight_fp8.to(compute_dtype) * self.weight_scale.to(compute_dtype)
        else:
            w = self._weight_fp32
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._effective_weight(x.dtype)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class NativeFP8ConvTranspose2d(nn.Module):
    def __init__(
        self,
        conv: nn.ConvTranspose2d,
        *,
        weight_scale: Optional[torch.Tensor] = None,
        store_native: bool = True,
    ):
        super().__init__()
        for attr in (
            "stride",
            "padding",
            "dilation",
            "groups",
            "in_channels",
            "out_channels",
            "kernel_size",
            "output_padding",
        ):
            setattr(self, attr, getattr(conv, attr))

        scale = weight_scale if weight_scale is not None else compute_fp8_weight_scale(conv.weight)
        self.register_buffer("weight_scale", scale.view(-1, 1, 1, 1))

        if store_native and FP8_DTYPE is not None:
            q = (conv.weight.data / self.weight_scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
            self.register_buffer("weight_fp8", q)
            self.store_native = True
        else:
            self.register_parameter("weight_fp8", None)  # type: ignore
            self.register_buffer("_weight_fp32", conv.weight.data.clone())
            self.store_native = False

        if conv.bias is not None:
            self.register_buffer("bias", conv.bias.data.clone())
        else:
            self.bias = None

    def _effective_weight(self, compute_dtype: torch.dtype) -> torch.Tensor:
        if self.store_native and self.weight_fp8 is not None:
            return self.weight_fp8.to(compute_dtype) * self.weight_scale.to(compute_dtype)
        return self._weight_fp32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._effective_weight(x.dtype)
        return F.conv_transpose2d(
            x,
            w,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )


class NativeFP8Linear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        *,
        weight_scale: Optional[torch.Tensor] = None,
        store_native: bool = True,
    ):
        super().__init__()
        scale = weight_scale if weight_scale is not None else compute_fp8_weight_scale(linear.weight)
        self.register_buffer("weight_scale", scale.view(-1, 1))

        if store_native and FP8_DTYPE is not None:
            q = (linear.weight.data / self.weight_scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
            self.register_buffer("weight_fp8", q)
            self.store_native = True
        else:
            self.register_parameter("weight_fp8", None)  # type: ignore
            self.register_buffer("_weight_fp32", linear.weight.data.clone())
            self.store_native = False

        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.clone())
        else:
            self.bias = None

    def _effective_weight(self, compute_dtype: torch.dtype) -> torch.Tensor:
        if self.store_native and self.weight_fp8 is not None:
            return self.weight_fp8.to(compute_dtype) * self.weight_scale.to(compute_dtype)
        return self._weight_fp32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._effective_weight(x.dtype)
        return F.linear(x, w, self.bias)


def _replace_module(root: nn.Module, name: str, new_mod: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)


def convert_to_native_fp8(
    model: nn.Module,
    *,
    activation_scales: Optional[Dict[str, float]] = None,
    use_act_scale: bool = False,
    store_native: bool = True,
) -> Tuple[nn.Module, str]:
    """Replace Conv/Linear with NativeFP8* modules (Conv2d-safe, unlike torchao static)."""
    activation_scales = activation_scales or {}
    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, (NativeFP8Conv2d, NativeFP8ConvTranspose2d, NativeFP8Linear)):
            continue
        act = activation_scales.get(name) if use_act_scale else None
        if isinstance(module, nn.Conv2d):
            wrapped = NativeFP8Conv2d(module, weight_scale=None, store_native=store_native)
            if act is not None:
                wrapped.weight_scale.copy_(compute_fp8_weight_scale(module.weight, act_scale=act).view(-1, 1, 1, 1))
            _replace_module(model, name, wrapped)
            replaced += 1
        elif isinstance(module, nn.ConvTranspose2d):
            wrapped = NativeFP8ConvTranspose2d(module, weight_scale=None, store_native=store_native)
            if act is not None:
                wrapped.weight_scale.copy_(compute_fp8_weight_scale(module.weight, act_scale=act).view(-1, 1, 1, 1))
            _replace_module(model, name, wrapped)
            replaced += 1
        elif isinstance(module, nn.Linear) and module.weight.dim() == 2:
            wrapped = NativeFP8Linear(module, weight_scale=None, store_native=store_native)
            if act is not None:
                wrapped.weight_scale.copy_(compute_fp8_weight_scale(module.weight, act_scale=act).view(-1, 1))
            _replace_module(model, name, wrapped)
            replaced += 1

    scheme = "fp8_e4m3fn_native" if store_native and FP8_DTYPE is not None else "fp8_e4m3fn_native_fp32_storage"
    return model, scheme
