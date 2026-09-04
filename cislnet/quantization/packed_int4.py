"""Packed int4 per-channel weight storage for integer PTQ."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

INT4_QMAX = 7  # signed 4-bit


def _iter_weight_tensors(model: nn.Module):
    for name, module in model.named_modules():
        w = getattr(module, "weight", None)
        if w is None or not isinstance(w, torch.Tensor):
            continue
        if w.dim() not in (2, 4):
            continue
        yield name, module, w


def quantize_weight_per_channel(
    w: torch.Tensor,
    *,
    act_scale: Optional[float] = None,
    percentile: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (int8 codes in [-7,7], per-row fp32 scales)."""
    flat = w.detach().float().reshape(w.shape[0], -1)
    if percentile >= 100.0:
        row_max = flat.abs().amax(dim=1, keepdim=True)
    else:
        q = max(0.0, min(100.0, percentile)) / 100.0
        row_max = torch.quantile(flat.abs(), q, dim=1, keepdim=True)
    row_max = row_max.clamp(min=1e-8)
    if act_scale is not None and act_scale > 0:
        row_max = torch.maximum(row_max, torch.tensor(act_scale, device=flat.device, dtype=flat.dtype))
    scale = (row_max / float(INT4_QMAX)).squeeze(1)
    q = torch.round(flat / scale.unsqueeze(1)).clamp(-INT4_QMAX, INT4_QMAX).to(torch.int8)
    return q.cpu(), scale.cpu().float()


def pack_nibbles(q: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 codes (int8 storage) into uint8 nibbles (2 weights / byte)."""
    q_u = (q.to(torch.int16) + 8).clamp(0, 15).to(torch.uint8).contiguous().view(-1)
    if q_u.numel() % 2 == 1:
        q_u = torch.cat([q_u, torch.zeros(1, dtype=torch.uint8)])
    lo = q_u[0::2]
    hi = q_u[1::2]
    return (lo | (hi << 4)).contiguous()


def unpack_nibbles(packed: torch.Tensor, numel: int) -> torch.Tensor:
    packed = packed.view(-1)
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    q_u = torch.empty(packed.numel() * 2, dtype=torch.uint8)
    q_u[0::2] = lo
    q_u[1::2] = hi
    q_u = q_u[:numel]
    return (q_u.to(torch.int16) - 8).to(torch.int8)


def dequantize_packed(entry: Dict[str, Any], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    shape = tuple(entry["shape"])
    n = int(torch.tensor(shape).prod())
    q = unpack_nibbles(entry["qpacked"], n).reshape(shape[0], -1)
    scale = entry["scale"].float()
    return (q.float() * scale.unsqueeze(1)).reshape(shape).to(dtype)


def pack_model_weights(
    model: nn.Module,
    activation_scales: Optional[Dict[str, float]] = None,
    *,
    use_act_scale: bool = False,
    percentile: float = 100.0,
    skip_fn=None,
) -> Dict[str, Dict[str, Any]]:
    activation_scales = activation_scales or {}
    packed: Dict[str, Dict[str, Any]] = {}
    with torch.no_grad():
        for name, _module, w in _iter_weight_tensors(model):
            if skip_fn is not None and skip_fn(name):
                continue
            act = activation_scales.get(name) if use_act_scale else None
            q, scale = quantize_weight_per_channel(
                w, act_scale=act, percentile=percentile
            )
            packed[name + ".weight"] = {
                "qpacked": pack_nibbles(q),
                "scale": scale,
                "shape": list(w.shape),
                "numel": int(q.numel()),
            }
    return packed


def collect_other_params(
    model: nn.Module,
    packed_keys: set[str],
    *,
    fp32: bool = True,
) -> Dict[str, torch.Tensor]:
    """Biases, BN buffers, and other non-packed tensors."""
    other: Dict[str, torch.Tensor] = {}
    for key, tensor in model.state_dict().items():
        if key in packed_keys:
            continue
        if not torch.is_tensor(tensor):
            continue
        t = tensor.detach().cpu()
        if tensor.is_floating_point() and not fp32:
            other[key] = t.half()
        else:
            other[key] = t.float() if tensor.is_floating_point() else t
    return other


def packed_to_state_dict(
    packed_weights: Dict[str, Dict[str, Any]],
    other_params: Dict[str, torch.Tensor],
    *,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for key, entry in packed_weights.items():
        state[key] = dequantize_packed(entry, dtype=dtype)
    for key, tensor in other_params.items():
        state[key] = tensor.float() if tensor.is_floating_point() and dtype == torch.float32 else tensor
    return state


def estimate_packed_bytes(packed_weights: Dict[str, Dict[str, Any]], other_params: Dict[str, torch.Tensor]) -> int:
    total = 0
    for entry in packed_weights.values():
        total += int(entry["qpacked"].numel())
        total += int(entry["scale"].numel()) * 4
    for t in other_params.values():
        total += t.numel() * t.element_size()
    return total
