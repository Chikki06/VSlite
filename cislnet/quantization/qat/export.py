"""Export QAT-trained generator to packed int4/int8 artifacts."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from cislnet.quantization.config import QuantAblationConfig
from cislnet.quantization import packed_int4, packed_int8
from cislnet.quantization.qat.fake_quant import FakeQuantConv2d, FakeQuantConvTranspose2d, FakeQuantLinear
from cislnet.quantization.qat.prepare import disable_fake_quant
from cislnet.quantization.schemes import normalize_precision


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Replace fake-quant wrappers with inner conv/linear for PTQ pack."""
    for name, module in list(model.named_modules()):
        inner = None
        if isinstance(module, FakeQuantConv2d):
            inner = module.conv
        elif isinstance(module, FakeQuantConvTranspose2d):
            inner = module.conv
        elif isinstance(module, FakeQuantLinear):
            inner = module.linear
        if inner is None:
            continue
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], inner)
    return model


def export_qat_generator(
    model: nn.Module,
    config: QuantAblationConfig,
    *,
    source_ckpt: Optional[str] = None,
    opt_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Unwrap fake-quant, pack weights, return meta dict for save_quantized_generator().
    """
    disable_fake_quant(model)
    model = _unwrap_model(model)
    bits = int(config.bits)
    prec = normalize_precision("int4" if bits == 4 else "int8")
    packer = packed_int4 if bits == 4 else packed_int8

    packed = packer.pack_model_weights(model, activation_scales=None)
    packed_keys = set(packed.keys())
    other = packer.collect_other_params(model, packed_keys, fp32=config.store_params_fp32)
    scheme = f"qat_int{bits}_packed_per_channel"
    meta: Dict[str, Any] = {
        "precision": prec,
        "scheme": scheme,
        "param_dtype": f"int{bits}",
        "storage": "packed",
        "packed_weights": packed,
        "other_params": other,
        "packed_bytes_est": packer.estimate_packed_bytes(packed, other),
        "qat": True,
        "quant_ablation": config.to_dict(),
    }
    if source_ckpt:
        meta["source_ckpt"] = source_ckpt
    if opt_dir:
        meta["opt_dir"] = opt_dir
    return meta
