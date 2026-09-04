"""Quantized generator inference helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from cislnet.quantization.config import QuantAblationConfig
from cislnet.quantization.fp8_native import (
    NativeFP8Conv2d,
    NativeFP8ConvTranspose2d,
    NativeFP8Linear,
)


def _model_has_native_fp8(netG: torch.nn.Module) -> bool:
    for m in netG.modules():
        if isinstance(m, (NativeFP8Conv2d, NativeFP8ConvTranspose2d, NativeFP8Linear)):
            return True
    return False


def run_generator(
    netG: torch.nn.Module,
    input_tensor: torch.Tensor,
    *,
    precision: Optional[str] = None,
    config: Optional[QuantAblationConfig] = None,
    packed_payload: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Forward pass with dtype/mode appropriate for ``precision``."""
    prec = (precision or "fp32").lower()
    device = input_tensor.device
    cfg = config or QuantAblationConfig.from_env()
    mode = cfg.effective_inference_mode(prec)

    if prec == "fp16":
        x = input_tensor.half() if input_tensor.dtype != torch.float16 else input_tensor
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            return netG(x)

    if prec == "fp8":
        if mode == "fp8_native" or _model_has_native_fp8(netG):
            # NativeFP8 modules dequant weight_fp8 -> compute dtype inside each layer.
            # On H100/H200 this uses e4m3fn storage; set QUANT_FP8_NATIVE=1 at quant time.
            with torch.inference_mode():
                return netG(input_tensor)
        return netG(input_tensor)

    if prec in ("int4", "int8") and mode == "integer_gemm" and packed_payload is not None:
        # Reference integer path for FPGA golden validation (slow; not for full eval by default).
        from cislnet.quantization.integer.gemm import int8_conv2d_reference

        raise NotImplementedError(
            "Full-model integer_gemm inference requires layer-wise graph rewrite. "
            "Use QUANT_INFERENCE_MODE=fp32_dequant for eval; integer GEMM is exported via fpga_manifest "
            "and validated per-layer with cislnet/quantization/integer/gemm.py."
        )

    with torch.inference_mode():
        return netG(input_tensor)
