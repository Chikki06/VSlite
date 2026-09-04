"""Modular quantization ablation configuration (PTQ, QAT, inference, FPGA export)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Tuple


@dataclass
class QuantAblationConfig:
    """
    Single config object for all quant paths. Toggle via env or CLI for ablations.

    Env prefix: QUANT_* (see jobs/_lib/QUANT_ABLATIONS.md).
    """

    # --- Precision ---
    bits: int = 8  # 4 | 8; fp8 uses separate fp8_native flag
    fp8_native: bool = True  # store/run e4m3fn weights (H100/H200), not fp32 bake

    # --- PTQ calibration ---
    use_act_scale: bool = False  # merge activation max into weight scale (SmoothQuant-style)
    act_quant_at_inference: bool = False  # quantize activations at runtime (requires use_act_scale)
    calib_percentile: float = 100.0  # 100=max; 99.9 clips outlier activations
    calib_num_patches: int = 512
    calib_batch_size: int = 4

    # --- Layer selection ---
    skip_cbam: bool = False  # keep CBAM MLP/conv at fp32
    skip_patterns: Tuple[str, ...] = ()  # substring match on module names
    weight_only_layers: Tuple[str, ...] = ()  # force weight-only scale even if use_act_scale

    # --- Storage ---
    store_params_fp32: bool = True  # BN buffers / biases in fp32 (not fp16)

    # --- QAT ---
    qat_enabled: bool = False
    qat_epochs: int = 5
    qat_lr: float = 1e-5
    qat_freeze_discriminator: bool = True
    qat_freeze_netf: bool = True
    qat_fake_quant_weight: bool = True
    qat_fake_quant_activation: bool = True

    # --- Inference ---
    inference_mode: str = "auto"  # auto | fp32_dequant | integer_gemm | fp8_native
    integer_gemm_validate: bool = False  # cross-check int GEMM vs dequant on sample batch

    # --- FPGA export ---
    fpga_export: bool = False
    fpga_export_format: str = "manifest_v1"  # manifest_v1 | vitis_hint

    extra: dict[str, Any] = field(default_factory=dict)

    def effective_inference_mode(self, precision: str) -> str:
        prec = str(precision).lower()
        if self.inference_mode != "auto":
            return self.inference_mode
        if prec == "fp8" and self.fp8_native:
            return "fp8_native"
        if prec in ("int4", "int8") and not self.act_quant_at_inference:
            return "integer_gemm" if os.environ.get("QUANT_INTEGER_GEMM", "0") == "1" else "fp32_dequant"
        return "fp32_dequant"

    def should_skip_layer(self, module_name: str) -> bool:
        if self.skip_cbam and "cbam" in module_name.lower():
            return True
        return any(p in module_name for p in self.skip_patterns)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["skip_patterns"] = list(self.skip_patterns)
        d["weight_only_layers"] = list(self.weight_only_layers)
        return d

    @classmethod
    def from_env(cls, env: Optional[dict[str, str]] = None) -> QuantAblationConfig:
        e = dict(os.environ) if env is None else dict(env)

        def _b(key: str, default: bool) -> bool:
            v = e.get(key, "")
            if v == "":
                return default
            return v.strip().lower() in ("1", "true", "yes", "on")

        def _f(key: str, default: float) -> float:
            v = e.get(key, "")
            return float(v) if v.strip() else default

        def _i(key: str, default: int) -> int:
            v = e.get(key, "")
            return int(v) if v.strip() else default

        skip = tuple(x.strip() for x in e.get("QUANT_SKIP_PATTERNS", "").split(",") if x.strip())
        wo = tuple(x.strip() for x in e.get("QUANT_WEIGHT_ONLY_LAYERS", "").split(",") if x.strip())

        bits = _i("QUANT_BITS", 8)
        if e.get("QUANT_PRECISION", "").lower() in ("int4", "fp4"):
            bits = 4
        elif e.get("QUANT_PRECISION", "").lower() == "int8":
            bits = 8

        return cls(
            bits=bits,
            fp8_native=_b("QUANT_FP8_NATIVE", True),
            use_act_scale=_b("QUANT_USE_ACT_SCALE", False),
            act_quant_at_inference=_b("QUANT_ACT_QUANT_INFERENCE", False),
            calib_percentile=_f("QUANT_CALIB_PERCENTILE", 100.0),
            calib_num_patches=_i("CALIB_NUM_PATCHES", _i("QUANT_CALIB_NUM_PATCHES", 512)),
            calib_batch_size=_i("CALIB_BATCH_SIZE", _i("QUANT_CALIB_BATCH_SIZE", 4)),
            skip_cbam=_b("QUANT_SKIP_CBAM", False),
            skip_patterns=skip,
            weight_only_layers=wo,
            store_params_fp32=_b("QUANT_STORE_PARAMS_FP32", True),
            qat_enabled=_b("QAT_ENABLED", False),
            qat_epochs=_i("QAT_EPOCHS", 5),
            qat_lr=_f("QAT_LR", 1e-5),
            qat_freeze_discriminator=_b("QAT_FREEZE_DISCRIMINATOR", True),
            qat_freeze_netf=_b("QAT_FREEZE_NETF", True),
            qat_fake_quant_weight=_b("QAT_FAKE_QUANT_WEIGHT", True),
            qat_fake_quant_activation=_b("QAT_FAKE_QUANT_ACTIVATION", True),
            inference_mode=e.get("QUANT_INFERENCE_MODE", "auto").lower(),
            integer_gemm_validate=_b("QUANT_INTEGER_GEMM_VALIDATE", False),
            fpga_export=_b("QUANT_FPGA_EXPORT", False),
            fpga_export_format=e.get("QUANT_FPGA_EXPORT_FORMAT", "manifest_v1"),
        )

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
