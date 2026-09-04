"""Precision-specific quantization schemes for AttUnetGenerator."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from cislnet.quantization.calibrate import (
    calibration_batch_count,
    collect_activation_scales,
    iter_calibration_batches,
)
from cislnet.quantization.config import QuantAblationConfig
from cislnet.quantization.fp8_native import convert_to_native_fp8
from cislnet.quantization import packed_int4, packed_int8
from cislnet.quantization.fpga_export import export_fpga_manifest


def normalize_precision(precision: str) -> str:
    """Map legacy aliases to canonical precision tags."""
    p = str(precision).lower()
    if p == "fp4":
        return "int4"
    if p == "fp32":
        return "int32"
    return p


def _weight_modules(model: nn.Module):
    for name, module in model.named_modules():
        w = getattr(module, "weight", None)
        if w is None or not isinstance(w, torch.Tensor):
            continue
        if w.dim() not in (2, 4):
            continue
        yield name, module, w


def _weight_filter(mod, *_args):
    w = getattr(mod, "weight", None)
    return isinstance(w, torch.Tensor) and w.dim() in (2, 4)


def _bake_fp8_weights(
    model: nn.Module,
    activation_scales: Optional[Dict[str, float]] = None,
    *,
    use_act_scale: bool = False,
) -> str:
    """Legacy portable fp8 e4m3fn round-trip (weights stored as fp32)."""
    activation_scales = activation_scales or {}
    fp8_max = 448.0
    with torch.no_grad():
        for name, module, w in _weight_modules(model):
            row_max = w.data.abs().amax().clamp(min=1e-8)
            act = activation_scales.get(name) if use_act_scale else None
            if act is not None and act > 0:
                row_max = torch.maximum(row_max, torch.tensor(act, device=w.device, dtype=w.dtype))
            scale = row_max / fp8_max
            q = (w.data / scale).clamp(-fp8_max, fp8_max)
            if hasattr(torch, "float8_e4m3fn"):
                q = q.to(torch.float8_e4m3fn).to(w.dtype)
            module.weight.data = (q * scale).to(dtype=w.dtype)
    return "fp8_e4m3fn_calibrated_baked"


def _torchao_fp8_static(
    model: nn.Module,
    calib_loader,
    device: torch.device,
    max_batches: int,
) -> Tuple[nn.Module, str]:
    """Observer-based FP8 static PTQ: prepare -> calibrate -> convert."""
    from torchao.prototype.quantization.quant_api import Float8StaticActivationFloat8WeightConfig
    from torchao.quantization import quantize_

    quantize_(model, Float8StaticActivationFloat8WeightConfig(step="prepare"), filter_fn=_weight_filter)

    model.eval()
    with torch.no_grad():
        for batch in iter_calibration_batches(calib_loader, device, max_batches):
            model(batch["label"])

    quantize_(model, Float8StaticActivationFloat8WeightConfig(step="convert"), filter_fn=_weight_filter)
    return model, "torchao_fp8_static"


def _apply_integer_ptq(
    model: nn.Module,
    *,
    bits: int,
    activation_scales: Optional[Dict[str, float]],
    config: QuantAblationConfig,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    if bits == 4:
        packer = packed_int4
        scheme = "int4_packed_per_channel"
    elif bits == 8:
        packer = packed_int8
        scheme = "int8_packed_per_channel"
    else:
        raise ValueError(f"Unsupported integer bits={bits}")

    skip_fn = config.should_skip_layer if (config.skip_cbam or config.skip_patterns) else None
    packed = packer.pack_model_weights(
        model,
        activation_scales,
        use_act_scale=config.use_act_scale,
        percentile=config.calib_percentile,
        skip_fn=skip_fn,
    )
    packed_keys = set(packed.keys())
    other = packer.collect_other_params(model, packed_keys, fp32=config.store_params_fp32)
    if config.use_act_scale:
        scheme += "_act_calibrated"
    if config.calib_percentile < 100.0:
        scheme += f"_p{config.calib_percentile:g}"
    meta = {
        "scheme": scheme,
        "param_dtype": f"int{bits}",
        "storage": "packed",
        "packed_weights": packed,
        "other_params": other,
        "packed_bytes_est": packer.estimate_packed_bytes(packed, other),
        "quant_ablation": config.to_dict(),
    }
    return packed, other, scheme


def _is_sm_at_least_89() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from torchao.utils import is_sm_at_least_89 as _fn

        return bool(_fn())
    except Exception:
        major, minor = torch.cuda.get_device_capability()
        return (major, minor) >= (8, 9)


def apply_precision(
    model: nn.Module,
    precision: str,
    *,
    calib_loader=None,
    device: Optional[torch.device] = None,
    calib_num_patches: int = 512,
    calib_batch_size: int = 4,
    fp8_static: bool = False,
    config: Optional[QuantAblationConfig] = None,
) -> Dict[str, Any]:
    """
    Apply post-training quantization in-place.

    Pass ``config=QuantAblationConfig.from_env()`` for modular ablations.
    """
    precision = normalize_precision(precision)
    if config is None:
        config = QuantAblationConfig(
            calib_num_patches=calib_num_patches,
            calib_batch_size=calib_batch_size,
        )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    max_batches = calibration_batch_count(config.calib_num_patches, config.calib_batch_size)
    meta: Dict[str, Any] = {
        "precision": precision,
        "calib_num_patches": int(config.calib_num_patches),
        "calib_batch_size": int(config.calib_batch_size),
        "fp8_static": bool(fp8_static),
        "quant_ablation": config.to_dict(),
    }

    if precision == "fp16":
        model.half()
        meta["scheme"] = "fp16_half"
        meta["param_dtype"] = "float16"
        meta["storage"] = "state_dict"
        return meta

    if precision == "int32":
        meta["scheme"] = "int32_fp32_baseline"
        meta["param_dtype"] = "float32"
        meta["storage"] = "state_dict"
        return meta

    if precision == "fp8" and calib_loader is None:
        raise ValueError("fp8 PTQ requires calibration data; pass --dataroot")

    activation_scales = None
    if calib_loader is not None and precision in ("int4", "int8", "fp8"):
        activation_scales = collect_activation_scales(
            model,
            calib_loader,
            device,
            max_batches=max_batches,
            percentile=config.calib_percentile,
        )
        meta["activation_layers_observed"] = len(activation_scales)

    if precision == "fp8":
        if config.fp8_native:
            model, scheme = convert_to_native_fp8(
                model,
                activation_scales=activation_scales,
                use_act_scale=config.use_act_scale,
                store_native=True,
            )
            meta["scheme"] = scheme
            meta["param_dtype"] = "float8_e4m3fn"
            meta["storage"] = "native_fp8_module"
            return meta

        if _is_sm_at_least_89() and fp8_static:
            try:
                _, scheme = _torchao_fp8_static(model, calib_loader, device, max_batches)
                meta["scheme"] = scheme
                meta["param_dtype"] = "float8"
                meta["storage"] = "torchao_module"
                return meta
            except Exception as exc:
                meta["torchao_fp8_static_error"] = str(exc)
        scheme = _bake_fp8_weights(model, activation_scales, use_act_scale=config.use_act_scale)
        meta["scheme"] = scheme
        meta["param_dtype"] = "float32"
        meta["storage"] = "state_dict"
        return meta

    if precision == "int4":
        if calib_loader is None:
            raise ValueError("int4 PTQ requires calibration data; pass --dataroot")
        _packed, _other, scheme = _apply_integer_ptq(
            model, bits=4, activation_scales=activation_scales, config=config
        )
        meta.update(
            {
                "scheme": scheme,
                "param_dtype": "int4",
                "storage": "packed",
                "packed_weights": _packed,
                "other_params": _other,
                "packed_bytes_est": packed_int4.estimate_packed_bytes(_packed, _other),
            }
        )
        if config.fpga_export:
            meta["fpga_manifest"] = export_fpga_manifest(
                _packed,
                _other,
                out_dir=str(config.extra.get("fpga_out_dir", "fpga_export")),
                precision="int4",
                config_dict=config.to_dict(),
                export_format=config.fpga_export_format,
            )
        return meta

    if precision == "int8":
        if calib_loader is None:
            raise ValueError("int8 PTQ requires calibration data; pass --dataroot")
        _packed, _other, scheme = _apply_integer_ptq(
            model, bits=8, activation_scales=activation_scales, config=config
        )
        meta.update(
            {
                "scheme": scheme,
                "param_dtype": "int8",
                "storage": "packed",
                "packed_weights": _packed,
                "other_params": _other,
                "packed_bytes_est": packed_int8.estimate_packed_bytes(_packed, _other),
            }
        )
        if config.fpga_export:
            meta["fpga_manifest"] = export_fpga_manifest(
                _packed,
                _other,
                out_dir=str(config.extra.get("fpga_out_dir", "fpga_export")),
                precision="int8",
                config_dict=config.to_dict(),
                export_format=config.fpga_export_format,
            )
        return meta

    raise ValueError(
        f"Unsupported precision={precision!r}. Expected fp16, fp8, int4, int8, or int32."
    )
