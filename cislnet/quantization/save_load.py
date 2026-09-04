"""Save and load quantized generator artifacts."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import torch

from cislnet.quantization.build import load_opt
from cislnet.quantization.packed_int4 import packed_to_state_dict as int4_to_state_dict
from cislnet.quantization.packed_int8 import packed_to_state_dict as int8_to_state_dict
from cislnet.quantization.schemes import normalize_precision


def artifact_paths(out_dir: str, precision: str) -> Dict[str, str]:
    prec = normalize_precision(precision)
    return {
        "dir": out_dir,
        "weights": os.path.join(out_dir, f"net_G_{prec}.pt"),
        "meta": os.path.join(out_dir, "quant_meta.json"),
    }


def _packed_to_state_dict(payload: dict, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    prec = normalize_precision(str(payload.get("precision", "int4")))
    packed = payload["packed_weights"]
    other = payload.get("other_params", {})
    if prec == "int8" or payload.get("format", "").startswith("int8"):
        return int8_to_state_dict(packed, other, dtype=dtype)
    return int4_to_state_dict(packed, other, dtype=dtype)


def save_quantized_generator(
    model: torch.nn.Module,
    out_dir: str,
    precision: str,
    meta: Dict[str, Any],
    *,
    source_ckpt: Optional[str] = None,
    opt_dir: Optional[str] = None,
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    prec = normalize_precision(precision)
    paths = artifact_paths(out_dir, prec)

    if prec in ("int4", "int8") and meta.get("packed_weights") is not None:
        payload = {
            "precision": prec,
            "format": meta.get("scheme", f"{prec}_packed_per_channel"),
            "packed_weights": meta["packed_weights"],
            "other_params": meta.get("other_params", {}),
        }
    elif prec == "fp8" and meta.get("storage") in ("torchao_module", "native_fp8_module"):
        payload = {
            "precision": "fp8",
            "format": meta.get("scheme", "fp8_e4m3fn_native"),
            "storage": meta.get("storage"),
            "model": model,
        }
    else:
        payload = {
            "state_dict": model.state_dict(),
            "precision": prec,
        }
    torch.save(payload, paths["weights"])

    full_meta = dict(meta)
    full_meta.pop("packed_weights", None)
    full_meta.pop("other_params", None)
    full_meta["precision"] = prec
    full_meta["weights_path"] = paths["weights"]
    full_meta["weights_bytes"] = os.path.getsize(paths["weights"])
    if source_ckpt:
        full_meta["source_ckpt"] = source_ckpt
    if opt_dir:
        full_meta["opt_dir"] = opt_dir

    with open(paths["meta"], "w") as f:
        json.dump(full_meta, f, indent=2)

    return paths


def _load_payload_state_dict(payload: dict, dtype: torch.dtype = torch.float32) -> Dict[str, torch.Tensor]:
    if payload.get("packed_weights") is not None:
        return _packed_to_state_dict(payload, dtype=dtype)
    if "state_dict" in payload:
        return payload["state_dict"]
    return payload


def load_quantized_generator(
    quant_ckpt: str,
    opt_dir: str,
    *,
    gpu_ids: Optional[list[int]] = None,
    source_ckpt: Optional[str] = None,
) -> Tuple[torch.nn.Module, Dict[str, Any], Any]:
    """
    Load quantized weights into a freshly built generator.

    int4/int8 packed checkpoints: dequant to fp32 by default (QUANT_INFERENCE_MODE=fp32_dequant).
    fp8 native_fp8_module: loads model with e4m3fn weight storage for true FP8 inference.
    """
    meta_path = os.path.join(os.path.dirname(quant_ckpt), "quant_meta.json")
    meta: Dict[str, Any] = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    precision = normalize_precision(str(meta.get("precision", "int32")))

    if os.path.isfile(quant_ckpt):
        payload = torch.load(quant_ckpt, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and payload.get("model") is not None and payload.get("storage") in (
            "torchao_module",
            "native_fp8_module",
        ):
            meta.setdefault("precision", payload.get("precision"))
            meta.setdefault("scheme", payload.get("format"))
            meta.setdefault("storage", "torchao_module")
            precision = normalize_precision(str(payload.get("precision", precision)))
            netG = payload["model"]
            netG.eval()
            if gpu_ids and torch.cuda.is_available():
                netG.cuda()
            opt = load_opt(opt_dir)
            opt.gpu_ids = gpu_ids or []
            return netG, meta, opt

    opt = load_opt(opt_dir)
    if gpu_ids is None:
        gpu_ids = [0] if torch.cuda.is_available() else []
    elif gpu_ids and not torch.cuda.is_available():
        gpu_ids = []
    opt.gpu_ids = gpu_ids

    from models.networks import define_G

    netG = define_G(opt)
    netG.eval()

    if os.path.isfile(quant_ckpt):
        payload = torch.load(quant_ckpt, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            meta.setdefault("precision", payload.get("precision"))
            precision = normalize_precision(str(payload.get("precision", precision)))
            load_dtype = torch.float16 if precision == "fp16" else torch.float32
            state = _load_payload_state_dict(payload, dtype=load_dtype)
            netG.load_state_dict(state, strict=False)
        else:
            netG.load_state_dict(payload, strict=False)
    elif source_ckpt and os.path.isfile(source_ckpt):
        weights = torch.load(source_ckpt, map_location="cpu", weights_only=True)
        netG.load_state_dict(weights, strict=False)

    if precision == "fp16":
        netG.half()

    if gpu_ids and torch.cuda.is_available():
        netG.cuda()

    netG.eval()
    return netG, meta, opt
