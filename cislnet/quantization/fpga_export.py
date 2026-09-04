"""FPGA export manifest for packed int8/int4 CISLNet generators."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import torch


def export_fpga_manifest(
    packed_weights: Dict[str, Dict[str, Any]],
    other_params: Dict[str, torch.Tensor],
    *,
    out_dir: str,
    precision: str,
    config_dict: Dict[str, Any],
    export_format: str = "manifest_v1",
) -> str:
    """
    Write layer-wise weight blobs + scales for custom FPGA GEMM or Vitis AI prep.

    Returns path to manifest JSON.
    """
    os.makedirs(out_dir, exist_ok=True)
    layers = []
    blobs_dir = os.path.join(out_dir, "blobs")
    os.makedirs(blobs_dir, exist_ok=True)

    for key, entry in sorted(packed_weights.items()):
        blob_name = key.replace(".", "_") + ".bin"
        blob_path = os.path.join(blobs_dir, blob_name)
        q = entry.get("q")
        if q is None and "qpacked" in entry:
            q = entry["qpacked"]
        if torch.is_tensor(q):
            q.cpu().numpy().tofile(blob_path)
        scale = entry["scale"]
        layers.append(
            {
                "name": key,
                "shape": entry.get("shape"),
                "blob": os.path.relpath(blob_path, out_dir),
                "scale_dtype": "float16",
                "scale_numel": int(scale.numel()) if torch.is_tensor(scale) else 0,
                "packed": "nibbles" if "qpacked" in entry else "int8",
            }
        )

    manifest: Dict[str, Any] = {
        "format": export_format,
        "model": "cislnet_attunet",
        "precision": precision,
        "patch_size": 1024,
        "quant_ablation": config_dict,
        "layers": layers,
        "other_param_keys": sorted(other_params.keys()),
        "notes": {
            "custom_fpga": "Use integer/gemm.py int8_conv2d_reference as golden; swap GEMM with your RTL kernel.",
            "vitis_ai": "Separate pathway: export ONNX -> Vitis-AI quantizer -> compile for DPUCAHX8H. Not direct from this manifest.",
            "dpu_op_support": "AttUnet (ResNet50 + CBAM + ConvTranspose) may need operator partitioning; DPU is INT8-focused.",
        },
    }
    path = os.path.join(out_dir, "fpga_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path
