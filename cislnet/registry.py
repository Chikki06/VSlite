"""Model registry for unified TensorBoard and results paths."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# model tag -> default results subdirectory name
MODEL_REGISTRY: dict[str, str] = {
    "cislnet": "cislnet",
    "swingan": "swingan",
    "restormer": "restormer",
    "nafnet": "nafnet",
    "vqgan": "vqgan",
    "bbdm": "bbdm",
    "i2sb": "i2sb",
}

# Legacy unified TB / results tags -> canonical
LEGACY_MODEL_ALIASES: dict[str, str] = {
    "baseline_contrastive_pix2pix": "cislnet",
    "swingan_i2i": "swingan",
    "restormer_i2i": "restormer",
    "nafnet_i2i": "nafnet",
    "vqgan_vs": "vqgan",
    "bbdm_vs": "bbdm",
    "i2sb_vs": "i2sb",
}


def canonical_model_tag(model: str) -> str:
    m = str(model).strip()
    return LEGACY_MODEL_ALIASES.get(m, m)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def results_dir(model: str, run: str, *, root: Optional[Path] = None) -> Path:
    tag = canonical_model_tag(model)
    sub = MODEL_REGISTRY.get(tag, tag)
    base = root if root is not None else repo_root() / "results"
    return base / sub / str(run)


def unified_tb_dir(model: str, run: str, *, root: Optional[Path] = None) -> Path:
    import os

    if root is None:
        env = os.environ.get("UNIFIED_TB_ROOT", "").strip()
        root = Path(env).expanduser() if env else repo_root() / "results" / "unified"
    tag = canonical_model_tag(model)
    return root / tag / str(run)
