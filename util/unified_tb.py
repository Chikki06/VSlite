from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _repo_root() -> Path:
    # This file lives at <repo>/util/unified_tb.py
    return Path(__file__).resolve().parents[1]


def _sanitize_component(s: str) -> str:
    s = str(s).strip().replace(" ", "_")
    # Keep it filesystem-friendly.
    keep = []
    for ch in s:
        if ch.isalnum() or ch in {"-", "_", ".", "="}:
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep)
    return out if out else "run"


@dataclass(frozen=True)
class UnifiedTB:
    """
    Small helper to standardize TensorBoard log layout across training scripts.

    Layout:
      <repo>/results/unified/<model>/<run>/events.out.tfevents...
    """

    model: str
    run: str
    root: Path

    @property
    def log_dir(self) -> Path:
        return self.root / _sanitize_component(self.model) / _sanitize_component(self.run)


def resolve_unified_tb(model: str, run: str, root: Optional[str] = None) -> UnifiedTB:
    """
    Resolve unified TensorBoard directory. The root can be overridden via:
      - function arg `root`, or
      - env var `UNIFIED_TB_ROOT`
    """
    if root is None or str(root).strip() == "":
        root = os.environ.get("UNIFIED_TB_ROOT", "")
    if root is None or str(root).strip() == "":
        root_path = _repo_root() / "results" / "unified"
    else:
        root_path = Path(str(root)).expanduser().resolve()
    return UnifiedTB(model=model, run=run, root=root_path)


def build_unified_writer(model: str, run: str):
    """
    Late import so scripts without tensorboard dependency can still import this module.
    """
    from torch.utils.tensorboard import SummaryWriter

    try:
        from cislnet.registry import canonical_model_tag

        model = canonical_model_tag(model)
    except ImportError:
        pass

    u = resolve_unified_tb(model=model, run=run)
    u.log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(u.log_dir)), u

