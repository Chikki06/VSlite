"""Multi-phase training schedules (patch size / epochs per phase)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ProgressivePhase:
    patch_size: int
    epochs: int


def parse_progressive_schedule(spec: str) -> List[ProgressivePhase]:
    """
    Parse ``PROGRESSIVE_SCHEDULE`` env form: ``256:10,512:40``.
    Returns empty list if spec is blank (single-phase training).
    """
    spec = str(spec or "").strip()
    if not spec:
        return []
    phases: List[ProgressivePhase] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid progressive phase {part!r}; expected PATCH:EPOCHS")
        ps, ep = part.split(":", 1)
        phases.append(ProgressivePhase(patch_size=int(ps), epochs=int(ep)))
    if not phases:
        return []
    return phases


def total_epochs_from_schedule(spec: str, default_epochs: int) -> int:
    phases = parse_progressive_schedule(spec)
    if not phases:
        return int(default_epochs)
    return sum(p.epochs for p in phases)


def phase_for_epoch(phases: List[ProgressivePhase], epoch_1based: int) -> Optional[ProgressivePhase]:
    """Return active phase for 1-based epoch index."""
    if not phases:
        return None
    acc = 0
    for ph in phases:
        acc += ph.epochs
        if epoch_1based <= acc:
            return ph
    return phases[-1]
