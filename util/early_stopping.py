"""Validation-metric early stopping and best checkpoint tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class EarlyStoppingState:
    patience: int
    min_delta: float = 1e-4
    best_value: float = float("inf")
    best_epoch: int = 0
    epochs_without_improvement: int = 0
    stopped: bool = False

    def update(self, value: float, epoch: int) -> bool:
        """
        Update state with new metric. Returns True if this is a new best.
        """
        if value < self.best_value - self.min_delta:
            self.best_value = float(value)
            self.best_epoch = int(epoch)
            self.epochs_without_improvement = 0
            return True
        self.epochs_without_improvement += 1
        if self.patience > 0 and self.epochs_without_improvement >= self.patience:
            self.stopped = True
        return False

    def should_stop(self) -> bool:
        return self.patience > 0 and self.stopped


class EarlyStopping:
    """Monitor val L1 (lower is better) and trigger stop + best ckpt save."""

    def __init__(
        self,
        patience: int = 0,
        min_delta: float = 1e-4,
        *,
        on_improve: Optional[Callable[[], None]] = None,
    ) -> None:
        self.state = EarlyStoppingState(
            patience=int(patience),
            min_delta=float(min_delta),
        )
        self._on_improve = on_improve

    def step(self, metric: float, epoch: int) -> bool:
        """Returns True if training should stop after this epoch."""
        improved = self.state.update(float(metric), int(epoch))
        if improved and self._on_improve is not None:
            self._on_improve()
        return self.state.should_stop()

    def should_stop(self) -> bool:
        """Whether patience has been exhausted (delegates to state)."""
        return self.state.should_stop()
