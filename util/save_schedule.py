"""Save-frequency helpers driving the unified ``~50 saves per run`` cadence.

Two schedule kinds are exposed:

- :func:`make_iter_schedule` for iter-counted training/test loops.
- :func:`make_epoch_schedule` for epoch-counted training (e.g. one save per
  epoch when ``total_epochs <= target_saves``).

Both return a ``should_save(step)`` callable. The first step (``0``) and the
last step (``total - 1``) always trigger a save so we have anchors at the
boundaries of a run.
"""

from __future__ import annotations

from typing import Callable


def _schedule_period(total: int, target_saves: int) -> int:
    if total <= 0:
        return 1
    target = max(1, int(target_saves))
    return max(1, int(total) // target)


def make_iter_schedule(
    total_iters: int,
    target_saves: int = 50,
) -> Callable[[int], bool]:
    """Return ``should_save(step_idx)`` for an iteration-counted loop.

    ``step_idx`` is 0-based. The schedule fires when:

    - ``step_idx == 0`` (start sanity save)
    - ``step_idx == total_iters - 1`` (end of run)
    - ``(step_idx + 1) % period == 0`` where
      ``period = max(1, total_iters // target_saves)``

    The end-of-run anchor is intentionally ``==`` (not ``>=``). On resume, the
    schedule is rebuilt from the current run's CLI args while ``global_step``
    is loaded from the checkpoint. If anything shrinks the computed total
    (e.g. ``batch_size`` was increased, or ``--epochs`` was lowered) then
    ``step_idx`` can land past ``last`` from the very first step — a ``>=``
    anchor would then fire on every single training step. Using ``==``
    contains the anchor to a single iteration and lets the periodic check
    drive cadence beyond ``last``.
    """
    period = _schedule_period(total_iters, target_saves)
    last = max(0, int(total_iters) - 1)

    def _should(step_idx: int) -> bool:
        if step_idx <= 0:
            return True
        if step_idx == last:
            return True
        return ((step_idx + 1) % period) == 0

    return _should


def make_epoch_schedule(
    total_epochs: int,
    target_saves: int = 50,
) -> Callable[[int], bool]:
    """Return ``should_save(epoch_idx)`` for an epoch-counted loop.

    ``epoch_idx`` is 0-based. See :func:`make_iter_schedule` for the reason
    the end anchor is ``==`` rather than ``>=`` (resume-safety).
    """
    period = _schedule_period(total_epochs, target_saves)
    last = max(0, int(total_epochs) - 1)

    def _should(epoch_idx: int) -> bool:
        if epoch_idx <= 0:
            return True
        if epoch_idx == last:
            return True
        return ((epoch_idx + 1) % period) == 0

    return _should


def iter_period(total_iters: int, target_saves: int = 50) -> int:
    """Return the iter period used by :func:`make_iter_schedule`."""
    return _schedule_period(total_iters, target_saves)


def epoch_period(total_epochs: int, target_saves: int = 50) -> int:
    """Return the epoch period used by :func:`make_epoch_schedule`."""
    return _schedule_period(total_epochs, target_saves)
