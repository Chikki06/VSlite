"""Tests for ``util.save_schedule`` cadence helpers.

The regression test of interest is the resume scenario for VQGAN/SwinGAN/
Restormer/NAFNet trainers: when training resumes from a checkpoint and the
freshly-computed ``total_iters`` is smaller than the loaded ``global_step``
(e.g. ``batch_size`` was increased between runs so ``len(train_loader)``
shrank), the schedule must NOT fire on every step.
"""

from __future__ import annotations

from util.save_schedule import make_epoch_schedule, make_iter_schedule


def test_iter_schedule_fires_at_boundaries_and_period() -> None:
    total = 1000
    target = 10
    should = make_iter_schedule(total, target_saves=target)

    assert should(0) is True
    assert should(total - 1) is True

    fires = [i for i in range(total) if should(i)]
    assert len(fires) >= target


def test_iter_schedule_does_not_fire_every_step_when_step_exceeds_total() -> None:
    """Regression: bs increased on resume so ``last`` is below ``global_step``.

    The historical ``step_idx >= last`` end-anchor caused every step beyond
    ``last`` to fire, producing one preview PNG per training iteration.
    """
    total_iters = 375_275  # post-resume (bs=8) total
    resume_step = 390_286  # global_step loaded from a bs=4 checkpoint
    should = make_iter_schedule(total_iters, target_saves=100)

    fires_in_window = sum(1 for s in range(resume_step, resume_step + 500) if should(s))
    assert fires_in_window <= 1, (
        f"schedule fired {fires_in_window} times in a 500-step window past `last`; "
        "expected at most 1 (period-driven, not every-step)."
    )


def test_iter_schedule_periodic_fires_continue_past_total() -> None:
    total = 1000
    should = make_iter_schedule(total, target_saves=10)
    period = total // 10

    far = 5 * total
    aligned = far + (period - ((far + 1) % period)) % period
    if (aligned + 1) % period != 0:
        aligned += 1
    assert should(aligned) is True


def test_epoch_schedule_does_not_fire_every_epoch_past_total() -> None:
    total_epochs = 50
    target = 10
    should = make_epoch_schedule(total_epochs, target_saves=target)

    window = 20
    fires_past_last = sum(1 for e in range(total_epochs, total_epochs + window) if should(e))
    period = max(1, total_epochs // target)
    expected_periodic = window // period + 1
    assert fires_past_last < window, (
        f"epoch schedule fired {fires_past_last} times in a {window}-epoch window past "
        "`last`; expected periodic cadence, not every-epoch."
    )
    assert fires_past_last <= expected_periodic, (
        f"epoch schedule fired {fires_past_last} times; periodic bound is "
        f"{expected_periodic} (period={period})."
    )
