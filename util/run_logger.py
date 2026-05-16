"""Lightweight run-logger: emits ``opt.txt``, ``loss_log.txt`` and ``iter.txt``.

Format mirrors what ``util/visualizer.py`` writes for the baseline so existing
plotting tools (``util/plot_loss_curves.py``) keep working.

Used by training scripts that previously had no on-disk text logs (SwinGAN,
NAFNet, Restormer, VQGAN, I2SB wrapper, BBDM wrapper).

**Unified ``loss_log.txt`` convention:** the ``step`` argument to ``append_loss``
is always **training samples processed in the current epoch** (resets to 0 at
each epoch start), matching ``util/iter_counter.py`` ``epoch_iter`` / baseline
pix2pix. Use :func:`should_log_epoch_samples` for print cadence.
"""

from __future__ import annotations

import os
import time
from typing import Any, Mapping, Optional

# Default match baseline ``TrainOptions.print_freq`` and historical VQGAN hard-code.
DEFAULT_TRAIN_LOG_PRINT_FREQ = 100


def should_log_epoch_samples(epoch_samples: int, last_batch_size: int, print_freq: int) -> bool:
    """True when a text log line should be emitted for this batch (epoch-local).

    After processing a batch, ``epoch_samples`` is the total samples seen this
    epoch and ``last_batch_size`` is that batch's size. Mirrors baseline
    ``(steps % print_freq) < batch_size`` but scoped per epoch so ``iters`` in
    ``loss_log.txt`` rises to ~dataset size then resets each epoch.
    """
    if print_freq <= 0:
        return False
    return (int(epoch_samples) % int(print_freq)) < int(last_batch_size)


class RunLogger:
    def __init__(
        self,
        run_dir: str,
        *,
        name: str = "run",
        opt_filename: str = "opt.txt",
        loss_filename: str = "loss_log.txt",
        iter_filename: str = "iter.txt",
    ) -> None:
        self.run_dir = run_dir
        self.name = name
        os.makedirs(run_dir, exist_ok=True)
        self.opt_path = os.path.join(run_dir, opt_filename)
        self.loss_path = os.path.join(run_dir, loss_filename)
        self.iter_path = os.path.join(run_dir, iter_filename)
        self._loss_header_written = os.path.exists(self.loss_path)

    # -- opt.txt -----------------------------------------------------------

    def dump_opts(self, opt: Mapping[str, Any]) -> None:
        """Write a sorted ``key: value`` listing of options to ``opt.txt``."""
        flat: dict[str, Any] = {}
        self._flatten(opt, prefix="", out=flat)
        with open(self.opt_path, "w") as f:
            f.write("------------ Options -------------\n")
            for k in sorted(flat):
                f.write(f"{k}: {flat[k]}\n")
            f.write("-------------- End ----------------\n")

    @classmethod
    def _flatten(cls, value: Any, *, prefix: str, out: dict[str, Any]) -> None:
        if isinstance(value, Mapping):
            for k, v in value.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                cls._flatten(v, prefix=key, out=out)
        else:
            out[prefix or "value"] = value

    # -- loss_log.txt ------------------------------------------------------

    def _ensure_loss_header(self) -> None:
        if self._loss_header_written:
            return
        ts = time.strftime("%c")
        with open(self.loss_path, "a") as f:
            f.write(f"================ Training Loss ({ts}) ================\n")
        self._loss_header_written = True

    def append_loss(
        self,
        epoch: int,
        step: int,
        losses: Mapping[str, float],
        *,
        lr: Optional[float] = None,
        time_per_iter: Optional[float] = None,
    ) -> None:
        """Append one line to ``loss_log.txt``.

        ``step`` must be **epoch-local**: number of training samples processed
        so far this epoch (not a global step across epochs).
        """
        self._ensure_loss_header()
        parts = [f"(epoch: {int(epoch)}, iters: {int(step)}"]
        if time_per_iter is not None:
            parts.append(f"time: {float(time_per_iter):.3f}")
        if lr is not None:
            parts.append(f"lr: {float(lr):.6f}")
        head = ", ".join(parts) + ") "
        body_parts = []
        for k, v in losses.items():
            try:
                body_parts.append(f"{k}: {float(v):.3f}")
            except (TypeError, ValueError):
                body_parts.append(f"{k}: {v}")
        line = head + " ".join(body_parts)
        with open(self.loss_path, "a") as f:
            f.write(line + "\n")

    # -- iter.txt ----------------------------------------------------------

    def write_iter(self, epoch: int, step: int) -> None:
        with open(self.iter_path, "w") as f:
            f.write(f"{int(epoch)},{int(step)}\n")

    def read_iter(self) -> Optional[tuple[int, int]]:
        if not os.path.exists(self.iter_path):
            return None
        try:
            with open(self.iter_path) as f:
                line = f.readline().strip()
            if not line:
                return None
            a, b = line.split(",", 1)
            return int(a), int(b)
        except Exception:
            return None
