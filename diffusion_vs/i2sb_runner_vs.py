"""I2SB Runner subclass that routes image saving through the unified saver.

This module exists so the wrapper (``scripts/train_i2sb_vs.py``) does not need
to edit anything under ``third_party/I2SB``: we override ``evaluation`` to
replace upstream tensorboard image logging with ``util.sample_grid``.

A small log-tee is also provided to mirror upstream ``train_it ...`` lines
into the canonical ``loss_log.txt`` so all models have the baseline-style
on-disk loss log.
"""

from __future__ import annotations

import re
from typing import Callable

import torch

from util.run_logger import RunLogger
from util.sample_grid import save_unified_sample, split_batch_to_list


def make_i2sb_runner_class():
    """Return a subclass of upstream ``i2sb.Runner`` (imported lazily)."""

    from i2sb import Runner  # type: ignore

    class I2SBVSRunner(Runner):  # type: ignore[misc]
        _samples_dir: str
        _run_logger: RunLogger
        _should_save_image: Callable[[int], bool]

        def attach_unified_saver(
            self,
            *,
            samples_dir: str,
            run_logger: RunLogger,
            should_save_image: Callable[[int], bool],
        ) -> None:
            self._samples_dir = samples_dir
            self._run_logger = run_logger
            self._should_save_image = should_save_image

        @torch.no_grad()
        def evaluation(self, opt, it, val_loader, corrupt_method):  # type: ignore[override]
            log = self.log
            log.info(f"========== Evaluation started: iter={it} ==========")

            img_clean, img_corrupt, mask, _y, cond = self.sample_batch(
                opt, val_loader, corrupt_method
            )
            x1 = img_corrupt.to(opt.device)
            xs, _pred_x0s = self.ddpm_sampling(
                opt,
                x1,
                mask=mask,
                cond=cond,
                clip_denoise=opt.clip_denoise,
                verbose=opt.global_rank == 0,
            )
            img_recon = xs[:, 0, ...].detach().cpu()

            if (
                getattr(self, "_should_save_image", None) is not None
                and self._should_save_image(int(it))
            ):
                reals = split_batch_to_list(img_clean.detach().cpu())[:8]
                preds = split_batch_to_list(img_recon)[:8]
                name = f"epoch001_step{int(it):09d}"
                save_unified_sample(self._samples_dir, name, reals, preds)
                self._run_logger.write_iter(1, int(it))

            log.info(f"========== Evaluation finished: iter={it} ==========")
            torch.cuda.empty_cache()

    return I2SBVSRunner


class LossLogTee:
    """Wrap an i2sb ``Logger`` so ``info`` lines also append to loss_log.txt."""

    _LOSS_RE = re.compile(
        r"train_it\s+(\d+)/(\d+).*?lr:([0-9.eE+\-]+).*?loss:([+\-0-9.eE]+)"
    )

    def __init__(self, base_log, run_logger: RunLogger) -> None:
        self._base = base_log
        self._run_logger = run_logger

    def __getattr__(self, name):
        return getattr(self._base, name)

    def info(self, *args, **kwargs):
        self._base.info(*args, **kwargs)
        if not args:
            return
        msg = args[0] if isinstance(args[0], str) else " ".join(str(a) for a in args)
        m = self._LOSS_RE.search(msg)
        if not m:
            return
        try:
            it = int(m.group(1))
            lr = float(m.group(3))
            loss = float(m.group(4))
            self._run_logger.append_loss(
                epoch=1,
                step=it,
                losses={"loss": loss},
                lr=lr,
            )
        except Exception:
            pass
"""I2SB Runner subclass that routes image saving through the unified saver.

This module exists so the wrapper (``scripts/train_i2sb_vs.py``) does not need
to edit anything under ``third_party/I2SB``: we override ``evaluation`` to
replace upstream tensorboard image logging with ``util.sample_grid``.

A small log-tee is also provided to mirror upstream ``train_it ...`` lines
into the canonical ``loss_log.txt`` so all models have the baseline-style
on-disk loss log.
"""

from __future__ import annotations

import re
from typing import Callable

import torch

from util.run_logger import RunLogger
from util.sample_grid import save_unified_sample, split_batch_to_list


def make_i2sb_runner_class():
    """Return a subclass of upstream ``i2sb.Runner`` (imported lazily)."""
    from i2sb import Runner  # type: ignore

    class I2SBVSRunner(Runner):  # type: ignore[misc]
        _samples_dir: str
        _run_logger: RunLogger
        _should_save_image: Callable[[int], bool]

        def attach_unified_saver(
            self,
            *,
            samples_dir: str,
            run_logger: RunLogger,
            should_save_image: Callable[[int], bool],
        ) -> None:
            self._samples_dir = samples_dir
            self._run_logger = run_logger
            self._should_save_image = should_save_image

        @torch.no_grad()
        def evaluation(self, opt, it, val_loader, corrupt_method):  # type: ignore[override]
            log = self.log
            log.info(f"========== Evaluation started: iter={it} ==========")
            img_clean, img_corrupt, mask, _y, cond = self.sample_batch(
                opt, val_loader, corrupt_method
            )
            x1 = img_corrupt.to(opt.device)
            xs, _pred_x0s = self.ddpm_sampling(
                opt,
                x1,
                mask=mask,
                cond=cond,
                clip_denoise=opt.clip_denoise,
                verbose=opt.global_rank == 0,
            )
            img_recon = xs[:, 0, ...].detach().cpu()
            if getattr(self, "_should_save_image", None) is not None and self._should_save_image(int(it)):
                reals = split_batch_to_list(img_clean.detach().cpu())[:8]
                preds = split_batch_to_list(img_recon)[:8]
                name = f"epoch001_step{int(it):09d}"
                save_unified_sample(self._samples_dir, name, reals, preds)
                self._run_logger.write_iter(1, int(it))
            log.info(f"========== Evaluation finished: iter={it} ==========")
            torch.cuda.empty_cache()

    return I2SBVSRunner


class LossLogTee:
    """Wrap an i2sb ``Logger`` so ``info`` lines also append to loss_log.txt."""

    _LOSS_RE = re.compile(
        r"train_it\s+(\d+)/(\d+).*?lr:([0-9.eE+\-]+).*?loss:([+\-0-9.eE]+)"
    )

    def __init__(self, base_log, run_logger: RunLogger) -> None:
        self._base = base_log
        self._run_logger = run_logger

    def __getattr__(self, name):
        return getattr(self._base, name)

    def info(self, *args, **kwargs):
        self._base.info(*args, **kwargs)
        if not args:
            return
        msg = args[0] if isinstance(args[0], str) else " ".join(str(a) for a in args)
        m = self._LOSS_RE.search(msg)
        if not m:
            return
        try:
            it = int(m.group(1))
            lr = float(m.group(3))
            loss = float(m.group(4))
            self._run_logger.append_loss(
                epoch=1,
                step=it,
                losses={"loss": loss},
                lr=lr,
            )
        except Exception:
            pass
