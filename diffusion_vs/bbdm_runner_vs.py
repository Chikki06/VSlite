"""BBDM Runner shim that uses the unified image saver.

We do not modify ``third_party/BBDM``. Instead, importing this module registers
``BBDMVSRunner`` under the existing registry name ``BBDMRunner`` so the wrapper
script can pick it up via ``third_party/BBDM/utils.get_runner``.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import torch

from util.run_logger import RunLogger
from util.sample_grid import save_unified_sample, split_batch_to_list
from cislnet.diffusion_hooks import batch_l1, make_diffusion_early_stop, make_diffusion_tb


def _first_n_from_batch(t: torch.Tensor, n: int = 8) -> torch.Tensor:
    if t.ndim == 3:
        return t.unsqueeze(0)[:n]
    if t.ndim != 4:
        raise ValueError(f"Expected CHW or BCHW, got {tuple(t.shape)}")
    return t[:n]


def register_bbdm_vs_runner() -> None:
    from Register import Registers  # type: ignore
    from runners.DiffusionBasedModelRunners.BBDMRunner import BBDMRunner  # type: ignore

    @Registers.runners.register_with_name("BBDMRunner")  # type: ignore[misc]
    class BBDMVSRunner(BBDMRunner):  # type: ignore[misc]
        _run_logger: Optional[RunLogger] = None
        _should_save_image: Optional[Callable[[int], bool]] = None
        _samples_dir: Optional[str] = None
        _tb: Optional[object] = None
        _early: Optional[object] = None
        _run_tag: str = "bbdm"
        _output_dir: Optional[str] = None

        def attach_unified_saver(
            self,
            *,
            run_logger: RunLogger,
            should_save_image: Callable[[int], bool],
            samples_dir: str,
            model_tag: str = "bbdm",
            run_tag: Optional[str] = None,
            output_dir: Optional[str] = None,
        ) -> None:
            self._run_logger = run_logger
            self._should_save_image = should_save_image
            self._samples_dir = samples_dir
            self._run_tag = str(model_tag)
            self._output_dir = output_dir
            run_name = run_tag or os.path.basename(str(output_dir or samples_dir).rstrip("/"))
            self._tb = make_diffusion_tb(self._run_tag, run_name, run_dir=output_dir)
            self._early = make_diffusion_early_stop()
            try:
                nc = int(getattr(self.config.model, "VS", {}).get("input_nc", 10))
            except Exception:
                nc = 10
            self._tb.log_meta(
                type("Cfg", (), {"num_channels": nc, "img_size": 256, "batch_size": 1, "grad_accum_steps": 1})(),
                epoch=0,
            )

        def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage="train", write=True):  # type: ignore[override]
            loss = super().loss_fn(net, batch, epoch, step, opt_idx=opt_idx, stage=stage, write=write)
            if (
                self.is_main_process
                and stage == "train"
                and self._run_logger is not None
                and (int(step) % 50 == 0)
            ):
                try:
                    lr = float(self.optimizer[0].param_groups[0]["lr"]) if self.optimizer else None
                except Exception:
                    lr = None
                self._run_logger.append_loss(
                    epoch=int(epoch) + 1,
                    step=int(step),
                    losses={"loss": float(loss.detach().mean().item())},
                    lr=lr,
                )
            return loss

        @torch.no_grad()
        def sample(self, net, batch, sample_path, stage="train"):  # type: ignore[override]
            # Only save once per sample_step; BaseRunner calls sample() twice (train+val).
            if stage != "val":
                return
            if self._samples_dir is None or self._should_save_image is None:
                return
            if not self._should_save_image(int(self.global_step)):
                return

            (x, _x_name), (x_cond, _x_cond_name) = batch
            x = _first_n_from_batch(x, 8).to(self.config.training.device[0])
            x_cond = _first_n_from_batch(x_cond, 8).to(self.config.training.device[0])

            sample = net.sample(x_cond, clip_denoised=self.config.testing.clip_denoised).detach().cpu()
            val_l1 = batch_l1(sample, x.detach().cpu())
            epoch = int(self.global_epoch) + 1
            if self._tb is not None:
                self._tb.log_scalar("l1/val", val_l1, epoch)
            if self._early is not None and self._early.state.patience > 0:
                should_stop = self._early.step(val_l1, epoch)
                if self._early.state.best_epoch == epoch and self._output_dir:
                    try:
                        from util.atomic_io import atomic_torch_save

                        atomic_torch_save(
                            torch.save,
                            {
                                "epoch": max(0, epoch - 1),
                                "global_step": int(self.global_step),
                                "model_state": net.state_dict(),
                            },
                            os.path.join(self._output_dir, "ckpt_best.pth"),
                        )
                    except Exception:
                        pass
                if should_stop and self._tb is not None:
                    self._tb.log_early_stop(self._early.state.best_value, epoch)
            reals = split_batch_to_list(x.detach().cpu())[:8]
            preds = split_batch_to_list(sample)[:8]
            name = f"epoch{epoch:03d}_step{int(self.global_step):09d}"
            paths = save_unified_sample(self._samples_dir, name, reals, preds)
            if self._tb is not None and paths:
                self._tb.log_preview_pngs(paths, epoch)

            if self._run_logger is not None:
                self._run_logger.write_iter(epoch, int(self.global_step))

        @torch.no_grad()
        def sample_to_eval(self, net, test_loader, sample_path):  # type: ignore[override]
            # Disable upstream per-sample PNG dumps. Unified test saving is handled by
            # our own test scripts under scripts/test_bbdm_vs.py.
            return


__all__ = ["register_bbdm_vs_runner"]

