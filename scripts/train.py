"""
CISLNet (contrastive pix2pix) training for VSlite.

Formerly called baseline; unified framework hooks for TB, preview, early stopping.
"""

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import data
import util.util as util
from options.train_options import TrainOptions
from trainers import get_trainer
from util.batch_device import move_to_device
from util.iter_counter import IterationCounter
from util.save_schedule import make_iter_schedule
from util.tb_logger import UnifiedTBLogger
from util.preview_manager import PreviewManager
from util.early_stopping import EarlyStopping
from util.visualizer import Visualizer
from util.eval_caps import eval_max_batches_train, eval_max_batches_val, eval_max_batches_test
from util.run_logger import should_log_epoch_samples
from util.lpips_convnext import build_lpips_convnext_tcga
from cislnet.train_interface import VSliteTrainConfig, validate_train_config

os.environ["WANDB_MODE"] = "dryrun"

opt = TrainOptions().parse()
print(" ".join(sys.argv))

# Fixed patch sizes (VS_dataset → image_size) → stable cuDNN algo search.
if len(getattr(opt, "gpu_ids", []) or []) > 0:
    torch.backends.cudnn.benchmark = True

# Resolve total epochs
if getattr(opt, "total_epochs", 0) and int(opt.total_epochs) > 0:
    te = int(opt.total_epochs)
    opt.niter = te
    opt.niter_decay = 0

num_channels = int(getattr(opt, "input_nc", 13))
patch_size = int(getattr(opt, "image_size", 1024))
vcfg = VSliteTrainConfig(
    model="cislnet",
    run=str(opt.name),
    num_channels=num_channels,
    patch_size=patch_size,
    batch_size=int(opt.batch_size),
    epochs=int(opt.niter) + int(opt.niter_decay),
    lr=float(opt.lr),
    weight_decay=float(getattr(opt, "weight_decay", 0.0)),
    he_color_space=str(getattr(opt, "he_color_space", "srgb")),
    augment=not bool(getattr(opt, "no_aug", False)),
    early_stop_patience=int(getattr(opt, "early_stop_patience", 0) or 0),
    early_stop_min_delta=float(getattr(opt, "early_stop_min_delta", 1e-4)),
    output_dir=os.path.join(opt.checkpoints_dir, opt.name),
)
validate_train_config(vcfg)

dataloader = data.create_dataloader(opt)
# Eval/preview loaders are infrequent and serial — keep them single-process so
# we don't multiply pinned-memory prefetch workers on top of the train loader
# (that combination OOMs 64G Slurm allocations at PreviewManager init).
_eval_kw = dict(serial_batches=True, isTrain=False, batch_size=1, nThreads=0)
test_dataloader = data.create_dataloader(util.copyconf(opt, phase="test", **_eval_kw))
val_dataloader = data.create_dataloader(util.copyconf(opt, phase="val", **_eval_kw))
train_eval_dataloader = data.create_dataloader(
    util.copyconf(opt, phase="train", **_eval_kw)
)


def _dataset_len(loader) -> int:
    try:
        return int(len(loader.dataset))
    except Exception:
        return 0

trainer = get_trainer(opt)

# Optional OPAL + local LLM on generator updates (OPAL_ENABLE=1).
_g_steps_per_batch = (
    int(getattr(opt, "num_G_steps", 1))
    if int(getattr(opt, "D_steps_per_G", 1)) > 0
    else 0
)
# G.step fires when batch_index % num_F_steps_per_G == 0 (see run_generator_one_step).
_f_div = max(1, int(getattr(opt, "num_F_steps_per_G", 1)))
_batches_per_epoch = max(1, int(len(dataloader)))
_g_updates_per_epoch = (_batches_per_epoch // max(1, int(getattr(opt, "D_steps_per_G", 1)))) * max(1, _g_steps_per_batch)
# Rough: fraction of those that actually call optimizer_G.step
_g_updates_per_epoch = max(1, (_g_updates_per_epoch + _f_div - 1) // _f_div)
_total_g_updates = _g_updates_per_epoch * (int(opt.niter) + int(getattr(opt, "niter_decay", 0) or 0))
try:
    from cislnet.opal_hook import enable_opal, close_opal
    enable_opal(trainer, opt, total_g_updates=_total_g_updates, run_dir=vcfg.output_dir)
except Exception as _opal_exc:
    if __import__("os").environ.get("OPAL_ENABLE", "0") == "1":
        raise
    print(f"[opal] skipped: {_opal_exc}")

iter_counter = IterationCounter(opt, len(dataloader.dataset))
visualizer = Visualizer(opt)
tb = UnifiedTBLogger("cislnet", str(opt.name), run_dir=vcfg.output_dir)
tb.log_meta(vcfg, epoch=0)

samples_dir = os.path.join(opt.checkpoints_dir, opt.name, "samples")
os.makedirs(samples_dir, exist_ok=True)

steps_per_epoch = int(len(dataloader)) * int(opt.batch_size)
total_steps_expected = int(iter_counter.total_epochs) * steps_per_epoch
should_save_image = make_iter_schedule(total_steps_expected, target_saves=50)


def _trainer_device(trainer) -> torch.device:
    try:
        return next(trainer.model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.no_grad()
def _eval_l1_lpips(trainer, loader, lpips_mod, max_batches: int = 0) -> tuple[float, float]:
    """Capped diagnostic: mean L1 and lpips_convnext_tcga. Inference only, no grad."""
    trainer.model.eval()
    lpips_mod.eval()
    device = _trainer_device(trainer)
    tot_l1 = 0.0
    tot_lp = 0.0
    n = 0
    try:
        for bi, batch in enumerate(loader):
            if max_batches and bi >= int(max_batches):
                break
            batch = move_to_device(batch, device)
            fake = trainer.model(batch, mode="inference")
            y = batch["target"]
            l1 = torch.mean(torch.abs(fake - y)).item()
            lp = lpips_mod(fake, y)
            bs = int(y.shape[0]) if hasattr(y, "shape") else 1
            tot_l1 += float(l1) * bs
            tot_lp += float(lp.sum().item()) if lp.ndim > 0 else float(lp.item()) * bs
            n += bs
    finally:
        trainer.model.train()
    denom = max(1, n)
    return tot_l1 / denom, tot_lp / denom


@torch.no_grad()
def _preview_forward(fixed_batch):
    """Inference-only preview on fixed test tiles (no grad / no optimizer step)."""
    trainer.model.eval()
    try:
        device = next(trainer.model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    fixed_batch = move_to_device(fixed_batch, device)
    fake = trainer.model(fixed_batch, mode="inference")
    trainer.model.train()
    return fixed_batch["target"], fake


# Locked preview tiles come from the held-out **test** split (not train).
# Same patches every epoch; across seeded runs the loader order is deterministic.
# Fallback to val only if test is empty.
_preview_loader = test_dataloader if _dataset_len(test_dataloader) > 0 else val_dataloader
preview = PreviewManager(
    preview_loader=_preview_loader,
    out_dir=samples_dir,
    tb=tb,
    should_save=should_save_image,
    forward_fn=_preview_forward,
    preview_tile_list=os.environ.get("PREVIEW_TILE_LIST"),
)

early = EarlyStopping(
    patience=int(vcfg.early_stop_patience),
    min_delta=float(vcfg.early_stop_min_delta),
)

# Frozen ConvNeXt-TCGA Zhang-lin0 metric. Eval-only; not a training loss.
lpips_mod = build_lpips_convnext_tcga(
    ckpt_path=getattr(opt, "tcga_feature_ckpt", None),
    device=_trainer_device(trainer),
)


def _parse_save_epochs(raw) -> set[int] | None:
    s = str(raw or "").strip()
    if not s:
        return None
    out: set[int] = set()
    for part in s.replace(":", ",").split(","):
        p = part.strip()
        if p:
            out.add(int(p))
    return out or None


def _should_save_epoch_ckpt(epoch: int, total_epochs: int, save_at: set[int] | None, freq: int) -> bool:
    if epoch == total_epochs:
        return True
    if save_at is not None:
        return epoch in save_at
    freq = max(1, int(freq))
    return epoch % freq == 0


save_at_epochs = _parse_save_epochs(getattr(opt, "save_epochs", ""))


def _save_best_ckpt():
    import shutil

    trainer.save("latest")
    src = os.path.join(opt.checkpoints_dir, opt.name, "latest_net_G.pth")
    dst = os.path.join(opt.checkpoints_dir, opt.name, "best_net_G.pth")
    if os.path.isfile(src):
        shutil.copy2(src, dst)


for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        if i % opt.D_steps_per_G == 0:
            for _ in range(opt.num_G_steps):
                trainer.run_generator_one_step(data_i, iter=i)
        else:
            # Stale G-step fake must not be reused on a different batch.
            trainer.generated = None

        trainer.run_discriminator_one_step(data_i)

        if should_log_epoch_samples(iter_counter.epoch_iter, opt.batch_size, opt.print_freq):
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(
                epoch,
                iter_counter.epoch_iter,
                losses,
                iter_counter.time_per_iter,
                trainer.old_lr,
            )
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)
            tb.log_train_scalars(
                {k: float(v.mean().item()) if hasattr(v, "mean") else float(v) for k, v in losses.items()},
                int(iter_counter.total_steps_so_far),
            )

        step = int(iter_counter.total_steps_so_far)
        if step <= total_steps_expected and should_save_image(step - 1):
            preview.maybe_save(epoch, step)
            if getattr(opt, "debug", False):
                debug_dir = Path(opt.checkpoints_dir) / opt.name / "debug_npys"
                debug_dir.mkdir(parents=True, exist_ok=True)
                np.save(str(debug_dir / f"target_{step}.npy"), data_i["target"].data.cpu().numpy())
                np.save(str(debug_dir / f"input_{step}.npy"), data_i["input"].data.cpu().numpy())

        if iter_counter.needs_saving():
            print(f"saving the latest model (epoch {epoch}, total_steps {iter_counter.total_steps_so_far})")
            trainer.save("latest")
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    eval_train_batches = int(eval_max_batches_train())
    eval_val_batches = int(eval_max_batches_val())
    eval_test_batches = int(eval_max_batches_test())
    train_l1, train_lpips = _eval_l1_lpips(
        trainer, train_eval_dataloader, lpips_mod, max_batches=eval_train_batches
    )
    if _dataset_len(val_dataloader) > 0:
        val_l1, val_lpips = _eval_l1_lpips(
            trainer, val_dataloader, lpips_mod, max_batches=eval_val_batches
        )
    else:
        val_l1, val_lpips = float("nan"), float("nan")
    test_l1, test_lpips = _eval_l1_lpips(
        trainer, test_dataloader, lpips_mod, max_batches=eval_test_batches
    )
    tb_step = int(iter_counter.total_steps_so_far)
    tb.log_epoch_l1(train_l1, val_l1, test_l1, tb_step)
    tb.log_epoch_lpips(train_lpips, val_lpips, test_lpips, tb_step)

    def _fmt_m(v: float) -> str:
        return f"{v:.6f}" if math.isfinite(v) else "n/a"

    print(
        f"End of epoch {epoch} / {iter_counter.total_epochs} "
        f"\t samples {tb_step} "
        f"\t train_l1: {_fmt_m(train_l1)} \t val_l1: {_fmt_m(val_l1)} \t test_l1: {_fmt_m(test_l1)} "
        f"\t train_lpips_convnext_tcga: {_fmt_m(train_lpips)} "
        f"\t val_lpips_convnext_tcga: {_fmt_m(val_lpips)} "
        f"\t test_lpips_convnext_tcga: {_fmt_m(test_lpips)}"
    )

    preview.save_epoch_end(epoch, iter_counter.total_steps_so_far)

    if math.isfinite(val_l1) and vcfg.early_stop_patience > 0:
        if early.step(val_l1, epoch):
            _save_best_ckpt()

    if early.should_stop():
        tb.log_early_stop(early.state.best_value, epoch)
        print(f"Early stopping at epoch {epoch} (best val L1 {early.state.best_value:.6f})")
        break

    if _should_save_epoch_ckpt(
        epoch, int(iter_counter.total_epochs), save_at_epochs, int(opt.save_epoch_freq)
    ):
        print(
            f"saving the model at the end of epoch {epoch} "
            f"(samples {iter_counter.total_steps_so_far})"
        )
        trainer.save(f"{epoch}")

print("Training was successfully finished.")
try:
    from cislnet.opal_hook import close_opal
    close_opal(trainer)
except Exception as _e:
    print(f"[opal] close: {_e}")
tb.close()
