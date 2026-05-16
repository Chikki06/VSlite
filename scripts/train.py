"""
Wrapper location for baseline contrastive pix2pix training.

This file contains the original repo-root `train.py` implementation, moved under `scripts/`.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure repo-root modules (e.g. `data/`) are importable even when invoked as
# `python scripts/train.py` (which otherwise prepends `scripts/` to sys.path).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import data
import evaluation.group_evaluator as group_evaluator
import util.util as util
from options.train_options import TrainOptions
from trainers import get_trainer
from util.fixed_preview import gather_fixed_preview_batch
from util.iter_counter import IterationCounter
from util.batch_device import move_to_device
from util.sample_grid import save_unified_sample, split_batch_to_list
from util.save_schedule import make_iter_schedule
from util.unified_tb import build_unified_writer
from util.visualizer import Visualizer
from util.eval_caps import eval_max_batches_train, eval_max_batches_val, eval_max_batches_test
from util.run_logger import should_log_epoch_samples

os.environ["WANDB_MODE"] = "dryrun"

opt = TrainOptions().parse()
print(" ".join(sys.argv))

dataloader = data.create_dataloader(opt)
test_dataloader = data.create_dataloader(util.copyconf(opt, phase="test", batch_size=1))
val_dataloader = data.create_dataloader(util.copyconf(opt, phase="val", batch_size=1))
train_eval_dataloader = data.create_dataloader(
    util.copyconf(opt, phase="train", serial_batches=True, isTrain=False, batch_size=1)
)

trainer = get_trainer(opt)
iter_counter = IterationCounter(opt, len(dataloader.dataset))
visualizer = Visualizer(opt)
evaluator = group_evaluator.GroupEvaluator(opt)
u_writer, _u_tb = build_unified_writer(model="baseline_contrastive_pix2pix", run=str(opt.name))

samples_dir = os.path.join(opt.checkpoints_dir, opt.name, "samples")
os.makedirs(samples_dir, exist_ok=True)

# Total work is measured in "samples processed" (matches IterationCounter.total_steps_so_far)
# NOTE: `IterationCounter.total_steps_so_far` increments by `opt.batch_size` per dataloader
# iteration, so the natural unit is "samples processed", not "batches".
# Use the *actual* number of samples processed per epoch (len(dataloader) * batch_size),
# which also respects drop_last behavior in the dataloader.
steps_per_epoch = int(len(dataloader)) * int(opt.batch_size)
total_steps_expected = int(iter_counter.total_epochs) * steps_per_epoch
should_save_image = make_iter_schedule(total_steps_expected, target_saves=50)

# Capture 8 deterministic preview samples once so successive PNGs are
# directly comparable across the run.
fixed_preview = gather_fixed_preview_batch(train_eval_dataloader, n=8)


@torch.no_grad()
def _eval_l1(trainer, loader, max_batches: int = 0) -> float:
    """
    Evaluate mean L1(|G(x)-y|) over a dataloader.
    max_batches=0 means full loader; otherwise limits to first N batches.
    """
    trainer.model.eval()
    try:
        device = next(trainer.model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    tot = 0.0
    n = 0
    for bi, batch in enumerate(loader):
        if max_batches and bi >= int(max_batches):
            break
        batch = move_to_device(batch, device)
        fake = trainer.model(batch, mode="inference")
        if not torch.is_tensor(fake):
            raise TypeError(f"Expected inference output to be a Tensor, got {type(fake)}")
        y = batch["target"]
        if not torch.is_tensor(y):
            raise TypeError(f"Expected batch['target'] to be a Tensor, got {type(y)}")
        l1 = torch.mean(torch.abs(fake - y)).item()
        bs = int(y.shape[0]) if hasattr(y, "shape") else 1
        tot += float(l1) * bs
        n += bs
    trainer.model.train()
    return tot / max(1, n)


@torch.no_grad()
def _save_unified_preview(trainer, fixed_batch, epoch: int, step: int) -> None:
    if fixed_batch is None:
        return
    trainer.model.eval()
    try:
        device = next(trainer.model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    fixed_batch = move_to_device(fixed_batch, device)
    fake = trainer.model(fixed_batch, mode="inference")
    target = fixed_batch["target"]
    reals = split_batch_to_list(target.detach().cpu())[:8]
    preds = split_batch_to_list(fake.detach().cpu())[:8]
    name = f"epoch{int(epoch):03d}_step{int(step):09d}"
    save_unified_sample(samples_dir, name, reals, preds)
    trainer.model.train()


for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        if i % opt.D_steps_per_G == 0:
            for _ in range(opt.num_G_steps):
                trainer.run_generator_one_step(data_i, iter=i)

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

        step = int(iter_counter.total_steps_so_far)
        # `make_iter_schedule` expects a 0-based index; `total_steps_so_far` starts at 0
        # and then jumps by `batch_size`, so the right aligned index is `(step - 1)`.
        # Also guard against any mismatch where `step` exceeds the expected total
        # (which would otherwise make the schedule fire continuously near the end).
        if step <= total_steps_expected and should_save_image(step - 1):
            _save_unified_preview(trainer, fixed_preview, epoch, step)
            if getattr(opt, "debug", False):
                debug_dir = Path(opt.checkpoints_dir) / opt.name / "debug_npys"
                debug_dir.mkdir(parents=True, exist_ok=True)
                np.save(
                    str(debug_dir / f"target_{step}.npy"),
                    data_i["target"].data.cpu().numpy(),
                )
                np.save(
                    str(debug_dir / f"input_{step}.npy"),
                    data_i["input"].data.cpu().numpy(),
                )

        if iter_counter.needs_saving():
            print(
                f"saving the latest model (epoch {epoch}, total_steps {iter_counter.total_steps_so_far})"
            )
            trainer.save("latest")
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    eval_train_batches = int(eval_max_batches_train())
    eval_val_batches = int(eval_max_batches_val())
    eval_test_batches = int(eval_max_batches_test())
    train_l1 = _eval_l1(trainer, train_eval_dataloader, max_batches=eval_train_batches)
    val_l1 = _eval_l1(trainer, val_dataloader, max_batches=eval_val_batches)
    test_l1 = _eval_l1(trainer, test_dataloader, max_batches=eval_test_batches)
    u_writer.add_scalar("l1/train", float(train_l1), epoch)
    u_writer.add_scalar("l1/val", float(val_l1), epoch)
    u_writer.add_scalar("l1/test", float(test_l1), epoch)

    # End-of-epoch composite save (always, even if the iter schedule didn't fire).
    _save_unified_preview(trainer, fixed_preview, epoch, iter_counter.total_steps_so_far)

    if epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
        print(
            f"saving the model at the end of epoch {epoch}, iters {iter_counter.total_steps_so_far}"
        )
        trainer.save(f"{epoch}")

print("Training was successfully finished.")
u_writer.close()
