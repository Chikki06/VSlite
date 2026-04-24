"""
Wrapper location for baseline contrastive pix2pix training.

This file contains the original repo-root `train.py` implementation, moved under `scripts/`.
"""

import os
import sys
from pathlib import Path
from collections import OrderedDict

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
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer

os.environ["WANDB_MODE"] = "dryrun"

opt = TrainOptions().parse()
print(" ".join(sys.argv))

dataloader = data.create_dataloader(opt)
test_dataloader = data.create_dataloader(util.copyconf(opt, phase="test", batch_size=1))

trainer = get_trainer(opt)
iter_counter = IterationCounter(opt, len(dataloader.dataset))
visualizer = Visualizer(opt)
evaluator = group_evaluator.GroupEvaluator(opt)


def _displayable_input(input_tensor):
    """Return an image-like tensor for visualization (1 or 3 channels)."""
    if input_tensor is None:
        return None
    if input_tensor.dim() < 4:
        return input_tensor
    c = input_tensor.shape[1]
    if c == 1 or c == 3:
        return input_tensor
    if c >= 3:
        return input_tensor[:, :3, ...]
    first = input_tensor[:, :1, ...]
    return first.repeat(1, 3, 1, 1)


for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    latest_visuals = None
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        if i % opt.D_steps_per_G == 0:
            for _ in range(opt.num_G_steps):
                trainer.run_generator_one_step(data_i, iter=i)

        trainer.run_discriminator_one_step(data_i)

        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(
                epoch,
                iter_counter.epoch_iter,
                losses,
                iter_counter.time_per_iter,
                trainer.old_lr,
            )
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            step = iter_counter.total_steps_so_far
            generated = trainer.get_latest_generated()
            if generated is not None:
                input_preview = _displayable_input(data_i["input"])
                latest_visuals = OrderedDict(
                    [
                        ("input", input_preview),
                        ("synthesized_image", generated),
                        ("real_image", data_i["target"]),
                    ]
                )
                visualizer.display_current_results(latest_visuals, epoch, step)
            if getattr(opt, "debug", False):
                os.makedirs("images/", exist_ok=True)
                np.save(f"images/target_{step}.npy", data_i["target"].data.cpu().numpy())
                np.save(f"images/input_{step}.npy", data_i["input"].data.cpu().numpy())
                np.save(
                    f"images/gen_{step}.npy", trainer.get_latest_generated().data.cpu().numpy()
                )

        generated = trainer.get_latest_generated()
        if generated is not None:
            input_preview = _displayable_input(data_i["input"])
            latest_visuals = OrderedDict(
                [
                    ("input", input_preview),
                    ("synthesized_image", generated),
                    ("real_image", data_i["target"]),
                ]
            )

        if iter_counter.needs_saving():
            print(
                f"saving the latest model (epoch {epoch}, total_steps {iter_counter.total_steps_so_far})"
            )
            trainer.save("latest")
            iter_counter.record_current_iter()
            if latest_visuals is not None:
                print(
                    f"saving training images (epoch {epoch}, total_steps {iter_counter.total_steps_so_far})"
                )
                visualizer.display_current_results(
                    latest_visuals, epoch, iter_counter.total_steps_so_far
                )

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if latest_visuals is not None:
        print(
            f"saving epoch-end images (epoch {epoch}, total_steps {iter_counter.total_steps_so_far})"
        )
        visualizer.display_current_results(latest_visuals, epoch, iter_counter.total_steps_so_far)

    if epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
        print(
            f"saving the model at the end of epoch {epoch}, iters {iter_counter.total_steps_so_far}"
        )
        trainer.save(f"{epoch}")

print("Training was successfully finished.")

