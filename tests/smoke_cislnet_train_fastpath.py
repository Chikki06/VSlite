#!/usr/bin/env python3
"""Smoke: single-GPU no-DP + D-step fake reuse (CISLNet train fast path).

Run on a GPU node, e.g.:
  sbatch --gres=gpu:A100:1 --mem=32G --cpus-per-task=4 --time=00:20:00 \\
    --wrap 'source jobs/_lib/vslite_env.sh && python tests/smoke_cislnet_train_fastpath.py'
"""
from __future__ import annotations

import os
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from models.networks.sync_batchnorm import DataParallelWithCallback
from trainers.contrastive_pix2pix_trainer import ContrastivePix2PixTrainer
from util.batch_device import move_to_device


def _rss_gb() -> float:
    # Linux: ru_maxrss is KiB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def test_move_to_device_non_blocking():
    t = torch.randn(2, 3)
    out = move_to_device({"x": t, "p": "keep"}, "cpu", non_blocking=True)
    assert out["p"] == "keep"
    assert torch.equal(out["x"], t)
    print("[ok] move_to_device(non_blocking) preserves structure")


def test_trainer_skips_dp_single_gpu():
    if not torch.cuda.is_available():
        print("[skip] CUDA unavailable — DP skip check needs cuda:0")
        return False

    # Minimal stub so create_model is not needed: patch create_model.
    import trainers.contrastive_pix2pix_trainer as mod

    fake_module = torch.nn.Linear(4, 4)
    create_calls = {"n": 0}

    def _fake_create(opt):
        create_calls["n"] += 1
        m = MagicMock(spec=torch.nn.Module)
        m.to = lambda device: fake_module.to(device) or fake_module
        # Real nn.Module for .to()
        return fake_module

    # Use a real tiny Module so .to(cuda) works; patch create_model.
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l = torch.nn.Linear(4, 4)

        def create_optimizers(self, opt):
            return None, None, None

        def forward(self, data, mode):
            raise AssertionError("should not be called in this test")

    orig = mod.create_model
    mod.create_model = lambda opt: Tiny()
    try:
        opt = SimpleNamespace(gpu_ids=[0], isTrain=True, lr=2e-4)
        trainer = ContrastivePix2PixTrainer(opt)
        assert not isinstance(trainer.pix2pix_model, DataParallelWithCallback), (
            "single GPU must not wrap DataParallel"
        )
        assert trainer._use_dp is False
        assert next(trainer.pix2pix_model.parameters()).device.type == "cuda"
        print("[ok] single-GPU trainer skips DataParallel")
        return True
    finally:
        mod.create_model = orig


def test_discriminator_reuses_fake():
    """compute_discriminator_loss must not call generate_fake when fake is given."""
    from models.contrastive_pix2pix_model import ContrastivePix2PixModel

    # Build a lightweight stand-in with the real method bound.
    obj = SimpleNamespace(
        opt=SimpleNamespace(no_gan_loss=False, he_color_space="srgb"),
        criterionGAN=lambda pred, real, for_discriminator=True: torch.tensor(0.0),
    )

    gen_calls = {"n": 0}

    def generate_fake(input_semantics, real_image, compute_kld_loss=False):
        gen_calls["n"] += 1
        return torch.zeros_like(real_image), None

    def _images_for_pretrained_heads(fake, real):
        return fake, real

    def discriminate(inp, fake, real):
        return torch.zeros(1), torch.zeros(1)

    obj.generate_fake = generate_fake
    obj._images_for_pretrained_heads = _images_for_pretrained_heads
    obj.discriminate = discriminate

    fake = torch.randn(1, 3, 8, 8)
    real = torch.randn(1, 3, 8, 8)
    inp = torch.randn(1, 3, 8, 8)

    ContrastivePix2PixModel.compute_discriminator_loss(
        obj, inp, real, fake_image=fake
    )
    assert gen_calls["n"] == 0, "must skip generate_fake when fake_image provided"

    ContrastivePix2PixModel.compute_discriminator_loss(obj, inp, real, fake_image=None)
    assert gen_calls["n"] == 1, "must regenerate when fake_image is None"
    print("[ok] D-step reuses fake / falls back to generate_fake")


def smoke_one_train_step_if_possible():
    """Optional: one real train step when DATAROOT + CUDA exist."""
    if not torch.cuda.is_available():
        print("[skip] no CUDA for live train step")
        return
    dataroot = os.environ.get("DATAROOT", "")
    if not dataroot or not Path(dataroot).is_dir():
        print("[skip] DATAROOT unset — logic smoke only")
        return

    # Keep this tiny: batch 1, small preview caps, few threads.
    os.environ.setdefault("EVAL_MAX_BATCHES_TRAIN", "0")
    os.environ.setdefault("EVAL_MAX_BATCHES_VAL", "0")
    os.environ.setdefault("EVAL_MAX_BATCHES_TEST", "0")

    from options.train_options import TrainOptions
    from trainers import get_trainer
    import data

    # Parse minimal argv for contrastive train.
    argv = [
        "smoke",
        "--dataroot",
        dataroot,
        "--dataset_mode",
        "VS",
        "--model",
        "contrastive_pix2pix",
        "--gpu_ids",
        "0",
        "--batch_size",
        "1",
        "--image_size",
        "256",
        "--nThreads",
        "0",
        "--name",
        "smoke_fastpath",
        "--checkpoints_dir",
        str(_REPO / "results" / "cislnet"),
        "--total_epochs",
        "1",
        "--input_nc",
        "13",
        "--num_ir_channels",
        "10",
    ]
    sys.argv = argv
    opt = TrainOptions().parse()
    torch.backends.cudnn.benchmark = True

    rss0 = _rss_gb()
    loader = data.create_dataloader(opt)
    trainer = get_trainer(opt)
    assert not isinstance(trainer.pix2pix_model, DataParallelWithCallback)

    # Monkeypatch generate_fake to count D-path calls.
    model = trainer.pix2pix_model_on_one_gpu
    real_gen = model.generate_fake
    counts = {"gen": 0}

    def counting_gen(*a, **k):
        counts["gen"] += 1
        return real_gen(*a, **k)

    model.generate_fake = counting_gen

    batch = next(iter(loader))
    t0 = time.perf_counter()
    trainer.run_generator_one_step(batch, iter=0)
    g_gens = counts["gen"]
    trainer.run_discriminator_one_step(batch)
    d_extra = counts["gen"] - g_gens
    dt = time.perf_counter() - t0
    rss1 = _rss_gb()

    assert d_extra == 0, f"D step should not call generate_fake; got {d_extra} extra"
    print(
        f"[ok] live step: {dt:.3f}s  D_extra_gen={d_extra}  "
        f"RSS {rss0:.2f}->{rss1:.2f} GiB (max)"
    )


def main():
    print(f"cuda={torch.cuda.is_available()}  rss={_rss_gb():.2f} GiB")
    test_move_to_device_non_blocking()
    test_discriminator_reuses_fake()
    test_trainer_skips_dp_single_gpu()
    smoke_one_train_step_if_possible()
    print(f"[done] final rss={_rss_gb():.2f} GiB")


if __name__ == "__main__":
    main()
