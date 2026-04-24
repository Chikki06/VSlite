"""
Train an I2SB-style conditional bridge for VSlite virtual staining.

This wraps NVlabs/I2SB's `i2sb.Runner` and uses its *conditional* ADM backbone.
We adapt by:
  - defining x0 as H&E (3ch) and x1 as a learned 3ch projection of VS inputs (10/13ch)
  - setting opt.corrupt='mixture' so Runner.sample_batch uses (x0, x1, y) directly
  - keeping VSlite-style checkpoints under results/i2sb_vs/<name>/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

_THIS_DIR = Path(__file__).resolve().parent
_VSLITE_ROOT = _THIS_DIR.parent
if str(_VSLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_VSLITE_ROOT))

from diffusion_vs.data import VSliteVSDiffusionDataset, collate_vs_batch
from diffusion_vs.i2sb_vs import CondToRGB, resolve_i2sb_paths


class _I2SBMixtureDataset(torch.utils.data.Dataset):
    """
    Present (x0, x1, y) samples to I2SB Runner.sample_batch when opt.corrupt == 'mixture'.
    """

    def __init__(self, base: VSliteVSDiffusionDataset, cond_to_rgb: CondToRGB):
        self.base = base
        self.cond_to_rgb = cond_to_rgb

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        b = self.base[idx]
        # x0: target HE, x1: projected conditioning
        x0 = b.x_he
        # CondToRGB expects BCHW.
        x1 = self.cond_to_rgb(b.x_cond.unsqueeze(0)).squeeze(0)
        y = torch.tensor(0, dtype=torch.long)
        return x0, x1, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, required=True, help="run name (e.g., i2sb_10)")
    ap.add_argument("--dataroot", type=str, required=True)
    ap.add_argument("--results_dir", type=str, default="/projects/illinois/eng/bioe/rxb/akshat/VSlite/results/i2sb_vs")
    ap.add_argument("--input_nc", type=int, required=True, choices=[10, 13])
    ap.add_argument("--num_ir_channels", type=int, default=10)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--microbatch", type=int, default=2)
    ap.add_argument("--num_itr", type=int, default=200000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--interval", type=int, default=1000)
    ap.add_argument("--beta_max", type=float, default=0.3)
    ap.add_argument("--ema", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpt", type=str, default=None, help="resume from previous run name (loads results/<ckpt>/latest.pt)")
    args = ap.parse_args()

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))

    # Import I2SB after sys.path adjustments.
    i2sb_root = _VSLITE_ROOT / "third_party" / "I2SB"
    if str(i2sb_root) not in sys.path:
        sys.path.insert(0, str(i2sb_root))

    from logger import Logger  # type: ignore
    from i2sb import Runner, download_ckpt  # type: ignore

    paths = resolve_i2sb_paths(_VSLITE_ROOT)
    # Ensure ADM checkpoints/pkls exist.
    download_ckpt(str(paths.third_party_i2sb / "data"))

    ckpt_path = Path(args.results_dir).resolve() / args.name
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # Build an opt compatible with I2SB Runner.
    opt = SimpleNamespace()
    opt.seed = int(args.seed)
    opt.name = str(args.name)
    opt.gpu = int(args.gpu)
    opt.device = "cuda" if torch.cuda.is_available() else "cpu"
    opt.n_gpu_per_node = 1
    opt.master_address = "localhost"
    opt.node_rank = 0
    opt.num_proc_node = 1
    opt.distributed = False
    opt.use_fp16 = False
    opt.global_rank = 0
    opt.local_rank = 0
    opt.global_size = 1

    # I2SB model / diffusion
    opt.image_size = int(args.image_size)
    opt.corrupt = "mixture"
    opt.t0 = 1e-4
    opt.T = 1.0
    opt.interval = int(args.interval)
    opt.beta_max = float(args.beta_max)
    opt.ot_ode = False
    opt.clip_denoise = False
    opt.cond_x1 = True
    opt.add_x1_noise = False

    # optimization
    opt.batch_size = int(args.batch_size)
    opt.microbatch = int(args.microbatch)
    opt.num_itr = int(args.num_itr)
    opt.lr = float(args.lr)
    opt.lr_gamma = 1.0
    opt.lr_step = 1000
    opt.l2_norm = 0.0
    opt.ema = float(args.ema)

    # paths / ckpt
    opt.log_dir = ckpt_path / "log"
    opt.log_dir.mkdir(parents=True, exist_ok=True)
    opt.ckpt_path = ckpt_path
    if args.ckpt:
        opt.load = Path(args.results_dir).resolve() / str(args.ckpt) / "latest.pt"
    else:
        opt.load = None

    log = Logger(opt.global_rank, opt.log_dir)  # type: ignore
    log.info(f"[VSlite] Training I2SB-VS: name={opt.name} input_nc={args.input_nc}")

    base_train = VSliteVSDiffusionDataset(
        dataroot=args.dataroot,
        split="train",
        input_nc=int(args.input_nc),
        num_ir_channels=int(args.num_ir_channels),
    )
    base_val = VSliteVSDiffusionDataset(
        dataroot=args.dataroot,
        split="test",
        input_nc=int(args.input_nc),
        num_ir_channels=int(args.num_ir_channels),
    )

    cond_to_rgb = CondToRGB(int(args.input_nc)).to(opt.device)
    train_ds = _I2SBMixtureDataset(base_train, cond_to_rgb)
    val_ds = _I2SBMixtureDataset(base_val, cond_to_rgb)

    # I2SB Runner uses its own infinite-loader util; datasets only need __getitem__.
    run = Runner(opt, log)
    # corrupt_method unused when corrupt == 'mixture'
    run.train(opt, train_ds, val_ds, corrupt_method=None)


if __name__ == "__main__":
    main()

