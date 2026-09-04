"""Build AttUnetGenerator from checkpoint + opt.pkl."""

from __future__ import annotations

import os
from argparse import Namespace
from typing import Optional

import torch

import util.util as util
from models.networks import define_G


def load_opt(opt_dir: str, name: Optional[str] = None) -> Namespace:
    """Load training options from ``opt_dir/opt.pkl``."""
    opt_path = os.path.join(opt_dir, "opt.pkl")
    if not os.path.isfile(opt_path):
        raise FileNotFoundError(f"Missing opt.pkl at {opt_path}")
    opt = util.load_obj(opt_path)
    if name is not None:
        opt.name = name
    opt.isTrain = False
    return opt


def build_generator_from_checkpoint(
    source_ckpt: str,
    opt_dir: str,
    *,
    gpu_ids: Optional[list[int]] = None,
    strict: bool = True,
) -> tuple[torch.nn.Module, Namespace]:
    """Instantiate ``AttUnetGenerator`` and load fp32 weights from ``source_ckpt``."""
    if not os.path.isfile(source_ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {source_ckpt}")

    opt = load_opt(opt_dir)
    if gpu_ids is None:
        gpu_ids = [0] if torch.cuda.is_available() else []
    opt.gpu_ids = gpu_ids

    netG = define_G(opt)
    weights = torch.load(source_ckpt, map_location="cpu", weights_only=True)
    netG.load_state_dict(weights, strict=strict)
    netG.eval()
    return netG, opt
