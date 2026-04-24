"""
Train BBDM/LBBDM on VSlite virtual staining.

This uses third_party/BBDM's runner stack but:
  - registers a new dataset type `vslite_vs_npy`
  - optionally swaps latent model class to support:
      - VQGAN-pytorch checkpoints (vqgan_impl=vqgan_pytorch)
      - ConvNeXt conditioning tokens (condition_key=convnext)
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import torch
import yaml

_THIS_DIR = Path(__file__).resolve().parent
_VSLITE_ROOT = _THIS_DIR.parent
if str(_VSLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_VSLITE_ROOT))

from diffusion_vs.bbdm_latent_vs_model import build_latent_bbdm_model_class
from diffusion_vs.bbdm_vs_dataset import register_vslite_dataset


def _dict2namespace(d: dict):
    import argparse as _argparse

    ns = _argparse.Namespace()
    for k, v in d.items():
        if isinstance(v, dict):
            v = _dict2namespace(v)
        setattr(ns, k, v)
    return ns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="yaml config (BBDM style)")
    ap.add_argument("--result_path", type=str, default="/projects/illinois/eng/bioe/rxb/akshat/VSlite/results/bbdm_vs")
    ap.add_argument("--gpu_ids", type=str, default="0")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--resume_model", type=str, default=None)
    ap.add_argument("--resume_optim", type=str, default=None)
    args = ap.parse_args()

    register_vslite_dataset(_VSLITE_ROOT)

    bbdm_root = _VSLITE_ROOT / "third_party" / "BBDM"
    if str(bbdm_root) not in sys.path:
        sys.path.insert(0, str(bbdm_root))

    from utils import get_runner  # type: ignore
    import runners.DiffusionBasedModelRunners.BBDMRunner as bbdm_runner_mod  # type: ignore

    with open(args.config, "r") as f:
        dict_config = yaml.load(f, Loader=yaml.FullLoader)

    nconfig = _dict2namespace(dict_config)
    # attach args like upstream main.py does
    nconfig.args = argparse.Namespace(
        train=True,
        seed=int(args.seed),
        gpu_ids=str(args.gpu_ids),
        port="12355",
        result_path=str(args.result_path),
        resume_model=args.resume_model,
        resume_optim=args.resume_optim,
        max_epoch=None,
        max_steps=None,
        sample_to_eval=False,
        sample_at_start=False,
        save_top=False,
        config=str(args.config),
    )

    # apply resume overrides
    if args.resume_model is not None:
        nconfig.model.model_load_path = args.resume_model
    if args.resume_optim is not None:
        nconfig.model.optim_sche_load_path = args.resume_optim

    # Device setup (single GPU path)
    gpu_list = str(args.gpu_ids).split(",")
    if len(gpu_list) == 1 and gpu_list[0] != "-1":
        nconfig.training.use_DDP = False
        nconfig.training.device = [torch.device(f"cuda:{gpu_list[0]}")]
    else:
        raise RuntimeError("This wrapper currently supports single-GPU runs; use upstream BBDM main.py for DDP.")

    # Monkeypatch latent model class if requested.
    if str(getattr(nconfig.model, "model_type", "")) == "LBBDM":
        LatentBrownianBridgeModelVS = build_latent_bbdm_model_class(_VSLITE_ROOT)
        bbdm_runner_mod.LatentBrownianBridgeModel = LatentBrownianBridgeModelVS  # type: ignore

    runner = get_runner(nconfig.runner, nconfig)  # type: ignore
    runner.train()


if __name__ == "__main__":
    main()

