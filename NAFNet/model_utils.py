from __future__ import annotations

from types import SimpleNamespace

import yaml

from .nafnet_wrappers import build_nafnet_i2i


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    return SimpleNamespace(**cfg_dict), cfg_dict


def build_split_fracs(cfg):
    fr = getattr(cfg, "split_fracs", [0.9, 0.05, 0.05])
    from SwinGAN.utils.dataset_vslite_i2i import SplitFractions  # type: ignore

    return SplitFractions(train=float(fr[0]), val=float(fr[1]), test=float(fr[2]))


def build_model(cfg, in_chans: int):
    return build_nafnet_i2i(cfg=cfg, raw_in_chans=int(in_chans))
