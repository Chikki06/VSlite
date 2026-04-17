import hashlib
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


def _sorted_npy_files(d: str) -> List[str]:
    files = [f for f in os.listdir(d) if f.endswith(".npy") and not f.startswith(".")]
    files.sort()
    return files


def _hash01(s: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{s}".encode("utf-8")).hexdigest()
    # 8 hex chars = 32-bit
    return int(h[:8], 16) / float(0xFFFFFFFF)


@dataclass(frozen=True)
class SplitFractions:
    train: float = 0.9
    val: float = 0.05
    test: float = 0.05

    def __post_init__(self):
        tot = self.train + self.val + self.test
        if not (abs(tot - 1.0) < 1e-6):
            raise ValueError(f"Split fractions must sum to 1.0, got {tot}")


class VSLiteI2IDataset(Dataset):
    """
    Paired dataset for VSlite patches:
      dataroot/ir/*.npy  (H,W,C_ir)
      dataroot/vis/*.npy (H,W,3)   optional depending on input_mode
      dataroot/he/*.npy  (H,W,3)

    Pairing uses identical filenames across ir/vis/he.

    Output:
      x: FloatTensor (C_in,H,W) where C_in is 10 or 13
      y: FloatTensor (3,H,W)
    """

    def __init__(
        self,
        dataroot: str,
        input_mode: str = "ir10",  # "ir10" or "irvis13"
        num_ir_channels: int = 10,
        img_size: Optional[int] = 256,
        ir_clip: Tuple[float, float] = (0.0, 5.0),
        ir_scale: str = "0_1",  # "0_1" or "none"
        he_scale: str = "minus1_1",  # "minus1_1" or "0_1"
        max_items: Optional[int] = None,
        split: Optional[str] = None,  # None|train|val|test
        split_seed: int = 1337,
        split_fracs: SplitFractions = SplitFractions(),
        indices: Optional[Sequence[int]] = None,
    ):
        self.dataroot = dataroot
        self.input_mode = input_mode
        self.num_ir_channels = int(num_ir_channels)
        self.img_size = None if img_size is None else int(img_size)
        self.ir_clip = (float(ir_clip[0]), float(ir_clip[1]))
        self.ir_scale = ir_scale
        self.he_scale = he_scale

        ir_dir = os.path.join(dataroot, "ir")
        he_dir = os.path.join(dataroot, "he")
        vis_dir = os.path.join(dataroot, "vis")
        if not os.path.isdir(ir_dir):
            raise FileNotFoundError(f"Missing IR directory: {ir_dir}")
        if not os.path.isdir(he_dir):
            raise FileNotFoundError(f"Missing HE directory: {he_dir}")
        if input_mode == "irvis13" and not os.path.isdir(vis_dir):
            raise FileNotFoundError(f"Missing VIS directory: {vis_dir}")

        self._ir_dir = ir_dir
        self._vis_dir = vis_dir
        self._he_dir = he_dir

        names = _sorted_npy_files(ir_dir)
        if max_items is not None:
            names = names[: int(max_items)]

        # Apply deterministic split by filename hash if requested.
        if split is not None:
            split = split.lower()
            if split not in {"train", "val", "test"}:
                raise ValueError(f"split must be one of train/val/test, got {split}")
            keep: List[str] = []
            for n in names:
                r = _hash01(n, split_seed)
                if r < split_fracs.train:
                    bucket = "train"
                elif r < split_fracs.train + split_fracs.val:
                    bucket = "val"
                else:
                    bucket = "test"
                if bucket == split:
                    keep.append(n)
            names = keep

        if indices is not None:
            names = [names[i] for i in indices]

        self._names = names

    def __len__(self) -> int:
        return len(self._names)

    def _load_npy(self, folder: str, name: str) -> np.ndarray:
        p = os.path.join(folder, name)
        return np.load(p, mmap_mode="r")

    def _to_float_chw(self, arr: np.ndarray) -> torch.Tensor:
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.ndim != 3:
            raise ValueError(f"Expected HxWxC or HxW, got shape {arr.shape}")
        # HWC -> CHW
        t = torch.from_numpy(np.array(arr, dtype=np.float32)).permute(2, 0, 1).contiguous()
        return t

    def _resize_chw(self, t: torch.Tensor) -> torch.Tensor:
        if self.img_size is None:
            return t
        c, h, w = t.shape
        if h == self.img_size and w == self.img_size:
            return t
        
        import torch.nn.functional as F
        return F.interpolate(t.unsqueeze(0), size=(self.img_size, self.img_size), mode='bilinear', align_corners=False).squeeze(0)
        return out

    def _prep_ir(self, ir: torch.Tensor) -> torch.Tensor:
        lo, hi = self.ir_clip
        ir = ir.clamp(lo, hi)
        if self.ir_scale == "0_1":
            denom = max(1e-6, hi - lo)
            ir = (ir - lo) / denom
        elif self.ir_scale == "none":
            pass
        else:
            raise ValueError(f"Unknown ir_scale: {self.ir_scale}")
        return ir

    def _scale_he(self, he: torch.Tensor) -> torch.Tensor:
        # he is CHW float32, likely 0..255 or 0..1 or already normalized.
        if he.max().item() > 1.5:
            he = he / 255.0
        he = he.clamp(0.0, 1.0)
        if self.he_scale == "0_1":
            return he
        if self.he_scale == "minus1_1":
            return (he - 0.5) / 0.5
        raise ValueError(f"Unknown he_scale: {self.he_scale}")

    def __getitem__(self, idx: int):
        name = self._names[idx]

        ir = self._load_npy(self._ir_dir, name)
        ir_t = self._to_float_chw(ir)
        if ir_t.shape[0] < self.num_ir_channels:
            raise ValueError(f"IR file {name} has {ir_t.shape[0]} channels, need >= {self.num_ir_channels}")
        ir_t = ir_t[: self.num_ir_channels]
        ir_t = self._prep_ir(ir_t)
        ir_t = self._resize_chw(ir_t)

        if self.input_mode == "ir10":
            x = ir_t
        elif self.input_mode == "irvis13":
            vis = self._load_npy(self._vis_dir, name)
            vis_t = self._to_float_chw(vis)
            if vis_t.shape[0] != 3:
                raise ValueError(f"VIS file {name} expected 3 channels, got {vis_t.shape[0]}")
            # normalize VIS if 0..255
            if vis_t.max().item() > 1.5:
                vis_t = (vis_t / 255.0).clamp(0.0, 1.0)
            vis_t = self._resize_chw(vis_t)
            x = torch.cat([ir_t, vis_t], dim=0)
        else:
            raise ValueError(f"Unknown input_mode: {self.input_mode}")

        he = self._load_npy(self._he_dir, name)
        he_t = self._to_float_chw(he)
        if he_t.shape[0] != 3:
            raise ValueError(f"HE file {name} expected 3 channels, got {he_t.shape[0]}")
        he_t = self._resize_chw(he_t)
        y = self._scale_he(he_t)

        return {"x": x, "y": y, "name": name}

