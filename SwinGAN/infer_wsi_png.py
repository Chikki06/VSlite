"""Whole-slide PNG inference using SwinGAN generator checkpoints.

This module also acts as the shared lazy WSI loader for every per-model WSI
script in the repo (Restormer, NAFNet, VQGAN). The key invariant is:

  * IR and VIS sources are kept as LAZY handles (`spectral.open_memmap` /
    `PIL.Image.open`) inside `FolderData`. They are NEVER fully materialized
    into RAM and NEVER bulk-moved to GPU.
  * `infer_folder` (and the per-model variants in Restormer/NAFNet/VQGAN)
    reads exactly one tile-sized window at a time and ships only that tile
    to GPU.

Earlier versions of this file did `np.array(image.load())` for the entire
WSI and `data.vis.to(device)` for the entire WSI, which OOM-killed the
slurm step on ~30k x 30k slides (≈10 GB VIS + ~12 GB IR + GPU copies). The
helpers below intentionally mirror `scripts/infer_wsi_png_baseline.py`.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import spectral
import torch
import torch.nn.functional as F
import yaml

Image.MAX_IMAGE_PIXELS = None


@dataclass
class FolderData:
    """Lazy IR/VIS handles. None of these fields hold full WSI pixel buffers."""

    ir: object  # spectral memmap, shape (H, W, C)
    ir_shape_hwc: Tuple[int, int, int]
    vis: Optional[object] = None  # PIL.Image OR spectral memmap (H, W, C)
    vis_shape_hwc: Optional[Tuple[int, int, int]] = None
    vis_pixel_size: Optional[float] = None


class VSLiteFolderLoader:
    """Folder resolution for IR/VIS inputs used by WSI inference jobs."""

    def __init__(self, parent: str, folder_name: str, model_type: int):
        self.parent = parent
        self.folder_name = folder_name
        self.model_type = int(model_type)
        self.folder_category = self._get_folder_category()

    def _get_folder_category(self) -> str:
        folder_path = os.path.join(self.parent, self.folder_name)

        ir_subfolder = os.path.join(folder_path, "IR")
        vis_subfolder = os.path.join(folder_path, "VIS")
        if os.path.isdir(ir_subfolder) and os.path.isdir(vis_subfolder):
            return "ir_vis_subfolders"

        parent_basename = os.path.basename(self.parent).upper()
        if parent_basename == "IR":
            vis_sibling_dir = os.path.join(os.path.dirname(self.parent), "VIS")
            vis_sibling_folder = os.path.join(vis_sibling_dir, self.folder_name)
            if os.path.isdir(vis_sibling_folder):
                return "processed_siblings"
            vis_tiff = os.path.join(vis_sibling_dir, f"{self.folder_name}_VIS_REG.tif")
            if os.path.isfile(vis_tiff):
                return "processed_siblings_tiff"

        if os.path.isdir(ir_subfolder):
            return "ir_subfolder_tiff"

        return "direct"

    def _find_data_file(self, base_name: str, files: list[str]) -> Optional[str]:
        for ext in [".bsq", ".bip", ".bil", ".dat", ".img"]:
            candidate = base_name + ext
            if candidate in files:
                return candidate
        return None

    def _load_ir_from_dir(self, ir_folder_path: str) -> Tuple[object, Tuple[int, int, int]]:
        files = os.listdir(ir_folder_path)
        hdr_files = [f for f in files if f.endswith(".hdr")]
        if not hdr_files:
            raise FileNotFoundError(f"No .hdr files in IR folder: {ir_folder_path}")
        hdr_file = hdr_files[0]
        base_name = hdr_file[:-4]
        data_file = self._find_data_file(base_name, files)
        if data_file is None:
            raise FileNotFoundError(f"No paired data file for {hdr_file} in {ir_folder_path}")

        image = spectral.envi.open(
            os.path.join(ir_folder_path, hdr_file),
            os.path.join(ir_folder_path, data_file),
        )
        mm = image.open_memmap()  # (H, W, C), backed by file -- no RAM blow-up
        h, w, c = int(mm.shape[0]), int(mm.shape[1]), int(mm.shape[2])
        return mm, (h, w, c)

    def _load_vis_from_dir(self, vis_folder_path: str) -> Tuple[object, Tuple[int, int, int], float]:
        files = os.listdir(vis_folder_path)
        hdr_files = [f for f in files if f.endswith(".hdr")]
        if not hdr_files:
            raise FileNotFoundError(f"No .hdr files in VIS folder: {vis_folder_path}")
        hdr_file = hdr_files[0]
        base_name = hdr_file[:-4]
        data_file = self._find_data_file(base_name, files)
        if data_file is None:
            raise FileNotFoundError(f"No paired VIS data file for {hdr_file} in {vis_folder_path}")

        vis_image = spectral.envi.open(
            os.path.join(vis_folder_path, hdr_file),
            os.path.join(vis_folder_path, data_file),
        )
        vis_mm = vis_image.open_memmap()
        h, w, c = int(vis_mm.shape[0]), int(vis_mm.shape[1]), int(vis_mm.shape[2])

        pixel_size = vis_image.metadata.get("pixel size")
        if pixel_size:
            if isinstance(pixel_size, (list, tuple)):
                vis_pixel_size = float(pixel_size[0]) * 1e6
            else:
                vis_pixel_size = float(pixel_size) * 1e6
        else:
            vis_pixel_size = 2.0
        return vis_mm, (h, w, c), vis_pixel_size

    def _get_vis_tiff_path(self) -> str:
        vis_dir = self.parent.replace("/IR", "/VIS")
        return os.path.join(vis_dir, f"{self.folder_name}_VIS_REG.tif")

    def _load_vis_tiff(self, tiff_path: str) -> Tuple[object, Tuple[int, int, int], float]:
        if not os.path.isfile(tiff_path):
            raise FileNotFoundError(f"VIS TIFF not found: {tiff_path}")
        img = Image.open(tiff_path)  # lazy: no pixels loaded
        w, h = int(img.size[0]), int(img.size[1])
        c = len(img.getbands())
        return img, (h, w, c), 0.69

    def load(self) -> FolderData:
        folder_path = os.path.join(self.parent, self.folder_name)

        if self.folder_category in ("direct", "processed_siblings", "processed_siblings_tiff"):
            ir_folder = os.path.join(self.parent, self.folder_name)
        else:
            ir_folder = os.path.join(folder_path, "IR")

        ir_src, ir_shape = self._load_ir_from_dir(ir_folder)

        if self.model_type == 0:
            return FolderData(ir=ir_src, ir_shape_hwc=ir_shape)

        if self.folder_category == "direct":
            vis_folder = os.path.join(self.parent, self.folder_name + "_VIS")
            vis_src, vis_shape, px = self._load_vis_from_dir(vis_folder)
        elif self.folder_category == "ir_vis_subfolders":
            vis_folder = os.path.join(folder_path, "VIS")
            vis_src, vis_shape, px = self._load_vis_from_dir(vis_folder)
        elif self.folder_category == "processed_siblings":
            vis_folder = os.path.join(os.path.dirname(self.parent), "VIS", self.folder_name)
            vis_src, vis_shape, px = self._load_vis_from_dir(vis_folder)
        elif self.folder_category in ("processed_siblings_tiff", "ir_subfolder_tiff"):
            vis_src, vis_shape, px = self._load_vis_tiff(self._get_vis_tiff_path())
        else:
            raise RuntimeError(f"Unsupported folder category for 13ch mode: {self.folder_category}")

        return FolderData(ir=ir_src, ir_shape_hwc=ir_shape, vis=vis_src, vis_shape_hwc=vis_shape, vis_pixel_size=px)


# --------------------------------------------------------------------------------------
# Tile-windowed read helpers (lazy; mirror scripts/infer_wsi_png_baseline.py)
# --------------------------------------------------------------------------------------


def read_ir_patch_hwc(mm: object, *, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """Read a (h, w, C) IR window from an ENVI memmap without materializing the cube."""
    y0 = max(0, int(y0))
    x0 = max(0, int(x0))
    y1 = max(y0, int(y1))
    x1 = max(x0, int(x1))
    return np.asarray(mm[y0:y1, x0:x1, :], dtype=np.float32)


def read_vis_patch_to_chw(
    vis_src: object,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    expect_channels: int = 3,
) -> torch.Tensor:
    """Read a (C, h, w) VIS window as float32 in [0, 1]. Lazy in both PIL and ENVI cases."""
    if isinstance(vis_src, Image.Image):
        pw = max(0, int(x1 - x0))
        ph = max(0, int(y1 - y0))
        if ph == 0 or pw == 0:
            arr = np.zeros((0, 0, expect_channels), dtype=np.float32)
        else:
            box = (int(x0), int(y0), int(x0 + pw), int(y0 + ph))
            tile = vis_src.crop(box)
            if tile.mode not in ("RGB", "RGBA"):
                tile = tile.convert("RGB")
            arr_u8 = np.asarray(tile)
            if arr_u8.ndim == 2:
                arr_u8 = np.stack([arr_u8] * 3, axis=-1)
            arr = arr_u8.astype(np.float32) / 256.0
    else:
        # ENVI memmap: (H, W, C)
        arr = read_ir_patch_hwc(vis_src, y0=y0, y1=y1, x0=x0, x1=x1).astype(np.float32) / 256.0

    if arr.ndim != 3:
        raise RuntimeError(f"VIS patch has invalid shape: {arr.shape}")
    if arr.shape[2] != expect_channels:
        if arr.shape[2] > expect_channels:
            arr = arr[:, :, :expect_channels]
        else:
            pad = np.zeros((arr.shape[0], arr.shape[1], expect_channels - arr.shape[2]), dtype=arr.dtype)
            arr = np.concatenate([arr, pad], axis=2)
    chw = torch.from_numpy(np.transpose(arr, (2, 0, 1))).contiguous()
    return chw.clamp(0.0, 1.0)


def output_geometry(data: FolderData, model_type: int) -> Tuple[int, int]:
    """Return (r, c) of the target output image at 0.69 um/px.

    For model_type=1 we mirror `scripts/infer_wsi_png_baseline.py`'s assumption
    that the VIS source is already at the 0.69 um/px target pixel size (this is
    true for `input/VIS/*_VIS_REG.tif`, where `_load_vis_tiff` hardcodes 0.69).
    Doing a full-image resample to enforce 0.69 -- as the previous `_prep_vis`
    did -- is exactly what caused the OOM.
    """
    ir_h, ir_w, _ = data.ir_shape_hwc
    if int(model_type) == 1:
        if data.vis is None or data.vis_shape_hwc is None:
            raise RuntimeError("13ch mode requires VIS data")
        vis_h, vis_w, _ = data.vis_shape_hwc
        return int(vis_h), int(vis_w)
    ratio = 2.0 / 0.69
    return int(ir_h * ratio), int(ir_w * ratio)


def read_ir_tile_to_device(
    data: FolderData,
    *,
    y0_tgt: int,
    y1_tgt: int,
    x0_tgt: int,
    x1_tgt: int,
    tile_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Read an IR tile from disk and return it (1, C, tile_size, tile_size) on device.

    Coordinates are in TARGET (0.69 um/px) pixel space; the IR source lives at
    2.0 um/px, so we scale by 0.69/2.0 and bicubic-upsample to `tile_size`.
    """
    scaling_factor = 0.69 / 2.0
    y0 = int(y0_tgt * scaling_factor)
    y1 = int(y1_tgt * scaling_factor)
    x0 = int(x0_tgt * scaling_factor)
    x1 = int(x1_tgt * scaling_factor)
    ir_patch_hwc = read_ir_patch_hwc(data.ir, y0=y0, y1=y1, x0=x0, x1=x1)
    ir_patch = torch.from_numpy(np.transpose(ir_patch_hwc, (2, 0, 1))).unsqueeze(0)
    ir_patch = ir_patch.to(device=device, dtype=torch.float32, non_blocking=True).clamp(0.0, 5.0)
    target_lr = int(tile_size * scaling_factor)
    pad_rows = max(0, target_lr - ir_patch.shape[2])
    pad_cols = max(0, target_lr - ir_patch.shape[3])
    if pad_rows or pad_cols:
        ir_patch = F.pad(ir_patch, (0, pad_cols, 0, pad_rows), mode="constant", value=0.0)
    ir_patch = F.interpolate(ir_patch, size=(tile_size, tile_size), mode="bicubic", align_corners=True)
    return ir_patch


def read_vis_tile_to_device(
    data: FolderData,
    *,
    y0_tgt: int,
    y1_tgt: int,
    x0_tgt: int,
    x1_tgt: int,
    tile_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Read a VIS tile from disk and return it (1, 3, tile_size, tile_size) on device, in [0, 1]."""
    if data.vis is None:
        raise RuntimeError("VIS data required")
    vis_chw = read_vis_patch_to_chw(
        data.vis, y0=int(y0_tgt), y1=int(y1_tgt), x0=int(x0_tgt), x1=int(x1_tgt), expect_channels=3
    )
    v_pad_rows = max(0, tile_size - vis_chw.shape[1])
    v_pad_cols = max(0, tile_size - vis_chw.shape[2])
    vis_patch = vis_chw.unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True)
    if v_pad_rows or v_pad_cols:
        vis_patch = F.pad(vis_patch, (0, v_pad_cols, 0, v_pad_rows), mode="constant", value=0.0)
    return vis_patch


# --------------------------------------------------------------------------------------
# Model builder + SwinGAN-specific infer_folder
# --------------------------------------------------------------------------------------


def load_config(path: str) -> SimpleNamespace:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return SimpleNamespace(**cfg)


def build_model(config_path: str, ckpt_path: str, model_type: int, device: torch.device):
    from Networks.vision_transformer import ProjectedSwinUnet

    cfg = load_config(config_path)
    in_chans = 13 if int(model_type) == 1 else 10
    use_cbam = bool(getattr(cfg, "use_cbam", False))
    G = ProjectedSwinUnet(cfg, in_chans=in_chans, out_chans=3, use_cbam=use_cbam).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "G_state" in ckpt:
        state = ckpt["G_state"]
    elif isinstance(ckpt, dict) and "G_model_state_dict" in ckpt:
        state = ckpt["G_model_state_dict"]
    else:
        state = ckpt

    G.load_state_dict(state, strict=True)
    G.eval()
    return G


def infer_folder(
    G,
    data: FolderData,
    model_type: int,
    tile_size: int,
    device: torch.device,
) -> np.ndarray:
    r, c = output_geometry(data, model_type)

    rn = int(math.ceil(r / tile_size))
    cn = int(math.ceil(c / tile_size))

    out = np.zeros((rn * tile_size + 1, cn * tile_size + 1, 3), dtype=np.uint8)
    I = list(range(0, rn * tile_size + 1, tile_size))
    J = list(range(0, cn * tile_size + 1, tile_size))

    with torch.no_grad():
        for ii in range(len(I) - 1):
            for jj in range(len(J) - 1):
                ir_patch = read_ir_tile_to_device(
                    data,
                    y0_tgt=I[ii],
                    y1_tgt=I[ii + 1],
                    x0_tgt=J[jj],
                    x1_tgt=J[jj + 1],
                    tile_size=tile_size,
                    device=device,
                )

                if int(model_type) == 1:
                    vis_patch = read_vis_tile_to_device(
                        data,
                        y0_tgt=I[ii],
                        y1_tgt=I[ii + 1],
                        x0_tgt=J[jj],
                        x1_tgt=J[jj + 1],
                        tile_size=tile_size,
                        device=device,
                    )
                    x = torch.cat([ir_patch, vis_patch], dim=1)
                else:
                    vis_patch = None
                    x = ir_patch

                y_hat = torch.tanh(G(x))
                y_np = (y_hat[0].clamp(-1.0, 1.0).cpu().numpy() * 0.5) + 0.5
                y_u8 = np.clip(y_np * 255.0, 0, 255).astype(np.uint8)
                out[I[ii] : I[ii + 1], J[jj] : J[jj + 1], :] = np.transpose(y_u8, (1, 2, 0))

                del ir_patch, x, y_hat, y_np, y_u8
                if vis_patch is not None:
                    del vis_patch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            gc.collect()

    out = out[:r, :c, :]
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Whole-slide PNG inference using SwinGAN generator checkpoints")
    p.add_argument("--ckpt", required=True, type=str, help="Path to SwinGAN checkpoint (.pth)")
    p.add_argument(
        "--config",
        default="/projects/illinois/eng/bioe/rxb/akshat/VSlite/SwinGAN/config_i2i_baseline.yaml",
        type=str,
        help="SwinGAN config used to build the generator",
    )
    p.add_argument("--model", type=int, default=0, choices=[0, 1], help="0: 10ch IR, 1: 13ch IR+VIS")
    p.add_argument("--folder", type=str, default=None, help="Optional single folder path to process")
    p.add_argument("--parent_folder", type=str, default=None, help="Parent folder containing sample folders")
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for PNG files (default: <ckpt_dir>/wsi_png)",
    )
    p.add_argument("--tile_size", type=int, default=None, help="Patch size for model inference")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = build_model(args.config, args.ckpt, args.model, device)

    cfg = load_config(args.config)
    tile_size = int(args.tile_size if args.tile_size is not None else getattr(cfg, "img_size", 256))

    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "wsi_png")
    else:
        output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.folder:
        folder_path = os.path.abspath(args.folder)
        parent = os.path.dirname(folder_path)
        folder_names = [os.path.basename(folder_path)]
    elif args.parent_folder:
        parent = os.path.abspath(args.parent_folder)
        folder_names = [
            d for d in sorted(os.listdir(parent)) if os.path.isdir(os.path.join(parent, d)) and not d.endswith("_VIS")
        ]
    else:
        raise ValueError("Provide --folder or --parent_folder")

    for folder_name in folder_names:
        print(f"Processing {folder_name} ...", flush=True)
        loader = VSLiteFolderLoader(parent, folder_name, args.model)
        data = loader.load()
        pred = infer_folder(G, data, args.model, tile_size=tile_size, device=device)

        suffix = "13ch" if int(args.model) == 1 else "10ch"
        out_path = os.path.join(output_dir, f"{folder_name}_{suffix}.png")
        Image.fromarray(pred).save(out_path)
        print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
