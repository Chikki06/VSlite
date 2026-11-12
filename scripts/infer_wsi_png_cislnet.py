import argparse
import math
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import spectral
import torch
import torch.nn.functional as F
import gc

# Ensure repo-root modules (e.g. `data/`, `models/`) are importable even when invoked as
# `python scripts/infer_wsi_png_baseline.py` (which otherwise prepends `scripts/` to sys.path).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from options.test_options import TestOptions  # noqa: E402
from models import create_model  # noqa: E402
from cislnet.wsi_local_padding import (  # noqa: E402
    LocalPaddingConfig,
    WSILocalPaddingState,
    add_local_padding_cli_args,
    effective_wsi_batch_size,
    resolve_local_padding_config,
)

Image.MAX_IMAGE_PIXELS = None

INT32_INDEX_MAX = 2_147_483_647

# PIL Image.crop is not reliably thread-safe on a shared handle.
_VIS_READ_LOCK = threading.Lock()


def resolve_wsi_batch_size(cli_batch_size: Optional[int], tile_size: int) -> int:
    """Resolve micro-batch size: CLI > WSI_BATCH_SIZE env > tile-size heuristic.

    CISLNet is fully convolutional; training uses ~10 @1024 on A100. Inference
    (eval / no-grad) can go higher — default 16 @1024 instead of the old BS=1.
    """
    if cli_batch_size is not None:
        return max(1, int(cli_batch_size))
    env = os.environ.get("WSI_BATCH_SIZE", "").strip()
    if env:
        return max(1, int(env))
    if tile_size <= 256:
        return 32
    if tile_size <= 512:
        return 16
    return 16


def resolve_wsi_num_workers(cli_num_workers: Optional[int] = None) -> int:
    """Parallel CPU tile readers: CLI > WSI_NUM_WORKERS env > 4."""
    if cli_num_workers is not None:
        return max(0, int(cli_num_workers))
    env = os.environ.get("WSI_NUM_WORKERS", "").strip()
    if env:
        return max(0, int(env))
    return 4


def _is_retryable_batch_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "32-bit index math" in msg:
        return True
    if "out of memory" in msg:
        return True
    if "cuda" in msg and "memory" in msg:
        return True
    return False


@dataclass
class FolderData:
    # IR/VIS are intentionally stored as lazy sources (memmap or PIL Image for TIFF)
    # to avoid loading whole-slide arrays into RAM.
    ir: object
    ir_shape_hwc: Tuple[int, int, int]
    vis: Optional[object]
    vis_shape_hwc: Optional[Tuple[int, int, int]]
    vis_pixel_size: Optional[float]


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
        mm = image.open_memmap()  # shape: (H, W, C), backed by file
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
        vis_mm = vis_image.open_memmap()  # shape: (H, W, C)
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
        # Pillow reads tiled TIFFs by region; Image.MAX_IMAGE_PIXELS is None at import.
        img = Image.open(tiff_path)
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
            return FolderData(ir=ir_src, ir_shape_hwc=ir_shape, vis=None, vis_shape_hwc=None, vis_pixel_size=None)

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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Whole-slide PNG inference using baseline contrastive pix2pix checkpoints")
    p.add_argument("--name", required=True, type=str, help="Run name (e.g. vs13, vs3)")
    p.add_argument("--checkpoints_dir", required=True, type=str, help="Baseline checkpoints dir (contains <name>/)")
    p.add_argument("--which_epoch", type=str, default="latest")
    p.add_argument("--results_dir", type=str, default=str(_REPO_ROOT / "results" / "baseline"))
    p.add_argument("--gpu_ids", type=str, default="0")

    # Must match the baseline model family.
    p.add_argument("--dataset_mode", type=str, default="VS")
    p.add_argument("--model", type=str, default="contrastive_pix2pix")

    # Input channel policy.
    p.add_argument("--input_nc", type=int, default=13)
    p.add_argument("--num_ir_channels", type=int, default=10)
    p.add_argument(
        "--ir_channel_indices",
        type=str,
        default="",
        help='Optional comma-separated IR indices (e.g. "0,8,9"). If set, overrides --num_ir_channels slicing.',
    )

    # WSI traversal + outputs.
    p.add_argument("--folder", type=str, default=None, help="Optional single folder path to process")
    p.add_argument("--parent_folder", type=str, default=None, help="Parent folder containing sample folders (e.g. input/IR)")
    p.add_argument(
        "--folders",
        type=str,
        default="",
        help='Optional comma-separated slide folder names under --parent_folder (e.g. "10442-1,3840-1").',
    )
    p.add_argument(
        "--max_folders",
        type=int,
        default=0,
        help="If >0, process only the first N folders (after --folders filter / sorted listing).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for PNG/JPEG files (default: results_dir/test/<name>/wsi_pred)",
    )
    p.add_argument(
        "--image_format",
        type=str,
        default="png",
        choices=("png", "jpeg", "jpg"),
        help="Output image format. jpeg/jpg recommended for large WSI previews.",
    )
    p.add_argument(
        "--jpeg_quality",
        type=int,
        default=92,
        help="JPEG quality 1-95 (slight compression; default 92).",
    )
    p.add_argument(
        "--png_compress_level",
        type=int,
        default=6,
        help="PNG zlib compress level 0-9 (default 6).",
    )
    p.add_argument(
        "--tile_size",
        type=int,
        default=1024,
        help="Tile size for generator inference (pixels); match `third_party/main.py` (1024) and VS_dataset image_size",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional inference micro-batch size in tiles; overrides WSI_BATCH_SIZE env when provided.",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Parallel CPU tile readers (IR/VIS). Overrides WSI_NUM_WORKERS (default 4).",
    )
    p.add_argument(
        "--optim",
        action="store_true",
        default=False,
        help="Move VIS tile interpolation onto GPU and pack batches into preallocated tensors.",
    )
    add_local_padding_cli_args(p)
    return p.parse_args()


def _save_wsi_image(pred: np.ndarray, out_path: str, *, image_format: str, jpeg_quality: int, png_compress_level: int) -> str:
    """Save WSI prediction with mild compression. Returns the path written."""
    fmt = str(image_format or "png").lower().strip()
    if fmt in ("jpg", "jpeg"):
        if not out_path.lower().endswith((".jpg", ".jpeg")):
            out_path = os.path.splitext(out_path)[0] + ".jpg"
        Image.fromarray(pred).convert("RGB").save(
            out_path, format="JPEG", quality=int(jpeg_quality), optimize=True
        )
    else:
        if not out_path.lower().endswith(".png"):
            out_path = os.path.splitext(out_path)[0] + ".png"
        Image.fromarray(pred).save(
            out_path, format="PNG", optimize=True, compress_level=int(png_compress_level)
        )
    return out_path


def _parse_ir_indices(s: str) -> list[int]:
    s = str(s or "").strip()
    if not s:
        return []
    idx = [int(x.strip()) for x in s.split(",") if x.strip() != ""]
    if not idx:
        raise ValueError(f"Invalid --ir_channel_indices={s!r}. No indices parsed.")
    return idx


def _select_ir_indices(c_tot: int, *, num_ir_channels: int, ir_channel_indices: str) -> tuple[list[int], int]:
    idx = _parse_ir_indices(ir_channel_indices)
    if idx:
        bad = [i for i in idx if i < 0 or i >= c_tot]
        if bad:
            raise ValueError(f"--ir_channel_indices out of range for IR channels={c_tot}: {bad}")
        return idx, len(idx)
    else:
        n = int(num_ir_channels)
        if n == 0:
            # VIS-only run (input_nc=3, no IR bands): empty IR index set; the
            # ir_vis tile path below then packs VIS RGB alone (vis_offset=0).
            return [], 0
        if n <= 0 or n > c_tot:
            raise ValueError(f"Invalid --num_ir_channels={n} for IR channels={c_tot}")
        return list(range(n)), n


def _loader_model_type(input_nc: int, num_ir_channels: int, ir_channel_indices: str) -> int:
    """0=IR-only loader, 1=load VIS. Must agree with `infer_folder` IR vs IR+VIS split."""
    s = str(ir_channel_indices or "").strip()
    if s:
        n_ir = len(_parse_ir_indices(s))
    else:
        n_ir = int(num_ir_channels)
    return 1 if int(input_nc) == int(n_ir) + 3 else 0


def _read_envi_patch_hwc(mm: np.ndarray, *, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    # mm is (H, W, C) memmap; slicing stays lazy / windowed.
    y0 = max(0, int(y0))
    x0 = max(0, int(x0))
    y1 = max(y0, int(y1))
    x1 = max(x0, int(x1))
    return np.asarray(mm[y0:y1, x0:x1, :], dtype=np.float32)


def _hwc_to_01_f32(arr: np.ndarray, *, expect_channels: int) -> np.ndarray:
    """Normalize HWC array to float32 roughly in [0, 1] (matches training `/256` on 8-bit paths)."""
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3:
        raise RuntimeError(f"VIS patch has invalid shape: {arr.shape}")
    if arr.shape[2] != expect_channels:
        if arr.shape[2] > expect_channels:
            arr = arr[:, :, :expect_channels]
        else:
            pad = np.zeros(
                (arr.shape[0], arr.shape[1], expect_channels - arr.shape[2]),
                dtype=arr.dtype,
            )
            arr = np.concatenate([arr, pad], axis=2)
    t = arr.astype(np.float32)
    dt = arr.dtype
    if dt == np.uint16:
        t = t * (1.0 / 65535.0)
    elif dt in (np.float32, np.float64):
        t = np.clip(t, 0.0, None)
        mx = float(np.max(t)) if t.size else 0.0
        if mx > 1.5:
            t = t / 256.0
    else:
        t = t / 256.0
    return np.clip(t, 0.0, 1.0)


def _read_vis_source_crop_hwc(
    vis_src: object,
    *,
    y0s: int,
    y1s: int,
    x0s: int,
    x1s: int,
    expect_channels: int = 3,
) -> np.ndarray:
    """Read VIS pixels from native storage (integer source box), return HWC float32 [0,1]."""
    y0s = max(0, int(y0s))
    x0s = max(0, int(x0s))
    y1s = max(y0s, int(y1s))
    x1s = max(x0s, int(x1s))
    if isinstance(vis_src, Image.Image):
        pw = x1s - x0s
        ph = y1s - y0s
        if ph <= 0 or pw <= 0:
            return np.zeros((0, 0, expect_channels), dtype=np.float32)
        box = (int(x0s), int(y0s), int(x0s + pw), int(y0s + ph))
        tile = vis_src.crop(box)
        if tile.mode not in ("RGB", "RGBA"):
            tile = tile.convert("RGB")
        raw = np.asarray(tile)
    else:
        # Keep native integer dtype so `_hwc_to_01_f32` can distinguish 8- vs 16-bit ENVI cubes.
        raw = np.asarray(vis_src[y0s:y1s, x0s:x1s, :])
    return _hwc_to_01_f32(raw, expect_channels=expect_channels)


def _read_vis_patch_to_chw(
    vis_src: object,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    expect_channels: int = 3,
    native_hw: Optional[Tuple[int, int]] = None,
    vis_scale: float = 1.0,
) -> torch.Tensor:
    """Return float32 CHW in [0,1] for a tile in *output* (0.69 µm) pixel coordinates.

    Mirrors `third_party/main.py`, which resamples VIS to 0.69 µm *before* tiling. Here we
    resample lazily per tile: map output rectangle to native VIS indices, read a window, then
    `interpolate` to the exact tile footprint when ``vis_scale != 1``.
    """
    h_tgt = max(0, int(y1 - y0))
    w_tgt = max(0, int(x1 - x0))
    if h_tgt == 0 or w_tgt == 0:
        return torch.zeros((expect_channels, 0, 0), dtype=torch.float32)

    scale = float(vis_scale) if float(vis_scale) > 0 else 1.0
    if native_hw is None:
        # Legacy 1:1 path (native pixels already match the output grid).
        arr = _read_vis_source_crop_hwc(
            vis_src, y0s=y0, y1s=y1, x0s=x0, x1s=x1, expect_channels=expect_channels
        )
        chw = torch.from_numpy(np.transpose(arr, (2, 0, 1))).contiguous()
        return chw.clamp(0.0, 1.0)

    vis_h, vis_w = int(native_hw[0]), int(native_hw[1])
    if abs(scale - 1.0) < 1e-6:
        y0s = max(0, min(vis_h, y0))
        y1s = max(y0s, min(vis_h, y1))
        x0s = max(0, min(vis_w, x0))
        x1s = max(x0s, min(vis_w, x1))
        arr = _read_vis_source_crop_hwc(
            vis_src, y0s=y0s, y1s=y1s, x0s=x0s, x1s=x1s, expect_channels=expect_channels
        )
    else:
        fy0 = float(y0) / scale
        fy1 = float(y1) / scale
        fx0 = float(x0) / scale
        fx1 = float(x1) / scale
        y0s = max(0, int(math.floor(fy0)))
        y1s = min(vis_h, max(y0s + 1, int(math.ceil(fy1))))
        x0s = max(0, int(math.floor(fx0)))
        x1s = min(vis_w, max(x0s + 1, int(math.ceil(fx1))))
        arr = _read_vis_source_crop_hwc(
            vis_src, y0s=y0s, y1s=y1s, x0s=x0s, x1s=x1s, expect_channels=expect_channels
        )

    chw = torch.from_numpy(np.transpose(arr, (2, 0, 1))).contiguous()
    if chw.shape[1] == 0 or chw.shape[2] == 0:
        return torch.zeros((expect_channels, h_tgt, w_tgt), dtype=torch.float32)
    if chw.shape[1] != h_tgt or chw.shape[2] != w_tgt:
        chw = F.interpolate(
            chw.unsqueeze(0),
            size=(h_tgt, w_tgt),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return chw.clamp(0.0, 1.0)


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
    """Read IR from ENVI memmap and return (1, C, tile_size, tile_size) on device.

    Target coords are at 0.69 µm/px; IR source is at 2.0 µm/px (same as third_party/main.py).
    """
    scaling_factor = 0.69 / 2.0
    y0 = int(y0_tgt * scaling_factor)
    y1 = int(y1_tgt * scaling_factor)
    x0 = int(x0_tgt * scaling_factor)
    x1 = int(x1_tgt * scaling_factor)
    ir_patch_hwc = _read_envi_patch_hwc(data.ir, y0=y0, y1=y1, x0=x0, x1=x1)
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
    """Read VIS and return (1, 3, tile_size, tile_size) on device in [0, 1]."""
    if data.vis is None:
        raise RuntimeError("VIS data required")
    vis_chw = _read_vis_patch_to_chw(
        data.vis,
        y0=int(y0_tgt),
        y1=int(y1_tgt),
        x0=int(x0_tgt),
        x1=int(x1_tgt),
        expect_channels=3,
    )
    v_pad_rows = max(0, tile_size - vis_chw.shape[1])
    v_pad_cols = max(0, tile_size - vis_chw.shape[2])
    vis_patch = vis_chw.unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True)
    if v_pad_rows or v_pad_cols:
        vis_patch = F.pad(vis_patch, (0, v_pad_cols, 0, v_pad_rows), mode="constant", value=0.0)
    return vis_patch


def _output_geometry(
    data: FolderData,
    *,
    mode: str,
    ir_h: int,
    ir_w: int,
) -> tuple[int, int, float, Optional[Tuple[int, int]]]:
    """Return (r, c, vis_scale, native_vis_hw) for the 0.69 µm output grid.

    Matches `third_party/main.py`: 13ch uses VIS resampled to 0.69 µm/px; IR-only uses IR×(2/0.69).
    """
    if mode == "ir_vis":
        assert data.vis_shape_hwc is not None and data.vis_pixel_size is not None
        vis_h, vis_w, _ = data.vis_shape_hwc
        target_ps = 0.69
        vis_scale = float(data.vis_pixel_size) / target_ps
        r = max(1, int(vis_h * vis_scale))
        c = max(1, int(vis_w * vis_scale))
        native_hw = (int(vis_h), int(vis_w))
        return r, c, vis_scale, native_hw
    ratio = 2.0 / 0.69
    return max(1, int(ir_h * ratio)), max(1, int(ir_w * ratio)), 1.0, None


def _fake_to_tile_u8(fake: torch.Tensor, *, dh: int, dw: int) -> np.ndarray:
    """Convert generator output (BCHW or CHW, [-1,1]) to uint8 HWC sized (dh, dw)."""
    if not torch.is_tensor(fake):
        raise TypeError(f"Expected inference tensor output; got {type(fake)}")
    y_chw = fake[0] if fake.dim() == 4 else fake
    if y_chw.dim() != 3 or int(y_chw.shape[0]) != 3:
        raise RuntimeError(f"Expected CHW with 3 channels; got {tuple(y_chw.shape)}")
    y_np = y_chw.clamp(-1.0, 1.0).detach().float().cpu().numpy()
    y_np = np.transpose(y_np, (1, 2, 0))  # HWC
    if y_np.shape[0] != dh or y_np.shape[1] != dw:
        # Upsample undersized tiles (avoids blank bands when stitch slots are larger than model output).
        t = torch.from_numpy(np.transpose(y_np, (2, 0, 1))).unsqueeze(0)
        t = F.interpolate(t, size=(dh, dw), mode="bilinear", align_corners=False)
        y_np = np.transpose(t.squeeze(0).numpy(), (1, 2, 0))
    y_np = y_np[:dh, :dw, :]
    return np.ascontiguousarray(np.clip((y_np * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8))


def _load_one_tile_cpu(
    data: FolderData,
    *,
    ii: int,
    jj: int,
    I: Sequence[int],
    J: Sequence[int],
    tile_size: int,
    ir_idx: Sequence[int],
    mode: str,
    optim: bool,
    vis_scale: float,
    native_vis_hw: Optional[Tuple[int, int]],
    vis: Optional[object],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Load one tile to CPU float32 CHW tensors (IR channel-selected, optional VIS)."""
    ir_patch = read_ir_tile_to_device(
        data,
        y0_tgt=I[ii],
        y1_tgt=I[ii + 1],
        x0_tgt=J[jj],
        x1_tgt=J[jj + 1],
        tile_size=tile_size,
        device=torch.device("cpu"),
    )
    ir_chw = ir_patch[0, list(ir_idx), :, :].contiguous()
    if mode != "ir_vis":
        return ir_chw, None

    assert vis is not None
    # Serialize VIS crops: shared PIL handle is not thread-safe.
    with _VIS_READ_LOCK:
        if optim or abs(float(vis_scale) - 1.0) >= 1e-6:
            vis_chw = _read_vis_patch_to_chw(
                vis,
                y0=int(I[ii]),
                y1=int(I[ii + 1]),
                x0=int(J[jj]),
                x1=int(J[jj + 1]),
                expect_channels=3,
                native_hw=native_vis_hw,
                vis_scale=vis_scale,
            )
            if vis_chw.shape[1] != tile_size or vis_chw.shape[2] != tile_size:
                vis_chw = F.interpolate(
                    vis_chw.unsqueeze(0),
                    size=(tile_size, tile_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            v_pad_rows = max(0, tile_size - vis_chw.shape[1])
            v_pad_cols = max(0, tile_size - vis_chw.shape[2])
            if v_pad_rows or v_pad_cols:
                vis_chw = F.pad(vis_chw.unsqueeze(0), (0, v_pad_cols, 0, v_pad_rows), mode="constant", value=0.0).squeeze(0)
        else:
            vis_patch = read_vis_tile_to_device(
                data,
                y0_tgt=I[ii],
                y1_tgt=I[ii + 1],
                x0_tgt=J[jj],
                x1_tgt=J[jj + 1],
                tile_size=tile_size,
                device=torch.device("cpu"),
            )
            vis_chw = vis_patch[0]
    return ir_chw, vis_chw.contiguous()


def infer_folder(
    model,
    data: FolderData,
    *,
    input_nc: int,
    num_ir_channels: int,
    ir_channel_indices: str,
    tile_size: int,
    batch_size: int,
    optim: bool,
    device: torch.device,
    local_padding: Optional[LocalPaddingConfig] = None,
    num_workers: Optional[int] = None,
) -> np.ndarray:
    ir_h, ir_w, ir_c = data.ir_shape_hwc
    ir_idx, n_ir = _select_ir_indices(int(ir_c), num_ir_channels=int(num_ir_channels), ir_channel_indices=str(ir_channel_indices or ""))

    # Decide whether this run is IR-only or IR+VIS based on channel arithmetic.
    if int(input_nc) == int(n_ir):
        mode = "ir_only"
    elif int(input_nc) == int(n_ir) + 3:
        mode = "ir_vis"
    else:
        raise ValueError(
            f"Invalid channel config: input_nc={input_nc} with num_ir_channels={n_ir}. "
            f"Use input_nc={n_ir} (IR-only) or input_nc={n_ir + 3} (IR+VIS)."
        )

    if mode == "ir_vis":
        if data.vis is None or data.vis_pixel_size is None:
            raise RuntimeError("IR+VIS mode requires VIS data + pixel size metadata.")
        assert data.vis_shape_hwc is not None
        if int(data.vis_shape_hwc[2]) < 3:
            raise RuntimeError(f"VIS must have >=3 channels; got {data.vis_shape_hwc}")
        vis = data.vis
    else:
        vis = None

    r, c, vis_scale, native_vis_hw = _output_geometry(data, mode=mode, ir_h=ir_h, ir_w=ir_w)

    rn = int(math.ceil(r / tile_size))
    cn = int(math.ceil(c / tile_size))
    # Store as uint8 directly to reduce RAM by 4x vs float32.
    out = np.zeros((rn * tile_size + 1, cn * tile_size + 1, 3), dtype=np.uint8)

    I = list(range(0, rn * tile_size + 1, tile_size))
    J = list(range(0, cn * tile_size + 1, tile_size))

    tile_coords = [(ii, jj, int(I[ii + 1] - I[ii]), int(J[jj + 1] - J[jj])) for ii in range(len(I) - 1) for jj in range(len(J) - 1)]

    # Conservative cap to avoid the 32-bit index math crash in PyTorch internals.
    max_bs_int32 = max(1, INT32_INDEX_MAX // (64 * tile_size * tile_size))
    requested_bs = max(1, int(batch_size))
    bs = min(requested_bs, max_bs_int32)
    if bs != requested_bs:
        print(
            f"[WSI] Requested batch_size={requested_bs} exceeds int32-safe cap={max_bs_int32} at tile_size={tile_size}; using {bs}."
        )

    vis_offset = len(ir_idx)
    lp_cfg = local_padding or LocalPaddingConfig(enabled=False)
    bs = effective_wsi_batch_size(bs, lp_cfg)
    lp_state = WSILocalPaddingState(lp_cfg) if lp_cfg.enabled else None
    nw = resolve_wsi_num_workers(num_workers)
    if lp_cfg.enabled:
        # Local padding needs sequential row/col state; keep IO on the main thread.
        nw = 0
        print(
            f"[WSI] local_padding enabled: padding_size={lp_cfg.padding_size} "
            f"outer={lp_cfg.outer_padding}",
            flush=True,
        )
    print(f"[WSI] infer_folder batch_size={bs} num_workers={nw} tiles={len(tile_coords)}", flush=True)

    def _load_tile(coord: Tuple[int, int, int, int]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        ii, jj, _, _ = coord
        return _load_one_tile_cpu(
            data,
            ii=ii,
            jj=jj,
            I=I,
            J=J,
            tile_size=tile_size,
            ir_idx=ir_idx,
            mode=mode,
            optim=optim,
            vis_scale=vis_scale,
            native_vis_hw=native_vis_hw,
            vis=vis,
        )

    def _load_sub_batch_cpu(
        sub_batch: Sequence[Tuple[int, int, int, int]],
        pool: Optional[ThreadPoolExecutor],
    ) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        if pool is None or len(sub_batch) == 1:
            return [_load_tile(c) for c in sub_batch]
        return list(pool.map(_load_tile, sub_batch))

    pool: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=nw) if nw > 0 else None
    try:
        with torch.no_grad():
            for b0 in range(0, len(tile_coords), bs):
                batch = tile_coords[b0 : b0 + bs]
                sub_bs = len(batch)

                while sub_bs > 0:
                    sub_batch = batch[:sub_bs]
                    x_channels = vis_offset + (3 if mode == "ir_vis" else 0)
                    cpu_tiles = _load_sub_batch_cpu(sub_batch, pool)
                    x_batch = torch.empty((sub_bs, x_channels, tile_size, tile_size), device=device, dtype=torch.float32)
                    y_dummy = torch.zeros((sub_bs, 3, tile_size, tile_size), device=device, dtype=torch.float32)

                    try:
                        for slot, ((ii, jj, _, _), (ir_chw, vis_chw)) in enumerate(zip(sub_batch, cpu_tiles)):
                            if lp_state is not None and jj == 0:
                                lp_state.on_new_row()
                            x_batch[slot, :vis_offset, :, :] = ir_chw.to(device=device, non_blocking=True)
                            if mode == "ir_vis":
                                assert vis_chw is not None
                                x_batch[slot, vis_offset:, :, :] = vis_chw.to(device=device, non_blocking=True)
                            if lp_state is not None:
                                x_batch[slot] = lp_state.apply_to_input(
                                    x_batch[slot], ii=ii, jj=jj, vis_offset=vis_offset
                                )

                        fake_batch = model({"input": x_batch, "target": y_dummy}, mode="inference")

                    except RuntimeError as e:
                        if _is_retryable_batch_oom(e) and sub_bs > 1:
                            new_sub_bs = max(1, sub_bs // 2)
                            print(
                                f"[WSI] batch OOM/index at sub_batch={sub_bs}; retrying with {new_sub_bs}.",
                                flush=True,
                            )
                            del x_batch, y_dummy, cpu_tiles
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            gc.collect()
                            sub_bs = new_sub_bs
                            continue
                        raise

                    if torch.is_tensor(fake_batch) and fake_batch.dim() == 3:
                        fake_batch = fake_batch.unsqueeze(0)

                    for kk, (ii, jj, dh, dw) in enumerate(sub_batch):
                        fake = fake_batch[kk] if torch.is_tensor(fake_batch) and fake_batch.dim() == 4 else fake_batch
                        y_u8 = _fake_to_tile_u8(fake, dh=dh, dw=dw)
                        out[I[ii] : I[ii + 1], J[jj] : J[jj + 1], :] = y_u8
                        if lp_state is not None:
                            n_ii = len(I) - 1
                            n_jj = len(J) - 1
                            y_chw = fake if fake.dim() == 3 else fake
                            lp_state.update_after_tile(
                                x_input=x_batch[kk],
                                y_output=y_chw,
                                vis_offset=vis_offset,
                                is_last_col=(jj == n_jj - 1),
                                is_last_row=(ii == n_ii - 1),
                            )

                    del x_batch, y_dummy, fake_batch, cpu_tiles
                    batch = batch[sub_bs:]
                    sub_bs = len(batch)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    out = out[:r, :c, :]
    return out


def main() -> None:
    args = _parse_args()
    lp_cfg = resolve_local_padding_config(
        local_padding=bool(getattr(args, "local_padding", False)),
        local_padding_size=int(getattr(args, "local_padding_size", 1)),
        local_padding_outer=str(getattr(args, "local_padding_outer", "replicate")),
    )

    # Construct a TestOptions parse that loads the model + opt defaults from the checkpoint folder.
    # This mirrors `scripts/vs_infer_and_metrics.py` behavior, but without the patch dataset.
    argv = [
        "infer_wsi_png_baseline.py",
        "--name",
        args.name,
        "--checkpoints_dir",
        args.checkpoints_dir,
        "--dataroot",
        str(_REPO_ROOT / "dataset" / "data_frozen"),  # unused by WSI loader; required by options parser
        "--dataset_mode",
        args.dataset_mode,
        "--model",
        args.model,
        "--which_epoch",
        args.which_epoch,
        "--results_dir",
        args.results_dir,
        "--gpu_ids",
        args.gpu_ids,
        "--num_test",
        str(10**9),
        "--input_nc",
        str(int(args.input_nc)),
        "--num_ir_channels",
        str(int(args.num_ir_channels)),
        "--load_from_opt_file",
        "--use_wandb",
        "false",
    ]
    if str(args.ir_channel_indices or "").strip():
        argv += ["--ir_channel_indices", str(args.ir_channel_indices)]
    sys.argv = argv

    opt = TestOptions().parse()
    opt.isTrain = False
    opt.phase = "test"
    opt.serial_batches = True
    opt.no_flip = True
    opt.no_flip_vert = True
    opt.no_rotate = True
    opt.batch_size = resolve_wsi_batch_size(args.batch_size, int(args.tile_size))
    nw = resolve_wsi_num_workers(getattr(args, "num_workers", None))
    print(f"[WSI] Using batch_size={opt.batch_size} num_workers={nw} (cli_bs={args.batch_size})")

    model = create_model(opt)
    model.eval()

    if args.output_dir is None:
        output_dir = os.path.join(os.path.abspath(args.results_dir), "test", str(args.name), "wsi_pred")
    else:
        output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.folder:
        folder_path = os.path.abspath(args.folder)
        parent = os.path.dirname(folder_path)
        folder_names = [os.path.basename(folder_path)]
    elif args.parent_folder:
        parent = os.path.abspath(args.parent_folder)
        want = [x.strip() for x in str(args.folders or "").split(",") if x.strip()]
        all_names = [
            d for d in sorted(os.listdir(parent)) if os.path.isdir(os.path.join(parent, d)) and not d.endswith("_VIS")
        ]
        if want:
            missing = [n for n in want if n not in set(all_names)]
            if missing:
                raise ValueError(f"--folders not found under {parent}: {missing}")
            folder_names = want
        else:
            folder_names = all_names
    else:
        raise ValueError("Provide --folder or --parent_folder")

    max_folders = int(args.max_folders or 0)
    if max_folders > 0:
        folder_names = folder_names[:max_folders]

    device = torch.device("cuda" if (torch.cuda.is_available() and str(args.gpu_ids) != "-1") else "cpu")

    for folder_name in folder_names:
        print(f"Processing {folder_name} ...")
        # model_type: 0=IR-only, 1=IR+VIS. Must use the same n_ir as `infer_folder` (indices vs slice).
        model_type = _loader_model_type(
            int(args.input_nc), int(args.num_ir_channels), str(args.ir_channel_indices or "")
        )
        loader = VSLiteFolderLoader(parent, folder_name, model_type=model_type)
        data = loader.load()
        pred = infer_folder(
            model,
            data,
            input_nc=int(args.input_nc),
            num_ir_channels=int(args.num_ir_channels),
            ir_channel_indices=str(args.ir_channel_indices or ""),
            tile_size=int(args.tile_size),
            batch_size=int(opt.batch_size),
            optim=bool(args.optim),
            device=device,
            local_padding=lp_cfg,
            num_workers=nw,
        )

        # Suffix reflects total generator input channels (Vis+ uses input_nc = n_ir + 3).
        suffix = f"{int(args.input_nc)}ch"
        ext = "jpg" if str(args.image_format).lower() in ("jpeg", "jpg") else "png"
        out_path = os.path.join(output_dir, f"{folder_name}_{suffix}.{ext}")
        out_path = _save_wsi_image(
            pred,
            out_path,
            image_format=str(args.image_format),
            jpeg_quality=int(args.jpeg_quality),
            png_compress_level=int(args.png_compress_level),
        )
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

