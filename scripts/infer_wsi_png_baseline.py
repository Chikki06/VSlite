import argparse
import math
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

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

Image.MAX_IMAGE_PIXELS = None


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
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for PNG files (default: results_dir/test/<name>/wsi_pred)",
    )
    p.add_argument("--tile_size", type=int, default=512, help="Tile size for generator inference (pixels)")
    return p.parse_args()


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
        if n <= 0 or n > c_tot:
            raise ValueError(f"Invalid --num_ir_channels={n} for IR channels={c_tot}")
        return list(range(n)), n

def _read_envi_patch_hwc(mm: np.ndarray, *, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    # mm is (H, W, C) memmap; slicing stays lazy / windowed.
    y0 = max(0, int(y0))
    x0 = max(0, int(x0))
    y1 = max(y0, int(y1))
    x1 = max(x0, int(x1))
    return np.asarray(mm[y0:y1, x0:x1, :], dtype=np.float32)


def _read_vis_patch_to_chw(
    vis_src: object,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    expect_channels: int = 3,
) -> torch.Tensor:
    # Returns float32 CHW in [0,1].
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
        arr = _read_envi_patch_hwc(vis_src, y0=y0, y1=y1, x0=x0, x1=x1).astype(np.float32) / 256.0

    if arr.ndim != 3:
        raise RuntimeError(f"VIS patch has invalid shape: {arr.shape}")
    if arr.shape[2] != expect_channels:
        # If VIS has extra bands, keep first 3; if fewer, pad.
        if arr.shape[2] > expect_channels:
            arr = arr[:, :, :expect_channels]
        else:
            pad = np.zeros((arr.shape[0], arr.shape[1], expect_channels - arr.shape[2]), dtype=arr.dtype)
            arr = np.concatenate([arr, pad], axis=2)
    chw = torch.from_numpy(np.transpose(arr, (2, 0, 1))).contiguous()
    return chw.clamp(0.0, 1.0)


def infer_folder(
    model,
    data: FolderData,
    *,
    input_nc: int,
    num_ir_channels: int,
    ir_channel_indices: str,
    tile_size: int,
    device: torch.device,
) -> np.ndarray:
    # Keep the same geometric relation from SwinGAN/Restormer/NAFNet WSI scripts.
    scaling_factor = 0.69 / 2.0

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
            raise RuntimeError("13ch mode requires VIS data + pixel size metadata.")
        assert data.vis_shape_hwc is not None
        vis_h, vis_w, vis_c = data.vis_shape_hwc
        if int(vis_c) < 3:
            raise RuntimeError(f"VIS must have >=3 channels; got {data.vis_shape_hwc}")

        # IMPORTANT: VIS source is already at target pixel size in most pipelines;
        # we avoid full-image resampling here to prevent huge memory spikes.
        r, c = int(vis_h), int(vis_w)
        vis = data.vis
    else:
        ratio = 2.0 / 0.69
        r = int(ir_h * ratio)
        c = int(ir_w * ratio)
        vis = None

    rn = int(math.ceil(r / tile_size))
    cn = int(math.ceil(c / tile_size))
    # Store as uint8 directly to reduce RAM by 4x vs float32.
    out = np.zeros((rn * tile_size + 1, cn * tile_size + 1, 3), dtype=np.uint8)

    I = list(range(0, rn * tile_size + 1, tile_size))
    J = list(range(0, cn * tile_size + 1, tile_size))

    # Baseline model expects NCHW, and returns fake image in [-1,1].
    y_dummy = torch.zeros((1, 3, tile_size, tile_size), device=device, dtype=torch.float32)
    with torch.no_grad():
        for ii in range(len(I) - 1):
            for jj in range(len(J) - 1):
                # IR patch (lower-resolution source), then upsample to model patch size.
                y0 = int(I[ii] * scaling_factor)
                y1 = int(I[ii + 1] * scaling_factor)
                x0 = int(J[jj] * scaling_factor)
                x1 = int(J[jj + 1] * scaling_factor)
                ir_patch_hwc = _read_envi_patch_hwc(data.ir, y0=y0, y1=y1, x0=x0, x1=x1)  # (h, w, c)
                ir_patch_hwc = ir_patch_hwc[:, :, ir_idx]  # select channels
                # to CHW torch on device
                ir_patch = torch.from_numpy(np.transpose(ir_patch_hwc, (2, 0, 1))).unsqueeze(0).to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                ir_patch = ir_patch.clamp(0.0, 5.0)
                target_lr = int(tile_size * scaling_factor)
                pad_rows = max(0, target_lr - ir_patch.shape[1])
                pad_cols = max(0, target_lr - ir_patch.shape[2])
                ir_patch = F.pad(ir_patch, (0, pad_cols, 0, pad_rows), mode="constant", value=0.0)
                ir_patch = F.interpolate(ir_patch, size=(tile_size, tile_size), mode="bicubic", align_corners=True)

                if mode == "ir_vis":
                    assert vis is not None
                    vis_chw = _read_vis_patch_to_chw(
                        vis,
                        y0=int(I[ii]),
                        y1=int(I[ii + 1]),
                        x0=int(J[jj]),
                        x1=int(J[jj + 1]),
                        expect_channels=3,
                    )
                    v_pad_rows = max(0, tile_size - vis_chw.shape[1])
                    v_pad_cols = max(0, tile_size - vis_chw.shape[2])
                    vis_patch = F.pad(
                        vis_chw.unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True),
                        (0, v_pad_cols, 0, v_pad_rows),
                        mode="constant",
                        value=0.0,
                    )
                    x_patch = torch.cat([ir_patch, vis_patch], dim=1)
                else:
                    x_patch = ir_patch

                fake = model({"input": x_patch, "target": y_dummy}, mode="inference")
                y_np = fake[0].clamp(-1.0, 1.0).detach().cpu().numpy()
                y_np = np.transpose(y_np, (1, 2, 0))  # HWC
                y_u8 = np.clip((y_np * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
                out[I[ii]: I[ii + 1], J[jj]: J[jj + 1], :] = y_u8

                # Keep host memory stable during long WSI runs.
                del ir_patch_hwc, ir_patch, x_patch, fake, y_np, y_u8
                if mode == "ir_vis":
                    del vis_chw, vis_patch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            gc.collect()

    out = out[:r, :c, :]
    return out


def main() -> None:
    args = _parse_args()

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
    opt.batch_size = 1

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
        folder_names = [
            d for d in sorted(os.listdir(parent)) if os.path.isdir(os.path.join(parent, d)) and not d.endswith("_VIS")
        ]
    else:
        raise ValueError("Provide --folder or --parent_folder")

    device = torch.device("cuda" if (torch.cuda.is_available() and str(args.gpu_ids) != "-1") else "cpu")

    for folder_name in folder_names:
        print(f"Processing {folder_name} ...")
        # model_type: 0=IR-only, 1=IR+VIS (13ch). Infer from input_nc.
        model_type = 1 if int(args.input_nc) > int(args.num_ir_channels) else 0
        loader = VSLiteFolderLoader(parent, folder_name, model_type=model_type)
        data = loader.load()
        pred = infer_folder(
            model,
            data,
            input_nc=int(args.input_nc),
            num_ir_channels=int(args.num_ir_channels),
            ir_channel_indices=str(args.ir_channel_indices or ""),
            tile_size=int(args.tile_size),
            device=device,
        )

        suffix = "13ch" if int(args.input_nc) > int(args.num_ir_channels) else f"{int(args.input_nc)}ch"
        out_path = os.path.join(output_dir, f"{folder_name}_{suffix}.png")
        Image.fromarray(pred).save(out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

