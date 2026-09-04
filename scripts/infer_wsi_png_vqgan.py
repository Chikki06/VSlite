"""Whole-slide PNG inference using a trained VQGAN H&E autoencoder.

VQGAN is an H&E -> H&E autoencoder. WSI inference therefore:

  1. Resolves IR and VIS sources from a sample folder using the same
     `VSLiteFolderLoader` used by every other WSI inference script (Baseline,
     SwinGAN, Restormer, NAFNet), so the input contract is identical (point
     at `input/IR/<sample>` and the loader finds the paired VIS at
     `input/VIS/<sample>_VIS_REG.tif` or its sibling variants). IR is
     intentionally ignored.
  2. Reads the VIS slide tile by tile via `SwinGAN.infer_wsi_png.read_vis_tile_to_device`
     (lazy `PIL.Image.open` + per-tile `.crop` for TIFFs, or `spectral.open_memmap`
     windowing for ENVI VIS). The full WSI never lives in RAM or GPU memory.
  3. Runs each tile through the VQGAN encoder/codebook/decoder and stitches
     the decoded tiles into a single uint8 RGB PNG.

Output layout (aligned with `scripts/infer_wsi_png_baseline.py`):

  <output_dir>/<sample>_vqgan.png

When `--output_dir` is omitted, defaults to:

  <results_dir>/test/<name>/wsi_pred/
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

_THIS_DIR = Path(__file__).resolve().parent
_VSLITE_ROOT = _THIS_DIR.parent
if str(_VSLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_VSLITE_ROOT))
_SWINGAN_DIR = _VSLITE_ROOT / "SwinGAN"
if str(_SWINGAN_DIR) not in sys.path:
    sys.path.insert(0, str(_SWINGAN_DIR))

from SwinGAN.infer_wsi_png import (  # noqa: E402
    FolderData,
    VSLiteFolderLoader,
    read_vis_tile_to_device,
)
from diffusion_vs.vqgan_vs import import_third_party_vqgan  # noqa: E402
from cislnet.wsi_local_padding import (  # noqa: E402
    LocalPaddingConfig,
    WSILocalPaddingState,
    add_local_padding_cli_args,
    resolve_local_padding_config,
)

Image.MAX_IMAGE_PIXELS = None


def _build_vqgan(
    *,
    ckpt_path: Path,
    image_size: int,
    latent_grid_size: int,
    latent_dim: int,
    num_codebook_vectors: int,
    device: torch.device,
):
    VQGAN, _Discriminator = import_third_party_vqgan(_VSLITE_ROOT)

    class _VQArgs:
        pass

    vq_args = _VQArgs()
    vq_args.latent_dim = int(latent_dim)
    vq_args.image_size = int(image_size)
    vq_args.latent_grid_size = int(latent_grid_size)
    vq_args.num_codebook_vectors = int(num_codebook_vectors)
    vq_args.beta = 0.25
    vq_args.image_channels = 3
    vq_args.device = str(device)

    vqgan = VQGAN(vq_args).to(device=device)  # type: ignore[call-arg]

    raw = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(raw, dict) and "vqgan" in raw:
        state = raw["vqgan"]
    elif isinstance(raw, dict) and "model_state" in raw:
        state = raw["model_state"]
    else:
        state = raw
    vqgan.load_state_dict(state, strict=True)
    vqgan.eval()
    return vqgan


def infer_folder(
    vqgan,
    data: FolderData,
    *,
    tile_size: int,
    device: torch.device,
    local_padding: Optional[LocalPaddingConfig] = None,
) -> np.ndarray:
    if data.vis is None or data.vis_shape_hwc is None:
        raise RuntimeError("VQGAN WSI inference requires VIS data.")

    vis_h, vis_w, _ = data.vis_shape_hwc
    r, c = int(vis_h), int(vis_w)

    rn = int(math.ceil(r / tile_size))
    cn = int(math.ceil(c / tile_size))

    out = np.zeros((rn * tile_size + 1, cn * tile_size + 1, 3), dtype=np.uint8)
    I = list(range(0, rn * tile_size + 1, tile_size))
    J = list(range(0, cn * tile_size + 1, tile_size))

    lp_cfg = local_padding or LocalPaddingConfig(enabled=False)
    lp_state = WSILocalPaddingState(lp_cfg) if lp_cfg.enabled else None
    n_ii = len(I) - 1
    n_jj = len(J) - 1
    if lp_cfg.enabled:
        print(
            f"[WSI] VQGAN local_padding: padding_size={lp_cfg.padding_size} "
            f"outer={lp_cfg.outer_padding}",
            flush=True,
        )

    with torch.no_grad():
        for ii in range(n_ii):
            if lp_state is not None:
                lp_state.on_new_row()
            for jj in range(n_jj):
                vis_patch = read_vis_tile_to_device(
                    data,
                    y0_tgt=I[ii],
                    y1_tgt=I[ii + 1],
                    x0_tgt=J[jj],
                    x1_tgt=J[jj + 1],
                    tile_size=tile_size,
                    device=device,
                )
                if lp_state is not None:
                    vis_patch = lp_state.apply_to_input(vis_patch, ii=ii, jj=jj, vis_offset=0)
                # VQGAN expects H&E targets in [-1, 1] (matches training).
                x_he = vis_patch.clamp(0.0, 1.0) * 2.0 - 1.0

                decoded, _z, _q = vqgan(x_he)
                y_np = (decoded[0].clamp(-1.0, 1.0).cpu().numpy() * 0.5) + 0.5
                y_u8 = np.clip(y_np * 255.0, 0, 255).astype(np.uint8)
                out[I[ii] : I[ii + 1], J[jj] : J[jj + 1], :] = np.transpose(y_u8, (1, 2, 0))
                if lp_state is not None:
                    lp_state.update_after_tile(
                        x_input=vis_patch,
                        y_output=decoded[0],
                        vis_offset=0,
                        is_last_col=(jj == n_jj - 1),
                        is_last_row=(ii == n_ii - 1),
                    )

                del vis_patch, x_he, decoded, y_np, y_u8
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            gc.collect()

    out = out[:r, :c, :]
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Whole-slide PNG inference using VQGAN H&E autoencoder checkpoints")
    p.add_argument("--name", required=True, type=str, help="VQGAN run name under <results_dir>/<name>/")
    p.add_argument(
        "--results_dir",
        type=str,
        default=str(_VSLITE_ROOT / "results" / "vqgan_vs"),
        help="Root results dir for VQGAN runs (contains <name>/checkpoints/...).",
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default="ckpt_latest.pt",
        help="Checkpoint filename under <results_dir>/<name>/checkpoints/ (or absolute path).",
    )
    p.add_argument("--image_size", type=int, default=512, help="Patch size VQGAN was constructed at.")
    p.add_argument("--latent_grid_size", type=int, default=16)
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--num_codebook_vectors", type=int, default=4096)
    p.add_argument("--folder", type=str, default=None, help="Optional single folder path to process")
    p.add_argument(
        "--parent_folder",
        type=str,
        default=None,
        help="Parent folder containing sample folders (e.g. input/IR), same contract as other WSI scripts.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for PNG files (default: <results_dir>/test/<name>/wsi_pred).",
    )
    p.add_argument(
        "--tile_size",
        type=int,
        default=None,
        help="Tile size for VQGAN inference (defaults to --image_size).",
    )
    p.add_argument("--gpu", type=int, default=0)
    add_local_padding_cli_args(p)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    lp_cfg = resolve_local_padding_config(
        local_padding=bool(getattr(args, "local_padding", False)),
        local_padding_size=int(getattr(args, "local_padding_size", 1)),
        local_padding_outer=str(getattr(args, "local_padding_outer", "replicate")),
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ckpt_arg = Path(str(args.ckpt))
    if ckpt_arg.is_absolute() or ckpt_arg.exists():
        ckpt_path = ckpt_arg.resolve()
    else:
        ckpt_path = Path(args.results_dir).resolve() / args.name / "checkpoints" / args.ckpt
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"VQGAN checkpoint not found: {ckpt_path}")

    tile_size = int(args.tile_size if args.tile_size is not None else args.image_size)

    vqgan = _build_vqgan(
        ckpt_path=ckpt_path,
        image_size=int(args.image_size),
        latent_grid_size=int(args.latent_grid_size),
        latent_dim=int(args.latent_dim),
        num_codebook_vectors=int(args.num_codebook_vectors),
        device=device,
    )

    if args.output_dir is None:
        output_dir = os.path.join(
            os.path.abspath(args.results_dir), "test", str(args.name), "wsi_pred"
        )
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
            d
            for d in sorted(os.listdir(parent))
            if os.path.isdir(os.path.join(parent, d)) and not d.endswith("_VIS")
        ]
    else:
        raise ValueError("Provide --folder or --parent_folder")

    for folder_name in folder_names:
        print(f"Processing {folder_name} ...", flush=True)
        loader = VSLiteFolderLoader(parent, folder_name, model_type=1)
        data = loader.load()
        pred = infer_folder(
            vqgan,
            data,
            tile_size=tile_size,
            device=device,
            local_padding=lp_cfg,
        )

        out_path = os.path.join(output_dir, f"{folder_name}_vqgan.png")
        Image.fromarray(pred).save(out_path)
        print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
