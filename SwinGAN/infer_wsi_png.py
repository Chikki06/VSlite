import argparse
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

from Networks.vision_transformer import ProjectedSwinUnet

Image.MAX_IMAGE_PIXELS = None


@dataclass
class FolderData:
    ir: torch.Tensor
    vis: Optional[torch.Tensor]
    vis_pixel_size: Optional[float]


class VSLiteFolderLoader:
    """Mirror of tests/main.py folder resolution for IR/VIS inputs."""

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

    def _load_ir_from_dir(self, ir_folder_path: str) -> torch.Tensor:
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
        img = np.array(image.load())
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img[img > 5.0] = 5.0
        img[img < 0.0] = 0.0
        return torch.from_numpy(img).float()

    def _load_vis_from_dir(self, vis_folder_path: str) -> Tuple[torch.Tensor, float]:
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
        vis_data = np.array(vis_image.load()).astype(np.float32) / 256.0
        vis_data = np.transpose(vis_data, (2, 0, 1))
        vis = torch.from_numpy(vis_data).float()

        pixel_size = vis_image.metadata.get("pixel size")
        if pixel_size:
            if isinstance(pixel_size, (list, tuple)):
                vis_pixel_size = float(pixel_size[0]) * 1e6
            else:
                vis_pixel_size = float(pixel_size) * 1e6
        else:
            vis_pixel_size = 2.0
        return vis, vis_pixel_size

    def _get_vis_tiff_path(self) -> str:
        vis_dir = self.parent.replace("/IR", "/VIS")
        return os.path.join(vis_dir, f"{self.folder_name}_VIS_REG.tif")

    def _load_vis_tiff(self, tiff_path: str) -> Tuple[torch.Tensor, float]:
        if not os.path.isfile(tiff_path):
            raise FileNotFoundError(f"VIS TIFF not found: {tiff_path}")
        with Image.open(tiff_path) as img:
            arr = np.array(img).astype(np.float32) / 256.0
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        elif arr.ndim == 3:
            arr = np.transpose(arr, (2, 0, 1))
        vis = torch.from_numpy(arr).float()
        return vis, 0.69

    def load(self) -> FolderData:
        folder_path = os.path.join(self.parent, self.folder_name)

        if self.folder_category in ("direct", "processed_siblings", "processed_siblings_tiff"):
            ir_folder = os.path.join(self.parent, self.folder_name)
        else:
            ir_folder = os.path.join(folder_path, "IR")

        ir = self._load_ir_from_dir(ir_folder)

        if self.model_type == 0:
            return FolderData(ir=ir, vis=None, vis_pixel_size=None)

        if self.folder_category == "direct":
            vis_folder = os.path.join(self.parent, self.folder_name + "_VIS")
            vis, px = self._load_vis_from_dir(vis_folder)
        elif self.folder_category == "ir_vis_subfolders":
            vis_folder = os.path.join(folder_path, "VIS")
            vis, px = self._load_vis_from_dir(vis_folder)
        elif self.folder_category == "processed_siblings":
            vis_folder = os.path.join(os.path.dirname(self.parent), "VIS", self.folder_name)
            vis, px = self._load_vis_from_dir(vis_folder)
        elif self.folder_category in ("processed_siblings_tiff", "ir_subfolder_tiff"):
            vis, px = self._load_vis_tiff(self._get_vis_tiff_path())
        else:
            raise RuntimeError(f"Unsupported folder category for 13ch mode: {self.folder_category}")

        return FolderData(ir=ir, vis=vis, vis_pixel_size=px)


def load_config(path: str) -> SimpleNamespace:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return SimpleNamespace(**cfg)


def build_model(config_path: str, ckpt_path: str, model_type: int, device: torch.device) -> ProjectedSwinUnet:
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


def _prep_vis(vis: torch.Tensor, vis_pixel_size: float) -> torch.Tensor:
    target_pixel_size = 0.69
    scale = float(vis_pixel_size) / target_pixel_size
    _, h, w = vis.shape
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    vis_resized = F.interpolate(
        vis.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
    ).squeeze(0)
    return vis_resized.clamp(0.0, 1.0)


def infer_folder(
    G: ProjectedSwinUnet,
    data: FolderData,
    model_type: int,
    tile_size: int,
    device: torch.device,
) -> np.ndarray:
    # Keep the same geometric relation from tests/main.py
    scaling_factor = 0.69 / 2.0

    ir = data.ir.to(device=device, dtype=torch.float32)
    if model_type == 1:
        if data.vis is None or data.vis_pixel_size is None:
            raise RuntimeError("13ch mode requires VIS data")
        vis = _prep_vis(data.vis.to(device=device, dtype=torch.float32), data.vis_pixel_size)
        _, r, c = vis.shape
    else:
        _, h_ir, w_ir = ir.shape
        ratio = 2.0 / 0.69
        r = int(h_ir * ratio)
        c = int(w_ir * ratio)
        vis = None

    rn = int(math.ceil(r / tile_size))
    cn = int(math.ceil(c / tile_size))

    out = np.zeros((3, rn * tile_size + 1, cn * tile_size + 1), dtype=np.float32)

    I = list(range(0, rn * tile_size + 1, tile_size))
    J = list(range(0, cn * tile_size + 1, tile_size))

    with torch.no_grad():
        for ii in range(len(I) - 1):
            for jj in range(len(J) - 1):
                # IR patch (lower-resolution source), then upsample to model patch size.
                ir_patch = ir[
                    :,
                    int(I[ii] * scaling_factor): int(I[ii + 1] * scaling_factor),
                    int(J[jj] * scaling_factor): int(J[jj + 1] * scaling_factor),
                ]
                target_lr = int(tile_size * scaling_factor)
                pad_rows = max(0, target_lr - ir_patch.shape[1])
                pad_cols = max(0, target_lr - ir_patch.shape[2])
                ir_patch = F.pad(ir_patch.unsqueeze(0), (0, pad_cols, 0, pad_rows), mode="constant", value=0.0)
                ir_patch = F.interpolate(ir_patch, size=(tile_size, tile_size), mode="bicubic", align_corners=True)

                if model_type == 1:
                    vis_patch = vis[
                        :,
                        int(I[ii]): int(I[ii + 1]),
                        int(J[jj]): int(J[jj + 1]),
                    ]
                    v_pad_rows = max(0, tile_size - vis_patch.shape[1])
                    v_pad_cols = max(0, tile_size - vis_patch.shape[2])
                    vis_patch = F.pad(
                        vis_patch.unsqueeze(0), (0, v_pad_cols, 0, v_pad_rows), mode="constant", value=0.0
                    )
                    x = torch.cat([ir_patch, vis_patch], dim=1)
                else:
                    x = ir_patch

                y_hat = torch.tanh(G(x))
                y_np = (y_hat[0].clamp(-1.0, 1.0).cpu().numpy() * 0.5) + 0.5
                out[:, I[ii]: I[ii + 1], J[jj]: J[jj + 1]] = y_np

    out = out[:, :r, :c]
    out = np.transpose(out, (1, 2, 0))
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
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
        print(f"Processing {folder_name} ...")
        loader = VSLiteFolderLoader(parent, folder_name, args.model)
        data = loader.load()
        pred = infer_folder(G, data, args.model, tile_size=tile_size, device=device)

        suffix = "13ch" if int(args.model) == 1 else "10ch"
        out_path = os.path.join(output_dir, f"{folder_name}_{suffix}.png")
        Image.fromarray(pred).save(out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
