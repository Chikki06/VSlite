import argparse
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_VSLITE_ROOT = _THIS_DIR.parent
if str(_VSLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_VSLITE_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from Networks.vision_transformer import ProjectedSwinUnet
from utils.dataset_vslite_i2i import SplitFractions, VSLiteI2IDataset
from SwinGAN.utils.sample_saver import save_palette5_pred_target_png
from util.i2i_metrics import compute_metrics_streaming


def _get_env(name: str, default=None):
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def load_config(path: str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return SimpleNamespace(**cfg), cfg


def build_split_fracs(cfg):
    fr = getattr(cfg, "split_fracs", [0.9, 0.05, 0.05])
    return SplitFractions(train=float(fr[0]), val=float(fr[1]), test=float(fr[2]))


def psnr(gt: torch.Tensor, pred: torch.Tensor) -> float:
    # gt/pred in [-1,1]
    gt01 = (gt + 1.0) / 2.0
    pr01 = (pred + 1.0) / 2.0
    mse = torch.mean((gt01 - pr01) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def ssim_stub(gt: torch.Tensor, pred: torch.Tensor) -> float:
    # Cheap approximation placeholder; real SSIM needs skimage (add later if desired).
    # Returns 1 - normalized L1 over [0,1].
    gt01 = (gt + 1.0) / 2.0
    pr01 = (pred + 1.0) / 2.0
    l1 = torch.mean(torch.abs(gt01 - pr01)).item()
    return float(max(0.0, 1.0 - l1))


def save_grid(out_dir: str, name: str, x: torch.Tensor, y: torch.Tensor, yhat: torch.Tensor):
    os.makedirs(out_dir, exist_ok=True)
    # Save numpy arrays for quick inspection
    np.save(os.path.join(out_dir, f"{name}_x.npy"), x.cpu().numpy())
    np.save(os.path.join(out_dir, f"{name}_y.npy"), y.cpu().numpy())
    np.save(os.path.join(out_dir, f"{name}_yhat.npy"), yhat.cpu().numpy())


def _tensor_to_png_array(tensor: torch.Tensor, prefer_last_three: bool = False) -> np.ndarray:
    tensor = tensor.detach().float().cpu()
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW or BCHW tensor, got shape {tuple(tensor.shape)}")
    if tensor.shape[0] > 3:
        tensor = tensor[-3:] if prefer_last_three else tensor[:3]
    if tensor.min().item() < 0.0:
        tensor = (tensor + 1.0) / 2.0
    tensor = tensor.clamp(0.0, 1.0)
    array = (tensor * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    if array.shape[2] == 1:
        array = array[:, :, 0]
    return array


def save_pngs(out_dir: str, name: str, x: torch.Tensor, y: torch.Tensor, yhat: torch.Tensor, input_mode: str):
    os.makedirs(out_dir, exist_ok=True)
    x_png = _tensor_to_png_array(x, prefer_last_three=(input_mode == "irvis13"))
    y_png = _tensor_to_png_array(y)
    yhat_png = _tensor_to_png_array(yhat)
    # Legacy filenames
    Image.fromarray(x_png).save(os.path.join(out_dir, f"{name}_x.png"))
    Image.fromarray(y_png).save(os.path.join(out_dir, f"{name}_y.png"))
    Image.fromarray(yhat_png).save(os.path.join(out_dir, f"{name}_yhat.png"))
    # Standardized filenames (match Restormer i2i naming)
    Image.fromarray(x_png).save(os.path.join(out_dir, f"{name}_input.png"))
    Image.fromarray(y_png).save(os.path.join(out_dir, f"{name}_target.png"))
    Image.fromarray(yhat_png).save(os.path.join(out_dir, f"{name}_pred.png"))


def _save_standard_dirs(root: str, name: str, x: torch.Tensor, y: torch.Tensor, yhat: torch.Tensor, input_mode: str):
    in_dir = os.path.join(root, "input")
    tg_dir = os.path.join(root, "target")
    pr_dir = os.path.join(root, "pred")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(tg_dir, exist_ok=True)
    os.makedirs(pr_dir, exist_ok=True)
    Image.fromarray(_tensor_to_png_array(x, prefer_last_three=(input_mode == "irvis13"))).save(
        os.path.join(in_dir, f"{name}.png")
    )
    Image.fromarray(_tensor_to_png_array(y)).save(os.path.join(tg_dir, f"{name}.png"))
    Image.fromarray(_tensor_to_png_array(yhat)).save(os.path.join(pr_dir, f"{name}.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config_i2i_baseline.yaml")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--num_samples", type=int, default=20)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--fid_max_items", type=int, default=None)
    ap.add_argument("--fid_batch_size", type=int, default=16)
    ap.add_argument(
        "--infer_batch_size",
        type=int,
        default=int(os.environ.get("VSLITE_INFER_BS", "16")),
        help="Mini-batch size for the generator forward pass (DataLoader batch_size). Overridable via VSLITE_INFER_BS.",
    )
    ap.add_argument("--save_every", type=int, default=0, help="If >0, save standardized PNG triplets every N patches.")
    ap.add_argument("--save_palette_every", type=int, default=0, help="If >0, save a palette preview every N patches.")
    args = ap.parse_args()

    cfg, cfg_dict = load_config(args.config)
    cfg.dataroot = _get_env("SWINGAN_DATAROOT", getattr(cfg, "dataroot", ""))
    cfg.input_mode = _get_env("SWINGAN_INPUT_MODE", getattr(cfg, "input_mode", "ir10"))
    cfg.gpu_id = _get_env("SWINGAN_GPU_ID", getattr(cfg, "gpu_id", "0"))

    if not cfg.dataroot:
        raise ValueError("Config 'dataroot' is empty. Set SWINGAN_DATAROOT or edit config.")

    default_out_dir = os.path.join(os.path.dirname(args.ckpt), "test_outputs")
    out_dir = args.out_dir or _get_env("SWINGAN_TEST_OUT", default_out_dir)
    os.makedirs(out_dir, exist_ok=True)
    try:
        shutil.copyfile(args.config, os.path.join(out_dir, "config.yaml"))
    except Exception:
        pass

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_fracs = build_split_fracs(cfg)
    ds = VSLiteI2IDataset(
        cfg.dataroot,
        input_mode=cfg.input_mode,
        num_ir_channels=int(getattr(cfg, "num_ir_channels", 10)),
        img_size=int(getattr(cfg, "img_size", 256)),
        he_scale=str(getattr(cfg, "he_scale", "minus1_1")),
        split=args.split,
        split_seed=int(getattr(cfg, "split_seed", 1337)),
        split_fracs=split_fracs,
        max_items=getattr(cfg, "max_items", None),
        test_includes_val=(args.split == "test"),
    )
    infer_bs = max(1, int(args.infer_batch_size))
    loader = DataLoader(ds, batch_size=infer_bs, shuffle=False, num_workers=0)

    in_chans = 10 if cfg.input_mode == "ir10" else 13
    use_cbam = bool(getattr(cfg, "use_cbam", False))
    G = ProjectedSwinUnet(cfg, in_chans=in_chans, out_chans=3, use_cbam=use_cbam).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    G.load_state_dict(ckpt["G_state"], strict=True)
    G.eval()

    # Streaming inference + metrics (avoid saving every patch).
    def _pairs():
        saved = 0
        buf5: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        global_idx = 0
        for batch in tqdm(loader, desc="Testing", unit="batch"):
            x = batch["x"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device, dtype=torch.float32)
            with torch.no_grad():
                yhat = torch.tanh(G(x))

            names = batch["name"]
            B = int(yhat.size(0))
            for j in range(B):
                name = names[j] if isinstance(names, list) else names
                y_j = y[j].detach().cpu()
                yhat_j = yhat[j].detach().cpu()
                buf5.append((name, y_j, yhat_j))
                if len(buf5) > 5:
                    buf5.pop(0)

                if saved < args.num_samples and len(buf5) == 5:
                    save_palette5_pred_target_png(out_dir, f"{name}_{saved:03d}", list(buf5), filename_suffix="palette5")
                    saved += 5

                if args.save_palette_every and (global_idx % int(args.save_palette_every) == 0) and len(buf5) == 5:
                    save_palette5_pred_target_png(
                        out_dir, f"{name}_every{global_idx}", list(buf5), filename_suffix="palette5_every"
                    )

                if args.save_every and (global_idx % int(args.save_every) == 0) and len(buf5) == 5:
                    save_palette5_pred_target_png(
                        out_dir, f"{name}_save{global_idx}", list(buf5), filename_suffix="palette5_save"
                    )

                global_idx += 1
                yield (y_j, yhat_j)

    res = compute_metrics_streaming(
        pairs=_pairs(),
        device="cuda" if torch.cuda.is_available() else "cpu",
        fid_max_items=args.fid_max_items,
        fid_batch_size=int(args.fid_batch_size),
    )
    summary = {
        "n": res.n,
        "psnr_mean": res.psnr_mean,
        "psnr_std": res.psnr_std,
        "ssim_mean": res.ssim_mean,
        "ssim_std": res.ssim_std,
        "fid": res.fid,
    }
    with open(os.path.join(out_dir, "metrics.yaml"), "w") as f:
        yaml.safe_dump(summary, f)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        import json

        json.dump(summary, f, indent=2)
    print(
        "MODE: EVAL | "
        + " ".join(
            [
                f"PSNR_test: {summary['psnr_mean']:.3f}" if summary["psnr_mean"] is not None else "PSNR_test: nan",
                f"SSIM_test: {summary['ssim_mean']:.3f}" if summary["ssim_mean"] is not None else "SSIM_test: nan",
                f"FID_test: {summary['fid']:.3f}" if summary["fid"] is not None else "FID_test: nan",
            ]
        )
    )


if __name__ == "__main__":
    main()

