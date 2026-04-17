import argparse
import os
import shutil
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from Networks.vision_transformer import ProjectedSwinUnet
from utils.dataset_vslite_i2i import SplitFractions, VSLiteI2IDataset


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
    Image.fromarray(_tensor_to_png_array(x, prefer_last_three=(input_mode == "irvis13"))).save(
        os.path.join(out_dir, f"{name}_x.png")
    )
    Image.fromarray(_tensor_to_png_array(y)).save(os.path.join(out_dir, f"{name}_y.png"))
    Image.fromarray(_tensor_to_png_array(yhat)).save(os.path.join(out_dir, f"{name}_yhat.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config_i2i_baseline.yaml")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--num_samples", type=int, default=20)
    ap.add_argument("--split", type=str, default="test")
    args = ap.parse_args()

    cfg, cfg_dict = load_config(args.config)
    cfg.dataroot = _get_env("SWINGAN_DATAROOT", getattr(cfg, "dataroot", ""))
    cfg.input_mode = _get_env("SWINGAN_INPUT_MODE", getattr(cfg, "input_mode", "ir10"))
    cfg.gpu_id = _get_env("SWINGAN_GPU_ID", getattr(cfg, "gpu_id", "0"))

    if not cfg.dataroot:
        raise ValueError("Config 'dataroot' is empty. Set SWINGAN_DATAROOT or edit config.")

    out_dir = _get_env("SWINGAN_TEST_OUT", os.path.join(os.path.dirname(args.ckpt), "test_outputs"))
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
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    in_chans = 10 if cfg.input_mode == "ir10" else 13
    use_cbam = bool(getattr(cfg, "use_cbam", False))
    G = ProjectedSwinUnet(cfg, in_chans=in_chans, out_chans=3, use_cbam=use_cbam).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    G.load_state_dict(ckpt["G_state"], strict=True)
    G.eval()

    psnrs = []
    ssims = []
    saved = 0
    for batch in tqdm(loader, desc="Testing", unit="img"):
        x = batch["x"].to(device=device, dtype=torch.float32)
        y = batch["y"].to(device=device, dtype=torch.float32)
        with torch.no_grad():
            yhat = torch.tanh(G(x))
        psnrs.append(psnr(y, yhat))
        ssims.append(ssim_stub(y, yhat))
        if saved < args.num_samples:
            save_grid(out_dir, batch["name"][0], x[0], y[0], yhat[0])
            save_pngs(out_dir, batch["name"][0], x[0], y[0], yhat[0], cfg.input_mode)
            saved += 1

    summary = {
        "psnr_mean": float(np.mean(psnrs)) if psnrs else None,
        "psnr_std": float(np.std(psnrs)) if psnrs else None,
        "ssim_mean": float(np.mean(ssims)) if ssims else None,
        "ssim_std": float(np.std(ssims)) if ssims else None,
        "n": len(psnrs),
    }
    with open(os.path.join(out_dir, "metrics.yaml"), "w") as f:
        yaml.safe_dump(summary, f)
    print(summary)


if __name__ == "__main__":
    main()

