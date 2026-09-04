"""
Patch inference + metrics for the VQGAN H&E autoencoder.

Writes metrics under:
  results/vqgan/test/<run_name>/patch_eval_test/metrics.{json,yaml}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_THIS_DIR = Path(__file__).resolve().parent
_VSLITE_ROOT = _THIS_DIR.parent
if str(_VSLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_VSLITE_ROOT))

from diffusion_vs.vqgan_vs import VQGANHEDataset, collate_vqgan_batch, import_third_party_vqgan
from util.i2i_metrics import compute_metrics_streaming


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, required=True, help="run name under results_dir (vqgan_vs/<name>/checkpoints/...)")
    ap.add_argument("--dataroot", type=str, required=True)
    ap.add_argument("--results_dir", type=str, default="/projects/illinois/eng/bioe/rxb/akshat/VSlite/results/vqgan")
    ap.add_argument("--ckpt", type=str, required=True, help="checkpoint filename under <run>/checkpoints/")
    ap.add_argument("--input_nc", type=int, default=13, choices=[10, 13])
    ap.add_argument("--num_ir_channels", type=int, default=10)
    ap.add_argument("--image_size", type=int, default=1024, help="Patch size for dataset resize during evaluation.")
    ap.add_argument("--latent_grid_size", type=int, default=16)
    ap.add_argument("--latent_dim", type=int, default=256)
    ap.add_argument("--num_codebook_vectors", type=int, default=4096)
    ap.add_argument("--split", type=str, default="test", choices=["train", "test"])
    ap.add_argument("--num_test", type=int, default=75000)
    ap.add_argument("--fid_max_items", type=int, default=None)
    ap.add_argument("--fid_batch_size", type=int, default=16)
    ap.add_argument(
        "--infer_batch_size",
        type=int,
        default=int(os.environ.get("VSLITE_INFER_BS", "16")),
        help="Mini-batch size for the VQGAN forward pass (DataLoader batch_size). Overridable via VSLITE_INFER_BS.",
    )
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    run_root = Path(args.results_dir).resolve() / args.name
    ckpt_path = run_root / "checkpoints" / args.ckpt
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"VQGAN checkpoint not found: {ckpt_path}")

    VQGAN, _Discriminator = import_third_party_vqgan(_VSLITE_ROOT)

    class _VQArgs:
        pass

    vq_args = _VQArgs()
    vq_args.latent_dim = int(args.latent_dim)
    vq_args.image_size = int(args.image_size)
    vq_args.latent_grid_size = int(args.latent_grid_size)
    vq_args.num_codebook_vectors = int(args.num_codebook_vectors)
    vq_args.beta = 0.25
    vq_args.image_channels = 3
    vq_args.device = str(device)

    vqgan = VQGAN(vq_args).to(device=device)  # type: ignore
    vqgan.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    vqgan.eval()

    ds = VQGANHEDataset(
        dataroot=args.dataroot,
        split=str(args.split),
        input_nc=int(args.input_nc),
        num_ir_channels=int(args.num_ir_channels),
        image_size=int(args.image_size),
        test_includes_val=(str(args.split) == "test"),
    )

    patch_eval_leaf = "patch_eval_test" if args.split == "test" else "patch_eval_train"
    out_dir = os.path.join(str(Path(args.results_dir).resolve()), "test", args.name, patch_eval_leaf)
    os.makedirs(out_dir, exist_ok=True)

    infer_bs = max(1, int(args.infer_batch_size))
    loader = DataLoader(
        ds,
        batch_size=infer_bs,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_vqgan_batch,
    )

    def _pairs():
        n_seen = 0
        for batch in loader:
            if args.num_test > 0 and n_seen >= int(args.num_test):
                break
            x = batch.x_he.to(device=device, dtype=torch.float32)
            with torch.no_grad():
                decoded, _z, _q = vqgan(x)
            B = int(x.size(0))
            for j in range(B):
                if args.num_test > 0 and n_seen >= int(args.num_test):
                    break
                y = x[j].detach().cpu()
                yhat = decoded[j].detach().cpu()
                n_seen += 1
                yield (y, yhat)

    res = compute_metrics_streaming(
        pairs=_pairs(),
        device="cuda" if torch.cuda.is_available() else "cpu",
        fid_max_items=args.fid_max_items,
        fid_batch_size=int(args.fid_batch_size),
    )
    metrics = {
        "n": res.n,
        "psnr_mean": res.psnr_mean,
        "psnr_std": res.psnr_std,
        "ssim_mean": res.ssim_mean,
        "ssim_std": res.ssim_std,
        "fid": res.fid,
        "lpips_convnext_tcga_mean": res.lpips_convnext_tcga_mean,
        "lpips_convnext_tcga_std": res.lpips_convnext_tcga_std,
        "split": str(args.split),
        "name": str(args.name),
        "ckpt": str(args.ckpt),
        "out_dir": out_dir,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    try:
        import yaml

        with open(os.path.join(out_dir, "metrics.yaml"), "w") as f:
            yaml.safe_dump(metrics, f)
    except Exception:
        pass

    print(
        "MODE: EVAL | "
        + " ".join(
            [
                f"PSNR_test: {metrics['psnr_mean']:.3f}" if metrics["psnr_mean"] is not None else "PSNR_test: nan",
                f"SSIM_test: {metrics['ssim_mean']:.3f}" if metrics["ssim_mean"] is not None else "SSIM_test: nan",
                f"FID_test: {metrics['fid']:.3f}" if metrics["fid"] is not None else "FID_test: nan",
                (
                    f"LPIPS_convnext_tcga_test: {metrics['lpips_convnext_tcga_mean']:.3f}"
                    if metrics["lpips_convnext_tcga_mean"] is not None
                    else "LPIPS_convnext_tcga_test: nan"
                ),
            ]
        )
    )


if __name__ == "__main__":
    main()

