"""
Patch inference + metrics for I2SB-VS (virtual staining).

Loads `results/i2sb_vs/<name>/latest.pt` (I2SB checkpoint format) and runs DDPM sampling
from x1 (projected VS conditioning) toward x0 (H&E).

Outputs:
  <results_dir>/test/<run_name>/patch_eval/metrics.{json,yaml}
in the same layout as VSlite baseline models.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_VSLITE_ROOT = _THIS_DIR.parent
if str(_VSLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_VSLITE_ROOT))

from diffusion_vs.data import VSliteVSDiffusionDataset
from diffusion_vs.i2sb_vs import CondToRGB
from util.i2i_metrics import compute_metrics_streaming


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, required=True, help="trained run name (folder under results_dir)")
    ap.add_argument("--dataroot", type=str, required=True)
    ap.add_argument("--results_dir", type=str, default="/projects/illinois/eng/bioe/rxb/akshat/VSlite/results/i2sb_vs")
    ap.add_argument("--which_epoch", type=str, default="latest", choices=["latest"], help="I2SB wrapper only saves latest.pt")
    ap.add_argument("--input_nc", type=int, required=True, choices=[10, 13])
    ap.add_argument("--num_ir_channels", type=int, default=10)
    ap.add_argument("--num_test", type=int, default=75000, help="0 = entire split")
    ap.add_argument("--split", type=str, default="test", choices=["train", "test"])
    ap.add_argument("--fid_max_items", type=int, default=None)
    ap.add_argument("--fid_batch_size", type=int, default=16)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--nfe", type=int, default=200, help="sampling steps (function evals)")
    args = ap.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))

    i2sb_root = _VSLITE_ROOT / "third_party" / "I2SB"
    if str(i2sb_root) not in sys.path:
        sys.path.insert(0, str(i2sb_root))

    from logger import Logger  # type: ignore
    from i2sb import Runner  # type: ignore

    ckpt_path = Path(args.results_dir).resolve() / args.name
    ckpt_file = ckpt_path / "latest.pt"
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"I2SB checkpoint not found: {ckpt_file}")

    # Build minimal opt compatible with Runner + ddpm_sampling.
    class _Opt:
        pass

    opt = _Opt()
    opt.device = "cuda" if torch.cuda.is_available() else "cpu"
    opt.distributed = False
    opt.global_rank = 0
    opt.global_size = 1
    opt.interval = 1000
    opt.ot_ode = False
    opt.clip_denoise = False
    opt.cond_x1 = True
    opt.use_fp16 = False

    log = Logger(0, ckpt_path / "log")  # type: ignore
    # Runner will read saved options.pkl if we pass save_opt=False and build from our opt,
    # but we want the original betas/noise_levels. Load options.pkl if present.
    options_pkl = ckpt_path / "options.pkl"
    if options_pkl.is_file():
        import pickle

        with open(options_pkl, "rb") as f:
            saved = pickle.load(f)
        for k, v in vars(saved).items():
            setattr(opt, k, v)
        opt.device = "cuda" if torch.cuda.is_available() else "cpu"
        opt.distributed = False
        opt.global_rank = 0
        opt.global_size = 1
        opt.load = ckpt_file
    else:
        opt.beta_max = 0.3
        opt.t0 = 1e-4
        opt.T = 1.0
        opt.load = ckpt_file

    runner = Runner(opt, log, save_opt=False)

    ds = VSliteVSDiffusionDataset(
        dataroot=args.dataroot,
        split=str(args.split),
        input_nc=int(args.input_nc),
        num_ir_channels=int(args.num_ir_channels),
    )

    # Output layout: results_dir/test/<run>/patch_eval
    run_name = args.name
    patch_eval_leaf = "patch_eval" if args.split == "test" else "patch_eval_train"
    out_dir = os.path.join(str(Path(args.results_dir).resolve()), "test", run_name, patch_eval_leaf)
    os.makedirs(out_dir, exist_ok=True)

    cond_to_rgb = CondToRGB(int(args.input_nc)).to(opt.device)

    def _pairs():
        n_seen = 0
        for i in range(len(ds)):
            if args.num_test > 0 and n_seen >= int(args.num_test):
                break
            b = ds[i]
            x0 = b.x_he.unsqueeze(0).to(opt.device)
            x1 = cond_to_rgb(b.x_cond.unsqueeze(0).to(opt.device))
            # Sample from x1 to bridge toward x0 distribution.
            xs, _pred_x0 = runner.ddpm_sampling(
                opt,
                x1,
                mask=None,
                cond=x1,  # conditional ADM expects cond concatenated; we use x1 as condition.
                clip_denoise=False,
                nfe=int(args.nfe),
                verbose=False,
            )
            # xs[:,0] is final recon at log_steps[0] (step 0), by Runner's convention.
            yhat = xs[:, 0, ...][0].detach().cpu()
            y = x0[0].detach().cpu()
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
        "split": str(args.split),
        "name": str(args.name),
        "nfe": int(args.nfe),
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
            ]
        )
    )


if __name__ == "__main__":
    main()

