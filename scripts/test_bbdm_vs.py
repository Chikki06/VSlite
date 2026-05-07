"""
Patch inference + metrics for BBDM/LBBDM-VS.

Rather than using upstream BBDM's PNG dump evaluation, we compute streaming
PSNR/SSIM/FID consistent with VSlite (`util/i2i_metrics.py`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml

_THIS_DIR = Path(__file__).resolve().parent
_VSLITE_ROOT = _THIS_DIR.parent
if str(_VSLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_VSLITE_ROOT))

from diffusion_vs.bbdm_latent_vs_model import build_latent_bbdm_model_class
from diffusion_vs.bbdm_vs_dataset import register_vslite_dataset
from util.i2i_metrics import compute_metrics_streaming


def _dict2namespace(d: dict):
    import argparse as _argparse

    ns = _argparse.Namespace()
    for k, v in d.items():
        if isinstance(v, dict):
            v = _dict2namespace(v)
        setattr(ns, k, v)
    return ns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--resume_model", type=str, required=True)
    ap.add_argument("--result_path", type=str, default="/projects/illinois/eng/bioe/rxb/akshat/VSlite/results/bbdm_vs")
    ap.add_argument("--run_name", type=str, required=True, help="short tag for output folder (e.g., bbdmA_10)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num_test", type=int, default=75000)
    ap.add_argument("--fid_max_items", type=int, default=None)
    ap.add_argument("--fid_batch_size", type=int, default=16)
    ap.add_argument("--split", type=str, default="test", choices=["train", "test"])
    args = ap.parse_args()

    register_vslite_dataset(_VSLITE_ROOT)

    bbdm_root = _VSLITE_ROOT / "third_party" / "BBDM"
    if str(bbdm_root) not in sys.path:
        sys.path.insert(0, str(bbdm_root))

    from utils import get_runner  # type: ignore
    import runners.DiffusionBasedModelRunners.BBDMRunner as bbdm_runner_mod  # type: ignore

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))

    with open(args.config, "r") as f:
        dict_config = yaml.load(f, Loader=yaml.FullLoader)
    nconfig = _dict2namespace(dict_config)
    nconfig.args = argparse.Namespace(
        train=False,
        seed=1234,
        gpu_ids=str(args.gpu),
        port="12355",
        result_path=str(args.result_path),
        resume_model=str(args.resume_model),
        resume_optim=None,
        max_epoch=None,
        max_steps=None,
        sample_to_eval=False,
        sample_at_start=False,
        save_top=False,
        config=str(args.config),
    )
    nconfig.training.use_DDP = False
    nconfig.training.device = [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
    nconfig.model.model_load_path = str(args.resume_model)

    # Monkeypatch latent model class for LBBDM.
    if str(getattr(nconfig.model, "model_type", "")) == "LBBDM":
        LatentBrownianBridgeModelVS = build_latent_bbdm_model_class(_VSLITE_ROOT)
        bbdm_runner_mod.LatentBrownianBridgeModel = LatentBrownianBridgeModelVS  # type: ignore

    runner = get_runner(nconfig.runner, nconfig)  # type: ignore
    # Ensure model is loaded.
    runner.load_model_from_checkpoint()
    net = runner.get_ema_net() if hasattr(runner, "get_ema_net") else runner.net
    net.eval()

    # Build dataset directly from registry (runner utility).
    from Register import Registers  # type: ignore

    ds = Registers.datasets[nconfig.data.dataset_type](nconfig.data.dataset_config, stage=("train" if args.split == "train" else "test"))

    out_dir = os.path.join(
        str(Path(args.result_path).resolve()),
        "test",
        str(args.run_name),
        "patch_eval_test" if args.split == "test" else "patch_eval_train",
    )
    os.makedirs(out_dir, exist_ok=True)

    def _pairs():
        n_seen = 0
        for i in range(len(ds)):
            if args.num_test > 0 and n_seen >= int(args.num_test):
                break
            (x, _xname), (x_cond, _cname) = ds[i]
            x = x.unsqueeze(0).to(nconfig.training.device[0], dtype=torch.float32)
            x_cond = x_cond.unsqueeze(0).to(nconfig.training.device[0], dtype=torch.float32)
            with torch.no_grad():
                yhat = net.sample(x_cond, clip_denoised=bool(getattr(nconfig.testing, "clip_denoised", False)))
            # net.sample returns BCHW in [-1,1]
            y = x[0].detach().cpu()
            yhat0 = yhat[0].detach().cpu()
            n_seen += 1
            yield (y, yhat0)

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
        "run_name": str(args.run_name),
        "resume_model": str(args.resume_model),
        "out_dir": out_dir,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    try:
        import yaml as _yaml

        with open(os.path.join(out_dir, "metrics.yaml"), "w") as f:
            _yaml.safe_dump(metrics, f)
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

