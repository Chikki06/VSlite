import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_THIS_DIR = Path(__file__).resolve().parent
_VSLITE_ROOT = _THIS_DIR.parent
if str(_VSLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_VSLITE_ROOT))

from SwinGAN.utils.sample_saver import save_palette5_pred_target_png
import util.util as util
from data import create_dataloader
from models import create_model
from options.test_options import TestOptions
from util.i2i_metrics import compute_metrics_streaming


def _tensor_to_png_array(tensor: torch.Tensor, prefer_last_three: bool = False) -> np.ndarray:
    t = tensor.detach().float().cpu()
    if t.ndim == 4:
        t = t[0]
    if t.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got {tuple(t.shape)}")
    if t.shape[0] > 3:
        t = t[-3:] if prefer_last_three else t[:3]
    if t.min().item() < 0.0:
        t = (t + 1.0) * 0.5
    t = t.clamp(0.0, 1.0)
    arr = (t * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument("--checkpoints_dir", type=str, required=True)
    ap.add_argument("--dataroot", type=str, required=True)
    ap.add_argument("--dataset_mode", type=str, default="VS")
    ap.add_argument("--model", type=str, default="contrastive_pix2pix")
    ap.add_argument("--which_epoch", type=str, default="latest")
    ap.add_argument("--results_dir", type=str, default="./results")
    ap.add_argument("--gpu_ids", type=str, default="0")
    ap.add_argument("--split", type=str, default="test", choices=["train", "test"])
    ap.add_argument(
        "--num_test",
        type=int,
        default=0,
        help="How many test patches to infer (PSNR/SSIM/FID). 0 = entire test split.",
    )
    ap.add_argument("--input_nc", type=int, default=13)
    ap.add_argument("--num_ir_channels", type=int, default=10)
    ap.add_argument(
        "--ir_channel_indices",
        type=str,
        default="",
        help='Optional comma-separated IR indices (e.g. "0,8,9"). If set, overrides --num_ir_channels slicing.',
    )
    ap.add_argument("--fid_max_items", type=int, default=None)
    ap.add_argument("--fid_batch_size", type=int, default=16)
    ap.add_argument("--save_every", type=int, default=0, help="If >0, save standardized PNG triplets every N patches.")
    args = ap.parse_args()

    # If explicit IR indices are provided, align num_ir_channels with them so option parsing stays consistent.
    num_ir_channels = int(args.num_ir_channels)
    ir_idx = str(args.ir_channel_indices or "").strip()
    if ir_idx:
        try:
            idx = [int(x.strip()) for x in ir_idx.split(",") if x.strip() != ""]
        except Exception:
            raise ValueError(f"Invalid --ir_channel_indices={ir_idx!r}. Expected comma-separated ints like '0,8,9'.")
        if not idx:
            raise ValueError(f"Invalid --ir_channel_indices={ir_idx!r}. No indices parsed.")
        num_ir_channels = len(idx)

    num_test_for_parser = int(args.num_test) if args.num_test > 0 else 10**9
    argv = [
        "vs_infer_and_metrics.py",
        "--name",
        args.name,
        "--checkpoints_dir",
        args.checkpoints_dir,
        "--dataroot",
        args.dataroot,
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
        str(num_test_for_parser),
        "--input_nc",
        str(args.input_nc),
        "--num_ir_channels",
        str(num_ir_channels),
        "--load_from_opt_file",
        "--use_wandb",
        "false",
    ]
    if ir_idx:
        argv += ["--ir_channel_indices", ir_idx]
    sys.argv = argv
    opt = TestOptions().parse()
    opt.isTrain = False
    opt.phase = str(args.split)
    opt.serial_batches = True
    opt.no_flip = True
    opt.no_flip_vert = True
    opt.no_rotate = True
    opt.num_test = float("inf") if args.num_test <= 0 else int(args.num_test)
    opt.batch_size = 1

    dataset = create_dataloader(opt)
    model = create_model(opt)
    model.eval()

    # Keep run names short and consistent:
    # - split=test:  <results_dir>/test/<name>/patch_eval_test
    # - split=train: <results_dir>/test/<name>/patch_eval_train
    # - if which_epoch != latest: same but with <name>_<which_epoch>
    run_name = opt.name if str(opt.which_epoch) == "latest" else f"{opt.name}_{opt.which_epoch}"
    patch_eval_leaf = "patch_eval_test" if args.split == "test" else "patch_eval_train"
    out_dir = os.path.join(opt.results_dir, "test", run_name, patch_eval_leaf)
    input_dir = os.path.join(out_dir, "input")
    target_dir = os.path.join(out_dir, "target")
    pred_dir = os.path.join(out_dir, "pred")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    def _pairs():
        n_seen = 0
        buf5: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        for i, data_i in enumerate(dataset):
            if n_seen >= opt.num_test:
                break
            data_i = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data_i.items()}
            with torch.no_grad():
                generated = model(data_i, mode="inference")
            bs = int(generated.shape[0]) if torch.is_tensor(generated) else 1
            for b in range(bs):
                if n_seen >= opt.num_test:
                    break
                name = os.path.basename(data_i["path"][b])
                name = os.path.splitext(name)[0]
                x = data_i["label"][b]
                y = data_i["image"][b]
                yhat = generated[b]

                buf5.append((name, y.detach().cpu(), yhat.detach().cpu()))
                if len(buf5) > 5:
                    buf5.pop(0)

                # Only palette saving (pred + target). No per-patch triplets.
                if args.save_every and (n_seen % int(args.save_every) == 0) and len(buf5) == 5:
                    save_palette5_pred_target_png(out_dir, f"{name}_save{n_seen}", list(buf5), filename_suffix="palette5_save")

                n_seen += 1
                yield (y.detach(), yhat.detach())

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
        "pred_dir": pred_dir,
        "target_dir": target_dir,
        "split": str(args.split),
    }
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
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    try:
        import yaml

        with open(os.path.join(out_dir, "metrics.yaml"), "w") as f:
            yaml.safe_dump(metrics, f)
    except Exception:
        pass


if __name__ == "__main__":
    main()

