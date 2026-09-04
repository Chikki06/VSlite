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
from cislnet.region_eval import tiled_generator_forward, validate_region_tiling


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
    ap.add_argument("--save_every", type=int, default=0, help="If >0, save palette PNGs every N patches.")
    ap.add_argument(
        "--save_triplets",
        action="store_true",
        help="Save input/target/pred PNG triplets under patch_eval/{input,target,pred}/",
    )
    ap.add_argument(
        "--region_eval_size",
        type=int,
        default=0,
        help="If >0, load patches at this canvas size and tile inference to the model's trained patch size.",
    )
    ap.add_argument(
        "--save_region_triplets",
        action="store_true",
        help="When --region_eval_size is set, save one stitched triplet per region (2048 default).",
    )
    ap.add_argument(
        "--test_list_path",
        type=str,
        default="",
        help="Override dataroot/test.txt for the eval split.",
    )
    ap.add_argument(
        "--val_list_path",
        type=str,
        default="",
        help="Override dataroot/val.txt (only used when test_includes_val is on).",
    )
    ap.add_argument(
        "--test_includes_val",
        type=int,
        default=None,
        help="1=union test+val for FID; 0=test list only. Default: 1 when --split test.",
    )
    ap.add_argument(
        "--results_name",
        type=str,
        default="",
        help="Optional results folder leaf under results_dir/test/. "
        "Checkpoint still loads from --name; default is --name.",
    )
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
    # Combine test.txt and val.txt into a single held-out set for FID/PSNR/SSIM
    # when running on the test split (default). Override with --test_includes_val 0/1.
    if args.test_includes_val is None:
        opt.test_includes_val = bool(args.split == "test")
    else:
        opt.test_includes_val = bool(int(args.test_includes_val))
    if str(args.test_list_path or "").strip():
        opt.test_list_path = str(args.test_list_path).strip()
        print(f"override test_list_path: {opt.test_list_path}")
    if str(args.val_list_path or "").strip():
        opt.val_list_path = str(args.val_list_path).strip()
        print(f"override val_list_path: {opt.val_list_path}")
    # Allow larger inference batches to better utilize big GPUs (e.g., 80GB A100).
    # Default to 16, overrideable via VSLITE_INFER_BS.
    infer_bs = int(os.environ.get("VSLITE_INFER_BS", "16"))
    opt.batch_size = max(1, infer_bs)

    model_tile_size = int(getattr(opt, "image_size", 1024))
    region_eval_size = int(args.region_eval_size or 0)
    use_region_eval = region_eval_size > 0
    if use_region_eval:
        validate_region_tiling(region_eval_size, model_tile_size)
        opt.image_size = region_eval_size

    dataset = create_dataloader(opt)
    model = create_model(opt)
    model.eval()
    net_g = getattr(model, "netG", None)
    if use_region_eval and net_g is None:
        raise RuntimeError("region_eval_size requires a model with netG (contrastive_pix2pix / AttUnetGenerator)")

    # Keep run names short and consistent:
    # - split=test:  <results_dir>/test/<name>/patch_eval_test
    # - split=train: <results_dir>/test/<name>/patch_eval_train
    # - if which_epoch != latest: same but with <name>_<which_epoch>
    # --results_name overrides the folder leaf (checkpoint still uses --name).
    results_leaf = str(args.results_name or "").strip() or str(opt.name)
    run_name = results_leaf if str(opt.which_epoch) == "latest" else f"{results_leaf}_{opt.which_epoch}"
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

    save_triplets = bool(args.save_triplets or (use_region_eval and args.save_region_triplets))

    def _infer_one(x: torch.Tensor) -> torch.Tensor:
        if use_region_eval:
            return tiled_generator_forward(net_g, x, model_tile_size)
        data_i = {"label": x.unsqueeze(0), "image": x.unsqueeze(0)[:3]}
        with torch.no_grad():
            generated = model(data_i, mode="inference")
        return generated[0]

    def _pairs():
        n_seen = 0
        buf5: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        for i, data_i in enumerate(dataset):
            if n_seen >= opt.num_test:
                break
            data_i = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data_i.items()}
            if use_region_eval:
                bs = int(data_i["label"].shape[0])
                generated_list = []
                for b in range(bs):
                    generated_list.append(_infer_one(data_i["label"][b]))
                generated = torch.stack(generated_list, dim=0)
            else:
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

                if args.save_every and (n_seen % int(args.save_every) == 0) and len(buf5) == 5:
                    save_palette5_pred_target_png(out_dir, f"{name}_save{n_seen}", list(buf5), filename_suffix="palette5_save")

                if save_triplets:
                    x_np = _tensor_to_png_array(x)
                    y_np = _tensor_to_png_array(y)
                    p_np = _tensor_to_png_array(yhat)
                    Image.fromarray(x_np).save(os.path.join(input_dir, f"{name}.png"))
                    Image.fromarray(y_np).save(os.path.join(target_dir, f"{name}.png"))
                    Image.fromarray(p_np).save(os.path.join(pred_dir, f"{name}.png"))

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
        "lpips_convnext_tcga_mean": res.lpips_convnext_tcga_mean,
        "lpips_convnext_tcga_std": res.lpips_convnext_tcga_std,
        "pred_dir": pred_dir,
        "target_dir": target_dir,
        "split": str(args.split),
        "region_eval_size": region_eval_size if use_region_eval else None,
        "model_tile_size": model_tile_size if use_region_eval else None,
    }
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

