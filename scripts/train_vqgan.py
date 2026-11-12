"""
Train a VQGAN autoencoder for H&E targets (virtual staining).

Uses `third_party/VQGAN-pytorch` model definitions, but trains with VSlite's dataset.
Checkpoints are saved under `results/vqgan/<name>/checkpoints/`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_THIS_DIR = Path(__file__).resolve().parent
_VSLITE_ROOT = _THIS_DIR.parent
if str(_VSLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_VSLITE_ROOT))

from diffusion_vs.vqgan_vs import (
    VQGANHEDataset,
    build_vqgan_discriminator,
    collate_vqgan_batch,
    import_third_party_vqgan,
)
from SwinGAN.utils.dataset_vslite_i2i import SplitFractions, VSLiteI2IDataset
from util.atomic_io import atomic_torch_save
from util.fixed_preview import gather_fixed_preview_batch
from util.run_logger import DEFAULT_TRAIN_LOG_PRINT_FREQ, RunLogger, should_log_epoch_samples
from util.sample_grid import save_unified_sample, split_batch_to_list
from util.save_schedule import make_iter_schedule
from util.unified_tb import build_unified_writer
from util.eval_caps import eval_max_batches_val, eval_max_batches_test


@torch.no_grad()
def _recon_l1(vqgan, loader, device, max_batches: int = 0) -> float:
    vqgan.eval()
    tot = 0.0
    n = 0
    for bi, batch in enumerate(loader):
        if max_batches and bi >= int(max_batches):
            break
        x = batch.x_he.to(device=device, dtype=torch.float32)
        decoded, _z, _q_loss = vqgan(x)
        rec_l1 = torch.abs(x - decoded).mean().item()
        bs = int(x.shape[0])
        tot += float(rec_l1) * bs
        n += bs
    return tot / max(1, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, required=True, help="run name under results/vqgan/")
    ap.add_argument("--dataroot", type=str, required=True)
    ap.add_argument("--results_dir", type=str, default="/projects/illinois/eng/bioe/rxb/akshat/VSlite/results/vqgan")
    ap.add_argument("--input_nc", type=int, default=13, choices=[10, 13], help="only used to read dataset; targets are H&E")
    ap.add_argument("--num_ir_channels", type=int, default=10)
    ap.add_argument("--image_size", type=int, default=512)
    # Keep the default latent grid at 16 for stability with this third_party
    # architecture. Increasing it (e.g. to 64 for 1024px inputs) also removes
    # downsampling stages and moves attention to a much larger map, which made
    # training collapse to near-constant outputs in practice.
    ap.add_argument("--latent_grid_size", type=int, default=16, help="Latent grid resolution (tokens) per side.")
    ap.add_argument("--latent_dim", type=int, default=256)
    ap.add_argument("--num_codebook_vectors", type=int, default=4096)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--grad_accum_steps", type=int, default=1, help="Accumulate gradients over N steps to simulate larger batch.")
    ap.add_argument("--amp", action="store_true", help="Use torch autocast (mixed precision) on CUDA.")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--learning_rate", type=float, default=2.25e-5)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--beta2", type=float, default=0.9)
    ap.add_argument("--disc_start", type=int, default=10000)
    ap.add_argument("--disc_factor", type=float, default=1.0)
    ap.add_argument("--rec_loss_factor", type=float, default=1.0)
    ap.add_argument("--perceptual_loss_factor", type=float, default=1.0, help="set >0 to use LPIPS if available")
    ap.add_argument(
        "--discriminator_variant",
        type=str,
        default="default",
        choices=["default", "patchd"],
        help="`patchd` explicitly selects PatchGAN discriminator behavior.",
    )
    # More saves → smaller interval between unified ``samples/*.png`` previews (see ``util/save_schedule.py``).
    ap.add_argument(
        "--sample_schedule_saves",
        type=int,
        default=100,
        help="Spread this many unified sample PNG saves across the full run (plus epoch-end snapshots).",
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument(
        "--resume",
        type=str,
        default="auto",
        choices=["auto", "none"],
        help="Resume from results_dir/name/checkpoints/ckpt_latest.pt if present (auto), or start fresh (none).",
    )
    args = ap.parse_args()

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    out_root = Path(args.results_dir).resolve() / args.name
    ckpt_dir = out_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    VQGAN, _Discriminator = import_third_party_vqgan(_VSLITE_ROOT)

    # Build args namespace expected by third_party VQGAN class.
    # It reads: latent_dim, image_size, latent_grid_size, num_codebook_vectors, beta, image_channels.
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
    vq_args.learning_rate = float(args.learning_rate)
    vq_args.beta1 = float(args.beta1)
    vq_args.beta2 = float(args.beta2)
    vq_args.disc_start = int(args.disc_start)
    vq_args.disc_factor = float(args.disc_factor)
    vq_args.rec_loss_factor = float(args.rec_loss_factor)
    vq_args.perceptual_loss_factor = float(args.perceptual_loss_factor)

    vqgan = VQGAN(vq_args).to(device=device)  # type: ignore
    disc = build_vqgan_discriminator(
        _VSLITE_ROOT,
        vq_args,
        variant=str(args.discriminator_variant),
    ).to(device=device)

    # Optional resume (default: auto).
    start_epoch = 0
    global_step = 0
    if str(args.resume).lower() == "auto":
        resume_path = ckpt_dir / "ckpt_latest.pt"
        if resume_path.is_file():
            payload = torch.load(resume_path, map_location=device)
            if isinstance(payload, dict):
                if "vqgan" in payload:
                    vqgan.load_state_dict(payload["vqgan"], strict=True)
                if "disc" in payload:
                    disc.load_state_dict(payload["disc"], strict=True)
                if "opt_vq" in payload:
                    # Optimizers are defined below; stash payload for later restore.
                    _resume_opt_vq = payload["opt_vq"]
                else:
                    _resume_opt_vq = None
                if "opt_disc" in payload:
                    _resume_opt_disc = payload["opt_disc"]
                else:
                    _resume_opt_disc = None
                _resume_scaler = payload.get("scaler", None)
                start_epoch = int(payload.get("epoch", -1)) + 1
                global_step = int(payload.get("global_step", 0))
            else:
                _resume_opt_vq = None
                _resume_opt_disc = None
                _resume_scaler = None
        else:
            _resume_opt_vq = None
            _resume_opt_disc = None
            _resume_scaler = None
    else:
        _resume_opt_vq = None
        _resume_opt_disc = None
        _resume_scaler = None

    # LPIPS is optional (avoid hard dependency).
    perceptual = None
    if float(args.perceptual_loss_factor) > 0:
        try:
            tp = _VSLITE_ROOT / "third_party" / "VQGAN-pytorch"
            if str(tp) not in sys.path:
                sys.path.insert(0, str(tp))
            from lpips import LPIPS  # type: ignore

            perceptual = LPIPS().eval().to(device=device)
        except Exception:
            perceptual = None

    opt_vq = torch.optim.Adam(
        list(vqgan.encoder.parameters())
        + list(vqgan.decoder.parameters())
        + list(vqgan.codebook.parameters())
        + list(vqgan.quant_conv.parameters())
        + list(vqgan.post_quant_conv.parameters()),
        lr=float(args.learning_rate),
        eps=1e-8,
        betas=(float(args.beta1), float(args.beta2)),
    )
    opt_disc = torch.optim.Adam(disc.parameters(), lr=float(args.learning_rate), eps=1e-8, betas=(float(args.beta1), float(args.beta2)))

    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if _resume_opt_vq is not None:
        try:
            opt_vq.load_state_dict(_resume_opt_vq)
        except Exception:
            pass
    if _resume_opt_disc is not None:
        try:
            opt_disc.load_state_dict(_resume_opt_disc)
        except Exception:
            pass
    if use_amp and _resume_scaler is not None:
        try:
            scaler.load_state_dict(_resume_scaler)
        except Exception:
            pass
    accum = max(1, int(args.grad_accum_steps))

    train_ds = VQGANHEDataset(
        dataroot=args.dataroot,
        split="train",
        input_nc=int(args.input_nc),
        num_ir_channels=int(args.num_ir_channels),
        image_size=int(args.image_size),
    )
    dl = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=collate_vqgan_batch,
    )

    val_ds = VQGANHEDataset(
        dataroot=args.dataroot,
        split="val",
        input_nc=int(args.input_nc),
        num_ir_channels=int(args.num_ir_channels),
        image_size=int(args.image_size),
    )
    test_ds = VQGANHEDataset(
        dataroot=args.dataroot,
        split="test",
        input_nc=int(args.input_nc),
        num_ir_channels=int(args.num_ir_channels),
        image_size=int(args.image_size),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_vqgan_batch,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_vqgan_batch,
    )

    u_writer, _u_tb = build_unified_writer(model="vqgan", run=str(args.name))
    run_logger = RunLogger(str(out_root), name=str(args.name))
    run_logger.dump_opts(vars(args))

    samples_dir = out_root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Align preview tiles with other stacks: held-out **test** split, first 8
    # items deterministically (shuffle=False). Same patches every epoch / seed.
    preview_input_mode = "irvis13" if int(args.input_nc) == 13 else "ir10"
    preview_ds = VSLiteI2IDataset(
        args.dataroot,
        input_mode=preview_input_mode,
        num_ir_channels=int(args.num_ir_channels),
        img_size=int(args.image_size),
        he_scale="minus1_1",
        split="test",
        split_seed=1337,
        split_fracs=SplitFractions(train=0.9, val=0.05, test=0.05),
    )
    preview_bs = 1
    preview_loader = DataLoader(
        preview_ds,
        batch_size=int(preview_bs),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    fixed_preview = gather_fixed_preview_batch(preview_loader, n=8)
    fixed_preview_x: torch.Tensor | None = None
    if fixed_preview is not None and torch.is_tensor(fixed_preview.get("y")):
        fixed_preview_x = fixed_preview["y"]

    iters_per_epoch = max(1, len(dl))
    total_iters = max(1, int(args.epochs) * iters_per_epoch)
    should_save_image = make_iter_schedule(total_iters, target_saves=int(args.sample_schedule_saves))
    train_log_freq = int(os.environ.get("VSLITE_TRAIN_LOG_PRINT_FREQ", str(DEFAULT_TRAIN_LOG_PRINT_FREQ)))

    @torch.no_grad()
    def _save_unified_preview(epoch_idx: int, step_idx: int) -> None:
        if fixed_preview_x is None:
            return
        was_train = vqgan.training
        vqgan.eval()
        x_eval = fixed_preview_x.to(device=device, dtype=torch.float32)
        decoded_eval, _z, _q = vqgan(x_eval)
        reals = split_batch_to_list(x_eval.detach().cpu())[:8]
        preds = split_batch_to_list(decoded_eval.detach().cpu())[:8]
        name = f"epoch{int(epoch_idx) + 1:03d}_step{int(step_idx):09d}"
        save_unified_sample(str(samples_dir), name, reals, preds)
        if was_train:
            vqgan.train()

    for epoch in range(int(start_epoch), int(args.epochs)):
        epoch_samples = 0
        for batch in dl:
            x = batch.x_he.to(device=device, dtype=torch.float32)
            bs = int(x.shape[0])
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                decoded, _z, q_loss = vqgan(x)

                # Generator path (keep graph).
                disc_fake_for_g = disc(decoded)

            # discriminator warmup
            disc_factor = vqgan.adopt_weight(float(args.disc_factor), global_step, threshold=int(args.disc_start))

            rec_l1 = torch.abs(x - decoded).mean()
            if perceptual is not None:
                lp = perceptual(x, decoded).mean()
            else:
                lp = torch.zeros((), device=device)

            perceptual_rec = float(args.rec_loss_factor) * rec_l1 + float(args.perceptual_loss_factor) * lp
            g_loss = -torch.mean(disc_fake_for_g)
            lam = vqgan.calculate_lambda(perceptual_rec, g_loss)
            vq_loss = perceptual_rec + q_loss + disc_factor * lam * g_loss

            if global_step % accum == 0:
                opt_vq.zero_grad(set_to_none=True)
                opt_disc.zero_grad(set_to_none=True)

            # Normalize losses so effective gradient matches larger batch.
            vq_loss_scaled = vq_loss / float(accum)
            scaler.scale(vq_loss_scaled).backward()

            # Discriminator path: detach generator output so we don't keep VQGAN graph alive.
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                disc_real = disc(x.detach())
                disc_fake = disc(decoded.detach())
                d_loss_real = torch.mean(F.relu(1.0 - disc_real))
                d_loss_fake = torch.mean(F.relu(1.0 + disc_fake))
                gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

            gan_loss_scaled = gan_loss / float(accum)
            scaler.scale(gan_loss_scaled).backward()

            if (global_step + 1) % accum == 0:
                scaler.step(opt_vq)
                scaler.step(opt_disc)
                scaler.update()

            global_step += 1
            epoch_samples += bs

            if should_log_epoch_samples(epoch_samples, bs, train_log_freq):
                decoded_det = decoded.detach()
                active_codes = torch.unique(_z.detach()).numel()
                run_logger.append_loss(
                    epoch + 1,
                    epoch_samples,
                    {
                        "rec_l1": float(rec_l1.detach().item()),
                        "lpips": float(lp.detach().item()),
                        "q_loss": float(q_loss.detach().item()),
                        "g_loss": float(g_loss.detach().item()),
                        "d_loss": float(gan_loss.detach().item()),
                        "lambda": float(lam.detach().item()),
                        "disc_factor": float(disc_factor),
                        "decoded_mean": float(decoded_det.mean().item()),
                        "decoded_std": float(decoded_det.std().item()),
                        "active_codes_batch": float(active_codes),
                    },
                    lr=float(args.learning_rate),
                )

            if should_save_image(global_step - 1):
                _save_unified_preview(epoch, global_step)
                run_logger.write_iter(epoch + 1, global_step)

        # Checkpoints must be atomic; corruption here ruins restarts.
        ckpt_payload = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "vqgan": vqgan.state_dict(),
            "disc": disc.state_dict(),
            "opt_vq": opt_vq.state_dict(),
            "opt_disc": opt_disc.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "args": vars(args),
        }
        atomic_torch_save(torch.save, ckpt_payload, ckpt_dir / f"ckpt_epoch_{epoch:03d}.pt")
        atomic_torch_save(torch.save, ckpt_payload, ckpt_dir / "ckpt_latest.pt")
        # Legacy single-module checkpoint (kept for compatibility with older scripts).
        atomic_torch_save(torch.save, vqgan.state_dict(), ckpt_dir / f"vqgan_epoch_{epoch:03d}.pt")
        # Split recon metrics at epoch end (overhead controlled by env).
        eval_val_batches = int(eval_max_batches_val())
        eval_test_batches = int(eval_max_batches_test())
        val_l1 = _recon_l1(vqgan, val_dl, device, max_batches=eval_val_batches)
        test_l1 = _recon_l1(vqgan, test_dl, device, max_batches=eval_test_batches)
        u_writer.add_scalar("recon_l1/val", float(val_l1), epoch)
        u_writer.add_scalar("recon_l1/test", float(test_l1), epoch)
        u_writer.add_scalar("meta/global_step", float(global_step), epoch)

        # End-of-epoch composite save (always).
        _save_unified_preview(epoch, global_step)
        run_logger.write_iter(epoch + 1, global_step)

    u_writer.close()


if __name__ == "__main__":
    main()

