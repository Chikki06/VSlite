"""
Train a VQGAN autoencoder for H&E targets (virtual staining).

Uses `third_party/VQGAN-pytorch` model definitions, but trains with VSlite's dataset.
Checkpoints are saved under `results/vqgan_vs/<name>/checkpoints/`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import utils as vutils

_THIS_DIR = Path(__file__).resolve().parent
_VSLITE_ROOT = _THIS_DIR.parent
if str(_VSLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_VSLITE_ROOT))

from diffusion_vs.vqgan_vs import VQGANHEDataset, import_third_party_vqgan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, required=True, help="run name under results/vqgan_vs/")
    ap.add_argument("--dataroot", type=str, required=True)
    ap.add_argument("--results_dir", type=str, default="/projects/illinois/eng/bioe/rxb/akshat/VSlite/results/vqgan_vs")
    ap.add_argument("--input_nc", type=int, default=13, choices=[10, 13], help="only used to read dataset; targets are H&E")
    ap.add_argument("--num_ir_channels", type=int, default=10)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--learning_rate", type=float, default=2.25e-5)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--beta2", type=float, default=0.9)
    ap.add_argument("--disc_start", type=int, default=10000)
    ap.add_argument("--disc_factor", type=float, default=1.0)
    ap.add_argument("--rec_loss_factor", type=float, default=1.0)
    ap.add_argument("--perceptual_loss_factor", type=float, default=0.0, help="set >0 to use LPIPS if available")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    out_root = Path(args.results_dir).resolve() / args.name
    ckpt_dir = out_root / "checkpoints"
    img_dir = out_root / "images"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    VQGAN, Discriminator = import_third_party_vqgan(_VSLITE_ROOT)

    # Build args namespace expected by third_party VQGAN class.
    # It reads: latent_dim, image_size, num_codebook_vectors, beta, image_channels.
    class _VQArgs:
        pass

    vq_args = _VQArgs()
    vq_args.latent_dim = 256
    vq_args.image_size = int(args.image_size)
    vq_args.num_codebook_vectors = 1024
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
    disc = Discriminator(vq_args).to(device=device)  # type: ignore

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

    train_ds = VQGANHEDataset(dataroot=args.dataroot, split="train", input_nc=int(args.input_nc), num_ir_channels=int(args.num_ir_channels))
    dl = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available(), drop_last=True)

    global_step = 0
    for epoch in range(int(args.epochs)):
        for batch in dl:
            x = batch.x_he.to(device=device, dtype=torch.float32)
            decoded, _z, q_loss = vqgan(x)

            disc_real = disc(x)
            disc_fake = disc(decoded)

            # discriminator warmup
            disc_factor = vqgan.adopt_weight(float(args.disc_factor), global_step, threshold=int(args.disc_start))

            rec_l1 = torch.abs(x - decoded).mean()
            if perceptual is not None:
                lp = perceptual(x, decoded).mean()
            else:
                lp = torch.zeros((), device=device)

            perceptual_rec = float(args.rec_loss_factor) * rec_l1 + float(args.perceptual_loss_factor) * lp
            g_loss = -torch.mean(disc_fake)
            lam = vqgan.calculate_lambda(perceptual_rec, g_loss)
            vq_loss = perceptual_rec + q_loss + disc_factor * lam * g_loss

            d_loss_real = torch.mean(F.relu(1.0 - disc_real))
            d_loss_fake = torch.mean(F.relu(1.0 + disc_fake))
            gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

            opt_vq.zero_grad(set_to_none=True)
            vq_loss.backward(retain_graph=True)
            opt_vq.step()

            opt_disc.zero_grad(set_to_none=True)
            gan_loss.backward()
            opt_disc.step()

            if global_step % 200 == 0:
                with torch.no_grad():
                    # Save a small grid: real then recon, mapped to [0,1] for visualization.
                    vis = torch.cat([x[:4], decoded[:4]], dim=0)
                    vutils.save_image((vis + 1) * 0.5, img_dir / f"e{epoch:03d}_s{global_step:07d}.png", nrow=4)

            global_step += 1

        torch.save(vqgan.state_dict(), ckpt_dir / f"vqgan_epoch_{epoch:03d}.pt")


if __name__ == "__main__":
    main()

