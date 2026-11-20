"""Smoke test for the UNI feature extractor (netF=uni) and the lpips_uni metric.

Run anywhere:
  python scripts/smoke_uni_netf.py                    # CPU, small sizes
  VSLITE_UNI_SMOKE_GPU=1 python scripts/smoke_uni_netf.py   # GPU, full 1024 size

Checks:
  1. UNIFeatureExtractor builds from the gated UNI v1 checkpoint (strict load).
  2. Feature maps: 4 blocks, shape (B, 1024, size/16, size/16).
  3. Gradient flows back to the query input through the frozen backbone
     (this is what the NCE loss needs; checkpointing must not block it).
  4. Full ContrastiveEncoder path (netF=uni) incl. PatchSampleH pooling.
  5. LPIPSUNI metric: zero distance for identical inputs, finite otherwise.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _build_encoder_opt(uni_input_size: int, crop_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        netF="uni",
        netH="mlp_sample",
        netH_nc=256,
        lambda_KL=0.0,
        nce_norm=2,
        nce_layers="1,2,3,4,5",
        freeze_netF=True,
        freeze_netH=False,
        uni_input_size=uni_input_size,
        uni_feature_ckpt=os.environ.get("VSLITE_UNI_CKPT", "")
        or str(_ROOT / "foundation_models" / "UNI" / "pytorch_model.bin"),
        uni_grad_checkpoint=True,
        crop_size=crop_size,
        aspect_ratio=1,
        init_type="xavier",
        init_variance=0.02,
        gpu_ids=[],
    )


def _train_probe(full: int, batch: int) -> None:
    """Training-realistic netF cost: key (real) no-grad + query (fake) grad path."""
    from models.networks.architecture import UNIFeatureExtractor

    device = torch.device("cuda:0")
    real = torch.randn(batch, 3, full, full, device=device)
    fake = torch.randn(batch, 3, full, full, device=device, requires_grad=True)

    for ckpt in (True, False):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            uni = UNIFeatureExtractor(
                opt=None,
                pretrained=True,
                freeze_weights=True,
                uni_input_size=full,
                grad_checkpoint=ckpt,
            ).to(device)
            # Key path (real, frozen data): no graph is built.
            t0 = time.time()
            with torch.no_grad():
                feats_k = uni(real)
            torch.cuda.synchronize()
            t_key = time.time() - t0
            # Query path (fake): grad forward + backward through the maps.
            t0 = time.time()
            feats_q = uni(fake)
            loss = sum(f.float().pow(2).mean() for f in feats_q)
            loss.backward()
            torch.cuda.synchronize()
            t_q = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(
                f"[train_probe] ckpt={ckpt} B={batch} @{full}px: "
                f"key_fwd={t_key:.2f}s query_fwd+bwd={t_q:.2f}s peak={peak:.2f} GiB"
            )
            fake.grad = None
            del uni, feats_k, feats_q, loss
        except torch.cuda.OutOfMemoryError as e:
            print(f"[train_probe] ckpt={ckpt} B={batch} @{full}px: OOM ({e})")
            torch.cuda.empty_cache()
    print("[train_probe] done")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=None, help="UNI input size override")
    ap.add_argument(
        "--train_probe",
        action="store_true",
        help="GPU training-realistic probe: B=10 @ full size, key no-grad fwd + "
        "query grad fwd/backward (with and without checkpointing).",
    )
    ap.add_argument("--batch", type=int, default=10, help="train_probe batch size")
    args = ap.parse_args()

    on_gpu = os.environ.get("VSLITE_UNI_SMOKE_GPU", "0") == "1" and torch.cuda.is_available()
    device = torch.device("cuda:0" if on_gpu else "cpu")
    print(f"[smoke] device={device}")

    full = int(args.size or 1024)
    small = 256 if not on_gpu else full

    if args.train_probe:
        if not on_gpu:
            raise SystemExit("--train_probe requires a GPU (VSLITE_UNI_SMOKE_GPU=1)")
        _train_probe(full, int(args.batch))
        return

    # ---- 1+2: standalone extractor ------------------------------------------
    from models.networks.architecture import UNIFeatureExtractor

    t0 = time.time()
    uni = UNIFeatureExtractor(
        opt=None,
        pretrained=True,
        freeze_weights=True,
        uni_input_size=full,
        grad_checkpoint=True,
    ).to(device)
    print(f"[smoke] UNI built in {time.time() - t0:.1f}s ckpt={uni.ckpt_path}")

    x = torch.randn(1, 3, small, small, device=device)
    t0 = time.time()
    feats = uni(x)
    torch.cuda.synchronize() if on_gpu else None
    shapes = [tuple(f.shape) for f in feats]
    print(f"[smoke] fwd({small}px -> fixed {full}px) {time.time() - t0:.2f}s shapes={shapes}")
    assert len(feats) == 4, f"expected 4 maps, got {len(feats)}"
    g = full // 16
    for f in feats:
        assert f.shape == (1, 1024, g, g), f"bad map {tuple(f.shape)} (extractor resizes to {full}px)"

    if full != small:
        x1024 = torch.randn(1, 3, full, full, device=device)
        t0 = time.time()
        feats1024 = uni(x1024)
        torch.cuda.synchronize() if on_gpu else None
        print(
            f"[smoke] fwd({full}px native) {time.time() - t0:.2f}s "
            f"shapes={[tuple(f.shape) for f in feats1024]}"
        )
        for f in feats1024:
            assert f.shape == (1, 1024, full // 16, full // 16)

    # ---- 3: gradient reaches the query input --------------------------------
    xq = torch.randn(1, 3, small, small, device=device, requires_grad=True)
    feats = uni(xq)
    loss = sum(f.float().pow(2).mean() for f in feats)
    loss.backward()
    assert xq.grad is not None and xq.grad.abs().sum() > 0, "no grad through UNI"
    grad_finite = torch.isfinite(xq.grad).all().item()
    print(f"[smoke] grad ok (finite={grad_finite}, |grad|={xq.grad.abs().mean().item():.3e})")
    assert grad_finite

    # ---- 4: ContrastiveEncoder integration ----------------------------------
    from models.networks.contrastive_encoder import ContrastiveEncoder

    opt = _build_encoder_opt(full, small)
    enc = ContrastiveEncoder(opt).to(device)
    xq2 = torch.randn(2, 3, small, small, device=device, requires_grad=True)
    feats, pools, pids = enc(xq2, num_patches=64)
    pool_shapes = [tuple(p.shape) for p in pools]
    print(f"[smoke] enc fwd pools={pool_shapes}")
    assert len(pools) == 4
    for p in pools:
        assert p.shape == (2, 64, 256), f"bad pool {tuple(p.shape)}"
    psum = sum(p.float().sum() for p in pools)
    psum.backward()
    assert xq2.grad is not None and torch.isfinite(xq2.grad).all().item()
    print("[smoke] enc grad ok")

    # frozen backbone really frozen
    enc_grads = [
        p.requires_grad
        for p in enc.encoder.model.parameters()
    ]
    assert not any(enc_grads), "UNI backbone params must be frozen"
    mlp_grads = [p.requires_grad for p in enc.projection.mlps.parameters()]
    assert any(mlp_grads), "projection MLPs must be trainable"
    print("[smoke] freeze pattern ok (backbone frozen, MLP H trainable)")

    # ---- 5: LPIPSUNI metric --------------------------------------------------
    from util.lpips_uni import build_lpips_uni

    lp = build_lpips_uni(device=device)
    a = torch.randn(1, 3, lp.input_size, lp.input_size, device=device)
    d_same = lp(a, a)
    b = torch.randn(1, 3, lp.input_size, lp.input_size, device=device)
    d_diff = lp(a, b)
    print(
        f"[smoke] lpips_uni same={d_same.item():.6f} diff={d_diff.item():.6f} "
        f"(size={lp.input_size})"
    )
    assert float(d_same.item()) < 1e-6 and torch.isfinite(d_diff).all()

    # ---- 6: create_network scramble -> reload pattern ------------------------
    # ContrastivePix2PixModel relies on pretrained_netF=True re-applying UNI
    # weights after BaseNetwork.init_weights xavier-scrambles every Linear/Conv.
    from models.networks import contrastive_encoder as ce

    enc2 = ce.ContrastiveEncoder(opt).to(device)
    ref_w = enc2.encoder.model.blocks[0].attn.qkv.weight.detach().clone()
    enc2.init_weights("xavier", 0.02)
    scrambled = not torch.allclose(ref_w, enc2.encoder.model.blocks[0].attn.qkv.weight)
    enc2.encoder.load_state_dict(
        UNIFeatureExtractor(
            opt=opt, pretrained=True, freeze_weights=True,
            uni_input_size=full, grad_checkpoint=False,
        ).state_dict()
    )
    restored = torch.allclose(
        ref_w, enc2.encoder.model.blocks[0].attn.qkv.weight
    )
    print(f"[smoke] init scrambles pretrained: {scrambled}; reload restores: {restored}")
    assert scrambled and restored, "UNI weights must survive the xavier init via reload"

    if on_gpu:
        print(f"[smoke] cuda max memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    print("[smoke] ALL OK")


if __name__ == "__main__":
    main()
