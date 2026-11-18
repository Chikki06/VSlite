"""Shared utilities for TCGA-UT encoder pretraining (SupCon / FCMAE).

Dataset: TCGA-UT (Komura et al. 2022, Zenodo 5889558): 1,608,060 256x256 H&E
patches, 32 cancer types, 6 resolution levels (0.5-1.0 um/px). Manifest built by
``scripts/build_tcga_ut_manifest.py``.

Checkpoint contract (drop-in for ``models.networks.architecture.ConvNeXtFeatureExtractor``
and all VSlite FID/LPIPS loaders): a bare fp32 ``torch.save(model.state_dict(), path)``
from a timm model built with ``num_classes=0`` (keys ``stem.*``, ``stages.*``,
``head.norm.*``; no classifier weights, no wrapper dict, no ``module.`` prefix).
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import random
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

VAL_PERMILLE = 25  # 2.5% of patients -> val (mirrors scripts/build_tcga_ut_manifest.py)


def split_of(patient: str) -> str:
    """Deterministic patient-level split (md5), shared by all TCGA-UT/U8K manifests."""
    h = int(hashlib.md5(patient.encode()).hexdigest()[:8], 16)
    return "val" if h % 1000 < VAL_PERMILLE else "train"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def read_manifest(
    manifest_csv: str | os.PathLike,
    splits: Sequence[str] = ("train",),
    res_filter: Sequence[int] | None = None,
    limit: int | None = None,
):
    """Read manifest CSV -> (rows, label_map).

    ``rows``: list of ``(relpath:str, label_idx:int, res_level:int)``.
    Rows are deterministically shuffled (seed 0) *before* applying ``limit``
    so truncated reads (smoke caps, the kNN bank) span all labels instead of
    only the first cancer type — the manifest CSV is written label-ordered.
    """
    splits = set(splits)
    rows: list[tuple[str, int, int]] = []
    labels: list[str] = []
    label_map: dict[str, int] = {}
    with open(manifest_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            if r["split"] not in splits:
                continue
            if res_filter is not None and int(r["res_level"]) not in res_filter:
                continue
            lab = r["label"]
            if lab not in label_map:
                label_map[lab] = len(labels)
                labels.append(lab)
            rows.append((r["relpath"], label_map[lab], int(r["res_level"])))
    random.Random(0).shuffle(rows)
    if limit is not None:
        rows = rows[:limit]
    return rows, label_map


# ---------------------------------------------------------------------------
# Dataset + transforms
# ---------------------------------------------------------------------------

class TwoCropTransform:
    """SupContrast-style: one image -> two independent augmentations."""

    def __init__(self, transform: Callable):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class TCGAUTDataset(Dataset):
    """Flat JPEG dataset driven by the manifest CSV.

    Returns ``(img, label_idx, idx)``; with a ``TwoCropTransform`` the img is a
    2-element list and the collate fn must be :func:`supcon_collate`.
    """

    def __init__(self, rows, extract_dir: str | os.PathLike, transform: Callable):
        self.rows = rows
        self.extract_dir = str(extract_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        relpath, label, _ = self.rows[idx]
        path = os.path.join(self.extract_dir, relpath)
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Corrupt/missing file: substitute another deterministic sample.
            for off in range(1, 16):
                j = (idx + off) % len(self.rows)
                try:
                    img = Image.open(os.path.join(self.extract_dir, self.rows[j][0])).convert("RGB")
                    break
                except Exception:
                    continue
            else:
                img = Image.new("RGB", (256, 256), (128, 128, 128))
        if self.transform is not None:
            img = self.transform(img)
        return img, label, idx


def supcon_collate(batch):
    imgs1 = torch.stack([b[0][0] for b in batch])
    imgs2 = torch.stack([b[0][1] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    idxs = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return imgs1, imgs2, labels, idxs


def single_collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    idxs = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return imgs, labels, idxs


def build_supcon_transforms(img_size: int = 256):
    """Simaugment (SupCon paper CIFAR aug) at 256 px, ImageNet-normalized."""
    from torchvision import transforms as T

    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_fcmae_transforms(img_size: int = 256):
    """FCMAE aug: random resized crop (+ flip) only — per ConvNeXt V2 paper §2.2."""
    from torchvision import transforms as T

    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_eval_transforms(img_size: int = 256):
    from torchvision import transforms as T

    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------------

def build_backbone(arch: str, img_size: int = 256, pretrained: bool = True):
    """Build the timm encoder (num_classes=0 -> pooled feature vector).

    arch: ``convnext_t`` | ``swin_t`` | ``convnextv2_t`` | ``convnextv2_b``
    Returns ``(model, feat_dim)``.
    """
    import timm

    specs = {
        "convnext_t": ("timm/convnext_tiny.in12k_ft_in1k", 768, {}),
        "swin_t": ("swin_tiny_patch4_window7_224.ms_in1k", 768, {"img_size": img_size}),
        "convnextv2_t": ("convnextv2_tiny.fcmae", 768, {}),
        "convnextv2_b": ("convnextv2_base.fcmae", 1024, {}),
    }
    if arch not in specs:
        raise ValueError(f"unknown arch '{arch}'; expected one of {sorted(specs)}")
    tag, feat_dim, extra = specs[arch]
    model = timm.create_model(tag, pretrained=pretrained, num_classes=0, **extra)
    return model, feat_dim


def save_encoder_state(model: nn.Module, path: str | os.PathLike) -> None:
    """Save the bare backbone state_dict (drop-in format, see module docstring)."""
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# SupCon loss (Khosla et al., NeurIPS 2020) — canonical formulation
# ---------------------------------------------------------------------------

class SupConLoss(nn.Module):
    """Supervised contrastive loss over 2 stacked views.

    ``features``: (2N, d) L2-normalized projections; ``labels``: (2N,) —
    views of one image share a label. Implementation follows the official
    SupContrast repo (HobbitLong) reduced to the multi-view case.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        n = labels.shape[0] // 2
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        eye = torch.eye(2 * n, dtype=torch.bool, device=device)
        logits_mask = ~eye
        mask = mask * logits_mask.float()
        exp_logits = torch.exp(logits) * logits_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        pos_counts = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_counts.clamp(min=1)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        # mean over views then anchors (official: reshape [views, n].mean)
        return loss.view(2, n).mean()


# ---------------------------------------------------------------------------
# LR schedules
# ---------------------------------------------------------------------------

def warmup_cosine_lambda(step: int, *, warmup_steps: int, total_steps: int,
                         base: float = 1.0, min_factor: float = 0.0,
                         warmup_start_factor: float = 0.0) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return warmup_start_factor + (base - warmup_start_factor) * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_factor + (base - min_factor) * cos


# ---------------------------------------------------------------------------
# Checkpoint / run-state IO
# ---------------------------------------------------------------------------

def save_run_state(
    run_dir: str | os.PathLike,
    backbone: nn.Module,
    extras: nn.Module | None,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    meta: dict,
    save_epoch_freq: int = 5,
) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "best_metric": best_metric,
        "backbone": backbone.state_dict(),
        "extras": None if extras is None else extras.state_dict(),
        "optimizer": optimizer.state_dict(),
        "meta": meta,
    }
    tmp = run_dir / "ckpt_latest.pth.tmp"
    torch.save(state, tmp)
    os.replace(tmp, run_dir / "ckpt_latest.pth")
    save_encoder_state(backbone, run_dir / "tcga_encoder.pt")
    if epoch % max(1, save_epoch_freq) == 0:
        torch.save(state, run_dir / f"ckpt_epoch{epoch}.pth")
    (run_dir / "iter.txt").write_text(f"{epoch}\n")


def load_run_state(run_dir: str | os.PathLike, backbone: nn.Module,
                   extras: nn.Module | None, optimizer: torch.optim.Optimizer,
                   map_location: str = "cpu") -> Optional[dict]:
    ckpt = Path(run_dir) / "ckpt_latest.pth"
    if not ckpt.is_file():
        return None
    state = torch.load(ckpt, map_location=map_location, weights_only=False)
    backbone.load_state_dict(state["backbone"])
    if extras is not None and state.get("extras") is not None:
        extras.load_state_dict(state["extras"])
    optimizer.load_state_dict(state["optimizer"])
    return state


# ---------------------------------------------------------------------------
# kNN sanity metric
# ---------------------------------------------------------------------------

@torch.no_grad()
def knn_accuracy(feat_train: torch.Tensor, lab_train: torch.Tensor,
                 feat_val: torch.Tensor, lab_val: torch.Tensor,
                 k: int = 5, chunk: int = 512) -> float:
    """Cosine 5-NN accuracy (features must be L2-normalized, on same device)."""
    correct = 0
    total = feat_val.shape[0]
    for i in range(0, total, chunk):
        fv = feat_val[i:i + chunk]                       # (C, d)
        sim = fv @ feat_train.T                          # (C, M)
        topv, topi = sim.topk(k, dim=1)
        votes = lab_train[topi]                          # (C, k)
        pred = torch.mode(votes, dim=1).values
        correct += (pred == lab_val[i:i + chunk]).sum().item()
    return correct / max(1, total)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# SupCon collapse guards (projection head, WD split, ln(2N-1) kill switch)
# ---------------------------------------------------------------------------

def uniform_collapse_loss(batch_size: int) -> float:
    """SupCon loss when every embedding is identical: ln(2N - 1)."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    return math.log(2 * batch_size - 1)


def build_supcon_projection_head(
    feat_dim: int, hidden: int = 768, out_dim: int = 128
) -> nn.Sequential:
    """Linear -> SyncBatchNorm -> GELU -> Linear. Caller L2-normalizes outputs."""
    return nn.Sequential(
        nn.Linear(feat_dim, hidden),
        nn.SyncBatchNorm(hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


_NORM_TYPES = (
    nn.LayerNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
)


def split_decay_param_groups(
    modules: Sequence[nn.Module], weight_decay: float
) -> list[dict]:
    """AdamW/SGD groups: decay vs wd=0 for LN/BN/bias/LayerScale gamma."""
    norm_param_ids: set[int] = set()
    for root in modules:
        for mod in root.modules():
            if isinstance(mod, _NORM_TYPES):
                for p in mod.parameters(recurse=False):
                    norm_param_ids.add(id(p))

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    seen: set[int] = set()
    for root in modules:
        for name, param in root.named_parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))
            if not param.requires_grad:
                continue
            leaf = name.rsplit(".", 1)[-1]
            if (
                leaf == "bias"
                or leaf == "gamma"
                or id(param) in norm_param_ids
            ):
                no_decay.append(param)
            else:
                decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class CollapseKillSwitch:
    """Abort when train loss stays at ln(2N-1) for ``patience`` epochs after ``min_epoch``.

    ``epoch`` is 0-based. The switch arms once ``epoch >= min_epoch`` (after epoch 5
    when ``min_epoch=5``).
    """

    def __init__(
        self,
        batch_size: int,
        *,
        min_epoch: int = 5,
        patience: int = 2,
        atol: float = 1e-3,
    ):
        self.target = uniform_collapse_loss(batch_size)
        self.min_epoch = min_epoch
        self.patience = patience
        self.atol = atol
        self.streak = 0

    def is_collapsed(self, loss: float) -> bool:
        return abs(loss - self.target) <= self.atol

    def update(self, epoch: int, loss: float) -> None:
        if epoch < self.min_epoch:
            self.streak = 0
            return
        if self.is_collapsed(loss):
            self.streak += 1
        else:
            self.streak = 0
        if self.streak >= self.patience:
            raise RuntimeError(
                f"SupCon representation collapse: train loss {loss:.5f} ~ "
                f"ln(2*bs-1)={self.target:.5f} for {self.streak} consecutive "
                f"epochs after epoch {self.min_epoch}. Aborting."
            )
