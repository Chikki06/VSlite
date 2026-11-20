"""PatchNCE separation diagnostics after mlp_sample pooling.

Tracks positive vs negative cosine similarities and top-1 NCE accuracy so a
short sanity train can tell whether the contrastive signal into G is weak.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

import torch


@dataclass(frozen=True)
class NceDiagStats:
    sim_pos: float
    sim_neg_mean: float
    sim_neg_hard: float
    margin: float
    margin_hard: float
    top1_acc: float

    def as_log_dict(self, prefix: str = "nce_") -> dict[str, float]:
        return {
            f"{prefix}sim_pos": self.sim_pos,
            f"{prefix}sim_neg_mean": self.sim_neg_mean,
            f"{prefix}sim_neg_hard": self.sim_neg_hard,
            f"{prefix}margin": self.margin,
            f"{prefix}margin_hard": self.margin_hard,
            f"{prefix}top1": self.top1_acc,
        }


def parse_int_list(raw: str | Sequence[int] | None, default: Sequence[int]) -> tuple[int, ...]:
    if raw is None or raw == "":
        return tuple(int(x) for x in default)
    if isinstance(raw, (list, tuple)):
        return tuple(int(x) for x in raw)
    parts = [p.strip() for p in str(raw).replace(":", ",").split(",") if p.strip()]
    return tuple(int(p) for p in parts)


def compute_patch_nce_diag(
    feat_q: torch.Tensor,
    feat_k: torch.Tensor,
    *,
    nce_fake_negatives: bool = True,
) -> NceDiagStats:
    """Cosine margins and top-1 accuracy for one layer of pooled patch features.

    ``feat_q`` / ``feat_k`` are ``[B, P, D]`` and assumed L2-normalized (as after
    ``mlp_sample``). Matches ``PatchNCELoss`` geometry: positives are matched
    pairs; negatives are other keys in the batch (plus queries when
    ``nce_fake_negatives``).
    """
    if feat_q.ndim != 3 or feat_k.ndim != 3:
        raise ValueError(f"expected [B,P,D] features, got {tuple(feat_q.shape)} / {tuple(feat_k.shape)}")
    if feat_q.shape != feat_k.shape:
        raise ValueError(f"feat_q/feat_k shape mismatch: {tuple(feat_q.shape)} vs {tuple(feat_k.shape)}")

    bsz, npatches, _dim = feat_q.shape
    # Positive cosine = dot product under L2 norm.
    sim_pos = (feat_q * feat_k).sum(dim=-1)  # [B, P]

    keys = feat_k
    if nce_fake_negatives:
        keys = torch.cat((feat_k, feat_q), dim=1)  # [B, 2P, D]

    sim_all = torch.bmm(feat_q, keys.transpose(2, 1))  # [B, P, K]
    eye = torch.eye(npatches, device=feat_q.device, dtype=torch.bool)[None, :, :]
    if nce_fake_negatives:
        pos_mask = torch.cat((eye, eye), dim=-1)
    else:
        pos_mask = eye

    sim_neg = sim_all.masked_fill(pos_mask, float("nan"))
    sim_neg_mean = torch.nanmean(sim_neg).item()
    # Hardest negative per query patch, then mean over patches/batch.
    sim_neg_hard = (
        torch.nan_to_num(sim_neg, nan=float("-inf")).amax(dim=-1).mean().item()
    )
    sim_pos_mean = sim_pos.mean().item()

    # Top-1: does the true positive beat all negatives? (pre-temperature logits)
    l_pos = sim_pos.reshape(-1, 1)
    l_neg = sim_all.masked_fill(pos_mask, -10.0).reshape(bsz * npatches, -1)
    logits = torch.cat((l_pos, l_neg), dim=1)
    top1 = (logits.argmax(dim=1) == 0).float().mean().item()

    return NceDiagStats(
        sim_pos=float(sim_pos_mean),
        sim_neg_mean=float(sim_neg_mean),
        sim_neg_hard=float(sim_neg_hard),
        margin=float(sim_pos_mean - sim_neg_mean),
        margin_hard=float(sim_pos_mean - sim_neg_hard),
        top1_acc=float(top1),
    )


def average_diag_stats(stats: Sequence[NceDiagStats]) -> NceDiagStats:
    if not stats:
        return NceDiagStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    n = float(len(stats))
    return NceDiagStats(
        sim_pos=sum(s.sim_pos for s in stats) / n,
        sim_neg_mean=sum(s.sim_neg_mean for s in stats) / n,
        sim_neg_hard=sum(s.sim_neg_hard for s in stats) / n,
        margin=sum(s.margin for s in stats) / n,
        margin_hard=sum(s.margin_hard for s in stats) / n,
        top1_acc=sum(s.top1_acc for s in stats) / n,
    )


def append_nce_diag_jsonl(
    path: str,
    *,
    step: int,
    epoch: int,
    stats: NceDiagStats,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    row = {"step": int(step), "epoch": int(epoch), **asdict(stats)}
    if extra:
        row.update(dict(extra))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def summarize_nce_diag_jsonl(
    path: str,
    *,
    margin_flat_eps: float = 0.02,
    top1_weak: float = 0.25,
) -> dict[str, Any]:
    """Read a diagnose JSONL and flag weak / flat contrastive signal."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return {"n": 0, "verdict": "empty", "weak_signal": True}

    margins = [float(r["margin"]) for r in rows]
    top1s = [float(r["top1_acc"]) for r in rows]
    mean_margin = sum(margins) / len(margins)
    mean_top1 = sum(top1s) / len(top1s)
    # Flat: last third within eps of first third mean.
    third = max(1, len(margins) // 3)
    early = sum(margins[:third]) / third
    late = sum(margins[-third:]) / third
    flat = abs(late - early) < margin_flat_eps
    weak = mean_margin < margin_flat_eps or mean_top1 < top1_weak or flat

    if mean_margin < 0.05 and mean_top1 < top1_weak:
        hint = "try_nce_t_0.05_or_deeper_blocks"
    elif mean_margin < 0.1:
        hint = "try_nce_t_0.10_0.15_and_blocks_4_10_16_22"
    else:
        hint = "margins_ok_still_sweep_t_and_blocks"

    return {
        "n": len(rows),
        "mean_margin": mean_margin,
        "mean_margin_hard": sum(float(r["margin_hard"]) for r in rows) / len(rows),
        "mean_top1": mean_top1,
        "mean_sim_pos": sum(float(r["sim_pos"]) for r in rows) / len(rows),
        "mean_sim_neg": sum(float(r["sim_neg_mean"]) for r in rows) / len(rows),
        "flat_margin": flat,
        "weak_signal": weak,
        "hint": hint,
        "verdict": "weak" if weak else "ok",
    }


def main_summarize(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Summarize nce_diag.jsonl from a sanity train")
    p.add_argument("jsonl", help="Path to nce_diag.jsonl")
    p.add_argument("--out", default="", help="Optional JSON summary path")
    args = p.parse_args(list(argv) if argv is not None else None)
    summary = summarize_nce_diag_jsonl(args.jsonl)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0 if not summary.get("weak_signal") else 0  # diagnose always succeeds


if __name__ == "__main__":
    raise SystemExit(main_summarize())
