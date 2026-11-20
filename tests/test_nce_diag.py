"""CPU unit tests for PatchNCE margin / top-1 diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from models.networks.nce_diag import (
    compute_patch_nce_diag,
    parse_int_list,
    summarize_nce_diag_jsonl,
)


def test_parse_int_list_accepts_commas_and_colons():
    assert parse_int_list("4,10,16,22", default=(2, 5, 8, 11)) == (4, 10, 16, 22)
    assert parse_int_list("2:5:8:11", default=(0,)) == (2, 5, 8, 11)
    assert parse_int_list(None, default=(2, 5, 8, 11)) == (2, 5, 8, 11)


def test_perfect_alignment_has_high_margin_and_top1():
    # Identical L2-normalized query/key → pos=1, other patches orthogonal-ish.
    torch.manual_seed(0)
    b, p, d = 2, 8, 16
    feat = torch.randn(b, p, d)
    feat = torch.nn.functional.normalize(feat, dim=-1)
    stats = compute_patch_nce_diag(feat, feat.clone(), nce_fake_negatives=True)
    assert stats.sim_pos > 0.99
    assert stats.margin > 0.5
    assert stats.top1_acc > 0.9


def test_identical_all_tokens_collapses_margin():
    # Every patch the same vector → pos and negs all ~1 (except masked diagonal).
    # Top-1 ties break toward index 0 (the positive), so only margin is asserted.
    feat = torch.nn.functional.normalize(torch.ones(1, 4, 8), dim=-1)
    stats = compute_patch_nce_diag(feat, feat.clone(), nce_fake_negatives=False)
    assert abs(stats.margin) < 0.05
    assert abs(stats.sim_pos - stats.sim_neg_mean) < 0.05


def test_summarize_flags_flat_weak(tmp_path: Path):
    path = tmp_path / "nce_diag.jsonl"
    rows = [
        {
            "step": i,
            "epoch": 1,
            "sim_pos": 0.2,
            "sim_neg_mean": 0.19,
            "sim_neg_hard": 0.21,
            "margin": 0.01,
            "margin_hard": -0.01,
            "top1_acc": 0.1,
        }
        for i in range(9)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    summary = summarize_nce_diag_jsonl(str(path))
    assert summary["weak_signal"] is True
    assert summary["verdict"] == "weak"
    assert summary["n"] == 9
