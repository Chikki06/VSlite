"""Guards against SupCon uniform-embedding collapse (ln(2N-1) plateau)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from util.tcga_ut import (
    CollapseKillSwitch,
    build_supcon_projection_head,
    split_decay_param_groups,
    uniform_collapse_loss,
)


def test_uniform_collapse_loss_bs512_is_ln1023():
    assert uniform_collapse_loss(512) == pytest.approx(math.log(1023))
    assert uniform_collapse_loss(512) == pytest.approx(6.930494765951626)


def test_kill_switch_ignores_collapse_before_epoch_5():
    ks = CollapseKillSwitch(512, min_epoch=5, patience=2)
    collapsed = uniform_collapse_loss(512)
    ks.update(3, collapsed)
    ks.update(4, collapsed)
    assert ks.streak == 0


def test_kill_switch_aborts_after_two_collapsed_epochs_past_min():
    ks = CollapseKillSwitch(512, min_epoch=5, patience=2)
    collapsed = 6.93049
    ks.update(5, collapsed)
    assert ks.streak == 1
    with pytest.raises(RuntimeError, match="representation collapse"):
        ks.update(6, collapsed)


def test_kill_switch_resets_on_noncollapsed_loss():
    ks = CollapseKillSwitch(512, min_epoch=5, patience=2)
    ks.update(5, 6.93049)
    ks.update(6, 5.0)
    assert ks.streak == 0
    ks.update(7, 6.93049)
    assert ks.streak == 1


def test_projection_head_syncbn_gelu_128():
    head = build_supcon_projection_head(768, hidden=768, out_dim=128)
    assert isinstance(head[0], nn.Linear)
    assert head[0].in_features == 768 and head[0].out_features == 768
    assert isinstance(head[1], nn.SyncBatchNorm)
    assert head[1].num_features == 768
    assert isinstance(head[2], nn.GELU)
    assert isinstance(head[3], nn.Linear)
    assert head[3].out_features == 128
    y = head(torch.randn(4, 768))
    assert y.shape == (4, 128)


def test_split_decay_exempts_syncbn_and_gamma():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 8)
            self.bn = nn.SyncBatchNorm(8)
            self.gamma = nn.Parameter(torch.ones(8))

    m = Toy()
    groups = split_decay_param_groups([m], weight_decay=0.05)
    assert len(groups) == 2
    decay, no_decay = groups
    assert decay["weight_decay"] == 0.05
    assert no_decay["weight_decay"] == 0.0
    no_decay_ids = {id(p) for p in no_decay["params"]}
    assert id(m.fc.bias) in no_decay_ids
    assert id(m.bn.weight) in no_decay_ids
    assert id(m.bn.bias) in no_decay_ids
    assert id(m.gamma) in no_decay_ids
    assert id(m.fc.weight) not in no_decay_ids
