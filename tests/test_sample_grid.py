"""Unit tests for pred|real pair sample saving."""

from __future__ import annotations

from pathlib import Path

import torch

from util.sample_grid import save_unified_sample


def test_save_unified_sample_writes_pred_left_real_right(tmp_path: Path) -> None:
    # Distinct solid colors so we can assert left vs right placement.
    pred = torch.zeros(3, 8, 8)
    pred[0] = 1.0  # red in [0,1]
    real = torch.zeros(3, 8, 8)
    real[2] = 1.0  # blue in [0,1]

    paths = save_unified_sample(str(tmp_path), "epoch001_step000000001", [real], [pred], pad=0)
    assert len(paths) == 1
    assert paths[0].endswith("epoch001_step000000001_p00.png")

    from PIL import Image
    import numpy as np

    arr = np.array(Image.open(paths[0]))
    assert arr.shape == (8, 16, 3)
    # Left half = pred (red), right half = real (blue)
    assert int(arr[4, 2, 0]) == 255 and int(arr[4, 2, 2]) == 0
    assert int(arr[4, 10, 2]) == 255 and int(arr[4, 10, 0]) == 0


def test_save_unified_sample_one_file_per_pair(tmp_path: Path) -> None:
    reals = [torch.zeros(3, 4, 4) for _ in range(3)]
    preds = [torch.ones(3, 4, 4) for _ in range(3)]
    paths = save_unified_sample(str(tmp_path), "epoch002_step000000010", reals, preds, pad=2)
    assert len(paths) == 3
    stems = sorted(Path(p).name for p in paths)
    assert stems == [
        "epoch002_step000000010_p00.png",
        "epoch002_step000000010_p01.png",
        "epoch002_step000000010_p02.png",
    ]
