import os
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from SwinGAN.utils.dataset_vslite_i2i import VSLiteI2IDataset  # noqa: E402
from data.VS_dataset import VSDataset  # noqa: E402


def _write_patch(dataroot: str, name: str, *, ir_ch: int = 10, hw: int = 8) -> None:
    os.makedirs(os.path.join(dataroot, "ir"), exist_ok=True)
    os.makedirs(os.path.join(dataroot, "vis"), exist_ok=True)
    os.makedirs(os.path.join(dataroot, "he"), exist_ok=True)
    np.save(os.path.join(dataroot, "ir", name), np.zeros((hw, hw, ir_ch), dtype=np.float32))
    np.save(os.path.join(dataroot, "vis", name), np.zeros((hw, hw, 3), dtype=np.float32))
    np.save(os.path.join(dataroot, "he", name), np.zeros((hw, hw, 3), dtype=np.float32))


def _samples_in_names(names) -> set[str]:
    out: set[str] = set()
    for n in names:
        base = os.path.splitext(os.path.basename(str(n)))[0]
        out.add(base.split("_", 1)[0])
    return out


def test_test_txt_split_no_leakage_sample_10440_1():
    with tempfile.TemporaryDirectory() as td:
        # Create a tiny dataset with 2 samples, multiple patches each.
        patches = [
            "10440-1_y00000_x00000.npy",
            "10440-1_y00000_x01024.npy",
            "99999-9_y00000_x00000.npy",
            "99999-9_y00000_x01024.npy",
        ]
        for p in patches:
            _write_patch(td, p)

        # Define test samples.
        with open(os.path.join(td, "test.txt"), "w", encoding="utf-8") as f:
            # Prefix rule: should capture 10440-1 (and any 10440-*)
            f.write("10440\n")

        # New split behavior: VSLiteI2IDataset
        ds_test = VSLiteI2IDataset(td, split="test")
        ds_train = VSLiteI2IDataset(td, split="train")
        ds_val = VSLiteI2IDataset(td, split="val")

        test_samples = _samples_in_names([ds_test[i]["name"] for i in range(len(ds_test))])
        train_samples = _samples_in_names([ds_train[i]["name"] for i in range(len(ds_train))])
        val_samples = _samples_in_names([ds_val[i]["name"] for i in range(len(ds_val))])

        assert "10440-1" in test_samples
        assert "10440-1" not in train_samples
        assert "10440-1" not in val_samples

        # Baseline VSDataset respects opt.phase and excludes test samples from train.
        class _Opt:
            pass

        opt = _Opt()
        opt.dataroot = td
        opt.max_dataset_size = 10**9
        opt.input_nc = 10
        opt.num_ir_channels = 10
        opt.ir_channel_indices = ""
        opt.image_size = 8
        opt.test_list_path = os.path.join(td, "test.txt")

        opt.phase = "test"
        ds_base_test = VSDataset()
        ds_base_test.initialize(opt)
        assert len(ds_base_test) > 0
        base_test_samples = _samples_in_names(ds_base_test.list_IR)
        assert "10440-1" in base_test_samples
        assert "99999-9" not in base_test_samples

        opt.phase = "train"
        ds_base_train = VSDataset()
        ds_base_train.initialize(opt)
        base_train_samples = _samples_in_names(ds_base_train.list_IR)
        assert "10440-1" not in base_train_samples
        assert "99999-9" in base_train_samples

