import os
import numpy as np

from SwinGAN.utils.dataset_vslite_i2i import VSLiteI2IDataset, SplitFractions


def _write_pair(root, name, ir_ch=10):
    os.makedirs(os.path.join(root, "ir"), exist_ok=True)
    os.makedirs(os.path.join(root, "vis"), exist_ok=True)
    os.makedirs(os.path.join(root, "he"), exist_ok=True)
    ir = np.random.rand(32, 32, ir_ch).astype(np.float32) * 6.0  # some values >5 to test clip
    vis = (np.random.rand(32, 32, 3).astype(np.float32) * 255.0)
    he = (np.random.rand(32, 32, 3).astype(np.float32) * 255.0)
    np.save(os.path.join(root, "ir", name), ir)
    np.save(os.path.join(root, "vis", name), vis)
    np.save(os.path.join(root, "he", name), he)


def test_ir10_and_irvis13_shapes(tmp_path):
    _write_pair(str(tmp_path), "a.npy", ir_ch=10)
    ds10 = VSLiteI2IDataset(str(tmp_path), input_mode="ir10", img_size=16, he_scale="minus1_1")
    ex = ds10[0]
    assert ex["x"].shape[0] == 10
    assert ex["y"].shape[0] == 3
    assert ex["x"].shape[1:] == (16, 16)
    assert ex["y"].shape[1:] == (16, 16)
    assert ex["y"].min().item() >= -1.0001 and ex["y"].max().item() <= 1.0001

    ds13 = VSLiteI2IDataset(str(tmp_path), input_mode="irvis13", img_size=16, he_scale="0_1")
    ex2 = ds13[0]
    assert ex2["x"].shape[0] == 13
    assert ex2["y"].min().item() >= -1e-6 and ex2["y"].max().item() <= 1.0001


def test_deterministic_split(tmp_path):
    for i in range(30):
        _write_pair(str(tmp_path), f"{i:03d}.npy", ir_ch=10)
    fr = SplitFractions(0.8, 0.1, 0.1)
    d1 = VSLiteI2IDataset(str(tmp_path), split="train", split_seed=123, split_fracs=fr)
    d2 = VSLiteI2IDataset(str(tmp_path), split="train", split_seed=123, split_fracs=fr)
    assert len(d1) == len(d2)
    assert [d1[i]["name"] for i in range(min(5, len(d1)))] == [d2[i]["name"] for i in range(min(5, len(d2)))]

