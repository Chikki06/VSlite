import os
import subprocess
import sys
import importlib.util

import pytest


def test_train_i2i_runs_one_epoch_cpu(tmp_path):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")
    # Build minimal fake dataroot
    import numpy as np

    dr = tmp_path / "data"
    (dr / "ir").mkdir(parents=True)
    (dr / "vis").mkdir(parents=True)
    (dr / "he").mkdir(parents=True)
    ir = (np.random.rand(32, 32, 10).astype(np.float32) * 6.0)
    vis = (np.random.rand(32, 32, 3).astype(np.float32) * 255.0)
    he = (np.random.rand(32, 32, 3).astype(np.float32) * 255.0)
    np.save(dr / "ir" / "0.npy", ir)
    np.save(dr / "vis" / "0.npy", vis)
    np.save(dr / "he" / "0.npy", he)

    # Minimal config written on the fly
    cfgp = tmp_path / "cfg.yaml"
    cfgp.write_text(
        "\n".join(
            [
                f"dataroot: '{dr.as_posix()}'",
                "input_mode: 'ir10'",
                "num_ir_channels: 10",
                "img_size: 32",
                "he_scale: 'minus1_1'",
                "split: null",
                "split_seed: 1337",
                "split_fracs: [1.0, 0.0, 0.0]",
                "max_items: 1",
                "output_dir: '" + (tmp_path / "out").as_posix() + "'",
                "pretrained_swin_ckpt: ''",
                "patch_size: 4",
                "embed_dim: 24",
                "depths: [1,1,1,1]",
                "num_heads: [1,1,1,1]",
                "window_size: 4",
                "mlp_ratio: 4.0",
                "qkv_bias: true",
                "qk_scale: null",
                "drop_rate: 0.0",
                "drop_path_rate: 0.0",
                "ape: false",
                "patch_norm: true",
                "use_checkpoint: false",
                "use_cbam: false",
                "use_perceptual: false",
                "device: 'cpu'",
                "gpu_id: '0'",
                "epochs_n: 1",
                "batch_size: 1",
                "lr_g: 0.0002",
                "lr_d: 0.0002",
                "train_num_workers: 0",
                "val_num_workers: 0",
                "gan_training: true",
                "lambda_l1: 1.0",
                "lambda_gan: 1.0",
            ]
        )
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [sys.executable, "SwinGAN/train_i2i.py", "--config", str(cfgp)]
    res = subprocess.run(cmd, cwd=os.getcwd(), env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-2000:]

