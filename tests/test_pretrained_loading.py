import os
import tempfile

import torch

from SwinGAN.Networks.vision_transformer import ProjectedSwinUnet


class _Cfg:
    img_size = 32
    patch_size = 4
    embed_dim = 24
    depths = [1, 1, 1, 1]
    num_heads = [1, 1, 1, 1]
    window_size = 4
    mlp_ratio = 4.0
    qkv_bias = True
    qk_scale = None
    drop_rate = 0.0
    drop_path_rate = 0.0
    ape = False
    patch_norm = True
    use_checkpoint = False


def test_pretrained_loader_ignores_projection_and_mismatched_shapes():
    cfg = _Cfg()
    m = ProjectedSwinUnet(cfg, in_chans=10, out_chans=3, use_cbam=False)
    before = m.proj.weight.detach().clone()

    # Build a fake checkpoint that matches one backbone key and mismatches another
    sd = m.swin_unet.state_dict()
    keys = list(sd.keys())
    good_k = keys[0]
    fake = {
        "model": {
            good_k: sd[good_k].clone(),
            # mismatched shape tensor
            keys[1]: torch.randn(1),
        }
    }
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "ckpt.pth")
        torch.save(fake, p)
        m.load_pretrained_backbone(p)

    after = m.proj.weight.detach().clone()
    assert torch.allclose(before, after)

