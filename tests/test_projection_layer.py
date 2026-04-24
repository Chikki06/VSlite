import pytest

torch = pytest.importorskip("torch")

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


def test_projection_maps_channels_and_grads():
    cfg = _Cfg()
    m10 = ProjectedSwinUnet(cfg, in_chans=10, out_chans=3, use_cbam=False)
    x10 = torch.randn(2, 10, cfg.img_size, cfg.img_size, requires_grad=True)
    y10 = torch.tanh(m10(x10))
    assert y10.shape == (2, 3, cfg.img_size, cfg.img_size)
    y10.mean().backward()
    assert m10.proj.weight.grad is not None

    m13 = ProjectedSwinUnet(cfg, in_chans=13, out_chans=3, use_cbam=False)
    x13 = torch.randn(1, 13, cfg.img_size, cfg.img_size, requires_grad=True)
    y13 = torch.tanh(m13(x13))
    assert y13.shape[1] == 3

