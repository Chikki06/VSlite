import torch

from SwinGAN.Networks.cbam import CBAM


def test_cbam_shape_invariance():
    m = CBAM(3)
    x = torch.randn(4, 3, 16, 16)
    y = m(x)
    assert y.shape == x.shape

