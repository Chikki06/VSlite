"""Smoke tests for region tiling helpers."""

import torch

from cislnet.region_eval import tiled_generator_forward, validate_region_tiling


class _DummyGen(torch.nn.Module):
    def forward(self, x):
        return x[:, :3]


def test_validate_region_tiling():
    assert validate_region_tiling(2048, 512) == 4
    assert validate_region_tiling(2048, 1024) == 2
    assert validate_region_tiling(1024, 1024) == 1


def test_tiled_generator_forward_stitch():
    gen = _DummyGen()
    x = torch.arange(13 * 2048 * 2048, dtype=torch.float32).view(13, 2048, 2048)
    out = tiled_generator_forward(gen, x, tile_size=512)
    assert out.shape == (3, 2048, 2048)
    # DummyGen returns first 3 input channels per tile; stitched corners should match.
    assert torch.allclose(out[:, 0:512, 0:512], x[:3, 0:512, 0:512])
    assert torch.allclose(out[:, 1536:2048, 1536:2048], x[:3, 1536:2048, 1536:2048])
