import pytest


torch = pytest.importorskip("torch")


from util.batch_device import move_to_device


def test_move_to_device_preserves_structure_and_moves_tensors():
    batch = {
        "input": torch.zeros(2, 3),
        "target": torch.ones(2, 3),
        "meta": {"name": "x", "idx": 7},
        "seq": [torch.arange(3), "y"],
        "tpl": (torch.randn(1), 5),
    }
    out = move_to_device(batch, torch.device("cpu"))

    assert isinstance(out, dict)
    assert out["meta"] == {"name": "x", "idx": 7}
    assert torch.is_tensor(out["input"]) and out["input"].device.type == "cpu"
    assert torch.is_tensor(out["target"]) and out["target"].device.type == "cpu"
    assert torch.is_tensor(out["seq"][0]) and out["seq"][0].device.type == "cpu"
    assert out["seq"][1] == "y"
    assert torch.is_tensor(out["tpl"][0]) and out["tpl"][0].device.type == "cpu"
    assert out["tpl"][1] == 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_move_to_device_prevents_cpu_gpu_mismatch_in_model_forward():
    class _Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 1, 1)

        def forward(self, batch):
            # mimic contrastive_pix2pix_model.preprocess_input usage
            x = batch["input"]
            return self.conv(x)

    m = _Tiny().cuda().eval()
    batch_cpu = {"input": torch.randn(1, 1, 4, 4), "target": torch.randn(1, 1, 4, 4)}

    with pytest.raises(RuntimeError):
        _ = m(batch_cpu)

    batch_gpu = move_to_device(batch_cpu, next(m.parameters()).device)
    y = m(batch_gpu)
    assert torch.is_tensor(y)
    assert y.device.type == "cuda"

