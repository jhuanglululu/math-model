import torch

from mathrl.device import get_device, seed_everything


def test_get_device_returns_torch_device():
    dev = get_device()
    assert isinstance(dev, torch.device)


def test_cpu_only_machine():
    # This dev machine (WSL, CPU-only) must resolve to cpu.
    if not torch.cuda.is_available() and not (
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    ):
        assert get_device().type == "cpu"


def test_seed_determinism():
    seed_everything(123)
    a = torch.randn(1000)
    seed_everything(123)
    b = torch.randn(1000)
    assert torch.equal(a, b)
