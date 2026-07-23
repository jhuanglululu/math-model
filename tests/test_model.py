import torch

from mathrl.model import GPT, GPTConfig
from mathrl.variations import get_model_variation


def _tiny_config() -> GPTConfig:
    return get_model_variation("tiny")


def test_forward_shape():
    cfg = _tiny_config()
    model = GPT(cfg)
    B, T = 3, 32
    assert T < cfg.block_size
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    logits = model(idx)
    assert logits.shape == (B, T, cfg.vocab_size)


def test_causality():
    # Changing a future token must not change logits at earlier positions.
    cfg = _tiny_config()
    model = GPT(cfg)
    model.eval()
    T = 16
    idx = torch.randint(0, cfg.vocab_size, (1, T))
    with torch.no_grad():
        base = model(idx)
        idx2 = idx.clone()
        # perturb the last token (a strictly future position for all t < T-1)
        idx2[0, -1] = (idx2[0, -1] + 1) % cfg.vocab_size
        perturbed = model(idx2)
    # positions strictly before the changed token are unaffected
    assert torch.allclose(base[:, :-1, :], perturbed[:, :-1, :], atol=1e-6)
    # sanity: the changed position itself does differ
    assert not torch.allclose(base[:, -1, :], perturbed[:, -1, :])


def test_param_count_tiny():
    cfg = _tiny_config()
    model = GPT(cfg)
    n = model.num_params()
    assert 0.5e6 <= n <= 2e6, f"tiny has {n} params, expected 0.5M-2M"
