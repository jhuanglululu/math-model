import torch
from safetensors.torch import load_model

from mathrl.checkpoint import Checkpointer, export, resume
from mathrl.device import seed_everything
from mathrl.model import GPT, GPTConfig


def _tiny_model() -> GPT:
    # small, dropout-free net so behaviour is deterministic
    return GPT(GPTConfig(vocab_size=28, block_size=64, n_layer=2, n_head=2, n_embd=32))


def _train_step(model: GPT, opt: torch.optim.Optimizer) -> None:
    # draw a batch from the global RNG so RNG restore is exercised
    idx = torch.randint(0, model.config.vocab_size, (4, 16))
    logits = model(idx)
    targets = torch.randint(0, model.config.vocab_size, (4, 16))
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
    )
    opt.zero_grad()
    loss.backward()
    opt.step()


def _snapshot(model: GPT) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def test_save_resume_step_equivalence(tmp_path):
    # --- uninterrupted run ---
    seed_everything(0)
    model_a = _tiny_model()
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
    _train_step(model_a, opt_a)  # step 1

    ckpt = Checkpointer("tiny", "smoke", 0, root=tmp_path)
    ckpt.save(model_a, opt_a, step=1, metric=0.5, config={"lr": 1e-3})
    weights_at_save = _snapshot(model_a)
    # optimizer moment at save
    moment_at_save = opt_a.state_dict()["state"][0]["exp_avg"].clone()

    _train_step(model_a, opt_a)  # step 2, uninterrupted
    uninterrupted = _snapshot(model_a)

    # --- resumed run ---
    model_b = _tiny_model()  # different init; weights will be overwritten
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3)
    step = resume(ckpt.dir, model_b, opt_b)
    assert step == 1

    # weights restored exactly
    restored = _snapshot(model_b)
    for k in weights_at_save:
        assert torch.allclose(restored[k], weights_at_save[k]), f"weight {k} mismatch"
    # optimizer state restored
    assert torch.allclose(opt_b.state_dict()["state"][0]["exp_avg"], moment_at_save)

    _train_step(model_b, opt_b)  # step 2 after resume
    resumed = _snapshot(model_b)

    # train-step-equivalence: one step after resume matches uninterrupted
    for k in uninterrupted:
        assert torch.allclose(uninterrupted[k], resumed[k], atol=1e-6), (
            f"step-equivalence failed for {k}"
        )


def test_export_weights_only(tmp_path):
    seed_everything(1)
    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt = Checkpointer("tiny", "smoke", 1, root=tmp_path)
    ckpt.save(model, opt, step=5, metric=0.1)

    out = export(ckpt.dir)
    assert out.exists() and out.name == "model.safetensors"

    fresh = _tiny_model()
    load_model(fresh, str(out))
    for k, v in model.state_dict().items():
        assert torch.allclose(fresh.state_dict()[k], v)


def test_keep_three_best(tmp_path):
    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt = Checkpointer("tiny", "smoke", 2, root=tmp_path, keep_best=3, higher_is_better=False)
    # lower metric is better
    for step, metric in [(10, 1.0), (20, 0.5), (30, 0.9), (40, 0.8), (50, 0.7)]:
        ckpt.save(model, opt, step=step, metric=metric)

    def exists(step: int) -> bool:
        return (ckpt.dir / f"{step}.safetensors").exists()

    # best three metrics (0.5, 0.7, 0.8) -> steps 20, 50, 40; 50 is also latest
    assert exists(20) and exists(40) and exists(50)
    assert not exists(10) and not exists(30)
    # current.* always maintained
    assert (ckpt.dir / "current.safetensors").exists()
    assert (ckpt.dir / "current.json").exists()
