"""GRPO: group-relative advantages and the clipped policy-gradient loss.

USER-IMPLEMENTED (RL core). Stubs only — the math, symbol-by-symbol, is in
docs/examples/rl-basics.html §4-§5; equation source: DeepSeekMath (Shao et
al. 2024) eq. 3.
"""

import torch

from mathrl.rollout import RolloutBatch


def grpo_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """A_i = (r_i - mean(group)) / (std(group) + eps).

    rewards, group_ids: (B*G,). Returns (B*G,) — one scalar per completion,
    later broadcast to its tokens. A group with identical rewards gets all-
    zero advantages (no signal — correct, don't "fix" it). Return a detached
    tensor: no gradient may flow through the baseline.
    """
    N = group_ids.max() + 1
    g = rewards.view(N, -1)

    stds = g.std(dim=-1, unbiased=False, keepdim=True) + eps
    return ((g - g.mean(dim=-1, keepdim=True)) / stds).view(-1)


def grpo_loss(
    model: torch.nn.Module,
    batch: RolloutBatch,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    kl_beta: float = 0.0,
    ref_model: torch.nn.Module | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Clipped surrogate loss over action-masked tokens.

    Sketch (you write it):
      1. logits = model(batch.tokens)  # (B*G, T, V); logits at t predict
         token t+1 — shift before gathering (rl-basics §7, off-by-one).
      2. logp = log_softmax gathered at the realized next tokens.
      3. ratio = exp(logp - logp_old) if batch.logp_old is not None else 1
         (single-step-per-batch case: loss reduces to -(A * logp)).
      4. per-token surrogate = min(ratio * A, clamp(ratio, 1-clip_eps,
         1+clip_eps) * A); A is the per-row advantage broadcast over T.
      5. Per-row mean over action_mask positions only, then mean over rows;
         negate (optimizers minimize).
      6. If kl_beta > 0: add kl_beta * KL(pi_theta || pi_ref) on the same
         masked positions, ref_model under no_grad.
    Returns (loss, stats) — stats at least: ratio_mean, clip_frac, kl,
    entropy (for the record.jsonl step line).
    """

    tokens = batch.tokens  # (R, T)
    logits = model(tokens)[:, :-1]  # (R, T-1, V)
    targets = tokens[:, 1:]
    mask = batch.action_mask[:, 1:].float()

    logprobs = torch.log_softmax(logits, dim=-1)
    logp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (R, T-1)

    A = advantages.detach().unsqueeze(-1)  # (R, 1)
    denom = mask.sum(-1).clamp_min(1.0)
    loss = -(((logp * A) * mask).sum(-1) / denom).mean()

    kl_val = 0.0
    if kl_beta > 0.0 and ref_model is not None:
        with torch.no_grad():
            ref_logp = (
                torch.log_softmax(ref_model(tokens)[:, :-1], dim=-1)
                .gather(-1, targets.unsqueeze(-1))
                .squeeze(-1)
            )
        d = ref_logp - logp
        kl = (((d.exp() - d - 1.0) * mask).sum(-1) / denom).mean()
        loss = loss + kl_beta * kl
        kl_val = kl.item()

    with torch.no_grad():
        ent = -(logprobs.exp() * logprobs).sum(-1)
        stats = {
            "ratio_mean": 1.0,
            "clip_frac": 0.0,
            "kl": kl_val,
            "entropy": ((ent * mask).sum() / mask.sum().clamp_min(1.0)).item(),
        }
    return loss, stats
