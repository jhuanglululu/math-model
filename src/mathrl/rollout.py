"""Rollout: sample completions with mid-generation env intervention.

Batched implementation. All B*G rows generate together: each step runs ONE
forward over the still-alive rows (right-padded to the current max length)
instead of one forward per row per token.

Why right-padding is safe here: padding sits AFTER each row's last real
token, and attention is causal — position i attends only to positions <= i —
so the logits at a row's own last real index never see the pads. We gather
exactly those per-row logits (`logits[j, len_j - 1]`) to sample each row's
next token. Learned positional embeddings are likewise unaffected (real
tokens keep positions 0..len_j-1).

Rows desynchronize (env injections lengthen some rows, others terminate
early); that's handled by tracking per-row lengths and re-padding each step.
"""

from dataclasses import dataclass

import torch

from mathrl.checker import reward
from mathrl.config import EnvConfig, RewardConfig
from mathrl.env import env_step
from mathrl.puzzles import Puzzle, prompt_tokens
from mathrl.tokenizer import MathTokenizer


@dataclass
class RolloutBatch:
    """One GRPO batch: B puzzles x G completions each, flattened to B*G rows.

    tokens:      (B*G, T) long — prompt + completion, right-padded with <pad>.
    action_mask: (B*G, T) bool — True ONLY on model-authored completion
                 tokens. False on prompt, padding, and every env-injected
                 token (result digits, </calculate>, <good>, <bad>). This is
                 the mask the loss sums over — getting it wrong is the
                 project's #1 silent bug (see rl-basics §2, §7).
    rewards:     (B*G,) float — checker.reward() total per completion.
    group_ids:   (B*G,) long — row i belongs to group group_ids[i]; all
                 completions of one puzzle share an id (for the group
                 baseline).
    logp_old:    (B*G, T) float or None — chosen-token log-probs under the
                 sampling policy, detached. Only needed if you take more than
                 one gradient step per rollout batch (the PPO ratio); with
                 one step per batch the ratio is exactly 1 and you can leave
                 this None.
    """

    tokens: torch.Tensor
    action_mask: torch.Tensor
    rewards: torch.Tensor
    group_ids: torch.Tensor
    logp_old: torch.Tensor | None = None


NEG_INF = float("-inf")


def sample_next(
    logits: torch.Tensor,
    disabled: list[int],
    temperature: float,
) -> torch.Tensor:
    """Sample one next token per row. logits (N, V) -> (N,) long."""
    logits = logits.clone() / temperature
    logits[:, MathTokenizer.PAD] = NEG_INF
    logits[:, MathTokenizer.BOS] = NEG_INF
    for t in disabled:
        logits[:, t] = NEG_INF
    probs = torch.softmax(logits.float(), dim=-1)  # float(): multinomial-safe on cpu
    return torch.multinomial(probs, 1).squeeze(-1)


def disabled_tool_ids(tools: str) -> list[int]:
    calc = [MathTokenizer.CALCULATE, MathTokenizer.RESULT, MathTokenizer.END_CALCULATE]
    verify = [
        MathTokenizer.VERIFY,
        MathTokenizer.END_VERIFY,
        MathTokenizer.GOOD,
        MathTokenizer.BAD,
    ]
    if tools == "none":
        return calc + verify
    if tools == "calculate":
        return verify
    return calc


@torch.no_grad()
def rollout(
    model: torch.nn.Module,
    puzzles: list[Puzzle],
    tok: MathTokenizer,
    env_cfg: EnvConfig,
    reward_cfg: RewardConfig,
    group_size: int,
    device: torch.device,
    temperature: float = 1.0,
) -> RolloutBatch:
    """Sample group_size completions per puzzle (batched) and score them.

    Row layout is group-major: rows [i*G, (i+1)*G) belong to puzzles[i] —
    grpo_advantages relies on this ordering. env_step gets each row's
    COMPLETION-ONLY token list, per the env.py contract.
    """
    disabled = disabled_tool_ids(env_cfg.tools)
    n_rows = len(puzzles) * group_size

    was_training = model.training
    model.eval()

    row_puzzle = [p for p in puzzles for _ in range(group_size)]
    rows = [list(prompt_tokens(p, tok)) for p in row_puzzle]  # prompt + completion
    masks = [[False] * len(r) for r in rows]
    completions: list[list[int]] = [[] for _ in range(n_rows)]  # env_step's view
    reasons: list[str] = [""] * n_rows

    alive = list(range(n_rows))
    while alive:
        max_len = max(len(rows[i]) for i in alive)
        x = torch.full((len(alive), max_len), MathTokenizer.PAD, dtype=torch.long, device=device)
        for j, i in enumerate(alive):
            x[j, : len(rows[i])] = torch.tensor(rows[i], dtype=torch.long, device=device)
        # one forward for all alive rows; gather each row's own last real position
        logits = model(x)  # (n_alive, max_len, V)
        last = torch.tensor([len(rows[i]) - 1 for i in alive], device=device)
        nxt = sample_next(
            logits[torch.arange(len(alive), device=device), last], disabled, temperature
        )

        still_alive = []
        for j, i in enumerate(alive):
            t = int(nxt[j])
            rows[i].append(t)
            masks[i].append(True)
            completions[i].append(t)

            action = env_step(completions[i], row_puzzle[i], tok, env_cfg)
            if action.kind == "inject":
                completions[i].extend(action.tokens)
                rows[i].extend(action.tokens)
                masks[i].extend([False] * len(action.tokens))
                still_alive.append(i)
            elif action.kind == "terminate":
                reasons[i] = action.reason
            else:
                still_alive.append(i)
        alive = still_alive

    rewards = [
        reward(row_puzzle[i], completions[i], tok, reward_cfg, reasons[i]).total
        for i in range(n_rows)
    ]

    max_len = max(len(r) for r in rows)
    for row, mask in zip(rows, masks):
        pad = max_len - len(row)
        row += [MathTokenizer.PAD] * pad
        mask += [False] * pad

    if was_training:
        model.train()

    return RolloutBatch(
        tokens=torch.tensor(rows, dtype=torch.long, device=device),
        action_mask=torch.tensor(masks, dtype=torch.bool, device=device),
        rewards=torch.tensor(rewards, dtype=torch.float, device=device),
        group_ids=torch.arange(len(puzzles), dtype=torch.long, device=device).repeat_interleave(
            group_size
        ),
        logp_old=None,
    )
