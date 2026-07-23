"""Rollout: sample completions with mid-generation env intervention.

USER-IMPLEMENTED (RL core). Stubs only — see docs/designs/token-protocol.md
and docs/examples/rl-basics.html. The scaffolded pieces you build on:
`env.env_step()` (token injection / termination), `checker.reward()`,
`puzzles.prompt_tokens()`.
"""

from mathrl.puzzles import prompt_tokens
from dataclasses import dataclass

import torch

from mathrl.config import EnvConfig, RewardConfig
from mathrl.puzzles import Puzzle
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
    """Sample group_size completions per puzzle and score them.

    Sketch (you write it):
      1. Build prompts with prompt_tokens(); replicate each G times.
      2. Autoregressive sampling loop under torch.no_grad(). After EVERY
         sampled token call env_step(completion_so_far, puzzle, tok, env_cfg)
         — completion_so_far is COMPLETION-ONLY (no prompt tokens; the
         prompt already ends with <reasoning>). env.py's length cap and
         reasoning-open logic depend on this:
           - CONTINUE: keep sampling.
           - INJECT(toks): append toks (mark them env-authored -> action_mask
             False), keep sampling after them.
           - TERMINATE(reason): stop this row; reason feeds reward().
      3. Hard-mask disabled-arm tool token logits to -inf before softmax
         (env_cfg.tools) — plus <pad>/<bos>, which are never valid actions.
      4. reward(puzzle, completion, tok, reward_cfg, terminated=reason).
      5. Right-pad rows to one T, assemble tensors.
    Sequences that hit max_completion_len without <eos> are terminated by
    env_step ('too_long'); do not silently truncate.
    """

    tokens = torch.Tensor([prompt_tokens(p, tok) for p in puzzles])
    tokens = tokens.repeat_interleave(tokens, group_size)

    raise NotImplementedError("RL core — user implements (see docstring sketch)")
