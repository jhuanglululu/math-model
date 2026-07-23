"""GRPO training loop — USER-IMPLEMENTED (RL core stub).

Usage (per project conventions — no other flags):
    uv run scripts/train_rl.py --model small --training <rl-variation> --seed 0

Outline (you write it; every piece it wires already exists):
    1. Parse --model/--training/--seed; get variations; seed_everything().
    2. get_device(); build model; load the SFT checkpoint as the starting
       policy (and optionally a frozen copy as ref_model when kl_beta > 0).
    3. Loop: sample puzzles (skip eval-set keys) -> rollout() ->
       grpo_advantages() -> grpo_loss() -> backward, grad-clip, step.
    4. RunRecord step lines with the RL fields from the design doc
       (reward_mean, solve_rate, tool_use_rate, tool_calls_per_ep,
       manual_steps_per_ep, format_viol_rate, neg_prefix_rate, ...).
    5. Checkpoints with resume, keep-3-best by eval solve rate on
       datasets/eval.jsonl (greedy decode).
"""

raise NotImplementedError("RL core — user implements (see module docstring)")
