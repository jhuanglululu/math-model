"""GRPO training loop.

Usage (per project conventions — only these flags):
    uv run scripts/train_rl.py --model small --training rl_calc --seed 0

The loop, records, and checkpointing are wired; the RL core of each
step lives in ``rl_step()`` — USER-IMPLEMENTED, marked below. Everything it
needs already exists and is verified: rollout(), grpo_advantages(),
grpo_loss().

Run order: the policy initializes from the SFT checkpoint named by the
variation's ``init_from`` (same model + seed), so train_sft must have run
first. Pipe-clean locally with:
    uv run scripts/train_sft.py --model tiny --training smoke
    uv run scripts/train_rl.py  --model tiny --training rl_smoke
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_model

from mathrl.grpo import grpo_loss, grpo_advantages
from mathrl.rollout import rollout
from mathrl.checkpoint import Checkpointer, run_dir
from mathrl.device import get_device, seed_everything
from mathrl.model import GPT, model_dtype
from mathrl.puzzles import Puzzle, canonical_key, generate_puzzle
from mathrl.records import RunRecord, TrainingProgress
from mathrl.tokenizer import MathTokenizer
from mathrl.variations import get_model_variation, get_training_variation

EVAL_JSONL = Path("datasets/eval.jsonl")


def load_eval_keys() -> set[str]:
    """Canonical keys of the held-out eval puzzles, to exclude from training."""
    keys: set[str] = set()
    if not EVAL_JSONL.exists():
        print(
            f"WARNING: {EVAL_JSONL} not found (run from repo root?) — "
            "RL puzzles will NOT exclude the held-out eval set"
        )
        return keys
    for line in EVAL_JSONL.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            keys.add(canonical_key(Puzzle(numbers=rec["numbers"], target=rec["target"])))
    return keys


def sample_puzzles(n: int, rng: random.Random, cfg, exclude_keys: set[str]) -> list[Puzzle]:
    out: list[Puzzle] = []
    while len(out) < n:
        p = generate_puzzle(rng, cfg)
        if canonical_key(p) not in exclude_keys:
            out.append(p)
    return out


def lr_at(step: int, base_lr: float, warmup: int, total: int) -> float:
    """Linear warmup then cosine decay to zero (same schedule as SFT)."""
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


# ========================================================================== #
# YOUR PART — the RL core of one training step.
# ========================================================================== #
def rl_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    puzzles: list[Puzzle],
    tok: MathTokenizer,
    tv,  # TrainingVariation: group_size, clip_eps, kl_beta, env, reward, grad_clip
    device: torch.device,
    ref_model: torch.nn.Module | None,
) -> dict[str, float]:
    """One GRPO update: rollout -> advantages -> loss -> backward -> clip -> step.

    Must return a flat stats dict for the record.jsonl step line. Required
    keys (the loop logs whatever you return, but these feed the progress
    display and best-checkpoint metric):
        reward_mean, reward_std, loss, grad_norm
    plus whatever grpo_loss's stats dict gives you (entropy, kl, ...).
    Useful extras when you get to them: solve_rate, tool_use_rate — needs
    RolloutBatch to carry reward reasons/tool_calls (your call).

    NOTE: take exactly ONE optimizer step per rollout batch here (logp_old is
    None — reusing a batch for several steps without the PPO ratio is
    off-policy and silently wrong).
    """

    rb = rollout(model, puzzles, tok, tv.env, tv.reward, tv.group_size, device)
    ad = grpo_advantages(rb.rewards, rb.group_ids)
    loss, stats = grpo_loss(model, rb, ad, tv.clip_eps, tv.kl_beta, ref_model)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tv.grad_clip)
    optimizer.step()

    n = len(rb.reasons)
    reason_counts = Counter(rb.reasons)
    return {
        "reward_mean": float(rb.rewards.mean()),
        "reward_std": float(rb.rewards.std()),
        "loss": float(loss),
        "grad_norm": float(grad_norm),
        "solve_rate": reason_counts.get("correct", 0) / n,
        "tool_use_rate": sum(1 for c in rb.tool_calls if c > 0) / n,
        "tool_calls_per_ep": sum(rb.tool_calls) / n,
        # flat per-reason fractions, e.g. reason_correct, reason_too_long ...
        **{f"reason_{r}": c / n for r, c in sorted(reason_counts.items())},
        **stats,
    }


# ========================================================================== #


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--training", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    dtype = model_dtype(device)
    print(f"device: {device}, dtype: {dtype}")

    model_cfg = get_model_variation(args.model)
    tv = get_training_variation(args.training)
    if not tv.init_from:
        raise SystemExit(f"training variation {tv.name!r} has no init_from — not an RL recipe?")
    tok = MathTokenizer()

    # --- policy, initialized from the SFT checkpoint (same model + seed) ---
    model = GPT(model_cfg).to(device=device, dtype=dtype)
    sft_ckpt = run_dir(args.model, tv.init_from, args.seed) / "current.safetensors"
    if not sft_ckpt.exists():
        raise SystemExit(
            f"no SFT checkpoint at {sft_ckpt} — run "
            f"`uv run scripts/train_sft.py --model {args.model} "
            f"--training {tv.init_from} --seed {args.seed}` first"
        )
    load_model(model, str(sft_ckpt))
    print(f"policy initialized from {sft_ckpt}")

    # Dropout must be OFF for RL: sampling (rollout) and scoring (grpo_loss)
    # must see the same distribution, and `small` has dropout=0.1. eval mode
    # does not block gradients — this is correct for the loss pass too.
    model.eval()

    # frozen reference for the optional KL penalty
    ref_model = None
    if tv.kl_beta > 0.0:
        ref_model = copy.deepcopy(model)
        ref_model.requires_grad_(False)
        ref_model.eval()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tv.lr, weight_decay=tv.weight_decay, betas=(0.9, 0.95)
    )

    # --- data / records / checkpoints ---
    rng = random.Random(args.seed)
    eval_keys = load_eval_keys()
    record = RunRecord(
        args.model,
        args.training,
        args.seed,
        config={"model": model_cfg.model_dump(), "training": tv.model_dump()},
        baseline=tv.init_from,
    )
    ckpt = Checkpointer(args.model, args.training, args.seed, keep_best=3, higher_is_better=True)

    progress = TrainingProgress(args.model, args.training, tv.steps, epoch=1)
    start_time = time.time()
    reward_window: list[float] = []  # rolling reward_mean since last checkpoint

    for step in range(tv.steps):
        lr = lr_at(step, tv.lr, tv.warmup_steps, tv.steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        puzzles = sample_puzzles(tv.rollout_batch, rng, tv.puzzle, eval_keys)

        t0 = time.time()
        stats = rl_step(model, optimizer, puzzles, tok, tv, device, ref_model)
        sec_per_step = time.time() - t0

        reward_window.append(stats.get("reward_mean", 0.0))
        progress.step(loss=stats.get("loss", 0.0))
        record.log_step(
            step,
            lr=lr,
            sec_per_step=round(sec_per_step, 4),
            episodes=len(puzzles) * tv.group_size,
            **{k: round(float(v), 5) for k, v in stats.items()},
        )

        if step % tv.eval_interval == 0 or step == tv.steps - 1:
            window_mean = sum(reward_window) / max(1, len(reward_window))
            reward_window.clear()
            elapsed = time.time() - start_time
            mm, ss = divmod(int(elapsed), 60)
            progress.bar.write(
                f"step {step:>5}/{tv.steps} | {mm:02d}:{ss:02d} | "
                f"reward {stats.get('reward_mean', 0.0):+6.3f} | "
                f"window {window_mean:+6.3f} | "
                f"solve {stats.get('solve_rate', 0.0):6.2%} | "
                f"entropy {stats.get('entropy', 0.0):6.3f}"
            )
            record.log_eval(step, reward_window_mean=round(window_mean, 5))
            # best-checkpoint metric: rolling reward mean (run scripts/eval.py
            # offline for the true greedy solve_rate)
            ckpt.save(model, optimizer, step, metric=window_mean, config={"lr": lr})

    progress.close()
    print("done. offline eval:")
    print(f"  uv run scripts/eval.py --model {args.model} --training {args.training} --n 200")


if __name__ == "__main__":
    main()
