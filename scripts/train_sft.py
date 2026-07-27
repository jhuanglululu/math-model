"""SFT warmup training.

Usage (project conventions — only these flags):
    uv run scripts/train_sft.py --model tiny --training smoke --seed 0

A run is fully determined by (model, training, seed): seed everything, build the
model from the model variation, generate the training set from the training
variation (puzzles + sft_trace, skipping the held-out eval keys), then train
with AdamW + warmup/cosine, grad clipping, JSONL records, a tqdm display with
periodic held-out validation, and keep-3-best checkpointing. `--model tiny --training smoke` finishes on CPU in ~2 minutes.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from mathrl.checkpoint import Checkpointer
from mathrl.device import get_device, seed_everything
from mathrl.model import GPT, model_dtype
from mathrl.puzzles import canonical_key, generate_puzzle
from mathrl.records import RunRecord, TrainingProgress
from mathrl.sft_data import build_batch
from mathrl.tokenizer import MathTokenizer
from mathrl.traces import sft_trace
from mathrl.variations import get_model_variation, get_training_variation

EVAL_JSONL = Path("datasets/eval.jsonl")


def load_eval_keys() -> set[str]:
    """Canonical keys of the held-out eval puzzles, to exclude from training."""
    keys: set[str] = set()
    if not EVAL_JSONL.exists():
        print(
            f"WARNING: {EVAL_JSONL} not found (run from repo root?) — "
            "training will NOT exclude the held-out eval puzzles"
        )
        return keys
    from mathrl.puzzles import Puzzle

    for line in EVAL_JSONL.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        keys.add(canonical_key(Puzzle(numbers=rec["numbers"], target=rec["target"])))
    return keys


def generate_traces(n, rng, tv, tok, exclude_keys, max_len):
    """Generate n (tokens, loss_mask) traces, skipping eval-set puzzles and any
    trace that would not fit in max_len."""
    out = []
    while len(out) < n:
        puzzle = generate_puzzle(rng, tv.puzzle)
        if canonical_key(puzzle) in exclude_keys:
            continue
        tokens, mask = sft_trace(puzzle, rng, tv.trace, tv.env, tok)
        if len(tokens) > max_len:
            continue
        out.append((tokens, mask))
    return out


def lr_at(step: int, base_lr: float, warmup: int, total: int) -> float:
    """Linear warmup then cosine decay to zero."""
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def validate(model, val_inputs, val_labels, device, batch_size) -> float:
    model.eval()
    total_loss, total_tok = 0.0, 0
    for i in range(0, val_inputs.size(0), batch_size):
        xb = val_inputs[i : i + batch_size].to(device)
        yb = val_labels[i : i + batch_size].to(device)
        logits = model(xb)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), yb.reshape(-1), ignore_index=-100, reduction="sum"
        )
        n = int((yb != -100).sum().item())
        total_loss += float(loss.item())
        total_tok += n
    model.train()
    return total_loss / max(1, total_tok)


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
    tok = MathTokenizer()
    max_len = model_cfg.block_size

    model = GPT(model_cfg).to(device=device, dtype=dtype)
    model.compile()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tv.lr, weight_decay=tv.weight_decay, betas=(0.9, 0.95)
    )

    # --- data ---
    rng = random.Random(args.seed)
    eval_keys = load_eval_keys()
    print(f"generating {tv.samples} training traces (excluding {len(eval_keys)} eval keys)...")
    train_traces = generate_traces(tv.samples, rng, tv, tok, eval_keys, max_len)
    n_val = min(128, max(8, tv.samples // 10))
    val_traces = generate_traces(n_val, rng, tv, tok, eval_keys, max_len)
    train_inputs, train_labels = build_batch(train_traces, max_len)
    val_inputs, val_labels = build_batch(val_traces, max_len)
    print(f"train examples: {train_inputs.size(0)}, val examples: {val_inputs.size(0)}")

    # --- records + checkpoints ---
    record = RunRecord(
        args.model,
        args.training,
        args.seed,
        config={"model": model_cfg.model_dump(), "training": tv.model_dump()},
        baseline="base",
    )
    ckpt = Checkpointer(args.model, args.training, args.seed, keep_best=3, higher_is_better=False)

    log_interval = max(1, min(50, tv.eval_interval // 2))
    tokens_per_step = tv.batch_size * max_len
    tokens_seen = 0

    progress = TrainingProgress(args.model, args.training, tv.steps, epoch=1)
    start_time = time.time()
    n_train = train_inputs.size(0)

    for step in range(tv.steps):
        lr = lr_at(step, tv.lr, tv.warmup_steps, tv.steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        idx = torch.randint(0, n_train, (tv.batch_size,))
        xb = train_inputs[idx].to(device)
        yb = train_labels[idx].to(device)

        t0 = time.time()
        logits = model(xb)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), yb.reshape(-1), ignore_index=-100
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tv.grad_clip)
        optimizer.step()
        sec_per_step = time.time() - t0
        tokens_seen += tokens_per_step

        loss_v = float(loss.item())
        progress.step(loss=loss_v)

        if step % log_interval == 0 or step == tv.steps - 1:
            record.log_step(
                step,
                loss=round(loss_v, 5),
                lr=lr,
                grad_norm=round(float(grad_norm), 5),
                tokens_seen=tokens_seen,
                sec_per_step=round(sec_per_step, 5),
            )

        if step % tv.eval_interval == 0 or step == tv.steps - 1:
            val_loss = validate(model, val_inputs, val_labels, device, tv.batch_size)
            elapsed = time.time() - start_time
            progress.validation(step, elapsed, loss_v, val_loss)
            record.log_eval(step, val_loss=round(val_loss, 5), train_loss=round(loss_v, 5))
            ckpt.save(model, optimizer, step, metric=val_loss, config={"lr": lr})

    progress.close()
    print("done.")


if __name__ == "__main__":
    main()
