"""Evaluate or sample a checkpoint with env-in-the-loop decoding.

The held-out puzzles are generated on the fly from the seeded eval stream
(puzzles.EVAL_SEED + the variation's PuzzleConfig) — the same stream whose
keys training excludes, so eval stays honest with no file on disk.

Usage:
    uv run scripts/eval.py --model small --training rl_calc            # greedy, 1000 puzzles
    uv run scripts/eval.py --model small --training rl_calc --n 200 --show-cases
    uv run scripts/eval.py --model small --training rl_calc --sample --n 5 --show-cases

Greedy (default) is the comparable metric — record.jsonl eval lines are only
appended for greedy runs. `--sample` decodes with top-k/top-p/repetition
penalty instead (qualitative inspection; pair it with --show-cases).
`--model`/`--training` are required: architecture and tool arm cannot be
inferred from a bare weights file. Without `--checkpoint`, the run's
current.safetensors is used.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_model
from tqdm import tqdm

from mathrl.checker import CORRECT, NEG_PREFIX, reward
from mathrl.checkpoint import run_dir
from mathrl.device import get_device, seed_everything
from mathrl.env import (
    DISABLED_TOOL,
    DONE,
    MALFORMED_TOOL,
    MAX_TOOL_CALLS,
    NEG_PREFIX_CALC,
    NESTED_TOOL,
    TOO_LONG,
    TOOL_OUT_OF_REASONING,
    env_step,
)
from mathrl.model import GPT, model_dtype
from mathrl.puzzles import EVAL_N, eval_puzzles, prompt_tokens
from mathrl.records import record_dir
from mathrl.tokenizer import MathTokenizer
from mathrl.variations import get_model_variation, get_training_variation

NEG_INF = float("-inf")

FORMAT_REASONS = frozenset(
    {
        MALFORMED_TOOL,
        TOO_LONG,
        MAX_TOOL_CALLS,
        TOOL_OUT_OF_REASONING,
        NESTED_TOOL,
        NEG_PREFIX_CALC,
        DISABLED_TOOL,
        "malformed",
        "no_reasoning",
        "no_eos",
    }
)


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
    return calc  # verify arm disables calculate


def pick_token(
    logits: torch.Tensor,
    generated: list[int],
    disabled: list[int],
    sampling: dict | None,
) -> int:
    """Greedy argmax when sampling is None, else top-k/top-p/rep-pen sampling."""
    logits = logits.clone()
    logits[MathTokenizer.PAD] = NEG_INF
    logits[MathTokenizer.BOS] = NEG_INF
    for t in disabled:
        logits[t] = NEG_INF
    if sampling is None:
        return int(torch.argmax(logits).item())

    if sampling["rep_pen"] != 1.0:
        for t in set(generated):
            if logits[t] > 0:
                logits[t] = logits[t] / sampling["rep_pen"]
            else:
                logits[t] = logits[t] * sampling["rep_pen"]
    if sampling["top_k"] > 0:
        k = min(sampling["top_k"], logits.numel())
        kth = torch.topk(logits, k).values[-1]
        logits[logits < kth] = NEG_INF
    probs = torch.softmax(logits.float(), dim=-1)
    if sampling["top_p"] < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        drop = cum - sorted_probs > sampling["top_p"]
        sorted_probs[drop] = 0.0
        probs = torch.zeros_like(probs).scatter(0, sorted_idx, sorted_probs)
    total = probs.sum()
    if total <= 0:
        return int(torch.argmax(logits).item())
    return int(torch.multinomial(probs / total, 1).item())


@torch.no_grad()
def episode(model, puzzle, tok, env_cfg, device, max_len, sampling=None):
    """Env-in-the-loop decode. Returns (completion, reason, model_token_count)."""
    disabled = disabled_tool_ids(env_cfg.tools)
    prompt = prompt_tokens(puzzle, tok)
    completion: list[int] = []
    model_tokens = 0
    reason = DONE
    hard_cap = env_cfg.max_completion_len + 8
    while True:
        context = (prompt + completion)[-max_len:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)[0, -1]
        nxt = pick_token(logits, completion, disabled, sampling)
        completion.append(nxt)
        model_tokens += 1
        action = env_step(completion, puzzle, tok, env_cfg)
        if action.kind == "inject":
            completion.extend(action.tokens)
        elif action.kind == "terminate":
            reason = action.reason
            break
        if len(completion) >= hard_cap:
            reason = TOO_LONG
            break
    return completion, reason, model_tokens


def count_tool_calls(completion: list[int]) -> int:
    return sum(1 for t in completion if t in (MathTokenizer.CALCULATE, MathTokenizer.VERIFY))


def count_manual_steps(completion: list[int]) -> int:
    # '=' only appears in model-authored manual steps, inside the reasoning block.
    if MathTokenizer.END_REASONING in completion:
        reasoning = completion[: completion.index(MathTokenizer.END_REASONING)]
    else:
        reasoning = completion
    return sum(1 for t in reasoning if t == MathTokenizer.EQUALS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--training", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default=None, help="explicit weights file or run dir")
    ap.add_argument("--n", type=int, default=EVAL_N, help="number of eval puzzles")
    ap.add_argument(
        "--show-cases",
        action="store_true",
        help="print each puzzle's completion/verdict (default: summary only)",
    )
    ap.add_argument("--sample", action="store_true", help="sample instead of greedy decode")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--rep-pen", type=float, default=1.1)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    model_cfg = get_model_variation(args.model)
    tv = get_training_variation(args.training)
    tok = MathTokenizer()
    max_len = model_cfg.block_size
    sampling = (
        {"top_k": args.top_k, "top_p": args.top_p, "rep_pen": args.rep_pen} if args.sample else None
    )

    if args.checkpoint is not None:
        p = Path(args.checkpoint)
        weights = p / "current.safetensors" if p.is_dir() else p
    else:
        weights = run_dir(args.model, args.training, args.seed) / "current.safetensors"
    if not weights.exists():
        raise FileNotFoundError(f"no checkpoint at {weights}")

    model = GPT(model_cfg).to(device=device, dtype=model_dtype(device))
    load_model(model, str(weights))
    model.eval()

    step = 0
    sidecar = weights.with_name(weights.name.replace(".safetensors", ".json"))
    if sidecar.exists():
        step = int(json.loads(sidecar.read_text()).get("step", 0))

    if args.n > EVAL_N:
        print(
            f"WARNING: --n {args.n} > {EVAL_N}; training only excluded the "
            f"first {EVAL_N} eval puzzles — the extra ones may overlap training data"
        )
    puzzles = eval_puzzles(tv.puzzle, args.n)
    n = len(puzzles)
    mode = "sampled" if sampling else "greedy"
    print(f"evaluating {n} puzzles ({mode}) on {weights} (arm={tv.env.tools})...")

    solved = fmt_viol = neg_prefix = tool_use = 0
    tool_calls_sum = manual_sum = len_sum = reward_sum = 0.0
    reason_counts: Counter[str] = Counter()
    model_tokens = 0
    t_start = time.time()

    bar = tqdm(puzzles, desc=f"eval {args.model}/{args.training}", unit="puzzle")
    for i, puzzle in enumerate(bar):
        completion, reason, n_tok = episode(model, puzzle, tok, tv.env, device, max_len, sampling)
        model_tokens += n_tok
        rb = reward(puzzle, completion, tok, tv.reward, terminated=reason)
        reason_counts[rb.reason] += 1
        if rb.reason == CORRECT:
            solved += 1
        if rb.reason in FORMAT_REASONS:
            fmt_viol += 1
        if rb.reason == NEG_PREFIX:
            neg_prefix += 1
        tc = count_tool_calls(completion)
        if tc > 0:
            tool_use += 1
        tool_calls_sum += tc
        manual_sum += count_manual_steps(completion)
        len_sum += len(completion)
        reward_sum += rb.total
        bar.set_postfix(solve=f"{solved / (i + 1):.2%}")

        if args.show_cases:
            verdict = "CORRECT" if rb.reason == CORRECT else f"WRONG ({rb.reason})"
            bar.write(f"=== puzzle {i + 1}: numbers={puzzle.numbers} target={puzzle.target}")
            bar.write(f"completion: {tok.decode(completion)}")
            bar.write(f"check:      {verdict} | reward {rb.total:+.2f}")
    bar.close()
    elapsed = time.time() - t_start

    metrics = {
        "solve_rate": solved / n,
        "format_viol_rate": fmt_viol / n,
        "neg_prefix_rate": neg_prefix / n,
        "tool_use_rate": tool_use / n,
        "tool_calls_per_ep": tool_calls_sum / n,
        "manual_steps_per_ep": manual_sum / n,
        "mean_completion_len": len_sum / n,
        "mean_reward": reward_sum / n,
    }

    print(f"\n{'metric':<22} value   (n={n}, {mode})")
    print("-" * 34)
    for k, v in metrics.items():
        print(f"{k:<22} {v:>10.4f}")

    print(f"\n{'reason':<22} count   frac")
    print("-" * 40)
    for r, c in reason_counts.most_common():
        print(f"{r:<22} {c:>5}  {c / n:>6.2%}")

    tok_per_sec = model_tokens / elapsed if elapsed > 0 else 0.0
    print(f"\ndecode speed: {tok_per_sec:.1f} model-tokens/sec ({model_tokens} tokens)")

    # greedy runs are the comparable metric — only those land in the record
    rec_path = record_dir(args.model, args.training, args.seed) / "record.jsonl"
    if sampling is None and rec_path.exists():
        line = {"type": "eval", "step": step, "n": n, "split": "heldout"}
        line.update({k: round(v, 5) for k, v in metrics.items()})
        line.update({f"reason_{r}": round(c / n, 5) for r, c in sorted(reason_counts.items())})
        with rec_path.open("a") as f:
            f.write(json.dumps(line) + "\n")
        print(f"appended eval line to {rec_path}")


if __name__ == "__main__":
    main()
