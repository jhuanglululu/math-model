"""Evaluate a checkpoint on the held-out puzzle set with greedy, env-in-the-loop
decoding.

Usage:
    uv run scripts/eval.py --model tiny --training smoke --seed 0 --n 50
    uv run scripts/eval.py --model tiny --training smoke --checkpoint <path.safetensors>

`--model`/`--training` are required: the architecture and the tool arm (env
config) cannot be inferred from a bare weights file. Without `--checkpoint`,
the run's `current.safetensors` is used. Metrics are printed as a table and, when
addressed by variation, an `{"type":"eval", ...}` line is appended to the run's
record.jsonl.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_model
from tqdm import tqdm

from mathrl.checker import CORRECT, NEG_PREFIX, reward
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
from mathrl.puzzles import Puzzle, prompt_tokens
from mathrl.records import record_dir
from mathrl.tokenizer import MathTokenizer
from mathrl.variations import get_model_variation, get_training_variation

EVAL_JSONL = Path("datasets/eval.jsonl")
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


def mask_logits(logits: torch.Tensor, disabled: list[int]) -> torch.Tensor:
    logits = logits.clone()
    logits[MathTokenizer.PAD] = NEG_INF
    logits[MathTokenizer.BOS] = NEG_INF
    for t in disabled:
        logits[t] = NEG_INF
    return logits


@torch.no_grad()
def greedy_rollout(model, puzzle, tok, env_cfg, device, max_len):
    """Greedy decode with env interaction. Returns (completion, reason)."""
    disabled = disabled_tool_ids(env_cfg.tools)
    prompt = prompt_tokens(puzzle, tok)
    completion: list[int] = []
    reason = DONE
    hard_cap = env_cfg.max_completion_len + 8
    while True:
        context = (prompt + completion)[-max_len:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)[0, -1]
        logits = mask_logits(logits, disabled)
        nxt = int(torch.argmax(logits).item())
        completion.append(nxt)
        action = env_step(completion, puzzle, tok, env_cfg)
        if action.kind == "inject":
            completion.extend(action.tokens)
        elif action.kind == "terminate":
            reason = action.reason
            break
        if len(completion) >= hard_cap:
            reason = TOO_LONG
            break
    return completion, reason


def count_tool_calls(completion: list[int]) -> int:
    return sum(1 for t in completion if t in (MathTokenizer.CALCULATE, MathTokenizer.VERIFY))


def count_manual_steps(completion: list[int]) -> int:
    # '=' only appears in model-authored manual steps, inside the reasoning block.
    if MathTokenizer.END_REASONING in completion:
        reasoning = completion[: completion.index(MathTokenizer.END_REASONING)]
    else:
        reasoning = completion
    return sum(1 for t in reasoning if t == MathTokenizer.EQUALS)


def load_puzzles(n: int | None) -> list[Puzzle]:
    puzzles = []
    for line in EVAL_JSONL.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        puzzles.append(Puzzle(numbers=rec["numbers"], target=rec["target"]))
    return puzzles if n is None else puzzles[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--training", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default=None, help="explicit weights file or run dir")
    ap.add_argument("--n", type=int, default=None, help="number of eval puzzles (default all)")
    ap.add_argument(
        "--show-cases",
        action="store_true",
        help="print each puzzle's prompt/completion/verdict (default: summary only)",
    )
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    model_cfg = get_model_variation(args.model)
    tv = get_training_variation(args.training)
    tok = MathTokenizer()
    max_len = model_cfg.block_size

    # resolve weights path
    from mathrl.checkpoint import run_dir

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

    puzzles = load_puzzles(args.n)
    n = len(puzzles)
    print(f"evaluating {n} puzzles on {weights} (arm={tv.env.tools})...")

    solved = fmt_viol = neg_prefix = tool_use = 0
    tool_calls_sum = manual_sum = len_sum = reward_sum = 0.0
    reason_counts: Counter[str] = Counter()

    bar = tqdm(puzzles, desc=f"eval {args.model}/{args.training}", unit="puzzle")
    for i, puzzle in enumerate(bar):
        completion, reason = greedy_rollout(model, puzzle, tok, tv.env, device, max_len)
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

    print(f"\n{'metric':<22} value   (n={n})")
    print("-" * 34)
    for k, v in metrics.items():
        print(f"{k:<22} {v:>10.4f}")

    print(f"\n{'reason':<22} count   frac")
    print("-" * 40)
    for r, c in reason_counts.most_common():
        print(f"{r:<22} {c:>5}  {c / n:>6.2%}")

    # append eval line to the run record (addressed by variation)
    rec_path = record_dir(args.model, args.training, args.seed) / "record.jsonl"
    if rec_path.exists():
        line = {"type": "eval", "step": step, "n": n, "split": "heldout"}
        line.update({k: round(v, 5) for k, v in metrics.items()})
        line.update({f"reason_{r}": round(c / n, 5) for r, c in sorted(reason_counts.items())})
        with rec_path.open("a") as f:
            f.write(json.dumps(line) + "\n")
        print(f"\nappended eval line to {rec_path}")


if __name__ == "__main__":
    main()
