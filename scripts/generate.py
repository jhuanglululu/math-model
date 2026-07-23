"""Interactive inference: sample full episodes from a checkpoint with the env in
the loop, printing prompt + generated tokens as decoded strings.

Usage:
    uv run scripts/generate.py --model tiny --training smoke --seed 0
    uv run scripts/generate.py --model tiny --training smoke --n-puzzles 5 --top-k 10

`--model`/`--training` give the architecture and the tool arm. Sampling knobs
(`--top-k`, `--top-p`, `--rep-pen`, `--seed`) all have sensible defaults so a
bare invocation just generates. Prints decode speed (model-authored tokens/sec)
at the end.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
from safetensors.torch import load_model

from mathrl.checker import reward
from mathrl.checkpoint import run_dir
from mathrl.device import get_device, seed_everything
from mathrl.env import DONE, TOO_LONG, env_step
from mathrl.model import GPT, model_dtype
from mathrl.puzzles import generate_puzzle, prompt_tokens
from mathrl.tokenizer import MathTokenizer
from mathrl.variations import get_model_variation, get_training_variation

NEG_INF = float("-inf")


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


def sample_next(
    logits: torch.Tensor,
    generated: list[int],
    disabled: list[int],
    top_k: int,
    top_p: float,
    rep_pen: float,
) -> int:
    logits = logits.clone()
    # hard masks: never sample pad/bos or disabled-arm tool tokens
    logits[MathTokenizer.PAD] = NEG_INF
    logits[MathTokenizer.BOS] = NEG_INF
    for t in disabled:
        logits[t] = NEG_INF
    # repetition penalty on already-generated tokens
    if rep_pen != 1.0:
        for t in set(generated):
            if logits[t] > 0:
                logits[t] = logits[t] / rep_pen
            else:
                logits[t] = logits[t] * rep_pen
    # top-k
    if top_k > 0:
        k = min(top_k, logits.numel())
        kth = torch.topk(logits, k).values[-1]
        logits[logits < kth] = NEG_INF
    probs = torch.softmax(logits, dim=-1)
    # top-p (nucleus)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        drop = cum - sorted_probs > top_p
        sorted_probs[drop] = 0.0
        probs = torch.zeros_like(probs).scatter(0, sorted_idx, sorted_probs)
    total = probs.sum()
    if total <= 0:
        return int(torch.argmax(logits).item())
    probs = probs / total
    return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def generate_episode(model, puzzle, tok, env_cfg, device, max_len, sampling):
    """Sample one episode. Returns (completion, reason, model_token_count)."""
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
        nxt = sample_next(logits, completion, disabled, **sampling)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--training", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default=None, help="explicit weights file or run dir")
    ap.add_argument("--n-puzzles", type=int, default=3)
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
    model.compile()
    print(f"loaded {weights} (arm={tv.env.tools})\n")

    sampling = {"top_k": args.top_k, "top_p": args.top_p, "rep_pen": args.rep_pen}
    rng = random.Random(args.seed)

    total_tokens = 0
    total_time = 0.0
    n_correct = 0
    for i in range(args.n_puzzles):
        puzzle = generate_puzzle(rng, tv.puzzle)
        prompt = prompt_tokens(puzzle, tok)
        t0 = time.time()
        completion, reason, model_tokens = generate_episode(
            model, puzzle, tok, tv.env, device, max_len, sampling
        )
        dt = time.time() - t0
        total_tokens += model_tokens
        total_time += dt

        rb = reward(puzzle, completion, tok, tv.reward, terminated=reason)
        n_correct += rb.reason == "correct"

        print(f"=== puzzle {i + 1}: numbers={puzzle.numbers} target={puzzle.target} ===")
        print(f"prompt:     {tok.decode(prompt)}")
        print(f"completion: {tok.decode(completion)}")
        print(f"end reason: {reason}")
        verdict = "CORRECT" if rb.reason == "correct" else f"WRONG ({rb.reason})"
        print(
            f"check:      {verdict} | reward {rb.total:+.2f} "
            f"(base {rb.base:+.2f}, tool_shaping {rb.tool_shaping:+.2f}, "
            f"tool_calls {rb.tool_calls})\n"
        )

    print(f"solved {n_correct}/{args.n_puzzles}")
    tok_per_sec = total_tokens / total_time if total_time > 0 else 0.0
    print(
        f"decode speed: {tok_per_sec:.1f} model-tokens/sec "
        f"({total_tokens} tokens in {total_time:.2f}s)"
    )


if __name__ == "__main__":
    main()
