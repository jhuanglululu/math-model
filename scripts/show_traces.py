"""Sample SFT traces and print them with the loss mask made visible.

Tokens the model is NOT trained on (loss_mask=False: the prompt and every
env-authored token) are shown dim and bracketed; trained-on tokens are bold.

Usage:
    uv run scripts/show_traces.py                          # sft_calc arm
    uv run scripts/show_traces.py --training sft_verify --n 8 --seed 3
"""

import argparse
import random
import sys

from mathrl.tokenizer import MathTokenizer
from mathrl.traces import sft_trace
from mathrl.puzzles import generate_puzzle
from mathrl.variations import get_training_variation

DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def render(tokens: list[int], mask: list[bool], tok: MathTokenizer, color: bool) -> str:
    parts = []
    for t, m in zip(tokens, mask, strict=True):
        s = tok.decode([t])
        if m:
            parts.append(f"{BOLD}{s}{RESET}" if color else s)
        else:
            parts.append(f"{DIM}[{s}]{RESET}" if color else f"[{s}]")
    return " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--training", default="sft_calc", help="training variation (sets tool arm + p_tool)"
    )
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    var = get_training_variation(args.training)
    tok = MathTokenizer()
    rng = random.Random(args.seed)
    color = sys.stdout.isatty()

    print(f"variation={args.training}  tools={var.env.tools}  p_tool={var.trace.p_tool}")
    print("legend: BOLD = in SFT loss;  [bracketed/dim] = masked out (prompt + env-authored)\n")

    for i in range(args.n):
        puzzle = generate_puzzle(rng, var.puzzle)
        tokens, mask = sft_trace(puzzle, rng, var.trace, var.env, tok)
        trained = sum(mask)
        print(
            f"--- example {i + 1}: numbers={puzzle.numbers} target={puzzle.target} "
            f"({len(tokens)} tokens, {trained} in loss)"
        )
        print(render(tokens, mask, tok, color))
        print()


if __name__ == "__main__":
    main()
