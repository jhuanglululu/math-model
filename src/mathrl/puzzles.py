"""Puzzle generation for the +/- countdown task.

A puzzle is a multiset of numbers plus a target. It is solvable by construction
(design doc "Puzzles"): sample numbers, a permutation, and signs (first term
positive), reject if any left-to-right prefix goes negative, then set the target
to the final value. The generating expression is stored on the puzzle as
`solution` (token ids) for the SFT trace builder.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .config import PuzzleConfig
from .tokenizer import MathTokenizer


@dataclass
class Puzzle:
    numbers: list[int]
    target: int
    # Token ids of the generating expression, e.g. `3 + 5 - 1 + 7`. Used by the
    # trace builder; None for puzzles loaded from the eval set (solution dropped).
    solution: list[int] | None = None


def generate_puzzle(rng: random.Random, cfg: PuzzleConfig) -> Puzzle:
    """Sample a solvable puzzle. Retries until the prefix and target-digit
    constraints hold.

    The retry loop is statistically bounded: for any multiset, sorting the
    inputs descending with alternating +/- signs yields non-negative prefixes
    and a target <= the largest input (hence <= max_input_digits digits, which
    is normally <= max_target_digits), so acceptance probability is bounded away
    from zero. No iteration cap is imposed.
    """
    max_value = 10**cfg.max_input_digits - 1  # inclusive upper bound on inputs
    while True:
        numbers = [rng.randint(cfg.min_value, max_value) for _ in range(cfg.n_numbers)]

        order = numbers[:]
        rng.shuffle(order)
        # First term is always positive; the rest get random +/- signs.
        signs = [1] + [rng.choice((1, -1)) for _ in range(cfg.n_numbers - 1)]

        acc = 0
        ok = True
        for s, v in zip(signs, order):
            acc += s * v
            if acc < 0:  # reject any negative left-to-right prefix
                ok = False
                break
        if not ok:
            continue

        # Reject if the target has too many digits (target 0 counts as 1 digit).
        if len(str(acc)) > cfg.max_target_digits:
            continue

        solution: list[int] = []
        for i, (s, v) in enumerate(zip(signs, order)):
            if i > 0:
                solution.append(MathTokenizer.PLUS if s == 1 else MathTokenizer.MINUS)
            solution.extend(MathTokenizer.encode_number(v))

        return Puzzle(numbers=numbers, target=acc, solution=solution)


def prompt_tokens(puzzle: Puzzle, tok: MathTokenizer) -> list[int]:
    """`<bos> n1 , n2 , ... <target> T <reasoning>` (numbers in stored order)."""
    out = [tok.BOS]
    for i, v in enumerate(puzzle.numbers):
        if i > 0:
            out.append(tok.COMMA)
        out.extend(tok.encode_number(v))
    out.append(tok.TARGET)
    out.extend(tok.encode_number(puzzle.target))
    out.append(tok.REASONING)
    return out


def canonical_key(puzzle: Puzzle) -> str:
    """Order-independent identity: sorted numbers + target. For eval holdout."""
    nums = ",".join(str(v) for v in sorted(puzzle.numbers))
    return f"{nums}|{puzzle.target}"
