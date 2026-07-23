"""Puzzle generation for the +/- countdown task.

A puzzle is a multiset of numbers plus a target. It is solvable by construction
(design doc "Puzzles"): sample a set size, numbers, a permutation, and signs
(first term positive), reject if any left-to-right prefix goes negative or the
target exceeds the digit cap, then set the target to the final value. The
generating expression is stored on the puzzle as `solution` (token ids) for the
SFT trace builder.
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


def _arrangement_tokens(order: list[int], signs: list[int]) -> list[int]:
    """Token ids for `n0 (op ni)*` given an ordering and its signs."""
    out: list[int] = []
    for i, (s, v) in enumerate(zip(signs, order)):
        if i > 0:
            out.append(MathTokenizer.PLUS if s == 1 else MathTokenizer.MINUS)
        out.extend(MathTokenizer.encode_number(v))
    return out


def _prefixes_ok_and_value(order: list[int], signs: list[int]) -> tuple[bool, int]:
    """Evaluate left to right; ok is False if any prefix goes negative."""
    acc = 0
    for s, v in zip(signs, order):
        acc += s * v
        if acc < 0:
            return False, acc
    return True, acc


def generate_puzzle(rng: random.Random, cfg: PuzzleConfig) -> Puzzle:
    """Sample a solvable puzzle. Retries until the prefix and target-digit
    constraints hold.

    The set size is sampled uniformly from [min_numbers, max_numbers] once per
    puzzle. The retry loop is statistically bounded: for any multiset, sorting
    the inputs descending with alternating +/- signs yields non-negative
    prefixes and a value <= the largest input (hence <= max_input_digits digits,
    normally <= max_target_digits), so acceptance probability is bounded away
    from zero. No iteration cap is imposed.
    """
    n = rng.randint(cfg.min_numbers, cfg.max_numbers)
    max_value = 10**cfg.max_input_digits - 1  # inclusive upper bound on inputs
    while True:
        numbers = [rng.randint(cfg.min_value, max_value) for _ in range(n)]

        order = numbers[:]
        rng.shuffle(order)
        # First term is always positive; the rest get random +/- signs.
        signs = [1] + [rng.choice((1, -1)) for _ in range(n - 1)]

        ok, acc = _prefixes_ok_and_value(order, signs)
        if not ok:  # negative left-to-right prefix
            continue
        # Reject if the target has too many digits (target 0 counts as 1 digit).
        if len(str(acc)) > cfg.max_target_digits:
            continue

        return Puzzle(numbers=numbers, target=acc, solution=_arrangement_tokens(order, signs))


def sample_wrong_arrangement(
    puzzle: Puzzle, rng: random.Random, max_tries: int = 64
) -> list[int] | None:
    """Token ids of a WRONG arrangement of the puzzle's full multiset, for retry
    demos: first term positive, all prefixes non-negative, final value != target.

    The final value may exceed the target digit cap (it is reasoning content, not
    an answer). Reject-samples up to `max_tries`; returns None if no wrong
    arrangement is found (possible in principle — e.g. every valid arrangement
    hits the target) so the trace builder can fall back to a no-retry trace. It
    never loops unboundedly.
    """
    numbers = puzzle.numbers
    n = len(numbers)
    for _ in range(max_tries):
        order = numbers[:]
        rng.shuffle(order)
        signs = [1] + [rng.choice((1, -1)) for _ in range(n - 1)]
        ok, acc = _prefixes_ok_and_value(order, signs)
        if not ok:
            continue
        if acc == puzzle.target:  # must be a WRONG attempt
            continue
        return _arrangement_tokens(order, signs)
    return None


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
