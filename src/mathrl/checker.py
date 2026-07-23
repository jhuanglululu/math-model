"""Answer evaluation, solution checking, and the reward function.

The answer is `NUM (OP NUM)*` with OP in {+,-}, evaluated strictly left to right
(design doc "Episode format"). It is correct iff it uses exactly the given
number multiset, every left-to-right prefix is >= 0, and the final value equals
the target. Correctness is order-independent by construction: we check the
multiset, not a string.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import RewardConfig
from .puzzles import Puzzle
from .tokenizer import MathTokenizer

# --- checker / reward reason strings (consumed by env.py and records) ---
CORRECT = "correct"
WRONG_VALUE = "wrong_value"
WRONG_MULTISET = "wrong_multiset"
NEG_PREFIX = "neg_prefix"
MALFORMED = "malformed"
NO_REASONING = "no_reasoning"
NO_EOS = "no_eos"


@dataclass
class EvalResult:
    value: int | None
    prefix_values: list[int]
    numbers: list[int]
    ok: bool
    reason: str  # "ok" | "malformed" | "neg_prefix"


@dataclass
class CheckResult:
    ok: bool
    reason: str  # CORRECT | WRONG_VALUE | WRONG_MULTISET | NEG_PREFIX | MALFORMED


@dataclass
class RewardBreakdown:
    base: float  # outcome reward (correct/wrong_value/.../format_violation)
    tool_shaping: float  # tool-call-count shaping; 0 on format-violation episodes
    total: float  # base + tool_shaping
    reason: str
    tool_calls: int  # completed tool blocks in the completion (logged by RL records)


def _parse_expr(tokens: list[int]) -> tuple[list[int], list[str]] | None:
    """Parse `NUM (OP NUM)*` into (numbers, ops). None if malformed.

    A NUM is a maximal run of digit tokens; OPs are PLUS/MINUS. Anything else
    (leading/trailing/double op, stray token, empty) is malformed.
    """
    if not tokens:
        return None
    numbers: list[int] = []
    ops: list[str] = []
    i = 0
    n = len(tokens)
    expect_num = True
    while i < n:
        t = tokens[i]
        if MathTokenizer.is_digit(t):
            if not expect_num:
                return None  # two numbers with no operator between
            j = i
            while j < n and MathTokenizer.is_digit(tokens[j]):
                j += 1
            numbers.append(MathTokenizer.digits_to_int(tokens[i:j]))
            i = j
            expect_num = False
        elif t in (MathTokenizer.PLUS, MathTokenizer.MINUS):
            if expect_num:
                return None  # op where a number was expected
            ops.append("+" if t == MathTokenizer.PLUS else "-")
            i += 1
            expect_num = True
        else:
            return None  # stray token (=, comma, special, ...)
    if expect_num:
        return None  # trailing operator
    return numbers, ops


def _evaluate(numbers: list[int], ops: list[str]) -> tuple[int, list[int], bool]:
    """Left-to-right value + prefix values; also whether any prefix went < 0."""
    acc = numbers[0]
    prefixes = [acc]
    neg = acc < 0
    for op, num in zip(ops, numbers[1:]):
        acc = acc + num if op == "+" else acc - num
        prefixes.append(acc)
        if acc < 0:
            neg = True
    return acc, prefixes, neg


def eval_left_to_right(tokens: list[int], tok: MathTokenizer) -> EvalResult:
    """Pure arithmetic evaluation of an expression token stream."""
    parsed = _parse_expr(tokens)
    if parsed is None:
        return EvalResult(value=None, prefix_values=[], numbers=[], ok=False, reason="malformed")
    numbers, ops = parsed
    value, prefixes, neg = _evaluate(numbers, ops)
    if neg:
        return EvalResult(
            value=value, prefix_values=prefixes, numbers=numbers, ok=False, reason="neg_prefix"
        )
    return EvalResult(value=value, prefix_values=prefixes, numbers=numbers, ok=True, reason="ok")


def check_solution(puzzle: Puzzle, answer_tokens: list[int], tok: MathTokenizer) -> CheckResult:
    """Full answer check. Reason priority: malformed > wrong_multiset >
    neg_prefix > wrong_value > correct."""
    parsed = _parse_expr(answer_tokens)
    if parsed is None:
        return CheckResult(ok=False, reason=MALFORMED)
    numbers, ops = parsed
    if sorted(numbers) != sorted(puzzle.numbers):
        return CheckResult(ok=False, reason=WRONG_MULTISET)
    value, _prefixes, neg = _evaluate(numbers, ops)
    if neg:
        return CheckResult(ok=False, reason=NEG_PREFIX)
    if value != puzzle.target:
        return CheckResult(ok=False, reason=WRONG_VALUE)
    return CheckResult(ok=True, reason=CORRECT)


# reasons that never come from a valid answer -> always the format reward
_FORMAT_REASONS = frozenset({MALFORMED, NO_REASONING, NO_EOS})


def _count_completed_tool_calls(completion_tokens: list[int]) -> int:
    """Number of CLOSED tool blocks: a <calculate>...</calculate> or
    <verify>...</verify> pair. Both arms count the same way. A closer with no
    matching opener is ignored (defensive)."""
    open_calc = False
    open_verify = False
    count = 0
    for t in completion_tokens:
        if t == MathTokenizer.CALCULATE:
            open_calc = True
        elif t == MathTokenizer.END_CALCULATE:
            if open_calc:
                count += 1
                open_calc = False
        elif t == MathTokenizer.VERIFY:
            open_verify = True
        elif t == MathTokenizer.END_VERIFY:
            if open_verify:
                count += 1
                open_verify = False
    return count


def _tool_shaping(n: int, cfg: RewardConfig) -> float:
    """Shaping for n completed tool calls: pure lookup. 0 at n=0; otherwise
    cfg.tool_reward[n - 1], clamping to the last entry for n beyond the list
    (the env terminates past max_tool_calls anyway, so this is a safety net)."""
    if n <= 0 or not cfg.tool_reward:
        return 0.0
    return cfg.tool_reward[min(n, len(cfg.tool_reward)) - 1]


def reward(
    puzzle: Puzzle,
    completion_tokens: list[int],
    tok: MathTokenizer,
    cfg: RewardConfig,
    terminated: str | None = None,
) -> RewardBreakdown:
    """One scalar per episode with a reason breakdown.

    `terminated` is the env termination reason (see env.env_step). Any reason
    other than "done" (or None) is an env-side format violation and maps
    straight to `cfg.format_violation`. Otherwise the answer (the tokens between
    </reasoning> and <eos>) is extracted and scored via check_solution.

    Tool-call-count shaping (the cfg.tool_reward lookup table) is added
    to the outcome reward on every episode EXCEPT format violations, which stay
    flat at cfg.format_violation so the bonus can't be farmed. `tool_calls`
    counts completed tool blocks and is always reported for records.
    """
    tool_calls = _count_completed_tool_calls(completion_tokens)

    def format_violation(reason: str) -> RewardBreakdown:
        return RewardBreakdown(
            base=cfg.format_violation,
            tool_shaping=0.0,
            total=cfg.format_violation,
            reason=reason,
            tool_calls=tool_calls,
        )

    if terminated is not None and terminated != "done":
        return format_violation(terminated)

    if MathTokenizer.END_REASONING not in completion_tokens:
        return format_violation(NO_REASONING)
    idx = completion_tokens.index(MathTokenizer.END_REASONING)
    rest = completion_tokens[idx + 1 :]
    if MathTokenizer.EOS not in rest:
        return format_violation(NO_EOS)
    answer = rest[: rest.index(MathTokenizer.EOS)]

    result = check_solution(puzzle, answer, tok)
    reason = result.reason
    if reason == MALFORMED:
        return format_violation(MALFORMED)

    base = {
        CORRECT: cfg.correct,
        WRONG_VALUE: cfg.wrong_value,
        WRONG_MULTISET: cfg.wrong_multiset,
        NEG_PREFIX: cfg.neg_prefix,
    }[reason]
    shaping = _tool_shaping(tool_calls, cfg)
    return RewardBreakdown(
        base=base,
        tool_shaping=shaping,
        total=base + shaping,
        reason=reason,
        tool_calls=tool_calls,
    )
