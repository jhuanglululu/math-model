"""SFT warmup trace generation.

Builds a full correct episode (prompt + completion) in the design-doc format:
manual left-to-right prefix steps (`3 + 5 = 8 <sep> 8 - 1 = 7 <sep> ...`), then
— with probability p_tool, and only when the env arm enables tools — a single
tool block that checks the FINAL expression (calculate or verify), then
`</reasoning>`, the answer expression, and `<eos>`.

Retry demos (p_retry): with probability p_retry, and when a wrong arrangement of
the puzzle's numbers exists, the trace first shows a FAILED attempt — manual
steps of a wrong-but-correctly-computed arrangement (final value != target) —
optionally with a tool block exposing the failure (calculate: `<result>` != the
target; verify: `<bad>`) when this trace also drew the tool demo, then the
correct attempt's manual steps, then the usual optional success tool check.
A single p_tool draw governs both the failing and the success tool blocks.
Retry breaks the "last manual step's RHS always equals the target" shortcut that
plain traces otherwise teach. No-retry traces are byte-identical to the
pre-retry behavior for a fixed seed/puzzle.

Returns (tokens, loss_mask). loss_mask is True exactly on model-authored tokens.
It is False on the whole prompt AND on env-authored tokens (calculate result
digits + `</calculate>`, verify `<good>`/`<bad>`), because SFT must mirror the
RL action mask — otherwise SFT teaches the model to predict tokens the env
injects at RL time.
"""

from __future__ import annotations

import random

from .checker import _parse_expr
from .config import EnvConfig, TraceConfig
from .puzzles import Puzzle, prompt_tokens, sample_wrong_arrangement
from .tokenizer import MathTokenizer


def sft_trace(
    puzzle: Puzzle,
    rng: random.Random,
    trace_cfg: TraceConfig,
    env_cfg: EnvConfig,
    tok: MathTokenizer,
) -> tuple[list[int], list[bool]]:
    if puzzle.solution is None:
        raise ValueError("sft_trace requires puzzle.solution (the generating expression)")

    tokens: list[int] = []
    mask: list[bool] = []

    def add(ids: list[int], author_is_model: bool) -> None:
        tokens.extend(ids)
        mask.extend([author_is_model] * len(ids))

    def emit_steps(numbers: list[int], ops: list[str]) -> int:
        """Emit `acc op num = new_acc <sep>` per operation; return final value."""
        acc = numbers[0]
        for op, num in zip(ops, numbers[1:]):
            new_acc = acc + num if op == "+" else acc - num
            op_id = MathTokenizer.PLUS if op == "+" else MathTokenizer.MINUS
            add(
                tok.encode_number(acc)
                + [op_id]
                + tok.encode_number(num)
                + [MathTokenizer.EQUALS]
                + tok.encode_number(new_acc)
                + [MathTokenizer.SEP],
                True,
            )
            acc = new_acc
        return acc

    def emit_tool(expr: list[int], result_value: int, is_good: bool) -> None:
        """Emit the tool block for the active arm (model tokens True, env False)."""
        if env_cfg.tools == "calculate":
            add([MathTokenizer.CALCULATE] + list(expr) + [MathTokenizer.RESULT], True)
            add(tok.encode_number(result_value) + [MathTokenizer.END_CALCULATE], False)
        else:  # "verify"
            add([MathTokenizer.VERIFY] + list(expr) + [MathTokenizer.END_VERIFY], True)
            add([MathTokenizer.GOOD if is_good else MathTokenizer.BAD], False)

    # --- prompt (through <reasoning>): never contributes to loss ---
    add(prompt_tokens(puzzle, tok), False)

    parsed = _parse_expr(puzzle.solution)
    if parsed is None:
        raise ValueError("puzzle.solution is not a valid expression")
    numbers, ops = parsed

    # Decision draws. `p_retry > 0` short-circuits so p_retry=0 consumes no RNG
    # and reproduces the pre-retry stream exactly. A single p_tool draw governs
    # both the failing and the success tool blocks.
    do_retry = trace_cfg.p_retry > 0 and rng.random() < trace_cfg.p_retry
    use_tool = env_cfg.tools != "none" and rng.random() < trace_cfg.p_tool

    wrong_expr = sample_wrong_arrangement(puzzle, rng) if do_retry else None

    # --- failed attempt (only when a wrong arrangement was found) ---
    if wrong_expr is not None:
        w_parsed = _parse_expr(wrong_expr)
        assert w_parsed is not None  # sample_wrong_arrangement always returns valid tokens
        w_numbers, w_ops = w_parsed
        w_final = emit_steps(w_numbers, w_ops)
        if use_tool:
            # calculate exposes the wrong result; verify returns <bad>.
            emit_tool(wrong_expr, w_final, is_good=False)

    # --- correct attempt ---
    emit_steps(numbers, ops)

    # --- optional success tool check of the final expression ---
    if use_tool:
        emit_tool(puzzle.solution, puzzle.target, is_good=True)

    # --- close reasoning, emit the answer, end (all model-authored) ---
    add([MathTokenizer.END_REASONING], True)
    add(list(puzzle.solution), True)
    add([MathTokenizer.EOS], True)

    return tokens, mask
