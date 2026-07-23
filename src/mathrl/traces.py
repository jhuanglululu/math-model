"""SFT warmup trace generation.

Builds a full correct episode (prompt + completion) in the design-doc format:
manual left-to-right prefix steps (`3 + 5 = 8 <sep> 8 - 1 = 7 <sep> ...`), then
— with probability p_tool, and only when the env arm enables tools — a single
tool block that checks the FINAL expression (calculate or verify), then
`</reasoning>`, the answer expression, and `<eos>`.

Returns (tokens, loss_mask). loss_mask is True exactly on model-authored tokens.
It is False on the whole prompt AND on env-authored tokens (calculate result
digits + `</calculate>`, and the verify `<good>`), because SFT must mirror the
RL action mask — otherwise SFT teaches the model to predict tokens the env
injects at RL time.
"""

from __future__ import annotations

import random

from .checker import _parse_expr
from .config import EnvConfig, TraceConfig
from .puzzles import Puzzle, prompt_tokens
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

    # --- prompt (through <reasoning>): never contributes to loss ---
    add(prompt_tokens(puzzle, tok), False)

    parsed = _parse_expr(puzzle.solution)
    if parsed is None:
        raise ValueError("puzzle.solution is not a valid expression")
    numbers, ops = parsed

    # --- manual reasoning steps: `acc op num = new_acc <sep>` (model-authored) ---
    acc = numbers[0]
    for op, num in zip(ops, numbers[1:]):
        new_acc = acc + num if op == "+" else acc - num
        op_id = MathTokenizer.PLUS if op == "+" else MathTokenizer.MINUS
        step = (
            tok.encode_number(acc)
            + [op_id]
            + tok.encode_number(num)
            + [MathTokenizer.EQUALS]
            + tok.encode_number(new_acc)
            + [MathTokenizer.SEP]
        )
        add(step, True)
        acc = new_acc

    # --- optional single tool block checking the final expression ---
    if env_cfg.tools != "none" and rng.random() < trace_cfg.p_tool:
        if env_cfg.tools == "calculate":
            # model: <calculate> <expr> <result> ; env: <digits> </calculate>
            add([MathTokenizer.CALCULATE] + list(puzzle.solution) + [MathTokenizer.RESULT], True)
            add(tok.encode_number(puzzle.target) + [MathTokenizer.END_CALCULATE], False)
        else:  # "verify"
            # model: <verify> <expr> </verify> ; env: <good>
            add([MathTokenizer.VERIFY] + list(puzzle.solution) + [MathTokenizer.END_VERIFY], True)
            add([MathTokenizer.GOOD], False)

    # --- close reasoning, emit the answer, end (all model-authored) ---
    add([MathTokenizer.END_REASONING], True)
    add(list(puzzle.solution), True)
    add([MathTokenizer.EOS], True)

    return tokens, mask
