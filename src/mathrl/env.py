"""Environment stepping: tool injection and episode termination.

The user's rollout loop calls `env_step` after every model-sampled token, with
the completion generated so far (the tokens AFTER the prompt; the prompt ends
with `<reasoning>`, so the completion begins inside the reasoning block).
Env-injected tokens are NOT sampled, so they never trigger a call — the last
token of `completion_so_far` is always the just-sampled model token.

The response is an EnvAction:

- continue  — nothing to do, keep sampling.
- inject    — append env-authored tokens (excluded from the action mask), then
              keep sampling. Emitted for calculate (`<result>` -> result digits
              + `</calculate>`) and verify (`</verify>` -> `<good>`/`<bad>`).
- terminate — end the episode with a reason string.

Termination reasons (consumed by the rollout loop and by checker.reward; every
reason except "done" maps to the format-violation reward):

    done                 <eos> reached — normal completion.
    too_long             completion reached max_completion_len without <eos>.
    max_tool_calls       opening a tool block beyond max_tool_calls.
    tool_out_of_reasoning tool token emitted after </reasoning> (outside the block).
    nested_tool          a tool opened while another is still open.
    malformed_tool       unparseable calculate expr, or a stray env-only tool
                         token the model shouldn't author (</calculate>, a
                         <result>/</verify> with no open block).
    neg_prefix_calc      a negative left-to-right prefix inside a <calculate> expr.
    disabled_tool        a tool token for a disabled arm (defense in depth; the
                         rollout also logit-masks these).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .checker import check_solution, eval_left_to_right
from .config import EnvConfig
from .puzzles import Puzzle
from .tokenizer import MathTokenizer

# termination reason constants
DONE = "done"
TOO_LONG = "too_long"
MAX_TOOL_CALLS = "max_tool_calls"
TOOL_OUT_OF_REASONING = "tool_out_of_reasoning"
NESTED_TOOL = "nested_tool"
MALFORMED_TOOL = "malformed_tool"
NEG_PREFIX_CALC = "neg_prefix_calc"
DISABLED_TOOL = "disabled_tool"

_CALC_TOKENS = frozenset(
    {MathTokenizer.CALCULATE, MathTokenizer.RESULT, MathTokenizer.END_CALCULATE}
)
_VERIFY_TOKENS = frozenset(
    {MathTokenizer.VERIFY, MathTokenizer.END_VERIFY, MathTokenizer.GOOD, MathTokenizer.BAD}
)
_TOOL_TOKENS = _CALC_TOKENS | _VERIFY_TOKENS


@dataclass
class EnvAction:
    kind: Literal["continue", "inject", "terminate"]
    tokens: list[int] = field(default_factory=list)
    reason: str | None = None


def _continue() -> EnvAction:
    return EnvAction(kind="continue")


def _inject(tokens: list[int]) -> EnvAction:
    return EnvAction(kind="inject", tokens=tokens)


def _terminate(reason: str) -> EnvAction:
    return EnvAction(kind="terminate", reason=reason)


def _scan_state(comp: list[int]) -> tuple[bool, tuple[str, int] | None, int]:
    """Reconstruct env state from the tokens BEFORE the last one.

    Returns (reasoning_open, open_tool, tool_calls) where open_tool is
    ("calc"|"verify", opening_index) or None, and tool_calls counts openings.
    """
    reasoning_open = True
    open_tool: tuple[str, int] | None = None
    tool_calls = 0
    for i, t in enumerate(comp):
        if t == MathTokenizer.END_REASONING:
            reasoning_open = False
        elif t == MathTokenizer.CALCULATE:
            open_tool = ("calc", i)
            tool_calls += 1
        elif t == MathTokenizer.END_CALCULATE:
            open_tool = None
        elif t == MathTokenizer.VERIFY:
            open_tool = ("verify", i)
            tool_calls += 1
        elif t == MathTokenizer.END_VERIFY:
            open_tool = None
    return reasoning_open, open_tool, tool_calls


def env_step(
    completion_so_far: list[int],
    puzzle: Puzzle,
    tok: MathTokenizer,
    cfg: EnvConfig,
) -> EnvAction:
    """Decide what the env does after the latest sampled token. See module doc."""
    if not completion_so_far:
        return _continue()

    last = completion_so_far[-1]

    # <eos> always ends the episode normally, even at the length cap.
    if last == MathTokenizer.EOS:
        return _terminate(DONE)

    # Length cap: reaching max_completion_len without <eos> is a truncation.
    if len(completion_so_far) >= cfg.max_completion_len:
        return _terminate(TOO_LONG)

    # Non-tool tokens (digits, +, -, =, <sep>, </reasoning>, <target>, ...) just
    # advance generation.
    if last not in _TOOL_TOKENS:
        return _continue()

    reasoning_open, open_tool, tool_calls = _scan_state(completion_so_far[:-1])

    # Any tool token after the reasoning block has closed is out of protocol.
    if not reasoning_open:
        return _terminate(TOOL_OUT_OF_REASONING)

    if cfg.tools == "none":
        return _terminate(DISABLED_TOOL)

    if cfg.tools == "calculate":
        if last in _VERIFY_TOKENS:
            return _terminate(DISABLED_TOOL)
        if last == MathTokenizer.CALCULATE:
            if open_tool is not None:
                return _terminate(NESTED_TOOL)
            if tool_calls >= cfg.max_tool_calls:
                return _terminate(MAX_TOOL_CALLS)
            return _continue()
        if last == MathTokenizer.RESULT:
            if open_tool is None or open_tool[0] != "calc":
                return _terminate(MALFORMED_TOOL)
            expr = completion_so_far[open_tool[1] + 1 : -1]
            res = eval_left_to_right(expr, tok)
            if not res.ok:
                if res.reason == "neg_prefix":
                    return _terminate(NEG_PREFIX_CALC)
                return _terminate(MALFORMED_TOOL)
            return _inject(tok.encode_number(res.value) + [MathTokenizer.END_CALCULATE])
        # </calculate> is env-authored; the model emitting it is out of protocol.
        return _terminate(MALFORMED_TOOL)

    # cfg.tools == "verify"
    if last in _CALC_TOKENS:
        return _terminate(DISABLED_TOOL)
    if last == MathTokenizer.VERIFY:
        if open_tool is not None:
            return _terminate(NESTED_TOOL)
        if tool_calls >= cfg.max_tool_calls:
            return _terminate(MAX_TOOL_CALLS)
        return _continue()
    if last == MathTokenizer.END_VERIFY:
        if open_tool is None or open_tool[0] != "verify":
            return _terminate(MALFORMED_TOOL)
        expr = completion_so_far[open_tool[1] + 1 : -1]
        # Symmetric with calculate: an unparseable/empty expr is malformed tool
        # use and terminates (a no-op <verify></verify> must not earn tool
        # shaping). A parseable expr that merely fails the check (wrong value,
        # wrong multiset, negative prefix) is a legitimate query -> <bad>.
        if eval_left_to_right(expr, tok).reason == "malformed":
            return _terminate(MALFORMED_TOOL)
        chk = check_solution(puzzle, expr, tok)
        return _inject([MathTokenizer.GOOD if chk.ok else MathTokenizer.BAD])
    # <good>/<bad> are env-authored; the model emitting them is out of protocol.
    return _terminate(MALFORMED_TOOL)
