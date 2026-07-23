"""Shared pydantic configs — the contract between library modules.

Composed into TrainingVariation (variations.py). Defaults here are the
project defaults; experiments override via named variations, never CLI flags.
"""

from typing import Literal

from pydantic import BaseModel


class PuzzleConfig(BaseModel):
    """Inputs are sampled from [min_value, 10**max_input_digits - 1];
    constructions whose target has more than max_target_digits digits are
    rejected (target 0 counts as one digit). Intermediate prefix values are
    intentionally NOT capped — only non-negativity is enforced."""

    min_numbers: int = 3
    max_numbers: int = 6  # set size sampled uniformly per puzzle (inclusive)
    min_value: int = 1
    max_input_digits: int = 1
    max_target_digits: int = 1


class RewardConfig(BaseModel):
    correct: float = 1.0
    wrong_value: float = 0.0
    wrong_multiset: float = -0.5
    neg_prefix: float = -0.5
    format_violation: float = -1.0
    # Tool-call shaping, added to the outcome reward except on
    # format_violation episodes. Pure lookup: n completed tool calls ->
    # 0 if n == 0 else tool_reward[n - 1]. Keep length == EnvConfig
    # .max_tool_calls (env terminates past that); n beyond the list clamps
    # to the last entry as a safety net.
    tool_reward: list[float] = [0.2, 0.1, 0.05, 0.0, -0.05, -0.1, -0.15, -0.2]


class TraceConfig(BaseModel):
    """SFT trace generation. p_tool: fraction of traces that demonstrate a
    single tool block checking the final expression (per the design doc,
    tools appear in SFT data only as an answer check)."""

    p_tool: float = 0.3
    # Fraction of traces demonstrating a failed attempt followed by a retry:
    # manual steps of a wrong-but-correctly-computed arrangement (final value
    # != target), then a fresh attempt that succeeds. Breaks the "last step's
    # RHS always equals the target" shortcut SFT otherwise teaches, and gives
    # RL nonzero support on try-again behavior.
    p_retry: float = 0.25


class EnvConfig(BaseModel):
    tools: Literal["none", "calculate", "verify"] = "none"
    max_tool_calls: int = 8
    max_completion_len: int = 192
