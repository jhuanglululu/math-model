from mathrl.config import EnvConfig
from mathrl.env import env_step
from mathrl.puzzles import Puzzle
from mathrl.tokenizer import MathTokenizer

TOK = MathTokenizer()
T = MathTokenizer
PUZZLE = Puzzle(numbers=[1, 3, 5, 7], target=14)
CORRECT_EXPR = [6, 13, 8, 14, 4, 13, 10]  # 3 + 5 - 1 + 7

CALC = EnvConfig(tools="calculate")
VERIFY = EnvConfig(tools="verify")
NONE = EnvConfig(tools="none")


# --- calculate injection ---


def test_calculate_injects_result_and_close():
    # <calculate> 3 + 5 <result>
    comp = [T.CALCULATE, 6, 13, 8, T.RESULT]
    act = env_step(comp, PUZZLE, TOK, CALC)
    assert act.kind == "inject"
    assert act.tokens == [11, T.END_CALCULATE]  # 8 </calculate>


def test_calculate_open_continues():
    act = env_step([T.CALCULATE], PUZZLE, TOK, CALC)
    assert act.kind == "continue"


def test_calculate_neg_prefix_terminates():
    # <calculate> 1 - 5 <result>  -> prefix -4
    comp = [T.CALCULATE, 4, 14, 8, T.RESULT]
    act = env_step(comp, PUZZLE, TOK, CALC)
    assert act.kind == "terminate" and act.reason == "neg_prefix_calc"


def test_calculate_malformed_expr_terminates():
    comp = [T.CALCULATE, 13, T.RESULT]  # "<calculate> + <result>"
    act = env_step(comp, PUZZLE, TOK, CALC)
    assert act.kind == "terminate" and act.reason == "malformed_tool"


def test_result_without_open_calc_terminates():
    act = env_step([T.RESULT], PUZZLE, TOK, CALC)
    assert act.kind == "terminate" and act.reason == "malformed_tool"


def test_nested_calc_terminates():
    comp = [T.CALCULATE, 6, T.CALCULATE]
    act = env_step(comp, PUZZLE, TOK, CALC)
    assert act.kind == "terminate" and act.reason == "nested_tool"


def test_max_tool_calls_terminates():
    cfg = EnvConfig(tools="calculate", max_tool_calls=1)
    comp = [T.CALCULATE, 6, T.RESULT, 9, T.END_CALCULATE, T.CALCULATE]
    act = env_step(comp, PUZZLE, TOK, cfg)
    assert act.kind == "terminate" and act.reason == "max_tool_calls"


# --- verify injection ---


def test_verify_good():
    comp = [T.VERIFY] + CORRECT_EXPR + [T.END_VERIFY]
    act = env_step(comp, PUZZLE, TOK, VERIFY)
    assert act.kind == "inject" and act.tokens == [T.GOOD]


def test_verify_bad():
    comp = [T.VERIFY, 6, 13, 8, T.END_VERIFY]  # 3 + 5, wrong multiset
    act = env_step(comp, PUZZLE, TOK, VERIFY)
    assert act.kind == "inject" and act.tokens == [T.BAD]


def test_verify_empty_expr_terminates_malformed():
    # a no-op <verify></verify> must not earn tool shaping (reviewer finding)
    comp = [T.VERIFY, T.END_VERIFY]
    act = env_step(comp, PUZZLE, TOK, VERIFY)
    assert act.kind == "terminate" and act.reason == "malformed_tool"


def test_verify_unparseable_expr_terminates_malformed():
    comp = [T.VERIFY, 13, 13, T.END_VERIFY]  # "+ +"
    act = env_step(comp, PUZZLE, TOK, VERIFY)
    assert act.kind == "terminate" and act.reason == "malformed_tool"


def test_verify_neg_prefix_expr_injects_bad():
    # parseable but failing expr is a legitimate query, not malformed
    comp = [T.VERIFY, 4, 14, 6, T.END_VERIFY]  # 1 - 3
    act = env_step(comp, PUZZLE, TOK, VERIFY)
    assert act.kind == "inject" and act.tokens == [T.BAD]


# --- termination reasons ---


def test_eos_done():
    act = env_step([6, 13, 8, T.EOS], PUZZLE, TOK, CALC)
    assert act.kind == "terminate" and act.reason == "done"


def test_too_long():
    cfg = EnvConfig(tools="calculate", max_completion_len=4)
    act = env_step([6, 13, 8, 6], PUZZLE, TOK, cfg)
    assert act.kind == "terminate" and act.reason == "too_long"


def test_eos_at_cap_is_done_not_too_long():
    cfg = EnvConfig(tools="calculate", max_completion_len=4)
    act = env_step([6, 13, 8, T.EOS], PUZZLE, TOK, cfg)
    assert act.kind == "terminate" and act.reason == "done"


def test_disabled_tool_none_arm():
    act = env_step([T.CALCULATE], PUZZLE, TOK, NONE)
    assert act.kind == "terminate" and act.reason == "disabled_tool"


def test_disabled_tool_wrong_arm():
    act = env_step([T.VERIFY], PUZZLE, TOK, CALC)
    assert act.kind == "terminate" and act.reason == "disabled_tool"
    act2 = env_step([T.CALCULATE], PUZZLE, TOK, VERIFY)
    assert act2.kind == "terminate" and act2.reason == "disabled_tool"


def test_tool_out_of_reasoning():
    comp = [T.END_REASONING, T.CALCULATE]
    act = env_step(comp, PUZZLE, TOK, CALC)
    assert act.kind == "terminate" and act.reason == "tool_out_of_reasoning"


def test_plain_token_continues():
    act = env_step([6, 13, 8], PUZZLE, TOK, CALC)
    assert act.kind == "continue"


def test_end_reasoning_continues():
    act = env_step([6, 13, 8, T.SEP, T.END_REASONING], PUZZLE, TOK, CALC)
    assert act.kind == "continue"
