from mathrl.checker import (
    _tool_shaping,
    check_solution,
    eval_left_to_right,
    reward,
)
from mathrl.config import RewardConfig
from mathrl.puzzles import Puzzle
from mathrl.tokenizer import MathTokenizer

TOK = MathTokenizer()
CFG = RewardConfig()
# puzzle {1,3,5,7} -> 14; a correct answer is "3 + 5 - 1 + 7"
PUZZLE = Puzzle(numbers=[1, 3, 5, 7], target=14)
CORRECT_ANSWER = [6, 13, 8, 14, 4, 13, 10]  # 3 + 5 - 1 + 7


def _episode(answer):
    """Wrap an answer as a minimal completion: </reasoning> answer <eos>."""
    return [MathTokenizer.END_REASONING] + answer + [MathTokenizer.EOS]


# --- eval_left_to_right ---


def test_eval_left_to_right_ok():
    r = eval_left_to_right(CORRECT_ANSWER, TOK)
    assert r.ok
    assert r.value == 14
    assert r.prefix_values == [3, 8, 7, 14]
    assert r.numbers == [3, 5, 1, 7]


def test_eval_left_to_right_neg_prefix():
    # 1 - 3 + 5 + 7 -> prefixes 1, -2, ...
    r = eval_left_to_right([4, 14, 6, 13, 8, 13, 10], TOK)
    assert not r.ok
    assert r.reason == "neg_prefix"
    assert r.prefix_values[1] == -2


def test_eval_left_to_right_malformed():
    assert eval_left_to_right([], TOK).reason == "malformed"
    assert eval_left_to_right([13], TOK).reason == "malformed"  # bare "+"
    assert eval_left_to_right([6, 13], TOK).reason == "malformed"  # trailing op
    assert eval_left_to_right([6, 13, 13, 8], TOK).reason == "malformed"  # double op
    assert eval_left_to_right([6, 16, 8], TOK).reason == "malformed"  # stray "="


# --- check_solution ---


def test_check_correct():
    assert check_solution(PUZZLE, CORRECT_ANSWER, TOK) == check_solution(
        PUZZLE, CORRECT_ANSWER, TOK
    )
    res = check_solution(PUZZLE, CORRECT_ANSWER, TOK)
    assert res.ok and res.reason == "correct"


def test_check_wrong_value():
    # 3 + 5 + 1 - 7 = 2, right multiset, non-negative prefixes
    res = check_solution(PUZZLE, [6, 13, 8, 13, 4, 14, 10], TOK)
    assert not res.ok and res.reason == "wrong_value"


def test_check_wrong_multiset():
    res = check_solution(PUZZLE, [6, 13, 8], TOK)  # 3 + 5, multiset {3,5}
    assert not res.ok and res.reason == "wrong_multiset"


def test_check_neg_prefix():
    # 1 - 3 + 5 + 7 uses {1,3,5,7} but prefix goes negative
    res = check_solution(PUZZLE, [4, 14, 6, 13, 8, 13, 10], TOK)
    assert not res.ok and res.reason == "neg_prefix"


def test_check_malformed():
    res = check_solution(PUZZLE, [13, 14], TOK)
    assert not res.ok and res.reason == "malformed"


# --- reward ---


def test_reward_correct():
    rb = reward(PUZZLE, _episode(CORRECT_ANSWER), TOK, CFG)
    assert rb.total == 1.0 and rb.reason == "correct"


def test_reward_wrong_value_is_zero():
    rb = reward(PUZZLE, _episode([6, 13, 8, 13, 4, 14, 10]), TOK, CFG)
    assert rb.total == 0.0 and rb.reason == "wrong_value"


def test_reward_wrong_multiset():
    rb = reward(PUZZLE, _episode([6, 13, 8]), TOK, CFG)
    assert rb.total == -0.5 and rb.reason == "wrong_multiset"


def test_reward_neg_prefix():
    rb = reward(PUZZLE, _episode([4, 14, 6, 13, 8, 13, 10]), TOK, CFG)
    assert rb.total == -0.5 and rb.reason == "neg_prefix"


def test_reward_malformed_answer_is_format_violation():
    rb = reward(PUZZLE, _episode([13]), TOK, CFG)
    assert rb.total == -1.0 and rb.reason == "malformed"


def test_reward_missing_reasoning_close():
    rb = reward(PUZZLE, CORRECT_ANSWER + [MathTokenizer.EOS], TOK, CFG)
    assert rb.total == -1.0 and rb.reason == "no_reasoning"


def test_reward_missing_eos():
    rb = reward(PUZZLE, [MathTokenizer.END_REASONING] + CORRECT_ANSWER, TOK, CFG)
    assert rb.total == -1.0 and rb.reason == "no_eos"


def test_reward_env_termination_maps_to_format_violation():
    rb = reward(PUZZLE, _episode(CORRECT_ANSWER), TOK, CFG, terminated="too_long")
    assert rb.total == -1.0 and rb.reason == "too_long"


def test_reward_done_termination_scores_normally():
    rb = reward(PUZZLE, _episode(CORRECT_ANSWER), TOK, CFG, terminated="done")
    assert rb.total == 1.0 and rb.reason == "correct"


# --- tool-call shaping ---

T = MathTokenizer


def _episode_with_calls(answer, n_calc=0, n_verify=0):
    """Completion with n completed tool blocks before </reasoning>."""
    blocks = []
    for _ in range(n_calc):
        blocks += [T.CALCULATE, T.END_CALCULATE]
    for _ in range(n_verify):
        blocks += [T.VERIFY, T.END_VERIFY]
    return blocks + _episode(answer)


def test_shaping_n0_no_bonus():
    rb = reward(PUZZLE, _episode_with_calls(CORRECT_ANSWER, 0), TOK, CFG)
    assert (rb.base, rb.tool_shaping, rb.total, rb.tool_calls) == (1.0, 0.0, 1.0, 0)


def test_shaping_default_lookup_table():
    # tool_reward = [0.2, 0.1, 0.05, 0.0, -0.05, -0.1, -0.15, -0.2]
    expected_total = {1: 1.2, 2: 1.1, 3: 1.05, 4: 1.0, 5: 0.95, 8: 0.8}
    for n, total in expected_total.items():
        rb = reward(PUZZLE, _episode_with_calls(CORRECT_ANSWER, n), TOK, CFG)
        assert rb.base == 1.0 and rb.tool_calls == n
        assert abs(rb.total - total) < 1e-9, (n, rb.total)


def test_shaping_clamps_beyond_list():
    # n=9 is past the 8-long table -> clamps to the last entry (== n=8)
    assert _tool_shaping(9, CFG) == _tool_shaping(8, CFG) == -0.2
    assert _tool_shaping(100, CFG) == -0.2


def test_shaping_counts_both_arms():
    rb = reward(PUZZLE, _episode_with_calls(CORRECT_ANSWER, n_calc=1, n_verify=1), TOK, CFG)
    assert rb.tool_calls == 2 and rb.total == 1.1


def test_shaping_applies_to_nonformat_outcomes():
    # wrong_multiset (-0.5) still gets shaping added
    rb = reward(PUZZLE, _episode_with_calls([6, 13, 8], 1), TOK, CFG)
    assert rb.reason == "wrong_multiset" and rb.base == -0.5 and rb.total == -0.3


def test_format_violation_with_tool_calls_stays_flat():
    # malformed answer + 2 completed calls -> flat -1.0, no shaping
    rb = reward(PUZZLE, _episode_with_calls([13], 2), TOK, CFG)
    assert rb.reason == "malformed"
    assert rb.base == -1.0 and rb.tool_shaping == 0.0 and rb.total == -1.0
    assert rb.tool_calls == 2  # still counted for records


def test_env_termination_with_tool_calls_stays_flat():
    comp = _episode_with_calls(CORRECT_ANSWER, 3)
    rb = reward(PUZZLE, comp, TOK, CFG, terminated="too_long")
    assert rb.total == -1.0 and rb.tool_shaping == 0.0 and rb.tool_calls == 3


def test_custom_reward_config_shorter_table_and_clamp():
    cfg = RewardConfig(tool_reward=[0.5, -0.2])
    rb1 = reward(PUZZLE, _episode_with_calls(CORRECT_ANSWER, 1), TOK, cfg)
    assert rb1.tool_shaping == 0.5 and rb1.total == 1.5
    rb2 = reward(PUZZLE, _episode_with_calls(CORRECT_ANSWER, 2), TOK, cfg)
    assert rb2.tool_shaping == -0.2 and abs(rb2.total - 0.8) < 1e-9
    # n=3 clamps to the last entry
    rb3 = reward(PUZZLE, _episode_with_calls(CORRECT_ANSWER, 3), TOK, cfg)
    assert rb3.tool_shaping == -0.2 and abs(rb3.total - 0.8) < 1e-9
