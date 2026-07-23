import random

from mathrl.config import EnvConfig, TraceConfig
from mathrl.puzzles import Puzzle
from mathrl.tokenizer import MathTokenizer
from mathrl.traces import sft_trace

TOK = MathTokenizer()

# Design-doc worked example: {1,3,5,7} -> 14, solution "3 + 5 - 1 + 7".
PUZZLE = Puzzle(numbers=[1, 3, 5, 7], target=14, solution=[6, 13, 8, 14, 4, 13, 10])

PROMPT = [1, 4, 15, 6, 15, 8, 15, 10, 17, 4, 7, 18]  # <bos> 1,3,5,7 <target> 14 <reasoning>
STEP1 = [6, 13, 8, 16, 11, 20]  # 3 + 5 = 8 <sep>
STEP2 = [11, 14, 4, 16, 10, 20]  # 8 - 1 = 7 <sep>
STEP3 = [10, 13, 10, 16, 4, 7, 20]  # 7 + 7 = 1 4 <sep>
ANSWER = [6, 13, 8, 14, 4, 13, 10]  # 3 + 5 - 1 + 7


def test_no_tool_trace_exact_tokens_and_mask():
    env_cfg = EnvConfig(tools="none")
    tokens, mask = sft_trace(
        PUZZLE, random.Random(0), TraceConfig(p_tool=1.0, p_retry=0.0), env_cfg, TOK
    )

    expected = PROMPT + STEP1 + STEP2 + STEP3 + [19] + ANSWER + [2]
    assert tokens == expected

    # False on the prompt, True on every model-authored token after it.
    expected_mask = [False] * len(PROMPT) + [True] * (
        len(STEP1) + len(STEP2) + len(STEP3) + 1 + len(ANSWER) + 1
    )
    assert mask == expected_mask
    assert len(tokens) == len(mask)


def test_with_tool_calculate_trace_exact_tokens_and_mask():
    env_cfg = EnvConfig(tools="calculate")
    tokens, mask = sft_trace(
        PUZZLE, random.Random(0), TraceConfig(p_tool=1.0, p_retry=0.0), env_cfg, TOK
    )

    calc_model = [21, 6, 13, 8, 14, 4, 13, 10, 22]  # <calculate> 3 + 5 - 1 + 7 <result>
    calc_env = [4, 7, 23]  # 1 4 </calculate>
    expected = PROMPT + STEP1 + STEP2 + STEP3 + calc_model + calc_env + [19] + ANSWER + [2]
    assert tokens == expected

    expected_mask = (
        [False] * len(PROMPT)
        + [True] * (len(STEP1) + len(STEP2) + len(STEP3))
        + [True] * len(calc_model)
        + [False] * len(calc_env)  # env-authored result digits + </calculate>
        + [True]  # </reasoning>
        + [True] * len(ANSWER)
        + [True]  # <eos>
    )
    assert mask == expected_mask
    assert len(tokens) == len(mask)


def test_with_tool_verify_trace_masks_injected_good():
    env_cfg = EnvConfig(tools="verify")
    tokens, mask = sft_trace(
        PUZZLE, random.Random(0), TraceConfig(p_tool=1.0, p_retry=0.0), env_cfg, TOK
    )

    verify_model = [24, 6, 13, 8, 14, 4, 13, 10, 25]  # <verify> expr </verify>
    verify_env = [26]  # <good>
    expected = PROMPT + STEP1 + STEP2 + STEP3 + verify_model + verify_env + [19] + ANSWER + [2]
    assert tokens == expected

    # the injected <good> must be masked out (env-authored)
    good_index = len(PROMPT) + len(STEP1) + len(STEP2) + len(STEP3) + len(verify_model)
    assert tokens[good_index] == 26
    assert mask[good_index] is False


def test_none_arm_never_emits_tool_even_with_p_tool_1():
    env_cfg = EnvConfig(tools="none")
    tokens, _ = sft_trace(
        PUZZLE, random.Random(0), TraceConfig(p_tool=1.0, p_retry=0.0), env_cfg, TOK
    )
    assert MathTokenizer.CALCULATE not in tokens
    assert MathTokenizer.VERIFY not in tokens


def test_p_tool_zero_never_emits_tool():
    env_cfg = EnvConfig(tools="calculate")
    tokens, _ = sft_trace(
        PUZZLE, random.Random(0), TraceConfig(p_tool=0.0, p_retry=0.0), env_cfg, TOK
    )
    assert MathTokenizer.CALCULATE not in tokens


def test_generated_puzzle_trace_is_self_consistent():
    # end-to-end: a generated puzzle's no-tool trace answer equals its solution
    from mathrl.config import PuzzleConfig
    from mathrl.puzzles import generate_puzzle

    p = generate_puzzle(random.Random(7), PuzzleConfig())
    tokens, mask = sft_trace(
        p, random.Random(0), TraceConfig(p_tool=0.0, p_retry=0.0), EnvConfig(), TOK
    )
    # answer is between </reasoning> and <eos>
    ri = tokens.index(MathTokenizer.END_REASONING)
    ei = tokens.index(MathTokenizer.EOS)
    assert tokens[ri + 1 : ei] == p.solution


# --- retry demos ---

# {3,5} -> 8, solution "3 + 5". The ONLY valid wrong arrangement (full multiset,
# first term positive, non-negative prefixes, value != 8) is "5 - 3" = 2, so
# sample_wrong_arrangement is deterministic here regardless of seed.
RETRY_PUZZLE = Puzzle(numbers=[3, 5], target=8, solution=[6, 13, 8])
R_PROMPT = [1, 6, 15, 8, 17, 11, 18]  # <bos> 3 , 5 <target> 8 <reasoning>
WRONG_STEP = [8, 14, 6, 16, 5, 20]  # 5 - 3 = 2 <sep>
CORRECT_STEP = [6, 13, 8, 16, 11, 20]  # 3 + 5 = 8 <sep>
R_ANSWER = [6, 13, 8]  # 3 + 5


def test_retry_calculate_trace_exact_tokens_and_mask():
    env_cfg = EnvConfig(tools="calculate")
    cfg = TraceConfig(p_tool=1.0, p_retry=1.0)
    tokens, mask = sft_trace(RETRY_PUZZLE, random.Random(0), cfg, env_cfg, TOK)

    fail_model = [21, 8, 14, 6, 22]  # <calculate> 5 - 3 <result>
    fail_env = [5, 23]  # 2 </calculate>  (env-authored wrong result)
    ok_model = [21, 6, 13, 8, 22]  # <calculate> 3 + 5 <result>
    ok_env = [11, 23]  # 8 </calculate>
    expected = (
        R_PROMPT
        + WRONG_STEP
        + fail_model
        + fail_env
        + CORRECT_STEP
        + ok_model
        + ok_env
        + [19]
        + R_ANSWER
        + [2]
    )
    assert tokens == expected

    expected_mask = (
        [False] * len(R_PROMPT)
        + [True] * len(WRONG_STEP)
        + [True] * len(fail_model)
        + [False] * len(fail_env)  # env-authored wrong result digits + </calculate>
        + [True] * len(CORRECT_STEP)
        + [True] * len(ok_model)
        + [False] * len(ok_env)
        + [True]  # </reasoning>
        + [True] * len(R_ANSWER)
        + [True]  # <eos>
    )
    assert mask == expected_mask
    assert len(tokens) == len(mask)


def test_retry_verify_trace_bad_then_good_masked():
    env_cfg = EnvConfig(tools="verify")
    cfg = TraceConfig(p_tool=1.0, p_retry=1.0)
    tokens, mask = sft_trace(RETRY_PUZZLE, random.Random(0), cfg, env_cfg, TOK)

    fail_model = [24, 8, 14, 6, 25]  # <verify> 5 - 3 </verify>
    fail_env = [27]  # <bad>
    ok_model = [24, 6, 13, 8, 25]  # <verify> 3 + 5 </verify>
    ok_env = [26]  # <good>
    expected = (
        R_PROMPT
        + WRONG_STEP
        + fail_model
        + fail_env
        + CORRECT_STEP
        + ok_model
        + ok_env
        + [19]
        + R_ANSWER
        + [2]
    )
    assert tokens == expected

    bad_index = len(R_PROMPT) + len(WRONG_STEP) + len(fail_model)
    assert tokens[bad_index] == 27 and mask[bad_index] is False  # <bad> masked
    good_index = bad_index + len(fail_env) + len(CORRECT_STEP) + len(ok_model)
    assert tokens[good_index] == 26 and mask[good_index] is False  # <good> masked


def test_p_retry_zero_reproduces_no_retry_trace():
    env_cfg = EnvConfig(tools="calculate")
    cfg = TraceConfig(p_tool=1.0, p_retry=0.0)
    tokens, _ = sft_trace(RETRY_PUZZLE, random.Random(0), cfg, env_cfg, TOK)

    ok_model = [21, 6, 13, 8, 22]  # single <calculate> 3 + 5 <result>
    ok_env = [11, 23]
    expected = R_PROMPT + CORRECT_STEP + ok_model + ok_env + [19] + R_ANSWER + [2]
    assert tokens == expected
    assert tokens.count(MathTokenizer.CALCULATE) == 1  # no failed attempt


def _manual_region(tokens):
    r = tokens.index(MathTokenizer.REASONING)
    e = tokens.index(MathTokenizer.END_REASONING)
    return tokens[r + 1 : e]


def _first_attempt_final(tokens):
    """Split manual steps by <sep>; find the accumulator break (attempt boundary)
    and return the first attempt's final value, or None if no break is visible."""
    region = _manual_region(tokens)
    steps, cur = [], []
    for t in region:
        if t == MathTokenizer.SEP:
            steps.append(cur)
            cur = []
        else:
            cur.append(t)
    parsed = []
    for seg in steps:
        i = 0
        lhs = []
        while i < len(seg) and MathTokenizer.is_digit(seg[i]):
            lhs.append(seg[i])
            i += 1
        i += 1  # skip op
        while i < len(seg) and MathTokenizer.is_digit(seg[i]):
            i += 1
        i += 1  # skip '='
        rhs = []
        while i < len(seg) and MathTokenizer.is_digit(seg[i]):
            rhs.append(seg[i])
            i += 1
        parsed.append((TOK.digits_to_int(lhs), TOK.digits_to_int(rhs)))
    for i in range(len(parsed) - 1):
        if parsed[i + 1][0] != parsed[i][1]:  # chain broke -> attempt boundary
            return parsed[i][1]
    return None


def test_p_retry_1_traces_contain_failed_attempt():
    from mathrl.config import PuzzleConfig
    from mathrl.puzzles import generate_puzzle

    cfg = TraceConfig(p_tool=0.0, p_retry=1.0)
    env_cfg = EnvConfig(tools="none")
    retries = 0
    n_seeds = 300
    for seed in range(n_seeds):
        p = generate_puzzle(random.Random(seed), cfg=PuzzleConfig())
        n = len(p.numbers)
        tokens, _ = sft_trace(p, random.Random(seed), cfg, env_cfg, TOK)
        eq = _manual_region(tokens).count(MathTokenizer.EQUALS)
        # exactly one of: retry (two attempts) or fallback (one attempt)
        assert eq in (n - 1, 2 * (n - 1))
        if eq == 2 * (n - 1):
            retries += 1
            fav = _first_attempt_final(tokens)
            if fav is not None:
                assert fav != p.target  # first attempt ends off-target
    # p_retry=1 makes nearly every trace a retry (fallback only when no wrong
    # arrangement exists, which is rare for 3..6 numbers)
    assert retries / n_seeds > 0.95
