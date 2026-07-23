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
    tokens, mask = sft_trace(PUZZLE, random.Random(0), TraceConfig(p_tool=1.0), env_cfg, TOK)

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
    tokens, mask = sft_trace(PUZZLE, random.Random(0), TraceConfig(p_tool=1.0), env_cfg, TOK)

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
    tokens, mask = sft_trace(PUZZLE, random.Random(0), TraceConfig(p_tool=1.0), env_cfg, TOK)

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
    tokens, _ = sft_trace(PUZZLE, random.Random(0), TraceConfig(p_tool=1.0), env_cfg, TOK)
    assert MathTokenizer.CALCULATE not in tokens
    assert MathTokenizer.VERIFY not in tokens


def test_p_tool_zero_never_emits_tool():
    env_cfg = EnvConfig(tools="calculate")
    tokens, _ = sft_trace(PUZZLE, random.Random(0), TraceConfig(p_tool=0.0), env_cfg, TOK)
    assert MathTokenizer.CALCULATE not in tokens


def test_generated_puzzle_trace_is_self_consistent():
    # end-to-end: a generated puzzle's no-tool trace answer equals its solution
    from mathrl.config import PuzzleConfig
    from mathrl.puzzles import generate_puzzle

    p = generate_puzzle(random.Random(7), PuzzleConfig())
    tokens, mask = sft_trace(p, random.Random(0), TraceConfig(p_tool=0.0), EnvConfig(), TOK)
    # answer is between </reasoning> and <eos>
    ri = tokens.index(MathTokenizer.END_REASONING)
    ei = tokens.index(MathTokenizer.EOS)
    assert tokens[ri + 1 : ei] == p.solution
