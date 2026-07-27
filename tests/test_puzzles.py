import random

from mathrl.checker import _parse_expr
from mathrl.config import PuzzleConfig
from mathrl.puzzles import (
    Puzzle,
    canonical_key,
    eval_keys,
    eval_puzzles,
    generate_puzzle,
    prompt_tokens,
    sample_wrong_arrangement,
)
from mathrl.tokenizer import MathTokenizer


def _assert_valid_puzzles(cfg, seeds=range(400)):
    max_value = 10**cfg.max_input_digits - 1
    for seed in seeds:
        rng = random.Random(seed)
        p = generate_puzzle(rng, cfg)
        assert cfg.min_numbers <= len(p.numbers) <= cfg.max_numbers
        # inputs within the derived digit-bounded range
        assert all(cfg.min_value <= v <= max_value for v in p.numbers)
        assert all(len(str(v)) <= cfg.max_input_digits for v in p.numbers)
        # target respects the digit cap (0 counts as one digit)
        assert p.target >= 0
        assert len(str(p.target)) <= cfg.max_target_digits
        # solution parses, uses exactly the puzzle multiset, and its left-to-right
        # prefixes never go negative, and it hits the target.
        parsed = _parse_expr(p.solution)
        assert parsed is not None
        numbers, ops = parsed
        assert sorted(numbers) == sorted(p.numbers)
        acc = numbers[0]
        assert acc >= 0
        for op, num in zip(ops, numbers[1:]):
            acc = acc + num if op == "+" else acc - num
            assert acc >= 0
        assert acc == p.target


def test_generator_default_config():
    # defaults are a project knob (currently 2-digit inputs / 3-digit targets);
    # assert validity against whatever they are, not a pinned value
    _assert_valid_puzzles(PuzzleConfig())


def test_generator_single_digit_config():
    _assert_valid_puzzles(PuzzleConfig(max_input_digits=1, max_target_digits=1))


def test_generator_two_digit_config_wider_ranges():
    cfg = PuzzleConfig(max_input_digits=2, max_target_digits=2)
    _assert_valid_puzzles(cfg)
    # sanity: this config actually reaches multi-digit inputs somewhere
    seen_two_digit_input = any(
        v >= 10 for s in range(200) for v in generate_puzzle(random.Random(s), cfg).numbers
    )
    assert seen_two_digit_input


def test_set_size_spans_full_default_range():
    cfg = PuzzleConfig()
    sizes = {len(generate_puzzle(random.Random(s), cfg).numbers) for s in range(400)}
    assert sizes == {3, 4, 5, 6}  # default min_numbers=3, max_numbers=6, inclusive


def test_min_equals_max_pins_size():
    cfg = PuzzleConfig(min_numbers=5, max_numbers=5)
    for s in range(100):
        assert len(generate_puzzle(random.Random(s), cfg).numbers) == 5


def test_sample_wrong_arrangement_is_wrong_but_valid():
    cfg = PuzzleConfig()
    for seed in range(300):
        p = generate_puzzle(random.Random(seed), cfg)
        wrong = sample_wrong_arrangement(p, random.Random(seed))
        if wrong is None:
            continue  # no wrong arrangement exists for this multiset (rare)
        parsed = _parse_expr(wrong)
        assert parsed is not None
        numbers, ops = parsed
        assert sorted(numbers) == sorted(p.numbers)  # full multiset
        acc = numbers[0]
        assert acc >= 0  # first term positive
        for op, num in zip(ops, numbers[1:]):
            acc = acc + num if op == "+" else acc - num
            assert acc >= 0  # non-negative prefixes
        assert acc != p.target  # final value must miss the target


def test_sample_wrong_arrangement_can_return_none_when_no_wrong_exists():
    # {1, 1}: arrangements are 1+1=2 and 1-1=0. If target is 2, the only other
    # non-negative arrangement (1-1=0) is available -> wrong exists. But for a
    # multiset where every non-negative arrangement equals the target, None.
    # {5}: single number, sole arrangement is "5" == target -> no wrong.
    p = Puzzle(numbers=[5], target=5, solution=[8])
    assert sample_wrong_arrangement(p, random.Random(0)) is None


def test_prompt_tokens_exact():
    tok = MathTokenizer()
    p = Puzzle(numbers=[1, 3, 5, 7], target=14)
    # <bos> 1 , 3 , 5 , 7 <target> 1 4 <reasoning>
    assert prompt_tokens(p, tok) == [1, 4, 15, 6, 15, 8, 15, 10, 17, 4, 7, 18]


def test_canonical_key_order_independent():
    a = Puzzle(numbers=[7, 1, 5, 3], target=14)
    b = Puzzle(numbers=[1, 3, 5, 7], target=14)
    c = Puzzle(numbers=[1, 3, 5, 7], target=13)
    assert canonical_key(a) == canonical_key(b)
    assert canonical_key(a) != canonical_key(c)
    assert canonical_key(b) == "1,3,5,7|14"


def test_generation_is_deterministic_by_seed():
    cfg = PuzzleConfig()
    p1 = generate_puzzle(random.Random(99), cfg)
    p2 = generate_puzzle(random.Random(99), cfg)
    assert (p1.numbers, p1.target, p1.solution) == (p2.numbers, p2.target, p2.solution)


def test_eval_stream_deterministic_and_distinct():
    cfg = PuzzleConfig()
    a = eval_puzzles(cfg, n=50)
    b = eval_puzzles(cfg, n=50)
    assert [(p.numbers, p.target) for p in a] == [(p.numbers, p.target) for p in b]
    assert len({canonical_key(p) for p in a}) == 50
    # smaller n is a prefix of larger n (training exclusion stays a superset)
    assert [(p.numbers, p.target) for p in eval_puzzles(cfg, n=10)] == [
        (p.numbers, p.target) for p in a[:10]
    ]
    assert eval_keys(cfg, n=50) == {canonical_key(p) for p in a}
