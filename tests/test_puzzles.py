import random

from mathrl.checker import _parse_expr
from mathrl.config import PuzzleConfig
from mathrl.puzzles import Puzzle, canonical_key, generate_puzzle, prompt_tokens
from mathrl.tokenizer import MathTokenizer


def test_generator_only_produces_solvable_nonneg_puzzles():
    cfg = PuzzleConfig()
    for seed in range(400):
        rng = random.Random(seed)
        p = generate_puzzle(rng, cfg)
        assert len(p.numbers) == cfg.n_numbers
        assert all(cfg.min_value <= v <= cfg.max_value for v in p.numbers)
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
