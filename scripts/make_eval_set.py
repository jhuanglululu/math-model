"""Write the fixed held-out eval set to datasets/eval.jsonl.

1000 distinct puzzles (by canonical key) generated from a fixed seed with the
default PuzzleConfig. One JSON object per line: {"numbers": [...], "target": T}.
The training generator skips any puzzle whose canonical key appears here.

Run: uv run scripts/make_eval_set.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from mathrl.config import PuzzleConfig
from mathrl.puzzles import canonical_key, generate_puzzle

SEED = 1234
N_PUZZLES = 1000
OUT_PATH = Path(__file__).resolve().parent.parent / "datasets" / "eval.jsonl"


def main() -> None:
    rng = random.Random(SEED)
    cfg = PuzzleConfig()

    seen: set[str] = set()
    rows: list[dict] = []
    while len(rows) < N_PUZZLES:
        puzzle = generate_puzzle(rng, cfg)
        key = canonical_key(puzzle)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"numbers": puzzle.numbers, "target": puzzle.target})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"wrote {len(rows)} puzzles to {OUT_PATH}")


if __name__ == "__main__":
    main()
