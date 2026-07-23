"""Thin wrapper over the shipped WordLevel tokenizer (src/mathrl/tokenizer.json).

The 28-token vocab and its ids are fixed by docs/designs/token-protocol.md and
asserted in tests. All named id constants below mirror that table exactly.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer

_TOKENIZER_PATH = Path(__file__).resolve().parent / "tokenizer.json"

# Digit tokens live at ids 3..12: digit d -> id (3 + d).
_DIGIT_BASE = 3


class MathTokenizer:
    """WordLevel tokenizer for the +/- countdown token protocol.

    encode/decode operate on the space-separated surface form (numbers are
    digit sequences, e.g. 14 -> "1 4"). Every special token has a named integer
    id constant so callers never hardcode magic numbers.
    """

    # --- special / structural token ids (design doc vocab table) ---
    PAD = 0
    BOS = 1
    EOS = 2
    # digits 0..9 -> ids 3..12 (see encode_number / digits_to_int)
    PLUS = 13
    MINUS = 14
    COMMA = 15
    EQUALS = 16
    TARGET = 17
    REASONING = 18
    END_REASONING = 19
    SEP = 20
    CALCULATE = 21
    RESULT = 22
    END_CALCULATE = 23
    VERIFY = 24
    END_VERIFY = 25
    GOOD = 26
    BAD = 27

    def __init__(self, path: str | Path | None = None) -> None:
        self._tok = Tokenizer.from_file(str(path or _TOKENIZER_PATH))
        # id -> token surface, for space-joined decode.
        vocab = self._tok.get_vocab()
        self._id_to_token = {i: t for t, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        """Encode a space-separated token string to ids."""
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        """Decode ids back to a space-joined token string."""
        return " ".join(self._id_to_token[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    # --- number helpers ---
    @staticmethod
    def encode_number(n: int) -> list[int]:
        """Encode a non-negative integer as its digit-token ids (14 -> [4, 7])."""
        if n < 0:
            raise ValueError(f"encode_number expects a non-negative int, got {n}")
        return [_DIGIT_BASE + int(d) for d in str(n)]

    @staticmethod
    def is_digit(token_id: int) -> bool:
        return _DIGIT_BASE <= token_id <= _DIGIT_BASE + 9

    @staticmethod
    def digits_to_int(ids: list[int]) -> int:
        """Turn a run of digit-token ids back into an int ([4, 7] -> 14)."""
        if not ids:
            raise ValueError("digits_to_int expects at least one digit id")
        value = 0
        for i in ids:
            if not MathTokenizer.is_digit(i):
                raise ValueError(f"not a digit token id: {i}")
            value = value * 10 + (i - _DIGIT_BASE)
        return value
