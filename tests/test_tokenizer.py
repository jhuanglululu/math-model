from mathrl.tokenizer import MathTokenizer

# Exact id table from docs/designs/token-protocol.md.
EXPECTED_IDS = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "0": 3,
    "1": 4,
    "2": 5,
    "3": 6,
    "4": 7,
    "5": 8,
    "6": 9,
    "7": 10,
    "8": 11,
    "9": 12,
    "+": 13,
    "-": 14,
    ",": 15,
    "=": 16,
    "<target>": 17,
    "<reasoning>": 18,
    "</reasoning>": 19,
    "<sep>": 20,
    "<calculate>": 21,
    "<result>": 22,
    "</calculate>": 23,
    "<verify>": 24,
    "</verify>": 25,
    "<good>": 26,
    "<bad>": 27,
}


def test_vocab_size():
    tok = MathTokenizer()
    assert tok.vocab_size == 28


def test_all_28_token_ids_exact():
    tok = MathTokenizer()
    for surface, expected in EXPECTED_IDS.items():
        assert tok.encode(surface) == [expected], surface


def test_named_constants_match_table():
    T = MathTokenizer
    assert (T.PAD, T.BOS, T.EOS) == (0, 1, 2)
    assert (T.PLUS, T.MINUS, T.COMMA, T.EQUALS) == (13, 14, 15, 16)
    assert (T.TARGET, T.REASONING, T.END_REASONING, T.SEP) == (17, 18, 19, 20)
    assert (T.CALCULATE, T.RESULT, T.END_CALCULATE) == (21, 22, 23)
    assert (T.VERIFY, T.END_VERIFY, T.GOOD, T.BAD) == (24, 25, 26, 27)


def test_round_trip_encode_decode():
    tok = MathTokenizer()
    text = "<bos> 1 , 3 , 5 , 7 <target> 1 4 <reasoning> 3 + 5 = 8 <sep>"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_encode_number():
    tok = MathTokenizer()
    assert tok.encode_number(0) == [3]
    assert tok.encode_number(7) == [10]
    assert tok.encode_number(14) == [4, 7]
    assert tok.encode_number(205) == [5, 3, 8]


def test_digits_to_int_round_trip():
    tok = MathTokenizer()
    for n in (0, 5, 14, 20, 199):
        assert tok.digits_to_int(tok.encode_number(n)) == n
