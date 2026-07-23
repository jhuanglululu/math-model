import torch

from mathrl.sft_data import IGNORE_INDEX, build_batch, build_example


def test_build_example_exact_shift_and_mask():
    # Hand-built tiny episode. loss_mask=False on the prompt (bos + given number)
    # and on one env-authored token; True on model-authored tokens.
    #
    # index:      0     1     2     3     4       5
    # token:     bos    9    <r>    9   <eos>   (env)
    # authored:  F      F     F     T     T       F
    tokens = [1, 12, 18, 12, 2, 26]
    loss_mask = [False, False, False, True, True, False]
    max_len = 8

    input_ids, labels = build_example(tokens, loss_mask, max_len)

    # input_ids: tokens padded with pad id 0
    assert input_ids == [1, 12, 18, 12, 2, 26, 0, 0]

    # labels[t] = tokens[t+1] iff loss_mask[t+1], else -100.
    #  t=0 -> next tok idx1 authored? F -> -100
    #  t=1 -> idx2 F -> -100
    #  t=2 -> idx3 T -> tokens[3]=12
    #  t=3 -> idx4 T -> tokens[4]=2
    #  t=4 -> idx5 F -> -100  (env-authored token not predicted)
    #  t=5 -> no next (last real) -> -100
    #  t=6,7 -> padding -> -100
    assert labels == [
        IGNORE_INDEX,
        IGNORE_INDEX,
        12,
        2,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
    ]


def test_build_example_all_authored_completion():
    # A fully model-authored two-token completion after a one-token prompt.
    tokens = [1, 5, 6]  # prompt=bos, then two authored tokens
    loss_mask = [False, True, True]
    input_ids, labels = build_example(tokens, loss_mask, max_len=4)
    assert input_ids == [1, 5, 6, 0]
    #  t=0 -> idx1 authored -> 5 ; t=1 -> idx2 authored -> 6 ; t=2 last -> -100
    assert labels == [5, 6, IGNORE_INDEX, IGNORE_INDEX]


def test_build_batch_shapes():
    ex = [([1, 5, 6], [False, True, True]), ([1, 7, 8, 9], [False, True, True, True])]
    inputs, labels = build_batch(ex, max_len=5)
    assert inputs.shape == (2, 5)
    assert labels.shape == (2, 5)
    assert inputs.dtype == torch.long and labels.dtype == torch.long
    # padding present on the shorter example
    assert inputs[0].tolist() == [1, 5, 6, 0, 0]
