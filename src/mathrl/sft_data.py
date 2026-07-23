"""SFT batching: turn ``(tokens, loss_mask)`` traces into padded, shifted
input/label tensors for next-token cross-entropy.

The single most bug-prone spot in LLM training is label/input alignment and
masking, so the shift-and-mask logic lives here as a pure, unit-tested function
(``build_example``) rather than inline in the training loop.

Convention: logits at position ``t`` predict token ``t+1``. So the label at
position ``t`` is ``tokens[t+1]``, supervised (not ``-100``) only when that
*next* token is model-authored (``loss_mask[t+1]`` is True). This mirrors the
RL action mask exactly — the prompt and env-authored tokens (result digits,
``</calculate>``, ``<good>``/``<bad>``) stay in the input context but never
receive gradient. Padding and the final position are ``-100``.
"""

from __future__ import annotations

import torch

IGNORE_INDEX = -100
PAD_ID = 0


def build_example(
    tokens: list[int],
    loss_mask: list[bool],
    max_len: int,
    pad_id: int = PAD_ID,
) -> tuple[list[int], list[int]]:
    """Build one padded ``(input_ids, labels)`` pair with the causal shift and
    loss mask already baked in.

    - ``input_ids``: ``tokens`` padded to ``max_len`` with ``pad_id``.
    - ``labels[t]``: ``tokens[t+1]`` if that next token exists and is
      model-authored (``loss_mask[t+1]``), else ``IGNORE_INDEX``. The last real
      position and all padding positions are ``IGNORE_INDEX``.

    Both lists have length ``max_len``. ``tokens`` must fit within ``max_len``.
    """
    assert len(tokens) == len(loss_mask), "tokens and loss_mask length mismatch"
    assert len(tokens) <= max_len, f"trace length {len(tokens)} exceeds max_len {max_len}"

    input_ids = list(tokens) + [pad_id] * (max_len - len(tokens))

    labels = [IGNORE_INDEX] * max_len
    for t in range(len(tokens) - 1):
        nxt = t + 1
        if loss_mask[nxt]:
            labels[t] = tokens[nxt]
    return input_ids, labels


def build_batch(
    examples: list[tuple[list[int], list[bool]]],
    max_len: int,
    pad_id: int = PAD_ID,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack a list of ``(tokens, loss_mask)`` traces into fixed-shape
    ``(input_ids, labels)`` long tensors of shape ``(B, max_len)``."""
    inputs: list[list[int]] = []
    labels: list[list[int]] = []
    for tokens, mask in examples:
        inp, lab = build_example(tokens, mask, max_len, pad_id)
        inputs.append(inp)
        labels.append(lab)
    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )
