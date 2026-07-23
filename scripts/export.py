"""Export a training checkpoint to weights-only model.safetensors.

Usage:
    uv run scripts/export.py checkpoints/small/sft_base/0            # uses current
    uv run scripts/export.py checkpoints/small/sft_base/0/1000.safetensors
"""

import argparse

from mathrl.checkpoint import export


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", help="run directory (uses current) or a <step>.safetensors")
    ap.add_argument(
        "--out", default=None, help="output path (default: model.safetensors beside source)"
    )
    args = ap.parse_args()
    out = export(args.checkpoint, args.out)
    print(f"exported weights-only checkpoint to {out}")


if __name__ == "__main__":
    main()
