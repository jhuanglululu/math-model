"""Device selection and global seeding.

One ``get_device()`` helper so model/data code never hardcodes ``"cuda:0"`` or
calls ``.cuda()`` directly. On a CUDA box (shared, per project setup) it picks
the freest GPU by parsing ``nvidia-smi`` and raises if none is free, rather
than blindly landing on GPU 0 and OOMing a neighbour's job.
"""

from __future__ import annotations

import random
import subprocess

import numpy as np
import torch


def _pick_free_cuda_device() -> torch.device:
    """Choose the freest visible GPU via ``nvidia-smi``.

    Prefers a GPU with near-empty memory AND 0% utilization (the box is
    shared). Raises RuntimeError if none looks free.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # nvidia-smi missing/failing but torch sees CUDA: fall back to device 0.
        raise RuntimeError(
            f"CUDA is available but `nvidia-smi` could not be queried to pick a free GPU: {e}"
        ) from e

    # thresholds: treat a GPU as free if it is nearly idle.
    mem_free_mib = 1024  # allow up to ~1 GiB resident (drivers, other users' scraps)
    util_free_pct = 10

    candidates: list[tuple[int, int, int]] = []  # (index, mem_used, util)
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        idx, mem_used, util = (int(parts[0]), int(parts[1]), int(parts[2]))
        candidates.append((idx, mem_used, util))

    if not candidates:
        raise RuntimeError("CUDA is available but nvidia-smi returned no GPUs")

    free = [c for c in candidates if c[1] <= mem_free_mib and c[2] <= util_free_pct]
    if not free:
        detail = ", ".join(f"gpu{idx}: {mem}MiB, {util}%util" for idx, mem, util in candidates)
        raise RuntimeError(
            "no free GPU found on this shared box "
            f"(need <= {mem_free_mib}MiB used and <= {util_free_pct}% util); "
            f"current state: {detail}"
        )

    # freest first: least memory used, then least utilization.
    free.sort(key=lambda c: (c[1], c[2]))
    return torch.device(f"cuda:{free[0][0]}")


def get_device() -> torch.device:
    """Return the best available device: a free CUDA GPU, else MPS, else CPU."""
    if torch.cuda.is_available():
        return _pick_free_cuda_device()
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` (CPU + all CUDA devices).

    Call before step 0 so that (model, training, seed) fully determines a run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
