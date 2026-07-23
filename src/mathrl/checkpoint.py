"""Checkpoint save / resume / export using safetensors.

Layout (shared addressing scheme with records):

    checkpoints/<model>/<training>/<seed>/
        <step>.safetensors        # model weights only
        <step>.opt.safetensors    # optimizer state (flattened tensors)
        <step>.json               # sidecar: step, RNG, config, metric, opt meta
        current.safetensors       # latest state, used by resume
        current.opt.safetensors
        current.json

Design choices, kept deliberately simple and documented:

- **safetensors, never .pt.** Model weights go through ``safetensors.torch``'s
  ``save_model`` / ``load_model`` so the tied ``wte`` / ``lm_head`` weight is
  handled correctly (a plain ``save_file`` rejects shared tensors).
- **Optimizer state lives in a separate ``<step>.opt.safetensors``.** Its
  ``state_dict`` is a nested dict of tensors keyed by parameter index; we
  *flatten* it to a flat ``{"<pid>|<key>": tensor}`` map for safetensors and
  keep the non-tensor structure (``param_groups``, which keys were tensors) in
  the sidecar JSON. See ``_flatten_optimizer`` / ``_unflatten_optimizer``.
- **Non-tensor state (step counter, RNG states, config snapshot, metric)**
  goes in the sidecar JSON with the same stem.
- **Keep the 3 best by a caller-supplied metric, plus the latest step**, and
  always overwrite ``current.*`` to the latest state for resume.

All paths derive from ``(model_name, training_name, seed)`` under a ``root``
that tests override to a tmp dir.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, load_model, save_file, save_model

# --------------------------------------------------------------------------- #
# RNG capture / restore (stored in the sidecar JSON)
# --------------------------------------------------------------------------- #


def _capture_rng() -> dict[str, Any]:
    py = random.getstate()
    np_state = np.random.get_state()
    rng: dict[str, Any] = {
        "python": [py[0], list(py[1]), py[2]],
        "numpy": [
            np_state[0],
            [int(x) for x in np_state[1]],
            int(np_state[2]),
            int(np_state[3]),
            float(np_state[4]),
        ],
        "torch": torch.get_rng_state().tolist(),
    }
    if torch.cuda.is_available():
        rng["torch_cuda"] = [s.tolist() for s in torch.cuda.get_rng_state_all()]
    return rng


def _restore_rng(rng: dict[str, Any]) -> None:
    py = rng["python"]
    random.setstate((py[0], tuple(py[1]), py[2]))
    nps = rng["numpy"]
    np.random.set_state((nps[0], np.array(nps[1], dtype=np.uint32), nps[2], nps[3], nps[4]))
    torch.set_rng_state(torch.tensor(rng["torch"], dtype=torch.uint8))
    if "torch_cuda" in rng and torch.cuda.is_available():
        states = [torch.tensor(s, dtype=torch.uint8) for s in rng["torch_cuda"]]
        torch.cuda.set_rng_state_all(states)


# --------------------------------------------------------------------------- #
# Optimizer flatten / unflatten
# --------------------------------------------------------------------------- #


def _flatten_optimizer(optimizer: torch.optim.Optimizer) -> tuple[dict, dict]:
    sd = optimizer.state_dict()
    tensors: dict[str, torch.Tensor] = {}
    meta_state: dict[str, dict[str, Any]] = {}
    for pid, st in sd["state"].items():
        meta_state[str(pid)] = {}
        for k, v in st.items():
            if isinstance(v, torch.Tensor):
                tensors[f"{pid}|{k}"] = v.detach().cpu().contiguous()
                meta_state[str(pid)][k] = "__tensor__"
            else:
                meta_state[str(pid)][k] = v
    opt_meta = {"param_groups": sd["param_groups"], "state": meta_state}
    return tensors, opt_meta


def _unflatten_optimizer(tensors: dict[str, torch.Tensor], opt_meta: dict) -> dict:
    state: dict[int, dict[str, Any]] = {}
    for pid_str, subkeys in opt_meta["state"].items():
        pid = int(pid_str)
        st: dict[str, Any] = {}
        for k, marker in subkeys.items():
            if marker == "__tensor__":
                st[k] = tensors[f"{pid}|{k}"]
            else:
                st[k] = marker
        state[pid] = st
    return {"state": state, "param_groups": opt_meta["param_groups"]}


# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #


def run_dir(
    model_name: str,
    training_name: str,
    seed: int,
    root: str | Path = "checkpoints",
) -> Path:
    return Path(root) / model_name / training_name / str(seed)


def _weights_to_stem(weights_path: Path) -> Path:
    """``dir/100.safetensors`` -> ``dir/100`` (also handles ``current``)."""
    name = weights_path.name
    assert name.endswith(".safetensors"), f"expected .safetensors, got {name}"
    return weights_path.with_name(name[: -len(".safetensors")])


# --------------------------------------------------------------------------- #
# Checkpointer
# --------------------------------------------------------------------------- #


class Checkpointer:
    """Manages numbered checkpoints for one run and prunes to the best few."""

    def __init__(
        self,
        model_name: str,
        training_name: str,
        seed: int,
        root: str | Path = "checkpoints",
        keep_best: int = 3,
        higher_is_better: bool = False,
    ) -> None:
        self.dir = run_dir(model_name, training_name, seed, root)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_best = keep_best
        self.higher_is_better = higher_is_better
        # (step, metric) history, rebuilt from existing sidecars so pruning
        # survives a restart.
        self._saved: dict[int, float | None] = {}
        for jf in self.dir.glob("*.json"):
            if jf.stem == "current":
                continue
            try:
                step = int(jf.stem)
            except ValueError:
                continue
            meta = json.loads(jf.read_text())
            self._saved[step] = meta.get("metric")

    def _write(
        self,
        stem: Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        sidecar: dict[str, Any],
    ) -> None:
        save_model(model, str(stem.with_name(stem.name + ".safetensors")))
        opt_meta = None
        if optimizer is not None:
            tensors, opt_meta = _flatten_optimizer(optimizer)
            # safetensors rejects an empty dict; only write when there is state.
            if tensors:
                save_file(tensors, str(stem.with_name(stem.name + ".opt.safetensors")))
        sidecar = {**sidecar, "opt_meta": opt_meta}
        stem.with_name(stem.name + ".json").write_text(json.dumps(sidecar))

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        metric: float | None = None,
        config: dict[str, Any] | None = None,
    ) -> Path:
        """Write ``<step>.*`` and refresh ``current.*``; then prune. Returns
        the path to the numbered weights file."""
        sidecar = {
            "step": step,
            "metric": metric,
            "config": config,
            "rng": _capture_rng(),
        }
        self._write(self.dir / str(step), model, optimizer, sidecar)
        self._write(self.dir / "current", model, optimizer, sidecar)
        self._saved[step] = metric
        self._prune()
        return self.dir / f"{step}.safetensors"

    def _sort_key(self, step: int) -> tuple:
        metric = self._saved[step]
        if metric is None:
            return (0, 0.0)  # unranked -> worst
        return (1, metric if self.higher_is_better else -metric)

    def _prune(self) -> None:
        if not self._saved:
            return
        latest = max(self._saved)
        keep = {latest}
        ranked = sorted(self._saved, key=self._sort_key, reverse=True)
        keep.update(ranked[: self.keep_best])
        for step in list(self._saved):
            if step in keep:
                continue
            stem = self.dir / str(step)
            for suffix in (".safetensors", ".opt.safetensors", ".json"):
                f = stem.with_name(stem.name + suffix)
                f.unlink(missing_ok=True)
            del self._saved[step]


# --------------------------------------------------------------------------- #
# Resume / export (standalone, given explicit paths)
# --------------------------------------------------------------------------- #


def resume(
    path_or_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    """Restore model (and optimizer, if given) plus step and RNG state.

    ``path_or_dir`` is either a run directory (resumes from ``current``) or a
    specific ``<step>.safetensors`` weights file. Returns the restored step.
    """
    p = Path(path_or_dir)
    weights = p / "current.safetensors" if p.is_dir() else p
    if not weights.exists():
        raise FileNotFoundError(f"no checkpoint weights at {weights}")
    stem = _weights_to_stem(weights)
    sidecar = json.loads(stem.with_name(stem.name + ".json").read_text())

    load_model(model, str(weights))

    if optimizer is not None and sidecar.get("opt_meta") is not None:
        opt_file = stem.with_name(stem.name + ".opt.safetensors")
        tensors = load_file(str(opt_file)) if opt_file.exists() else {}
        optimizer.load_state_dict(_unflatten_optimizer(tensors, sidecar["opt_meta"]))

    _restore_rng(sidecar["rng"])
    return int(sidecar["step"])


def export(checkpoint: str | Path, out_path: str | Path | None = None) -> Path:
    """Strip a checkpoint down to weights-only ``model.safetensors``.

    ``checkpoint`` is a run directory (uses ``current``) or a specific
    ``<step>.safetensors``. Because model weights already live in their own
    file (optimizer state is separate), export is a copy of that weights file
    to ``model.safetensors`` next to the source (or ``out_path``).
    """
    p = Path(checkpoint)
    weights = p / "current.safetensors" if p.is_dir() else p
    if not weights.exists():
        raise FileNotFoundError(f"no checkpoint weights at {weights}")
    out = Path(out_path) if out_path is not None else weights.parent / "model.safetensors"
    shutil.copyfile(weights, out)
    return out
