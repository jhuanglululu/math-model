"""Experiment record writer and training progress display.

Every run appends to ``records/<model>/<training>/<seed>/record.jsonl``: a
metadata line first, then one flat JSON object per logging interval (``step``
and ``eval`` lines). The record.jsonl is for post-hoc analysis
(``json.loads`` / pandas); tqdm is what the user watches live.

Reproducibility: a run is identified by code + model variation + training
variation + seed, so the meta line captures the config snapshot and the git
commit.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm


def git_commit() -> str:
    """Current commit hash, or ``"unknown"`` outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def record_dir(
    model_name: str,
    training_name: str,
    seed: int,
    root: str | Path = "records",
) -> Path:
    return Path(root) / model_name / training_name / str(seed)


class RunRecord:
    """Append-only writer for one run's ``record.jsonl``.

    Writing the meta line is the first thing that happens; subsequent
    ``log_step`` / ``log_eval`` calls append flat objects, always including
    ``step``.
    """

    def __init__(
        self,
        model_name: str,
        training_name: str,
        seed: int,
        config: dict[str, Any],
        baseline: str | None = None,
        root: str | Path = "records",
    ) -> None:
        self.model_name = model_name
        self.training_name = training_name
        self.seed = seed
        self.dir = record_dir(model_name, training_name, seed, root)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "record.jsonl"
        meta = {
            "type": "meta",
            "model": model_name,
            "training": training_name,
            "seed": seed,
            "config": config,
            "git_commit": git_commit(),
            "baseline": baseline,
            "started": datetime.now().isoformat(),
        }
        self._append(meta)

    def _append(self, obj: dict[str, Any]) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(obj) + "\n")

    def log_step(self, step: int, **fields: Any) -> None:
        """Append a flat ``{"type": "step", "step": step, ...}`` line."""
        self._append({"type": "step", "step": step, **fields})

    def log_eval(self, step: int, **fields: Any) -> None:
        """Append a flat ``{"type": "eval", "step": step, ...}`` line."""
        self._append({"type": "eval", "step": step, **fields})


# --------------------------------------------------------------------------- #
# Progress display (tqdm)
# --------------------------------------------------------------------------- #


def _fmt_time(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_validation_line(
    epoch: int,
    step: int,
    total: int,
    elapsed_sec: float,
    train_loss: float,
    val_loss: float,
) -> str:
    """The persistent validation line, digit-aligned per the skill spec:

    ``e2 | step  1350/3000 | 09:13 | loss 2.431 | val 2.512 | diff +0.081``

    ``diff = val - train`` is the generalization gap — a steadily growing diff
    is the earliest visible sign of overfitting, so it gets a fixed column.
    """
    diff = val_loss - train_loss
    return (
        f"e{epoch} | step {step:>5}/{total} | {_fmt_time(elapsed_sec)} | "
        f"loss {train_loss:6.3f} | val {val_loss:6.3f} | diff {diff:+6.3f}"
    )


class TrainingProgress:
    """Thin wrapper over tqdm implementing the skill's display conventions.

    - ``desc`` is ``<model>/<training> e<epoch>``.
    - Live train/val shown via ``set_postfix``.
    - On every validation a persistent, aligned line is written with
      ``tqdm.write`` so the val history stays readable above the moving bar.
    """

    def __init__(
        self,
        model_name: str,
        training_name: str,
        total_per_epoch: int,
        epoch: int = 1,
        **tqdm_kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.training_name = training_name
        self.total_per_epoch = total_per_epoch
        self.epoch = epoch
        self.bar = tqdm(total=total_per_epoch, **tqdm_kwargs)
        self._set_desc()

    def _set_desc(self) -> None:
        self.bar.set_description(f"{self.model_name}/{self.training_name} e{self.epoch}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.bar.reset(total=self.total_per_epoch)
        self._set_desc()

    def step(self, loss: float | None = None, val: float | None = None) -> None:
        """Advance the bar one step, updating the live loss/val postfix."""
        postfix: dict[str, str] = {}
        if loss is not None:
            postfix["loss"] = f"{loss:.3f}"
        if val is not None:
            postfix["val"] = f"{val:.3f}"
        if postfix:
            self.bar.set_postfix(postfix)
        self.bar.update(1)

    def validation(
        self,
        step: int,
        elapsed_sec: float,
        train_loss: float,
        val_loss: float,
    ) -> str:
        """Write the persistent validation line and refresh the postfix.

        Returns the written line (handy for tests / re-logging)."""
        self.bar.set_postfix({"loss": f"{train_loss:.3f}", "val": f"{val_loss:.3f}"})
        line = format_validation_line(
            self.epoch, step, self.total_per_epoch, elapsed_sec, train_loss, val_loss
        )
        self.bar.write(line)
        return line

    def close(self) -> None:
        self.bar.close()
