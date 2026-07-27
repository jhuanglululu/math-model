"""Named model and training variations — the two independent axes.

Configuration is hardcoded here as named pydantic variations rather than
exposed through free-form CLI flags: a variation that lives in code is
documented, diffable, and reviewable, and (model, training, seed) fully
determines a run. New experiment = new named variation, never a flag combo.

- Model variations (architecture): ``get_model_variation(name)`` -> GPTConfig.
- Training variations (recipe + composed env/puzzle/reward/trace configs, plus
  RL knobs for the later GRPO phase): ``get_training_variation(name)``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from mathrl.config import EnvConfig, PuzzleConfig, RewardConfig, TraceConfig
from mathrl.model import GPTConfig

# --------------------------------------------------------------------------- #
# Model variations (architecture axis)
# --------------------------------------------------------------------------- #

MODEL_VARIATIONS: dict[str, GPTConfig] = {
    # ~1M params — the required tiny model; `--training smoke --model tiny`
    # finishes on CPU in ~2 min.
    "tiny": GPTConfig(
        vocab_size=28,
        block_size=256,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.0,
    ),
    # ~15M params — the workhorse for real runs.
    "small": GPTConfig(
        vocab_size=28,
        block_size=256,
        n_layer=8,
        n_head=6,
        n_embd=384,
        dropout=0.1,
    ),
}


def get_model_variation(name: str) -> GPTConfig:
    """Resolve a model variation name to its GPTConfig."""
    try:
        return MODEL_VARIATIONS[name].model_copy(deep=True)
    except KeyError:
        raise KeyError(
            f"unknown model variation {name!r}; available: {sorted(MODEL_VARIATIONS)}"
        ) from None


# --------------------------------------------------------------------------- #
# Training variations (recipe axis)
# --------------------------------------------------------------------------- #


class TrainingVariation(BaseModel):
    """An SFT recipe plus composed env/puzzle/reward/trace configs and the RL
    knobs used later by the GRPO phase.

    The recipe is seed-independent; the seed is supplied at run time and,
    together with the model and training variation, fully determines the run.
    """

    name: str

    # --- SFT recipe ---
    steps: int = 20_000
    batch_size: int = 64
    lr: float = 3e-4
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 500
    samples: int = 100_000  # dataset size (number of generated SFT traces)

    # --- composed shared configs (from mathrl.config) ---
    puzzle: PuzzleConfig = Field(default_factory=PuzzleConfig)
    trace: TraceConfig = Field(default_factory=TraceConfig)
    env: EnvConfig = Field(default_factory=EnvConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)

    # --- RL knobs (used by the later GRPO phase; sensible defaults now) ---
    group_size: int = 8  # G: completions sampled per puzzle
    clip_eps: float = 0.2  # PPO/GRPO ratio clip epsilon
    kl_beta: float = 0.0  # KL-to-reference penalty coefficient
    rollout_batch: int = 32  # puzzles per rollout batch
    # Training-variation name whose checkpoint initializes the policy
    # (RL runs start from an SFT checkpoint: same model variation + seed,
    # checkpoints/<model>/<init_from>/<seed>/current.safetensors).
    init_from: str = ""


TRAINING_VARIATIONS: dict[str, TrainingVariation] = {
    # Required tiny/fast recipe: finishes ~2 min on CPU with the tiny model.
    "smoke": TrainingVariation(
        name="smoke",
        steps=50,
        batch_size=8,
        lr=3e-4,
        warmup_steps=5,
        weight_decay=0.1,
        grad_clip=1.0,
        eval_interval=10,
        samples=32,
    ),
    # Real SFT recipe, no tools.
    "sft_base": TrainingVariation(
        name="sft_base",
        steps=20_000,
        batch_size=64,
        lr=3e-4,
        warmup_steps=200,
        weight_decay=0.1,
        grad_clip=1.0,
        eval_interval=2000,
        samples=100_000,
        env=EnvConfig(tools="none"),
    ),
    # SFT arm with the calculate tool enabled.
    "sft_calc": TrainingVariation(
        name="sft_calc",
        steps=20_000,
        batch_size=64,
        lr=3e-4,
        warmup_steps=200,
        weight_decay=0.1,
        grad_clip=1.0,
        eval_interval=2000,
        samples=100_000,
        env=EnvConfig(tools="calculate"),
    ),
    # SFT arm with the verify tool enabled.
    "sft_verify": TrainingVariation(
        name="sft_verify",
        steps=20_000,
        batch_size=64,
        lr=3e-4,
        warmup_steps=200,
        weight_decay=0.1,
        grad_clip=1.0,
        eval_interval=2000,
        samples=100_000,
        env=EnvConfig(tools="verify"),
    ),
    # RL pipe-cleaner: run AFTER `--training smoke` (it initializes from the
    # smoke SFT checkpoint); finishes on CPU in ~1 min with the tiny model.
    "rl_smoke": TrainingVariation(
        name="rl_smoke",
        steps=3,
        lr=1e-5,
        warmup_steps=0,
        weight_decay=0.0,
        grad_clip=1.0,
        eval_interval=1,
        env=EnvConfig(tools="calculate", max_completion_len=64),
        group_size=2,
        clip_eps=0.2,
        kl_beta=0.0,
        rollout_batch=2,
        init_from="smoke",
    ),
    # Real RL recipe, no tools.
    "rl_base": TrainingVariation(
        name="rl_base",
        steps=1_000,
        lr=1e-5,
        warmup_steps=100,
        weight_decay=0.0,
        grad_clip=1.0,
        eval_interval=100,
        env=EnvConfig(tools="none"),
        group_size=8,
        clip_eps=0.2,
        kl_beta=0.0,
        rollout_batch=32,
        init_from="sft_base",
    ),
    # RL arm with the calculate tool enabled.
    "rl_calc": TrainingVariation(
        name="rl_calc",
        steps=1_000,
        lr=1e-5,
        warmup_steps=100,
        weight_decay=0.0,
        grad_clip=1.0,
        eval_interval=100,
        env=EnvConfig(tools="calculate"),
        group_size=8,
        clip_eps=0.2,
        kl_beta=0.0,
        rollout_batch=32,
        init_from="sft_calc",
    ),
    # RL arm with the verify tool enabled.
    "rl_verify": TrainingVariation(
        name="rl_verify",
        steps=1_000,
        lr=1e-5,
        warmup_steps=100,
        weight_decay=0.0,
        grad_clip=1.0,
        eval_interval=100,
        env=EnvConfig(tools="verify"),
        group_size=8,
        clip_eps=0.2,
        kl_beta=0.0,
        rollout_batch=32,
        init_from="sft_verify",
    ),
}


def get_training_variation(name: str) -> TrainingVariation:
    """Resolve a training variation name to its TrainingVariation."""
    try:
        return TRAINING_VARIATIONS[name].model_copy(deep=True)
    except KeyError:
        raise KeyError(
            f"unknown training variation {name!r}; available: {sorted(TRAINING_VARIATIONS)}"
        ) from None
