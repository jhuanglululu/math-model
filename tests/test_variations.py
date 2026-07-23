import pytest

from mathrl.config import EnvConfig, PuzzleConfig, RewardConfig, TraceConfig
from mathrl.model import GPTConfig
from mathrl.variations import (
    TrainingVariation,
    get_model_variation,
    get_training_variation,
)


def test_model_variations_resolve():
    tiny = get_model_variation("tiny")
    small = get_model_variation("small")
    assert isinstance(tiny, GPTConfig)
    assert isinstance(small, GPTConfig)
    assert tiny.n_layer == 4 and tiny.n_embd == 128 and tiny.block_size == 256
    assert small.n_layer == 8 and small.n_embd == 384 and small.block_size == 256


def test_training_variations_resolve():
    for name in ("smoke", "sft_base", "sft_calc", "sft_verify"):
        tv = get_training_variation(name)
        assert isinstance(tv, TrainingVariation)
        assert tv.name == name


def test_smoke_recipe_is_fast():
    tv = get_training_variation("smoke")
    assert tv.samples == 32
    assert tv.steps == 50
    assert tv.batch_size == 8


def test_composed_configs_present():
    tv = get_training_variation("sft_base")
    assert isinstance(tv.puzzle, PuzzleConfig)
    assert isinstance(tv.trace, TraceConfig)
    assert isinstance(tv.env, EnvConfig)
    assert isinstance(tv.reward, RewardConfig)
    # RL knobs exist with defaults
    assert tv.group_size > 0
    assert 0 < tv.clip_eps < 1
    assert tv.kl_beta >= 0
    assert tv.rollout_batch > 0


def test_tool_arms():
    assert get_training_variation("sft_base").env.tools == "none"
    assert get_training_variation("sft_calc").env.tools == "calculate"
    assert get_training_variation("sft_verify").env.tools == "verify"


def test_unknown_model_raises():
    with pytest.raises(KeyError, match="unknown model variation"):
        get_model_variation("nope")


def test_unknown_training_raises():
    with pytest.raises(KeyError, match="unknown training variation"):
        get_training_variation("nope")
