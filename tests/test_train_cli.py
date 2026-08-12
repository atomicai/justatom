from __future__ import annotations

import pytest

from justatom.api import train


def test_train_cli_accepts_only_method_and_dotted_overrides():
    parsed = train._parse_args(
        [
            "--config",
            "configs/train.yaml",
            "--method",
            "atomic",
            "--memory-bank.size",
            "256",
        ]
    )

    assert parsed["overrides"]["method"] == "atomic"
    assert parsed["overrides"]["memory_bank"]["size"] == 256


@pytest.mark.parametrize("retired", ["atom", "e_alpha_gate", "atom_gate_bank", "atom_gate_dynamic"])
def test_train_cli_rejects_retired_method_aliases(retired):
    with pytest.raises(SystemExit):
        train._parse_args(["--method", retired])


def test_resolve_train_config_applies_dataset_preset_and_method_profile():
    config = train.resolve_train_config(config={"method": "atomic", "dataset": {"id": "justatom"}})

    assert config.method.value == "atomic"
    assert config.dataset.name_or_path == ".data/polaroids.ai.data.json"
    assert config.dataset.lazy is False
    assert config.memory_bank.enabled


def test_qwen_lora_vanilla_bank_recipe_resolves():
    config = train.resolve_train_config(
        config_path="configs/experiments/qwen3-06b-lora-vanilla-bank.yaml",
    )

    assert config.method.value == "vanilla"
    assert config.experiment.role.value == "ablation"
    assert config.model.lora.enabled
    assert config.optimization.lr_encoder == pytest.approx(2e-5)
    assert config.memory_bank.enabled
    assert not config.memory_bank.adaptive.enabled
    assert config.memory_bank.margin.mode.value == "off"
