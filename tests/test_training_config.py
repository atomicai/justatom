from __future__ import annotations

import pytest

from justatom.training.config import (
    ExperimentRole,
    MarginMode,
    TrainingMethod,
    parse_train_config,
    train_config_to_dict,
)


def test_parse_train_config_builds_typed_atomic_config():
    config = parse_train_config(
        {
            "method": "atomic",
            "experiment": {"role": "canonical", "seed": 42},
            "model": {"name_or_path": "intfloat/multilingual-e5-small"},
            "dataset": {"name_or_path": "justatom/mmarco-ru-selected"},
        }
    )

    assert config.method is TrainingMethod.ATOMIC
    assert config.experiment.role is ExperimentRole.CANONICAL
    assert config.experiment.seed == 42
    assert config.alpha_gate.enabled
    assert config.memory_bank.enabled
    assert config.memory_bank.margin.mode is MarginMode.QUERY


def test_parse_train_config_rejects_unknown_nested_field():
    with pytest.raises(ValueError, match=r"memory_bank\.mystery"):
        parse_train_config(
            {
                "method": "atomic",
                "memory_bank": {"mystery": 7},
            }
        )


def test_parse_train_config_rejects_invalid_beta_before_model_load():
    with pytest.raises(ValueError, match=r"memory_bank\.adaptive\.collision_beta"):
        parse_train_config(
            {
                "method": "atomic",
                "memory_bank": {"adaptive": {"collision_beta": 0.0}},
            }
        )


def test_parse_train_config_keeps_dataset_preset_metadata():
    config = parse_train_config(
        {
            "method": "vanilla",
            "dataset": {
                "id": "mmarco-ru-selected",
                "name_or_path": "justatom/mmarco-ru-selected",
                "labels_field": "query",
                "content_field": "positive",
                "display_name": "mMARCO-ru-selected",
                "selection": {"seed": 42, "train_rows": 50_000},
                "corpus": {"split": "corpus"},
            },
        }
    )

    assert config.dataset.id == "mmarco-ru-selected"
    assert config.dataset.metadata["display_name"] == "mMARCO-ru-selected"
    assert config.dataset.metadata["selection"]["train_rows"] == 50_000


def test_train_config_serialization_is_plain_and_round_trippable():
    config = parse_train_config({"method": "atomic"})

    payload = train_config_to_dict(config)
    restored = parse_train_config(payload)

    assert payload["method"] == "atomic"
    assert payload["experiment"]["role"] == "canonical"
    assert payload["memory_bank"]["margin"]["mode"] == "query"
    assert restored == config
