from __future__ import annotations

import pytest

from justatom.training.config import ExperimentRole, MarginMode, TrainingMethod, parse_train_config, train_config_to_dict


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
    assert not config.alpha_gate.enabled
    assert config.memory_bank.enabled
    assert config.memory_bank.margin.mode is MarginMode.OFF
    assert config.gradient_projection.enabled
    assert not config.objective.decoupled


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
    assert payload["memory_bank"]["margin"]["mode"] == "off"
    assert payload["gradient_projection"]["enabled"] is True
    assert payload["reranker"]["enabled"] is False
    assert restored == config


def test_optional_transformers_reranker_is_typed_and_round_trippable(tmp_path):
    config = parse_train_config(
        {
            "method": "atomic",
            "reranker": {
                "enabled": True,
                "backend": "transformers",
                "local_files_only": True,
                "prefilter_hard_negatives": 16,
                "prefilter_random_negatives": 4,
                "negatives": 8,
                "strategy": "teacher_weighted",
                "teacher_temperature": 0.07,
                "teacher_weight_floor": 0.001,
                "cache": {"mode": "read-write", "path": str(tmp_path / "teacher.sqlite")},
            },
        }
    )

    assert config.reranker.enabled
    assert config.reranker.backend == "transformers"
    assert config.reranker.negatives == 8
    assert config.reranker.strategy == "teacher_weighted"
    assert config.reranker.teacher_temperature == 0.07
    assert config.reranker.teacher_weight_floor == 0.001
    assert parse_train_config(train_config_to_dict(config)) == config


def test_reranker_rejects_invalid_teacher_weighting_parameters():
    with pytest.raises(ValueError, match=r"reranker\.strategy"):
        parse_train_config({"method": "atomic", "reranker": {"strategy": "mystery"}})
    with pytest.raises(ValueError, match=r"reranker\.teacher_temperature"):
        parse_train_config({"method": "atomic", "reranker": {"teacher_temperature": 0.0}})
    with pytest.raises(ValueError, match=r"reranker\.teacher_weight_floor"):
        parse_train_config({"method": "atomic", "reranker": {"teacher_weight_floor": 1.1}})


def test_reranker_requires_a_memory_bank():
    with pytest.raises(ValueError, match="requires memory_bank.enabled"):
        parse_train_config({"method": "vanilla", "reranker": {"enabled": True}})


def test_read_only_reranker_cache_cannot_score_misses():
    with pytest.raises(ValueError, match="does not permit"):
        parse_train_config(
            {
                "method": "atomic",
                "reranker": {"enabled": True, "cache": {"mode": "read-only", "on_miss": "score"}},
            }
        )


def test_dataset_loader_options_are_typed_and_round_trippable():
    config = parse_train_config(
        {
            "method": "vanilla",
            "dataset": {
                "name_or_path": "owner/data",
                "lazy": True,
                "config": "russian",
                "drop_columns": ["photos", "embedding"],
            },
        }
    )

    assert config.dataset.lazy is True
    assert config.dataset.config == "russian"
    assert config.dataset.drop_columns == ("photos", "embedding")
    assert parse_train_config(train_config_to_dict(config)) == config


def test_dataset_lazy_must_be_boolean():
    with pytest.raises(ValueError, match=r"dataset\.lazy"):
        parse_train_config({"method": "vanilla", "dataset": {"lazy": "yes"}})


def test_lora_config_is_model_agnostic_and_round_trippable():
    config = parse_train_config(
        {
            "method": "atomic",
            "model": {
                "name_or_path": "intfloat/multilingual-e5-small",
                "lora": {
                    "enabled": True,
                    "rank": 8,
                    "alpha": 16,
                    "dropout": 0.05,
                    "target_modules": ["query", "value"],
                    "use_rslora": False,
                },
            },
        }
    )

    assert config.model.lora.enabled is True
    assert config.model.lora.target_modules == ("query", "value")
    assert parse_train_config(train_config_to_dict(config)) == config


def test_lora_rejects_non_hugging_face_pfbert():
    with pytest.raises(ValueError, match=r"pfbert is not supported"):
        parse_train_config(
            {
                "method": "vanilla",
                "model": {
                    "name_or_path": "justatom/pfbert",
                    "lora": {"enabled": True},
                },
            }
        )


@pytest.mark.parametrize("target_modules", [[], [""], 7])
def test_lora_rejects_invalid_target_modules(target_modules):
    with pytest.raises(ValueError, match=r"model\.lora\.target_modules"):
        parse_train_config(
            {
                "method": "vanilla",
                "model": {"lora": {"enabled": True, "target_modules": target_modules}},
            }
        )
