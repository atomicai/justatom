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
    assert config.memory_bank.mass_ratio == pytest.approx(0.5)
    assert config.memory_bank.mass_ramp_steps == 20
    assert config.memory_bank.margin.mode is MarginMode.OFF
    assert config.gradient_projection.enabled
    assert not config.objective.decoupled


def test_alpha_supervision_schema_round_trips():
    config = parse_train_config({"method": "atom_gate", "alpha_gate": {"supervision_weight": 0.4}})
    payload = train_config_to_dict(config)

    assert config.alpha_gate.supervision_weight == pytest.approx(0.4)
    assert "mix_weight" not in payload["alpha_gate"]
    assert "pairwise_margin" not in payload["objective"]
    assert parse_train_config(payload) == config


def test_auxiliary_temperatures_are_optional_and_round_trip():
    legacy = parse_train_config({"method": "atom_gate"})
    assert legacy.objective.simcse_temperature is None
    assert legacy.alpha_gate.target_temperature is None

    config = parse_train_config(
        {
            "method": "atom_gate",
            "objective": {"simcse_temperature": 0.2},
            "alpha_gate": {"target_temperature": 0.3},
        }
    )
    payload = train_config_to_dict(config)

    assert config.objective.simcse_temperature == pytest.approx(0.2)
    assert config.alpha_gate.target_temperature == pytest.approx(0.3)
    assert payload["objective"]["simcse_temperature"] == pytest.approx(0.2)
    assert payload["alpha_gate"]["target_temperature"] == pytest.approx(0.3)
    assert parse_train_config(payload) == config


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("objective", "simcse_temperature"),
        ("alpha_gate", "target_temperature"),
    ],
)
@pytest.mark.parametrize("value", [0.0, -0.1, float("nan"), float("inf"), float("-inf")])
def test_auxiliary_temperatures_must_be_finite_and_positive(section, field, value):
    with pytest.raises(ValueError, match=rf"{section}\.{field}"):
        parse_train_config({"method": "atom_gate", section: {field: value}})


@pytest.mark.parametrize("value", [1e-4, 1.01])
def test_simcse_temperature_must_fit_the_contrastive_kernel_range(value):
    with pytest.raises(ValueError, match=r"objective\.simcse_temperature"):
        parse_train_config({"method": "atom_gate", "objective": {"simcse_temperature": value}})


@pytest.mark.parametrize("value", [1e-3, 1.0])
def test_simcse_temperature_accepts_contrastive_kernel_boundaries(value):
    config = parse_train_config({"method": "atom_gate", "objective": {"simcse_temperature": value}})
    assert config.objective.simcse_temperature == pytest.approx(value)


@pytest.mark.parametrize("field", ["mix_weight", "mix_weight_warmup_steps", "entropy_weight"])
def test_retired_alpha_fields_are_rejected(field):
    with pytest.raises(ValueError, match=rf"unknown configuration field: alpha_gate\.{field}"):
        parse_train_config({"method": "atom_gate", "alpha_gate": {field: 0}})


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_alpha_supervision_weight_is_finite_and_non_negative(value):
    with pytest.raises(ValueError, match=r"alpha_gate\.supervision_weight"):
        parse_train_config({"method": "atom_gate", "alpha_gate": {"supervision_weight": value}})


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_memory_mass_ratio_is_finite_and_non_negative(value):
    with pytest.raises(ValueError, match=r"memory_bank\.mass_ratio"):
        parse_train_config({"method": "atomic", "memory_bank": {"mass_ratio": value}})


def test_memory_mass_ramp_is_positive():
    with pytest.raises(ValueError, match=r"memory_bank\.mass_ramp_steps"):
        parse_train_config({"method": "atomic", "memory_bank": {"mass_ramp_steps": 0}})


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
    assert restored == config


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
