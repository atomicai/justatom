from __future__ import annotations

from dataclasses import replace

import pytest

from justatom.training.config import ExperimentConfig, ExperimentRole, MarginMode, TrainingMethod
from justatom.training.methods import canonical_method_config, resolve_method


def test_canonical_profiles_have_exact_structural_components():
    vanilla = canonical_method_config(TrainingMethod.VANILLA)
    gate = canonical_method_config(TrainingMethod.ATOM_GATE)
    atomic = canonical_method_config(TrainingMethod.ATOMIC)

    assert not vanilla.alpha_gate.enabled and not vanilla.memory_bank.enabled
    assert gate.alpha_gate.enabled and not gate.memory_bank.enabled
    assert atomic.alpha_gate.enabled and atomic.memory_bank.enabled
    assert atomic.memory_bank.adaptive.enabled
    assert atomic.memory_bank.margin.mode is MarginMode.QUERY


def test_canonical_atom_gate_rejects_bank():
    gate = canonical_method_config(TrainingMethod.ATOM_GATE)
    invalid = replace(gate, memory_bank=canonical_method_config(TrainingMethod.ATOMIC).memory_bank)

    with pytest.raises(ValueError, match="atom_gate.*memory_bank"):
        resolve_method(invalid)


def test_atomic_fixed_margin_requires_ablation_role():
    atomic = canonical_method_config(TrainingMethod.ATOMIC)
    constant = replace(
        atomic,
        memory_bank=replace(
            atomic.memory_bank,
            margin=replace(atomic.memory_bank.margin, mode=MarginMode.CONSTANT),
        ),
    )

    with pytest.raises(ValueError, match="experiment.role"):
        resolve_method(constant)

    ablation = replace(
        constant,
        experiment=ExperimentConfig(role=ExperimentRole.ABLATION, seed=42),
    )
    assert resolve_method(ablation).memory_bank.margin.mode is MarginMode.CONSTANT


def test_vanilla_rejects_alpha_even_for_ablation():
    vanilla = canonical_method_config(TrainingMethod.VANILLA)
    gate = canonical_method_config(TrainingMethod.ATOM_GATE)
    invalid = replace(
        vanilla,
        experiment=ExperimentConfig(role=ExperimentRole.ABLATION, seed=42),
        alpha_gate=gate.alpha_gate,
    )

    with pytest.raises(ValueError, match="vanilla.*alpha_gate"):
        resolve_method(invalid)
