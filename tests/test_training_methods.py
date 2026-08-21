from __future__ import annotations

from dataclasses import replace

import pytest

from justatom.training.config import (
    AuxiliaryGradientConfig,
    AuxiliaryGradientMode,
    ExperimentConfig,
    ExperimentRole,
    MarginMode,
    TrainingMethod,
)
from justatom.training.methods import canonical_method_config, resolve_method


def test_canonical_profiles_have_exact_structural_components():
    vanilla = canonical_method_config(TrainingMethod.VANILLA)
    gate = canonical_method_config(TrainingMethod.ATOM_GATE)
    atomic = canonical_method_config(TrainingMethod.ATOMIC)

    assert not vanilla.objective.decoupled
    assert not gate.objective.decoupled
    assert not atomic.objective.decoupled
    assert not vanilla.alpha_gate.enabled and not vanilla.memory_bank.enabled
    assert not vanilla.anchor_bank.enabled
    assert gate.alpha_gate.enabled and not gate.memory_bank.enabled
    assert not gate.anchor_bank.enabled
    assert gate.alpha_gate.supervision_weight == pytest.approx(0.3)
    assert not hasattr(gate.alpha_gate, "mix_weight")
    assert not hasattr(gate.objective, "pairwise_margin")
    assert not atomic.alpha_gate.enabled and atomic.memory_bank.enabled
    assert not atomic.anchor_bank.enabled
    assert atomic.gradient_projection.enabled
    assert not atomic.objective.decoupled
    assert atomic.memory_bank.mining == "random"
    assert atomic.memory_bank.random_negatives == 12
    assert atomic.memory_bank.mass_ratio == pytest.approx(0.5)
    assert atomic.memory_bank.mass_ramp_steps == 20
    assert not atomic.memory_bank.adaptive.enabled
    assert atomic.memory_bank.margin.mode is MarginMode.OFF


@pytest.mark.parametrize("method", list(TrainingMethod))
def test_canonical_methods_reject_dcl_but_ablation_allows_it(method):
    config = canonical_method_config(method)
    dcl = replace(config, objective=replace(config.objective, decoupled=True))

    with pytest.raises(ValueError, match="coupled InfoNCE.*experiment.role=ablation"):
        resolve_method(dcl)

    ablation = replace(dcl, experiment=ExperimentConfig(role=ExperimentRole.ABLATION, seed=42))
    assert resolve_method(ablation).objective.decoupled


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


def test_atomic_requires_projection_and_rejects_alpha_gate():
    atomic = canonical_method_config(TrainingMethod.ATOMIC)
    gate = canonical_method_config(TrainingMethod.ATOM_GATE)

    with pytest.raises(ValueError, match="gradient_projection.enabled=true"):
        resolve_method(
            replace(
                atomic,
                gradient_projection=replace(atomic.gradient_projection, enabled=False),
            )
        )
    with pytest.raises(ValueError, match="does not permit alpha_gate"):
        resolve_method(replace(atomic, alpha_gate=gate.alpha_gate))


def test_atomic_anchor_bank_can_replace_memory_only_as_lora_ablation():
    atomic = canonical_method_config(TrainingMethod.ATOMIC)
    anchor = replace(
        atomic,
        experiment=ExperimentConfig(role=ExperimentRole.ABLATION, seed=42),
        model=replace(atomic.model, lora=replace(atomic.model.lora, enabled=True)),
        memory_bank=replace(atomic.memory_bank, enabled=False, size=0),
        anchor_bank=replace(atomic.anchor_bank, enabled=True, size=512),
    )

    resolved = resolve_method(anchor)

    assert resolved.anchor_bank.enabled
    assert not resolved.memory_bank.enabled
    with pytest.raises(ValueError, match="experiment.role=ablation"):
        resolve_method(replace(anchor, experiment=ExperimentConfig()))


@pytest.mark.parametrize("method", [TrainingMethod.VANILLA, TrainingMethod.ATOM_GATE])
def test_non_atomic_methods_reject_anchor_bank(method):
    config = canonical_method_config(method)
    invalid = replace(
        config,
        experiment=ExperimentConfig(role=ExperimentRole.ABLATION),
        model=replace(config.model, lora=replace(config.model.lora, enabled=True)),
        anchor_bank=replace(config.anchor_bank, enabled=True, size=8),
    )

    with pytest.raises(ValueError, match="anchor_bank"):
        resolve_method(invalid)


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


def test_vanilla_memory_bank_requires_ablation_role():
    vanilla = canonical_method_config(TrainingMethod.VANILLA)
    atomic = canonical_method_config(TrainingMethod.ATOMIC)
    bank = replace(
        atomic.memory_bank,
        adaptive=replace(atomic.memory_bank.adaptive, enabled=False),
        margin=replace(atomic.memory_bank.margin, mode=MarginMode.OFF, regularization_weight=0.0),
    )

    with pytest.raises(ValueError, match="canonical vanilla.*experiment.role=ablation"):
        resolve_method(replace(vanilla, memory_bank=bank))

    ablation = replace(
        vanilla,
        experiment=ExperimentConfig(role=ExperimentRole.ABLATION, seed=42),
        memory_bank=bank,
    )
    resolved = resolve_method(ablation)

    assert resolved.memory_bank.enabled
    assert resolved.memory_bank.size == 512
    assert not resolved.memory_bank.adaptive.enabled
    assert resolved.memory_bank.margin.mode is MarginMode.OFF


@pytest.mark.parametrize("method", list(TrainingMethod))
@pytest.mark.parametrize("mode", [AuxiliaryGradientMode.OBSERVE, AuxiliaryGradientMode.SAFE])
def test_non_off_auxiliary_gradient_requires_atom_gate_ablation(method, mode):
    config = replace(
        canonical_method_config(method),
        auxiliary_gradient=AuxiliaryGradientConfig(mode=mode),
    )

    with pytest.raises(ValueError, match="atom_gate.*ablation"):
        resolve_method(config)


@pytest.mark.parametrize("mode", [AuxiliaryGradientMode.OBSERVE, AuxiliaryGradientMode.SAFE])
def test_atom_gate_ablation_accepts_non_off_auxiliary_gradient(mode):
    config = replace(
        canonical_method_config(TrainingMethod.ATOM_GATE),
        experiment=ExperimentConfig(role=ExperimentRole.ABLATION, seed=42),
        auxiliary_gradient=AuxiliaryGradientConfig(mode=mode),
    )

    assert resolve_method(config).auxiliary_gradient.mode is mode
