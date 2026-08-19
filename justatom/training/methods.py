from __future__ import annotations

from dataclasses import replace

from justatom.training.config import (
    AlphaGateConfig,
    AnchorControlMode,
    AuxiliaryGradientMode,
    ExperimentRole,
    GradientProjectionConfig,
    MarginMode,
    MemoryBankConfig,
    ObjectiveConfig,
    TrainConfig,
    TrainingMethod,
)


def canonical_method_config(method: TrainingMethod | str) -> TrainConfig:
    method = TrainingMethod(method)
    objective = ObjectiveConfig(decoupled=False)
    if method is TrainingMethod.VANILLA:
        return TrainConfig(method=method, objective=objective)

    atom_gate_objective = replace(objective, simcse_dropout_weight=0.1)
    alpha_gate = AlphaGateConfig(enabled=True, supervision_weight=0.3)
    if method is TrainingMethod.ATOM_GATE:
        return TrainConfig(method=method, objective=atom_gate_objective, alpha_gate=alpha_gate)

    memory_bank = MemoryBankConfig(
        enabled=True,
        size=512,
        warmup_steps=50,
        mass_ratio=0.5,
        mass_ramp_steps=20,
        mining="random",
        hard_negatives=0,
        random_negatives=12,
    )
    return TrainConfig(
        method=method,
        objective=objective,
        memory_bank=memory_bank,
        gradient_projection=GradientProjectionConfig(enabled=True),
    )


def resolve_method(config: TrainConfig) -> TrainConfig:
    method = config.method
    role = config.experiment.role
    gate = config.alpha_gate
    bank = config.memory_bank
    anchor = config.anchor_bank
    auxiliary = config.auxiliary_gradient

    if auxiliary.mode is not AuxiliaryGradientMode.OFF and (
        method is not TrainingMethod.ATOM_GATE or role is not ExperimentRole.ABLATION
    ):
        raise ValueError("auxiliary_gradient mode requires atom_gate with experiment.role=ablation")

    if role is ExperimentRole.CANONICAL and config.objective.decoupled:
        raise ValueError(f"canonical {method.value} requires coupled InfoNCE; use experiment.role=ablation for DCL")

    if method is TrainingMethod.VANILLA:
        if gate.enabled:
            raise ValueError("vanilla does not permit alpha_gate.enabled")
        if anchor.enabled:
            raise ValueError("vanilla does not permit anchor_bank.enabled")
        if config.gradient_projection.enabled:
            raise ValueError("vanilla does not permit gradient_projection.enabled")
        if bank.enabled:
            if role is not ExperimentRole.ABLATION:
                raise ValueError("canonical vanilla does not permit memory_bank.enabled; use experiment.role=ablation")
            if bank.size <= 0:
                raise ValueError("vanilla bank ablation requires memory_bank.size > 0")
            if bank.margin.mode is MarginMode.QUERY and not bank.adaptive.enabled:
                raise ValueError("memory_bank.margin.mode=query requires memory_bank.adaptive.enabled=true")
        return config

    if not gate.enabled and method is TrainingMethod.ATOM_GATE:
        raise ValueError("atom_gate requires alpha_gate.enabled=true")

    if method is TrainingMethod.ATOM_GATE:
        if anchor.enabled:
            raise ValueError("atom_gate does not permit anchor_bank.enabled")
        if config.gradient_projection.enabled:
            raise ValueError("atom_gate does not permit gradient_projection.enabled")
        if bank.enabled:
            raise ValueError("atom_gate does not permit memory_bank.enabled")
        return config

    if gate.enabled:
        raise ValueError("atomic does not permit alpha_gate.enabled; memory gradients are controlled by projection")
    if not bank.enabled and not anchor.enabled:
        raise ValueError("atomic requires memory_bank.enabled=true or anchor_bank.enabled=true")
    if bank.enabled and bank.size <= 0:
        raise ValueError("atomic requires memory_bank.size > 0")
    additive_anchor = anchor.enabled and anchor.control is AnchorControlMode.ADDITIVE
    if not config.gradient_projection.enabled and not additive_anchor:
        raise ValueError("atomic requires gradient_projection.enabled=true unless anchor_bank.control=additive")

    if anchor.enabled:
        if role is not ExperimentRole.ABLATION:
            raise ValueError("atomic anchor bank requires experiment.role=ablation")
        if anchor.size <= 0:
            raise ValueError("atomic anchor bank requires anchor_bank.size > 0")
        if not config.model.lora.enabled:
            raise ValueError("atomic anchor bank requires model.lora.enabled=true")
        if additive_anchor:
            if bank.enabled:
                raise ValueError("additive anchor ablation does not permit memory_bank.enabled")
            if config.gradient_projection.enabled:
                raise ValueError("anchor_bank.control=additive requires gradient_projection.enabled=false")
        elif not config.gradient_projection.enabled:
            raise ValueError("anchor_bank.control=projection requires gradient_projection.enabled=true")

    if role is ExperimentRole.CANONICAL:
        if bank.margin.mode is not MarginMode.OFF:
            raise ValueError("canonical atomic keeps memory margin off; use experiment.role=ablation to enable it")
        if bank.adaptive.enabled:
            raise ValueError("canonical atomic keeps adaptive bank weights off; use experiment.role=ablation to enable them")

    if bank.margin.mode is MarginMode.QUERY and not bank.adaptive.enabled:
        raise ValueError("memory_bank.margin.mode=query requires memory_bank.adaptive.enabled=true")
    return config
