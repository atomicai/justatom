from __future__ import annotations

from dataclasses import replace

from justatom.training.config import (
    AdaptiveBankConfig,
    AlphaGateConfig,
    ExperimentRole,
    MarginConfig,
    MarginMode,
    MemoryBankConfig,
    ObjectiveConfig,
    TrainConfig,
    TrainingMethod,
)


def canonical_method_config(method: TrainingMethod | str) -> TrainConfig:
    method = TrainingMethod(method)
    if method is TrainingMethod.VANILLA:
        return TrainConfig(method=method)

    objective = ObjectiveConfig(simcse_dropout_weight=0.1)
    alpha_gate = AlphaGateConfig(enabled=True, mix_weight=0.3)
    if method is TrainingMethod.ATOM_GATE:
        return TrainConfig(method=method, objective=objective, alpha_gate=alpha_gate)

    memory_bank = MemoryBankConfig(
        enabled=True,
        size=512,
        warmup_steps=50,
        mining="mixed",
        hard_negatives=4,
        random_negatives=12,
        hard_warmup_steps=120,
        hard_ramp_steps=200,
        adaptive=AdaptiveBankConfig(
            enabled=True,
            collision_threshold=0.0,
            collision_beta=0.05,
        ),
        margin=MarginConfig(
            mode=MarginMode.QUERY,
            base=0.05,
            scale=0.02,
            minimum=0.0,
            maximum=0.15,
            admission_beta=0.05,
            regularization_weight=50.0,
        ),
    )
    return TrainConfig(
        method=method,
        objective=objective,
        alpha_gate=alpha_gate,
        memory_bank=memory_bank,
    )


def resolve_method(config: TrainConfig) -> TrainConfig:
    method = config.method
    role = config.experiment.role
    gate = config.alpha_gate
    bank = config.memory_bank

    if method is TrainingMethod.VANILLA:
        if gate.enabled:
            raise ValueError("vanilla does not permit alpha_gate.enabled")
        if bank.enabled:
            raise ValueError("vanilla does not permit memory_bank.enabled")
        return config

    if not gate.enabled:
        raise ValueError(f"{method.value} requires alpha_gate.enabled=true")

    if method is TrainingMethod.ATOM_GATE:
        if bank.enabled:
            raise ValueError("atom_gate does not permit memory_bank.enabled")
        return config

    if not bank.enabled:
        raise ValueError("atomic requires memory_bank.enabled=true")
    if bank.size <= 0:
        raise ValueError("atomic requires memory_bank.size > 0")

    if role is ExperimentRole.CANONICAL:
        if not bank.adaptive.enabled:
            raise ValueError(
                "canonical atomic requires memory_bank.adaptive.enabled=true; use experiment.role=ablation for controls"
            )
        if bank.margin.mode is not MarginMode.QUERY:
            raise ValueError("canonical atomic requires query margin; set experiment.role=ablation for a fixed-margin control")

    if bank.margin.mode is MarginMode.QUERY and not bank.adaptive.enabled:
        raise ValueError("memory_bank.margin.mode=query requires memory_bank.adaptive.enabled=true")
    return config
