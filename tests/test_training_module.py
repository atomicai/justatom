from __future__ import annotations

from dataclasses import replace

import pytest
import pytorch_lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader

from justatom.training import telemetry
from justatom.training.alpha_gate import QueryAlphaGate
from justatom.training.config import (
    ExperimentConfig,
    ExperimentRole,
    MarginMode,
    TrainingMethod,
    train_config_to_dict,
)
from justatom.training.methods import canonical_method_config
from justatom.training.module import ContrastiveTrainingModule
from justatom.training.objective import ObjectiveOutput


class TinyEncoder(nn.Module):
    output_dims = 4

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4)

    def encode_pair(self, batch):
        return (
            torch.nn.functional.normalize(self.projection(batch["queries"]), dim=-1),
            torch.nn.functional.normalize(self.projection(batch["documents"]), dim=-1),
        )

    def encode_queries(self, batch):
        return torch.nn.functional.normalize(self.projection(batch["queries"]), dim=-1)


def tiny_batch():
    return {
        "queries": torch.eye(2, 4),
        "documents": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        ),
        "doc_key_id": torch.tensor([1, 2]),
        "content_key_id": torch.tensor([11, 12]),
        "query_key_id": torch.tensor([21, 22]),
    }


def finite_output():
    loss = torch.tensor(1.0, requires_grad=True)
    return ObjectiveOutput(
        loss=loss,
        primary_loss=loss,
        memory_loss=None,
        main_per_row=torch.ones(2),
        memory_per_row=None,
        simcse_per_row=None,
        soft_fn_per_row=None,
        alpha_target=None,
        alpha_supervision_per_row=None,
        metrics={},
    )


def test_module_constructs_only_components_required_by_method():
    vanilla = ContrastiveTrainingModule.build(TinyEncoder(), canonical_method_config(TrainingMethod.VANILLA))
    gate = ContrastiveTrainingModule.build(TinyEncoder(), canonical_method_config(TrainingMethod.ATOM_GATE))
    atomic = ContrastiveTrainingModule.build(TinyEncoder(), canonical_method_config(TrainingMethod.ATOMIC))

    assert vanilla.alpha_gate is None and vanilla.memory_bank is None and vanilla.margin_head is None
    assert isinstance(gate.alpha_gate, QueryAlphaGate) and gate.memory_bank is None
    assert atomic.alpha_gate is None and atomic.memory_bank is not None and atomic.margin_head is None
    assert atomic.config.gradient_projection.enabled
    assert atomic.automatic_optimization is False


def test_load_schema_v1_checkpoint_migrates_historical_canonical_dcl_to_ablation(tmp_path, monkeypatch):
    config = canonical_method_config(TrainingMethod.ATOM_GATE)
    original = ContrastiveTrainingModule.build(TinyEncoder(), config)
    historical_config = train_config_to_dict(config)
    historical_config["objective"].update(decoupled=True, pairwise_margin=0.2)
    historical_config["alpha_gate"].pop("supervision_weight")
    historical_config["alpha_gate"].update(
        mix_weight=0.7,
        mix_weight_warmup_steps=10,
        entropy_weight=0.1,
    )
    checkpoint = tmp_path / "historical-checkpoint.pt"
    payload = {
        "schema_version": 1,
        "resolved_config": historical_config,
        "state_dict": original.state_dict(),
        "optimizer_states": [],
        "epoch": 0,
        "global_step": 1,
    }
    monkeypatch.setattr("justatom.training.module.torch.load", lambda *_args, **_kwargs: payload)

    restored, optimizer_states = ContrastiveTrainingModule.load_research_checkpoint(
        checkpoint,
        encoder=TinyEncoder(),
    )

    assert restored.config.experiment.role is ExperimentRole.ABLATION
    assert restored.config.objective.decoupled
    assert restored.config.alpha_gate.supervision_weight == 0.7
    assert "pairwise_margin" not in restored.config.objective.__dict__
    assert payload["resolved_config"]["objective"]["pairwise_margin"] == 0.2
    assert payload["resolved_config"]["alpha_gate"] == {
        "enabled": True,
        "head": {"layers": 1, "hidden_dim": None, "dropout": 0.0, "activation": "gelu"},
        "mix_weight": 0.7,
        "mix_weight_warmup_steps": 10,
        "entropy_weight": 0.1,
    }
    for name, value in original.alpha_gate.state_dict().items():
        torch.testing.assert_close(restored.alpha_gate.state_dict()[name], value)
    assert optimizer_states == []


def test_load_schema_v1_checkpoint_migrates_historical_atom_gate_to_ablation(tmp_path, monkeypatch):
    config = canonical_method_config(TrainingMethod.ATOM_GATE)
    original = ContrastiveTrainingModule.build(TinyEncoder(), config)
    payload = {
        "schema_version": 1,
        "resolved_config": train_config_to_dict(config),
        "state_dict": original.state_dict(),
    }
    monkeypatch.setattr("justatom.training.module.torch.load", lambda *_args, **_kwargs: payload)

    restored, _ = ContrastiveTrainingModule.load_research_checkpoint(
        tmp_path / "atom-gate-checkpoint.pt",
        encoder=TinyEncoder(),
    )

    assert restored.config.experiment.role is ExperimentRole.ABLATION


@pytest.mark.parametrize("method", [TrainingMethod.VANILLA, TrainingMethod.ATOMIC])
def test_load_schema_v1_checkpoint_preserves_canonical_coupled_methods(tmp_path, monkeypatch, method):
    config = canonical_method_config(method)
    original = ContrastiveTrainingModule.build(TinyEncoder(), config)
    payload = {
        "schema_version": 1,
        "resolved_config": train_config_to_dict(config),
        "state_dict": original.state_dict(),
    }
    monkeypatch.setattr("justatom.training.module.torch.load", lambda *_args, **_kwargs: payload)

    restored, optimizer_states = ContrastiveTrainingModule.load_research_checkpoint(
        tmp_path / "canonical-checkpoint.pt",
        encoder=TinyEncoder(),
    )

    assert restored.config.experiment.role is ExperimentRole.CANONICAL
    assert not restored.config.objective.decoupled
    assert optimizer_states == []


def test_research_checkpoint_uses_schema_v2(tmp_path):
    module = ContrastiveTrainingModule.build(
        TinyEncoder(),
        canonical_method_config(TrainingMethod.VANILLA),
    )

    checkpoint = module.save_research_checkpoint(tmp_path / "checkpoint.pt")

    payload = torch.load(checkpoint, weights_only=False)
    assert payload["schema_version"] == 2

    restored, optimizer_states = ContrastiveTrainingModule.load_research_checkpoint(
        checkpoint,
        encoder=TinyEncoder(),
    )

    assert restored.config == module.config
    assert optimizer_states == []


def test_vanilla_bank_ablation_constructs_bank_without_gate_or_margin_head():
    vanilla = canonical_method_config(TrainingMethod.VANILLA)
    atomic = canonical_method_config(TrainingMethod.ATOMIC)
    config = replace(
        vanilla,
        experiment=ExperimentConfig(role=ExperimentRole.ABLATION, seed=42),
        memory_bank=replace(
            atomic.memory_bank,
            adaptive=replace(atomic.memory_bank.adaptive, enabled=False),
            margin=replace(atomic.memory_bank.margin, mode=MarginMode.OFF, regularization_weight=0.0),
        ),
    )

    module = ContrastiveTrainingModule.build(TinyEncoder(), config)

    assert module.alpha_gate is None
    assert module.memory_bank is not None
    assert module.margin_head is None


def test_optimizer_contains_all_atomic_parameters_once():
    module = ContrastiveTrainingModule.build(TinyEncoder(), canonical_method_config(TrainingMethod.ATOMIC))

    optimizer = module.configure_optimizers()
    parameter_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    expected = [id(parameter) for parameter in module.parameters() if parameter.requires_grad]

    assert sorted(parameter_ids) == sorted(expected)
    assert len(parameter_ids) == len(set(parameter_ids))


def test_atomic_enqueues_documents_after_objective(monkeypatch):
    module = ContrastiveTrainingModule.build(TinyEncoder(), canonical_method_config(TrainingMethod.ATOMIC))
    events: list[str] = []
    monkeypatch.setattr(module, "_encode_dropout_query_view", lambda batch: module.encoder.encode_queries(batch))
    monkeypatch.setattr(
        module.objective,
        "forward",
        lambda *args, **kwargs: events.append("loss") or finite_output(),
    )
    monkeypatch.setattr(module.memory_bank, "enqueue", lambda *args, **kwargs: events.append("enqueue"))

    module.compute_training_step(tiny_batch(), step=0)

    assert events == ["loss", "enqueue"]


def test_atom_gate_bce_updates_only_head_parameters():
    module = ContrastiveTrainingModule.build(
        TinyEncoder(),
        canonical_method_config(TrainingMethod.ATOM_GATE),
    )

    output = module.compute_training_step(tiny_batch(), step=0)
    assert output.alpha_supervision_per_row is not None
    output.alpha_supervision_per_row.mean().backward()

    assert all(parameter.grad is None for parameter in module.encoder.parameters())
    assert module.objective.kernel.log_tau.grad is None
    assert module.alpha_gate is not None
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() and float(parameter.grad.abs().sum()) > 0.0
        for parameter in module.alpha_gate.parameters()
    )


def test_atom_gate_reports_calibration():
    module = ContrastiveTrainingModule.build(
        TinyEncoder(),
        canonical_method_config(TrainingMethod.ATOM_GATE),
    )

    output = module.compute_training_step(tiny_batch(), step=0)

    assert 0.0 <= output.metrics["alpha_target/mean"] <= 1.0
    assert output.metrics["alpha/absolute_error_mean"] >= 0.0


def test_retrieval_metrics_are_bucketed_by_target_confidence():
    scores = torch.tensor([[3.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
    confidence = torch.tensor([0.2, 0.5, 0.8])

    metrics = telemetry.retrieval_metrics_by_confidence(scores, confidence)

    assert metrics["alpha_target_bucket/low/count"] == 1.0
    assert metrics["alpha_target_bucket/low/hit_rate_at_1"] == 1.0
    assert metrics["alpha_target_bucket/medium/count"] == 1.0
    assert metrics["alpha_target_bucket/medium/hit_rate_at_1"] == 0.0
    assert metrics["alpha_target_bucket/high/count"] == 1.0
    assert metrics["alpha_target_bucket/high/hit_rate_at_1"] == 1.0


def test_retrieval_metrics_report_nan_for_empty_confidence_bucket():
    metrics = telemetry.retrieval_metrics_by_confidence(
        torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        torch.tensor([0.2, 0.25]),
    )

    for bucket in ("low", "medium", "high"):
        assert f"alpha_target_bucket/{bucket}/count" in metrics
        assert f"alpha_target_bucket/{bucket}/hit_rate_at_1" in metrics
        assert f"alpha_target_bucket/{bucket}/mrr" in metrics
    assert metrics["alpha_target_bucket/medium/count"] == 0.0
    assert metrics["alpha_target_bucket/high/count"] == 0.0
    assert torch.isnan(torch.tensor(metrics["alpha_target_bucket/medium/hit_rate_at_1"]))
    assert torch.isnan(torch.tensor(metrics["alpha_target_bucket/medium/mrr"]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_atom_gate_alpha_supervision_runs_under_cuda_fp16_autocast():
    module = ContrastiveTrainingModule.build(
        TinyEncoder().to("cuda"),
        canonical_method_config(TrainingMethod.ATOM_GATE),
    ).to("cuda")
    batch = {key: value.to("cuda") for key, value in tiny_batch().items()}

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = module.compute_training_step(batch, step=0)
        assert output.alpha_supervision_per_row is not None
        output.alpha_supervision_per_row.mean().backward()

    assert module.alpha_gate is not None
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.alpha_gate.parameters()
    )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_atom_gate_two_steps_stay_finite_on_mps():
    module = ContrastiveTrainingModule.build(
        TinyEncoder().to("mps"),
        canonical_method_config(TrainingMethod.ATOM_GATE),
    ).to("mps")
    for _ in range(2):
        batch = {key: value.to("mps") for key, value in tiny_batch().items()}
        output = module.compute_training_step(batch, step=0)
        output.loss.backward()
        assert torch.isfinite(output.loss)
        assert output.alpha_target is not None and torch.isfinite(output.alpha_target).all()
        module.zero_grad(set_to_none=True)


def test_atom_gate_reports_effective_detached_auxiliary_weight():
    module = ContrastiveTrainingModule.build(
        TinyEncoder(),
        canonical_method_config(TrainingMethod.ATOM_GATE),
    )

    output = module.compute_training_step(tiny_batch(), step=0)

    assert output.metrics["alpha_aux_weight/mean"] == pytest.approx(1.0 - output.metrics["alpha/mean"])
    assert output.metrics["alpha_aux_weight/min"] == pytest.approx(1.0 - output.metrics["alpha/max"])
    assert output.metrics["alpha_aux_weight/max"] == pytest.approx(1.0 - output.metrics["alpha/min"])


def test_compute_training_step_rejects_nonfinite_loss(monkeypatch):
    module = ContrastiveTrainingModule.build(TinyEncoder(), canonical_method_config(TrainingMethod.VANILLA))
    bad_output = ObjectiveOutput(
        loss=torch.tensor(float("nan")),
        primary_loss=torch.tensor(float("nan")),
        memory_loss=None,
        main_per_row=torch.ones(2),
        memory_per_row=None,
        simcse_per_row=None,
        soft_fn_per_row=None,
        alpha_target=None,
        alpha_supervision_per_row=None,
        metrics={},
    )
    monkeypatch.setattr(module.objective, "forward", lambda *args, **kwargs: bad_output)

    with pytest.raises(RuntimeError, match="Non-finite loss"):
        module.compute_training_step(tiny_batch(), step=7)


def test_lightning_automatic_optimization_keeps_tau_version_valid(tmp_path):
    module = ContrastiveTrainingModule.build(
        TinyEncoder(),
        canonical_method_config(TrainingMethod.VANILLA),
    )
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )

    trainer.fit(module, train_dataloaders=DataLoader([tiny_batch()], batch_size=None))

    assert trainer.global_step == 1
    assert torch.isfinite(module.objective.kernel.log_tau)


def test_lightning_atomic_manual_optimization_steps_with_live_bank(tmp_path):
    config = canonical_method_config(TrainingMethod.ATOMIC)
    config = replace(
        config,
        memory_bank=replace(config.memory_bank, size=8, warmup_steps=0, random_negatives=1),
    )
    module = ContrastiveTrainingModule.build(TinyEncoder(), config)
    second = {key: value.clone() for key, value in tiny_batch().items()}
    for key in ("doc_key_id", "content_key_id", "query_key_id"):
        second[key] += 100
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )

    trainer.fit(module, train_dataloaders=DataLoader([tiny_batch(), second], batch_size=None))

    assert trainer.global_step == 2
    assert module.memory_bank is not None and module.memory_bank.current_size == 4
    assert torch.isfinite(module.objective.kernel.log_tau)


def test_lightning_atomic_handles_gradient_accumulation_and_final_partial_step(tmp_path):
    config = canonical_method_config(TrainingMethod.ATOMIC)
    config = replace(
        config,
        optimization=replace(config.optimization, grad_acc_steps=2),
        memory_bank=replace(config.memory_bank, size=8, warmup_steps=0, random_negatives=1),
    )
    module = ContrastiveTrainingModule.build(TinyEncoder(), config)
    batches = []
    for offset in (0, 100, 200):
        batch = {key: value.clone() for key, value in tiny_batch().items()}
        for key in ("doc_key_id", "content_key_id", "query_key_id"):
            batch[key] += offset
        batches.append(batch)
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )

    trainer.fit(module, train_dataloaders=DataLoader(batches, batch_size=None))

    assert trainer.global_step == 2
