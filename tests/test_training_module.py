from __future__ import annotations

from dataclasses import replace

import pytest
import pytorch_lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader

from justatom.training.alpha_gate import QueryAlphaGate
from justatom.training.config import ExperimentConfig, ExperimentRole, MarginMode, TrainingMethod
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


class RecordingTokenizer:
    def batch_decode(self, input_ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        values = {
            1: "query: alpha",
            2: "query: beta",
            3: "passage: alpha document",
            4: "passage: beta document",
        }
        return [values[int(row[0])] for row in input_ids]


class TokenizedTinyEncoder(TinyEncoder):
    def __init__(self):
        super().__init__()
        self.processor = type("Processor", (), {"tokenizer": RecordingTokenizer()})()


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
    return ObjectiveOutput(
        loss=torch.tensor(1.0, requires_grad=True),
        main_per_row=torch.ones(2),
        simcse_per_row=None,
        soft_fn_per_row=None,
        metrics={},
    )


def test_module_constructs_only_components_required_by_method():
    vanilla = ContrastiveTrainingModule.build(TinyEncoder(), canonical_method_config(TrainingMethod.VANILLA))
    gate = ContrastiveTrainingModule.build(TinyEncoder(), canonical_method_config(TrainingMethod.ATOM_GATE))
    atomic = ContrastiveTrainingModule.build(TinyEncoder(), canonical_method_config(TrainingMethod.ATOMIC))

    assert vanilla.alpha_gate is None and vanilla.memory_bank is None and vanilla.margin_head is None
    assert isinstance(gate.alpha_gate, QueryAlphaGate) and gate.memory_bank is None
    assert atomic.alpha_gate is not None and atomic.memory_bank is not None and atomic.margin_head is not None


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


def test_optimizer_contains_encoder_temperature_alpha_and_margin_parameters_once():
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
    monkeypatch.setattr(module, "_semantic_pair_scores", lambda q, p, batch: torch.ones(2, 2))
    monkeypatch.setattr(module, "_lexical_pair_scores", lambda batch: torch.ones(2, 2))
    monkeypatch.setattr(
        module.objective,
        "forward",
        lambda *args, **kwargs: events.append("loss") or finite_output(),
    )
    monkeypatch.setattr(module.memory_bank, "enqueue", lambda *args, **kwargs: events.append("enqueue"))

    module.compute_training_step(tiny_batch(), step=0)

    assert events == ["loss", "enqueue"]


def test_alpha_mix_weight_uses_exact_linear_warmup():
    config = canonical_method_config(TrainingMethod.ATOM_GATE)
    config = replace(
        config,
        alpha_gate=replace(config.alpha_gate, mix_weight=0.3, mix_weight_warmup_steps=10),
    )
    module = ContrastiveTrainingModule.build(TinyEncoder(), config)

    assert module.effective_alpha_mix_weight(0) == 0.0
    assert module.effective_alpha_mix_weight(5) == pytest.approx(0.15)
    assert module.effective_alpha_mix_weight(10) == pytest.approx(0.3)
    assert module.effective_alpha_mix_weight(20) == pytest.approx(0.3)


def test_compute_training_step_rejects_nonfinite_loss(monkeypatch):
    module = ContrastiveTrainingModule.build(TinyEncoder(), canonical_method_config(TrainingMethod.VANILLA))
    bad_output = ObjectiveOutput(
        loss=torch.tensor(float("nan")),
        main_per_row=torch.ones(2),
        simcse_per_row=None,
        soft_fn_per_row=None,
        metrics={},
    )
    monkeypatch.setattr(module.objective, "forward", lambda *args, **kwargs: bad_output)

    with pytest.raises(RuntimeError, match="Non-finite loss"):
        module.compute_training_step(tiny_batch(), step=7)


def test_alpha_lexical_scores_are_recovered_from_tokenized_batch():
    module = ContrastiveTrainingModule.build(
        TokenizedTinyEncoder(),
        canonical_method_config(TrainingMethod.ATOM_GATE),
        lexical_lookup={
            "alpha document": "alpha",
            "beta document": "beta",
        },
    )
    batch = {
        "input_ids": torch.tensor([[1], [2]]),
        "pos_input_ids": torch.tensor([[3], [4]]),
    }
    module._last_negative_indices = torch.tensor([1, 0])

    scores = module._lexical_pair_scores(batch)

    assert scores is not None
    assert scores[:, 0].tolist() == pytest.approx([1.0, 1.0])
    assert scores[:, 1].tolist() == pytest.approx([0.0, 0.0])


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
