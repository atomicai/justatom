from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from justatom.training.config import ExperimentRole, TrainingMethod
from justatom.training.job import (
    RunManifest,
    TrainingJob,
    artifact_paths,
    build_lightning_trainer,
    write_run_manifest,
)
from justatom.training.methods import canonical_method_config


def test_manifest_contains_resolved_method_seed_and_git_state(tmp_path: Path):
    config = canonical_method_config(TrainingMethod.ATOMIC)
    config = replace(config, optimization=replace(config.optimization, batch_size=8, grad_acc_steps=4))
    manifest = RunManifest.from_config(config, git_commit="abc123", git_dirty=True)

    path = write_run_manifest(tmp_path, manifest)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert loaded["schema_version"] == 1
    assert loaded["method"] == "atomic"
    assert loaded["experiment"]["seed"] == 42
    assert loaded["git_commit"] == "abc123"
    assert loaded["git_dirty"] is True
    assert loaded["objective_contract"] == {
        "contrastive_kernel": "coupled_infonce",
        "alpha_aux_gradient": "not_applicable",
        "memory_mass": "count_normalized",
    }
    assert loaded["resolved_config"]["memory_bank"]["mass_ratio"] == pytest.approx(0.5)
    assert loaded["batch_contract"] == {
        "contrastive_microbatch": 8,
        "gradient_accumulation": 4,
        "optimizer_effective_batch": 32,
    }
    assert loaded["resolved_config"]["memory_bank"]["margin"]["mode"] == "off"
    assert loaded["resolved_config"]["gradient_projection"]["enabled"] is True


def test_disabled_memory_bank_manifest_marks_mass_not_applicable():
    manifest = RunManifest.from_config(
        canonical_method_config(TrainingMethod.VANILLA),
        git_commit="abc123",
        git_dirty=False,
    )

    assert manifest.objective_contract["memory_mass"] == "not_applicable"


def test_atom_gate_manifest_records_detached_auxiliary_control():
    manifest = RunManifest.from_config(
        canonical_method_config(TrainingMethod.ATOM_GATE),
        git_commit="abc123",
        git_dirty=False,
    )

    assert manifest.objective_contract == {
        "contrastive_kernel": "coupled_infonce",
        "alpha_aux_gradient": "detached",
        "memory_mass": "not_applicable",
        "alpha_target": "detached_positive_softmax_confidence",
        "alpha_head_input_gradient": "detached",
    }


def test_dcl_ablation_manifest_records_decoupled_kernel():
    config = canonical_method_config(TrainingMethod.VANILLA)
    config = replace(
        config,
        experiment=replace(config.experiment, role=ExperimentRole.ABLATION),
        objective=replace(config.objective, decoupled=True),
    )

    manifest = RunManifest.from_config(config, git_commit="abc123", git_dirty=False)

    assert manifest.objective_contract == {
        "contrastive_kernel": "decoupled_infonce",
        "alpha_aux_gradient": "not_applicable",
        "memory_mass": "not_applicable",
    }


def test_artifact_directories_are_distinct(tmp_path: Path):
    paths = artifact_paths(tmp_path)

    assert paths.deployable_encoder == tmp_path / "encoder"
    assert paths.adapter == tmp_path / "adapter"
    assert paths.research_checkpoint == tmp_path / "research" / "checkpoint.pt"
    assert paths.manifest == tmp_path / "run_manifest.yaml"


def test_atomic_trainer_delegates_gradient_accumulation_to_manual_optimization():
    atomic = canonical_method_config(TrainingMethod.ATOMIC)
    atomic = replace(atomic, optimization=replace(atomic.optimization, grad_acc_steps=4))
    vanilla = canonical_method_config(TrainingMethod.VANILLA)
    vanilla = replace(vanilla, optimization=replace(vanilla.optimization, grad_acc_steps=4))

    assert build_lightning_trainer(atomic).accumulate_grad_batches == 1
    assert build_lightning_trainer(vanilla).accumulate_grad_batches == 4


def test_training_job_writes_manifest_before_fit_and_returns_artifacts(tmp_path: Path):
    events: list[str] = []
    config = canonical_method_config(TrainingMethod.VANILLA)
    config = replace(
        config,
        dataset=replace(config.dataset, name_or_path="dummy"),
        artifacts=replace(config.artifacts, save_dir=str(tmp_path)),
    )

    class FakeModule:
        metrics_path = tmp_path / "batch_metrics.csv"

        def save_deployable_encoder(self, destination):
            events.append("encoder")
            destination.mkdir(parents=True)
            return destination

        def save_lora_adapter(self, destination):
            events.append("adapter")

        def save_research_checkpoint(self, destination):
            events.append("checkpoint")
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"checkpoint")
            return destination

    class FakeTrainer:
        def fit(self, module, train_dataloaders):
            assert (tmp_path / "run_manifest.yaml").exists()
            events.append("fit")

    job = TrainingJob(
        config,
        loader_factory=lambda _: ("loader", "processor"),
        encoder_factory=lambda *_: "encoder",
        module_factory=lambda *_args: FakeModule(),
        trainer_factory=lambda _: FakeTrainer(),
    )

    result = job.run()

    assert events == ["fit", "checkpoint", "adapter", "encoder"]
    assert result.encoder_dir == tmp_path / "encoder"
    assert result.research_checkpoint == tmp_path / "research" / "checkpoint.pt"
    assert result.adapter_dir is None
