from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import yaml

from justatom.training.config import TrainingMethod
from justatom.training.job import RunManifest, TrainingJob, artifact_paths, write_run_manifest
from justatom.training.methods import canonical_method_config


def test_manifest_contains_resolved_method_seed_and_git_state(tmp_path: Path):
    config = canonical_method_config(TrainingMethod.ATOMIC)
    manifest = RunManifest.from_config(config, git_commit="abc123", git_dirty=True)

    path = write_run_manifest(tmp_path, manifest)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert loaded["schema_version"] == 1
    assert loaded["method"] == "atomic"
    assert loaded["experiment"]["seed"] == 42
    assert loaded["git_commit"] == "abc123"
    assert loaded["git_dirty"] is True
    assert loaded["resolved_config"]["memory_bank"]["margin"]["mode"] == "query"


def test_artifact_directories_are_distinct(tmp_path: Path):
    paths = artifact_paths(tmp_path)

    assert paths.deployable_encoder == tmp_path / "encoder"
    assert paths.adapter == tmp_path / "adapter"
    assert paths.research_checkpoint == tmp_path / "research" / "checkpoint.pt"
    assert paths.manifest == tmp_path / "run_manifest.yaml"


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
            return None

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
        loader_factory=lambda _: ("loader", "processor", {}),
        encoder_factory=lambda *_: "encoder",
        module_factory=lambda *_args, **_kwargs: FakeModule(),
        trainer_factory=lambda _: FakeTrainer(),
    )

    result = job.run()

    assert events == ["fit", "checkpoint", "adapter", "encoder"]
    assert result.encoder_dir == tmp_path / "encoder"
    assert result.research_checkpoint == tmp_path / "research" / "checkpoint.pt"
    assert result.adapter_dir is None
