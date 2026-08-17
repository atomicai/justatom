from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytorch_lightning as L
import torch
import yaml

from justatom.modeling.mask import ILanguageModel
from justatom.processing import ITokenizer, igniset
from justatom.processing.loader import NamedDataLoader
from justatom.processing.prime import TrainWithContrastiveProcessor
from justatom.running.encoders import EncoderRunner
from justatom.tooling.collections import build_collection_metadata, resolve_artifact_dirname, write_collection_metadata
from justatom.training.config import (
    AuxiliaryGradientMode,
    LoraAdapterConfig,
    RuntimeConfig,
    TrainConfig,
    TrainingMethod,
    train_config_to_dict,
)
from justatom.training.data import prepare_training_data_from_config
from justatom.training.methods import resolve_method
from justatom.training.module import ContrastiveTrainingModule


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    deployable_encoder: Path
    adapter: Path
    research_checkpoint: Path
    manifest: Path


def artifact_paths(root: Path) -> ArtifactPaths:
    return ArtifactPaths(
        root=root,
        deployable_encoder=root / "encoder",
        adapter=root / "adapter",
        research_checkpoint=root / "research" / "checkpoint.pt",
        manifest=root / "run_manifest.yaml",
    )


def resolve_run_dir(config: TrainConfig) -> Path:
    if config.artifacts.save_dir:
        return Path(config.artifacts.save_dir)
    dirname = resolve_artifact_dirname(config.artifacts.collection_name or config.method.value)
    return Path.cwd() / "weights" / dirname


def _git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError:
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def objective_contract(config: TrainConfig) -> dict[str, str]:
    auxiliary_gradient, auxiliary_norm = {
        AuxiliaryGradientMode.OFF: ("off", "unbounded"),
        AuxiliaryGradientMode.OBSERVE: ("observe", "unbounded"),
        AuxiliaryGradientMode.SAFE: ("cosine_safe", "retrieval_relative"),
    }[config.auxiliary_gradient.mode]
    contract = {
        "contrastive_kernel": ("decoupled_infonce" if config.objective.decoupled else "coupled_infonce"),
        "alpha_aux_gradient": ("detached" if config.method is TrainingMethod.ATOM_GATE else "not_applicable"),
        "memory_mass": ("count_normalized" if config.memory_bank.enabled else "not_applicable"),
        "auxiliary_gradient": auxiliary_gradient,
        "auxiliary_norm": auxiliary_norm,
    }
    if config.method is TrainingMethod.ATOM_GATE:
        contract.update(
            alpha_target="detached_positive_softmax_confidence",
            alpha_head_input_gradient="detached",
        )
    if config.anchor_bank.enabled:
        contract.update(
            geometry_anchor="frozen_base_relational_kl",
            geometry_control="one_sided_task_projection",
        )
    return contract


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    method: str
    experiment: dict[str, Any]
    created_at: str
    git_commit: str | None
    git_dirty: bool | None
    objective_contract: dict[str, str]
    batch_contract: dict[str, int]
    resolved_config: dict[str, Any]

    @classmethod
    def from_config(
        cls,
        config: TrainConfig,
        *,
        git_commit: str | None,
        git_dirty: bool | None,
    ) -> RunManifest:
        payload = train_config_to_dict(config)
        return cls(
            schema_version=1,
            method=config.method.value,
            experiment=dict(payload["experiment"]),
            created_at=datetime.now(timezone.utc).isoformat(),
            git_commit=git_commit,
            git_dirty=git_dirty,
            objective_contract=objective_contract(config),
            batch_contract={
                "contrastive_microbatch": config.optimization.batch_size,
                "gradient_accumulation": config.optimization.grad_acc_steps,
                "optimizer_effective_batch": (config.optimization.batch_size * config.optimization.grad_acc_steps),
            },
            resolved_config=payload,
        )

    @classmethod
    def capture(cls, config: TrainConfig) -> RunManifest:
        commit = _git_output("rev-parse", "HEAD")
        status = _git_output("status", "--porcelain")
        dirty = None if commit is None else bool(status)
        return cls.from_config(config, git_commit=commit, git_dirty=dirty)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "experiment": self.experiment,
            "created_at": self.created_at,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "objective_contract": self.objective_contract,
            "batch_contract": self.batch_contract,
            "resolved_config": self.resolved_config,
        }


def write_run_manifest(run_dir: Path, manifest: RunManifest) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    destination = run_dir / "run_manifest.yaml"
    temporary = destination.with_suffix(".yaml.tmp")
    temporary.write_text(
        yaml.safe_dump(manifest.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def write_collection_metadata_from_config(run_dir: Path, config: TrainConfig) -> None:
    if config.artifacts.collection_name is None:
        return
    metadata = build_collection_metadata(
        collection_name=config.artifacts.collection_name,
        collection_tag=config.artifacts.collection_tag,
        model_name_or_path=config.model.name_or_path,
        dataset_name_or_path=config.dataset.name_or_path,
        save_dir=run_dir,
        payload=train_config_to_dict(config),
    )
    write_collection_metadata(save_dir=run_dir, metadata=metadata)


def build_training_loader(config: TrainConfig):
    rows = prepare_training_data_from_config(config)
    tokenizer = ITokenizer.from_pretrained(config.model.name_or_path)
    processor = TrainWithContrastiveProcessor(
        tokenizer=tokenizer,
        max_seq_len=config.model.max_seq_len,
        max_query_seq_len=config.model.max_query_seq_len,
        queries_field="queries",
        queries_prefix=config.model.query_prefix,
        pos_queries_prefix=config.model.content_prefix,
    )
    dataset, tensor_names = igniset(
        rows,
        processor=processor,
        batch_size=config.optimization.batch_size,
        streaming=True,
    )
    loader = NamedDataLoader(
        dataset=dataset,
        tensor_names=tensor_names,
        batch_size=config.optimization.batch_size,
    )
    return loader, processor


def resolve_torch_device(runtime: RuntimeConfig) -> str:
    if runtime.accelerator != "auto":
        if runtime.accelerator in {"gpu", "cuda"}:
            return "cuda:0"
        return runtime.accelerator
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_training_precision(runtime: RuntimeConfig) -> str:
    if runtime.precision != "auto":
        return runtime.precision
    if resolve_torch_device(runtime).startswith("cuda"):
        return "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
    return "32-true"


def apply_lora_adapter(language_model: ILanguageModel, config: LoraAdapterConfig) -> ILanguageModel:
    if not config.enabled:
        return language_model

    from peft import LoraConfig, TaskType, get_peft_model

    target_modules = list(config.target_modules) if isinstance(config.target_modules, tuple) else config.target_modules
    language_model.model = get_peft_model(
        language_model.model,
        LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=config.rank,
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
            target_modules=target_modules,
            use_rslora=config.use_rslora,
            bias=config.bias,
        ),
    )
    return language_model


def load_encoder(config: TrainConfig, processor: TrainWithContrastiveProcessor) -> EncoderRunner:
    language_model = ILanguageModel.load(model_name_or_path=config.model.name_or_path)
    if config.runtime.gradient_checkpointing:
        backbone = language_model.model
        if not hasattr(backbone, "gradient_checkpointing_enable"):
            raise ValueError(f"{config.model.name_or_path} does not support gradient checkpointing")
        backbone.gradient_checkpointing_enable()
        if hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()
        if hasattr(backbone, "config") and hasattr(backbone.config, "use_cache"):
            backbone.config.use_cache = False
    apply_lora_adapter(language_model, config.model.lora)
    return EncoderRunner(
        model=language_model,
        processor=processor,
        prediction_heads=[],
        device=resolve_torch_device(config.runtime),
    )


def build_training_logger(config: TrainConfig):
    if config.telemetry.backend == "csv":
        return False
    if config.telemetry.backend == "wandb":
        from pytorch_lightning.loggers import WandbLogger

        return WandbLogger(
            project=config.telemetry.wandb_project,
            name=config.telemetry.run_name,
        )
    raise ValueError(f"Unsupported telemetry.backend={config.telemetry.backend!r}")


def build_lightning_trainer(config: TrainConfig) -> L.Trainer:
    return L.Trainer(
        max_epochs=config.optimization.epochs,
        accelerator=config.runtime.accelerator,
        devices=config.runtime.devices,
        precision=resolve_training_precision(config.runtime),
        # Manual paths control each microbatch before accumulating its update.
        accumulate_grad_batches=(
            1
            if (config.gradient_projection.enabled or config.auxiliary_gradient.mode is not AuxiliaryGradientMode.OFF)
            else config.optimization.grad_acc_steps
        ),
        logger=build_training_logger(config),
        log_every_n_steps=1,
        enable_checkpointing=False,
        enable_model_summary=False,
    )


@dataclass(frozen=True)
class TrainingResult:
    run_dir: Path
    encoder_dir: Path
    research_checkpoint: Path | None
    metrics_path: Path | None
    adapter_dir: Path | None = None


class TrainingJob:
    def __init__(
        self,
        config: TrainConfig,
        *,
        loader_factory: Callable = build_training_loader,
        encoder_factory: Callable = load_encoder,
        module_factory: Callable = ContrastiveTrainingModule.build,
        trainer_factory: Callable = build_lightning_trainer,
    ):
        self.config = config
        self.loader_factory = loader_factory
        self.encoder_factory = encoder_factory
        self.module_factory = module_factory
        self.trainer_factory = trainer_factory

    def run(self) -> TrainingResult:
        config = resolve_method(self.config)
        paths = artifact_paths(resolve_run_dir(config))
        if config.telemetry.backend == "csv" and config.telemetry.metrics_path is None:
            config = replace(
                config,
                telemetry=replace(config.telemetry, metrics_path=str(paths.root / "batch_metrics.csv")),
            )
        L.seed_everything(config.experiment.seed, workers=True)
        write_run_manifest(paths.root, RunManifest.capture(config))
        write_collection_metadata_from_config(paths.root, config)

        loader, processor = self.loader_factory(config)
        encoder = self.encoder_factory(config, processor)
        module = self.module_factory(encoder, config)
        trainer = self.trainer_factory(config)
        trainer.fit(module, train_dataloaders=loader)

        checkpoint = module.save_research_checkpoint(paths.research_checkpoint)
        adapter_dir = module.save_lora_adapter(paths.adapter)
        encoder_dir = module.save_deployable_encoder(paths.deployable_encoder)
        metrics_path = None if module.metrics_path is None else Path(module.metrics_path)
        return TrainingResult(
            run_dir=paths.root,
            encoder_dir=encoder_dir,
            research_checkpoint=checkpoint,
            metrics_path=metrics_path,
            adapter_dir=adapter_dir,
        )
