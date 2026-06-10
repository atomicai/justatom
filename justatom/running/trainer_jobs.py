from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from justatom.tooling.collections import (
    build_collection_metadata,
    build_collection_tag_payload,
    resolve_artifact_dirname,
    write_collection_metadata,
)


def _normalize_recipe_name(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "atom": "atom_gate",
        "atom_gate": "atom_gate",
        "justatom_gate": "atom_gate",
    }
    return aliases.get(text, text)


def maybe_cuda_or_mps() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_training_mode(
    *,
    freeze_encoder: bool,
    include_semantic_gamma: bool,
    include_keywords_gamma: bool,
) -> str:
    enabled_count = int(include_semantic_gamma) + int(include_keywords_gamma)
    if enabled_count == 0:
        if freeze_encoder:
            raise ValueError("encoder-only training requires freeze_encoder=False because otherwise nothing is trainable")
        return "encoder-only"
    if freeze_encoder:
        return "gamma-only"
    return "encoder+gamma"


@dataclass
class BaseTrainingJob:
    dataset_name_or_path: str | Path
    prepare_training_data_fn: Any
    roll_metrics_path_fn: Any
    recipe: str | None = None
    model_name_or_path: str = "intfloat/multilingual-e5-small"
    collection_name: str | None = None
    collection_tag: str | None = None
    loss: str = "contrastive"
    temperature: float | None = None
    num_samples: int = 100
    batch_size: int = 4
    max_seq_len: int = 512
    freeze_encoder: bool = True
    gamma_joint: bool = False
    include_semantic_gamma: bool = True
    include_keywords_gamma: bool = True
    add_alpha_gate: bool = False
    alpha_entropy_weight: float = 0.0
    alpha_train_only: bool = False
    alpha_mix_weight: float = 0.3
    alpha_mix_weight_warmup_steps: int = 0
    alpha_residual: bool = False
    alpha_prior: float = 0.5
    alpha_residual_scale: float = 0.25
    alpha_head_input: str = "query"
    alpha_head_layers: int = 1
    alpha_head_hidden_dim: int | str | None = None
    alpha_head_dropout: float = 0.0
    alpha_head_activation: str = "gelu"
    alpha_head_include_doc: bool | None = None
    query_diagonal_gate: bool = False
    query_diagonal_gate_scale: float = 0.25
    query_diagonal_gate_mode: str = "raw"
    query_diagonal_identity_weight: float = 0.0
    query_diagonal_saturation_weight: float = 0.0
    activation_fn: str = "sigmoid"
    margin: float = 0.5
    contrastive_temperature: float = 0.03
    soft_contrastive_temperature: float = 1.0
    grad_acc_steps: int = 6
    optimizer: str = "adamw"
    max_negative_inverse_idf_recall: float | None = None
    contrastive_learnable_temperature: bool = True
    contrastive_decoupled: bool = True
    contrastive_simcse_dropout_weight: float = 0.0
    contrastive_soft_fn_attract_weight: float = 0.0
    contrastive_soft_fn_topk: int = 1
    contrastive_loss_alpha_gate: bool = False
    contrastive_loss_alpha_gate_mode: str = "augment"
    memory_bank_size: int = 0
    memory_bank_warmup_steps: int = 0
    memory_bank_mining_mode: str = "all"
    memory_bank_hard_negatives: int = 0
    memory_bank_random_negatives: int = 0
    memory_bank_hard_warmup_steps: int = 0
    memory_bank_hard_ramp_steps: int = 1
    memory_bank_too_hard_margin: float | None = None
    min_negative_inverse_idf_recall: float | None = None
    negative_sampling_mode: str = "safe-random"
    hard_negative_top_k: int | None = None
    focal_gamma: float = 2.0
    log_backend: str = "csv"
    wandb_project: str = "justatom-gamma"
    wandb_run_name: str | None = None
    n_epochs: int = 1
    content_field: str = "content"
    labels_field: str = "queries"
    split: str | None = None
    limit: int | None = None
    chunk_id_col: str | None = None
    keywords_or_phrases_field: str | None = "keywords_or_phrases"
    keywords_nested_col: str | None = None
    explanation_nested_col: str | None = None
    filters: dict | None = None
    lr_gamma: float = 1e-2
    lr_encoder: float = 2e-5
    weight_decay: float = 0.01
    save_dir: str | Path | None = None
    metrics_path: str | Path | None = None
    sample_training_rows_fn: Any | None = None

    def __post_init__(self):
        recipe = _normalize_recipe_name(self.recipe)
        self.recipe = recipe or None
        if self.temperature is None:
            self.temperature = float(self.contrastive_temperature)
        else:
            self.temperature = float(self.temperature)
            self.contrastive_temperature = float(self.temperature)
        training_mode = _resolve_training_mode(
            freeze_encoder=self.freeze_encoder,
            include_semantic_gamma=self.include_semantic_gamma,
            include_keywords_gamma=self.include_keywords_gamma,
        )
        if training_mode != "encoder-only" and self.loss not in {"soft-contrastive", "contrastive", "focal-contrastive"}:
            raise ValueError(
                f"training_mode={training_mode} only supports loss in {{soft-contrastive, contrastive, focal-contrastive}}, got loss={self.loss}"
            )
        if self.alpha_residual and not self.gamma_joint:
            raise ValueError("alpha_residual=True requires gamma_joint=True")
        if not 0.0 < float(self.alpha_prior) < 1.0:
            raise ValueError(f"alpha_prior must be strictly between 0 and 1, got {self.alpha_prior}")
        if float(self.alpha_residual_scale) < 0.0:
            raise ValueError(f"alpha_residual_scale must be >= 0, got {self.alpha_residual_scale}")
        self.alpha_head_input = str(self.alpha_head_input).strip().lower()
        if self.alpha_head_input in {"q", "alpha_q", "alpha(q)"}:
            self.alpha_head_input = "query"
        if self.alpha_head_input in {"pair", "query_doc", "query_document", "q_doc", "q_d", "alpha(q,d+)"}:
            self.alpha_head_input = "query_doc"
        if self.alpha_head_input not in {"query", "query_doc"}:
            raise ValueError("alpha_head_input must be one of: query, query_doc")
        self.alpha_head_include_doc = self.alpha_head_input == "query_doc" if self.alpha_head_include_doc is None else bool(self.alpha_head_include_doc)
        self.alpha_head_layers = max(int(self.alpha_head_layers), 0)
        if isinstance(self.alpha_head_hidden_dim, str):
            hidden_text = self.alpha_head_hidden_dim.strip().lower()
            self.alpha_head_hidden_dim = None if hidden_text in {"", "auto", "none", "null"} else int(hidden_text)
        elif self.alpha_head_hidden_dim is not None:
            self.alpha_head_hidden_dim = int(self.alpha_head_hidden_dim)
        self.alpha_head_dropout = float(self.alpha_head_dropout)
        if not 0.0 <= self.alpha_head_dropout < 1.0:
            raise ValueError(f"alpha_head_dropout must be in [0, 1), got {self.alpha_head_dropout}")
        self.alpha_head_activation = str(self.alpha_head_activation).strip().lower()
        if self.query_diagonal_gate and not self.gamma_joint:
            raise ValueError("query_diagonal_gate=True requires gamma_joint=True")
        self.memory_bank_size = int(self.memory_bank_size)
        if self.memory_bank_size < 0:
            raise ValueError(f"memory_bank_size must be >= 0, got {self.memory_bank_size}")
        self.memory_bank_warmup_steps = max(int(self.memory_bank_warmup_steps), 0)
        self.memory_bank_hard_negatives = max(int(self.memory_bank_hard_negatives), 0)
        self.memory_bank_random_negatives = max(int(self.memory_bank_random_negatives), 0)
        self.memory_bank_hard_warmup_steps = max(int(self.memory_bank_hard_warmup_steps), 0)
        self.memory_bank_hard_ramp_steps = max(int(self.memory_bank_hard_ramp_steps), 1)
        self.memory_bank_mining_mode = str(self.memory_bank_mining_mode).strip().lower()
        if self.memory_bank_mining_mode not in {"all", "random", "hard", "mixed"}:
            raise ValueError("memory_bank_mining_mode must be one of: all, random, hard, mixed")

    @property
    def training_mode(self) -> str:
        raise NotImplementedError

    def build_lightning_module(
        self,
        *,
        processor,
        lexical_text_by_content,
        save_dir: Path,
        metrics_path: str | None,
    ):
        raise NotImplementedError

    def _resolved_save_dir(self) -> Path:
        if self.save_dir is not None:
            save_dir = Path(self.save_dir)
        else:
            dirname = resolve_artifact_dirname(self.collection_name)
            save_dir = Path(os.getcwd()) / "weights" / dirname
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def _resolved_metrics_path(self, save_dir: Path) -> str | None:
        default_metrics_path = save_dir / "gamma_metrics.csv"
        if self.metrics_path is not None:
            return str(self.roll_metrics_path_fn(Path(self.metrics_path)))
        if self.log_backend == "csv":
            return str(self.roll_metrics_path_fn(default_metrics_path))
        return None

    def _collection_payload(self) -> dict[str, object]:
        return build_collection_tag_payload(
            recipe=self.recipe,
            model_name_or_path=self.model_name_or_path,
            dataset_name_or_path=self.dataset_name_or_path,
            loss=self.loss,
            temperature=self.temperature,
            grad_acc_steps=self.grad_acc_steps,
            freeze_encoder=self.freeze_encoder,
            include_semantic_gamma=self.include_semantic_gamma,
            include_keywords_gamma=self.include_keywords_gamma,
            activation_fn=self.activation_fn,
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            lr_gamma=self.lr_gamma,
            lr_encoder=self.lr_encoder,
            weight_decay=self.weight_decay,
            add_alpha_gate=self.add_alpha_gate,
            alpha_entropy_weight=self.alpha_entropy_weight,
            alpha_train_only=self.alpha_train_only,
            alpha_mix_weight=self.alpha_mix_weight,
            alpha_mix_weight_warmup_steps=self.alpha_mix_weight_warmup_steps,
            alpha_residual=self.alpha_residual,
            alpha_prior=self.alpha_prior,
            alpha_residual_scale=self.alpha_residual_scale,
            alpha_head_input=self.alpha_head_input,
            alpha_head_layers=self.alpha_head_layers,
            alpha_head_hidden_dim=self.alpha_head_hidden_dim,
            alpha_head_dropout=self.alpha_head_dropout,
            alpha_head_activation=self.alpha_head_activation,
            alpha_head_include_doc=self.alpha_head_include_doc,
            query_diagonal_gate=self.query_diagonal_gate,
            query_diagonal_gate_scale=self.query_diagonal_gate_scale,
            query_diagonal_gate_mode=self.query_diagonal_gate_mode,
            query_diagonal_identity_weight=self.query_diagonal_identity_weight,
            query_diagonal_saturation_weight=self.query_diagonal_saturation_weight,
            contrastive_learnable_temperature=self.contrastive_learnable_temperature,
            contrastive_decoupled=self.contrastive_decoupled,
            contrastive_simcse_dropout_weight=self.contrastive_simcse_dropout_weight,
            contrastive_soft_fn_attract_weight=self.contrastive_soft_fn_attract_weight,
            contrastive_soft_fn_topk=self.contrastive_soft_fn_topk,
            contrastive_loss_alpha_gate=self.contrastive_loss_alpha_gate,
            contrastive_loss_alpha_gate_mode=self.contrastive_loss_alpha_gate_mode,
            memory_bank_size=self.memory_bank_size,
            memory_bank_warmup_steps=self.memory_bank_warmup_steps,
            memory_bank_mining_mode=self.memory_bank_mining_mode,
            memory_bank_hard_negatives=self.memory_bank_hard_negatives,
            memory_bank_random_negatives=self.memory_bank_random_negatives,
            memory_bank_hard_warmup_steps=self.memory_bank_hard_warmup_steps,
            memory_bank_hard_ramp_steps=self.memory_bank_hard_ramp_steps,
            memory_bank_too_hard_margin=self.memory_bank_too_hard_margin,
            focal_gamma=self.focal_gamma,
            min_negative_inverse_idf_recall=self.min_negative_inverse_idf_recall,
            negative_sampling_mode=self.negative_sampling_mode,
            hard_negative_top_k=self.hard_negative_top_k,
            collection_tag=self.collection_tag,
        )

    def _write_collection_metadata(self, *, save_dir: Path) -> None:
        if self.collection_name is None:
            return
        metadata = build_collection_metadata(
            collection_name=self.collection_name,
            collection_tag=self.collection_tag,
            model_name_or_path=self.model_name_or_path,
            dataset_name_or_path=self.dataset_name_or_path,
            save_dir=save_dir,
            payload=self._collection_payload(),
        )
        local_meta_path, registry_path = write_collection_metadata(
            save_dir=save_dir,
            metadata=metadata,
        )
        logger.info(f"Collection metadata saved to: {local_meta_path}")
        if registry_path is not None:
            logger.info(f"Collection tag registry saved to: {registry_path}")

    def _build_loader(self):
        from justatom.processing import ITokenizer, igniset
        from justatom.processing.loader import NamedDataLoader
        from justatom.processing.prime import TrainWithContrastiveProcessor

        prepare_kwargs = {
            "dataset_name_or_path": self.dataset_name_or_path,
            "num_samples": self.num_samples,
            "content_field": self.content_field,
            "labels_field": self.labels_field,
            "split": self.split,
            "limit": self.limit,
            "chunk_id_col": self.chunk_id_col,
            "keywords_or_phrases_field": self.keywords_or_phrases_field,
            "keywords_nested_col": self.keywords_nested_col,
            "explanation_nested_col": self.explanation_nested_col,
            "filters": self.filters,
        }

        if self.sample_training_rows_fn is not None:
            js_data, lexical_text_by_content = self.sample_training_rows_fn(
                **prepare_kwargs,
            )
        else:
            _, js_data, lexical_text_by_content = self.prepare_training_data_fn(
                **prepare_kwargs,
            )
        logger.info(f"Prepared rows K=[{len(js_data)}]")

        tokenizer = ITokenizer.from_pretrained(self.model_name_or_path)
        processor = TrainWithContrastiveProcessor(
            tokenizer=tokenizer,
            max_seq_len=self.max_seq_len,
            queries_field="queries",
        )
        dataset, tensor_names = igniset(
            js_data,
            processor=processor,
            batch_size=self.batch_size,
            streaming=True,
        )
        loader = NamedDataLoader(
            dataset=dataset,
            tensor_names=tensor_names,
            batch_size=self.batch_size,
        )
        return loader, processor, lexical_text_by_content

    def _build_pl_logger(self, *, save_dir: Path):
        if self.log_backend != "wandb":
            return False
        from pytorch_lightning.loggers import WandbLogger

        run_name = self.wandb_run_name or self.collection_name
        pl_logger = WandbLogger(project=self.wandb_project, name=run_name)
        wandb_config = {
            "recipe": self.recipe,
            "dataset_name_or_path": str(self.dataset_name_or_path),
            "model_name_or_path": self.model_name_or_path,
            "collection_name": self.collection_name,
            "collection_tag": self.collection_tag,
            "loss": self.loss,
            "temperature": self.temperature,
            "num_samples": self.num_samples,
            "batch_size": self.batch_size,
            "max_seq_len": self.max_seq_len,
            "n_epochs": self.n_epochs,
            "freeze_encoder": self.freeze_encoder,
            "gamma_joint": self.gamma_joint,
            "include_semantic_gamma": self.include_semantic_gamma,
            "include_keywords_gamma": self.include_keywords_gamma,
            "add_alpha_gate": self.add_alpha_gate,
            "alpha_entropy_weight": self.alpha_entropy_weight,
            "alpha_train_only": self.alpha_train_only,
            "alpha_mix_weight": self.alpha_mix_weight,
            "alpha_mix_weight_warmup_steps": self.alpha_mix_weight_warmup_steps,
            "alpha_head_input": self.alpha_head_input,
            "alpha_head_layers": self.alpha_head_layers,
            "alpha_head_hidden_dim": self.alpha_head_hidden_dim,
            "alpha_head_dropout": self.alpha_head_dropout,
            "alpha_head_activation": self.alpha_head_activation,
            "alpha_head_include_doc": self.alpha_head_include_doc,
            "query_diagonal_gate": self.query_diagonal_gate,
            "query_diagonal_gate_scale": self.query_diagonal_gate_scale,
            "query_diagonal_gate_mode": self.query_diagonal_gate_mode,
            "query_diagonal_identity_weight": self.query_diagonal_identity_weight,
            "query_diagonal_saturation_weight": self.query_diagonal_saturation_weight,
            "activation_fn": self.activation_fn,
            "margin": self.margin,
            "contrastive_temperature": self.contrastive_temperature,
            "soft_contrastive_temperature": self.soft_contrastive_temperature,
            "grad_acc_steps": self.grad_acc_steps,
            "optimizer": self.optimizer,
            "max_negative_inverse_idf_recall": self.max_negative_inverse_idf_recall,
            "contrastive_learnable_temperature": self.contrastive_learnable_temperature,
            "contrastive_decoupled": self.contrastive_decoupled,
            "contrastive_simcse_dropout_weight": self.contrastive_simcse_dropout_weight,
            "contrastive_soft_fn_attract_weight": self.contrastive_soft_fn_attract_weight,
            "contrastive_soft_fn_topk": self.contrastive_soft_fn_topk,
            "contrastive_loss_alpha_gate": self.contrastive_loss_alpha_gate,
            "contrastive_loss_alpha_gate_mode": self.contrastive_loss_alpha_gate_mode,
            "memory_bank_size": self.memory_bank_size,
            "memory_bank_warmup_steps": self.memory_bank_warmup_steps,
            "memory_bank_mining_mode": self.memory_bank_mining_mode,
            "memory_bank_hard_negatives": self.memory_bank_hard_negatives,
            "memory_bank_random_negatives": self.memory_bank_random_negatives,
            "memory_bank_hard_warmup_steps": self.memory_bank_hard_warmup_steps,
            "memory_bank_hard_ramp_steps": self.memory_bank_hard_ramp_steps,
            "memory_bank_too_hard_margin": self.memory_bank_too_hard_margin,
            "min_negative_inverse_idf_recall": self.min_negative_inverse_idf_recall,
            "negative_sampling_mode": self.negative_sampling_mode,
            "hard_negative_top_k": self.hard_negative_top_k,
            "focal_gamma": self.focal_gamma,
            "lr_gamma": self.lr_gamma,
            "lr_encoder": self.lr_encoder,
            "weight_decay": self.weight_decay,
            "content_field": self.content_field,
            "labels_field": self.labels_field,
            "split": self.split,
            "limit": self.limit,
            "chunk_id_col": self.chunk_id_col,
            "keywords_or_phrases_field": self.keywords_or_phrases_field,
            "keywords_nested_col": self.keywords_nested_col,
            "explanation_nested_col": self.explanation_nested_col,
            "filters": self.filters,
            "save_dir": str(save_dir),
            "metrics_path": self.metrics_path,
            "log_backend": self.log_backend,
            "training_mode": self.training_mode,
            "resolved_wandb_run_name": run_name,
        }
        pl_logger.experiment.config.update(wandb_config, allow_val_change=True)
        return pl_logger

    def train(self) -> str:
        import pytorch_lightning as L

        save_dir = self._resolved_save_dir()
        metrics_path = self._resolved_metrics_path(save_dir)
        self._write_collection_metadata(save_dir=save_dir)
        if self.collection_name is not None:
            logger.info(f"Derived collection name for downstream indexing: {self.collection_name}")
        if metrics_path is not None:
            logger.info(f"Batch metrics CSV will be written to: {metrics_path}")

        loader, processor, lexical_text_by_content = self._build_loader()
        lightning_module = self.build_lightning_module(
            processor=processor,
            lexical_text_by_content=lexical_text_by_content,
            save_dir=save_dir,
            metrics_path=metrics_path,
        )
        pl_logger = self._build_pl_logger(save_dir=save_dir)
        pl_trainer = L.Trainer(
            max_epochs=self.n_epochs,
            accelerator="auto",
            devices="auto",
            logger=pl_logger,
            log_every_n_steps=1,
            enable_checkpointing=False,
            enable_model_summary=False,
        )
        pl_trainer.fit(model=lightning_module, train_dataloaders=loader)
        return "" if metrics_path is None else str(metrics_path)


class GammaOnlyTrainingJob(BaseTrainingJob):
    @property
    def training_mode(self) -> str:
        return "gamma-only"

    def build_lightning_module(
        self,
        *,
        processor,
        lexical_text_by_content,
        save_dir: Path,
        metrics_path: str | None,
    ):
        from justatom.modeling.mask import ILanguageModel
        from justatom.running.encoders import GammaHybridRunner
        from justatom.running.trainer import (
            BiGammaLightningTrainer,
            UniGammaLightningTrainer,
        )

        lm_model = ILanguageModel.load(model_name_or_path=self.model_name_or_path)
        device = maybe_cuda_or_mps()
        runner = GammaHybridRunner(
            model=lm_model,
            processor=processor,
            prediction_heads=[],
            device=device,
            include_semantic_gamma=self.include_semantic_gamma,
            include_keywords_gamma=self.include_keywords_gamma,
            gamma_joint=self.gamma_joint,
            activation_fn=self.activation_fn,
            query_diagonal_gate=self.query_diagonal_gate,
            query_diagonal_gate_scale=self.query_diagonal_gate_scale,
            query_diagonal_gate_mode=self.query_diagonal_gate_mode,
            alpha_residual=self.alpha_residual,
            alpha_prior=self.alpha_prior,
            alpha_residual_scale=self.alpha_residual_scale,
            alpha_head_hidden_dim=self.alpha_head_hidden_dim,
            alpha_head_layers=self.alpha_head_layers,
            alpha_head_dropout=self.alpha_head_dropout,
            alpha_head_activation=self.alpha_head_activation,
            alpha_include_doc=self.alpha_head_include_doc,
        )
        enabled_count = int(self.include_semantic_gamma) + int(self.include_keywords_gamma)
        trainer_cls = BiGammaLightningTrainer if enabled_count == 2 else UniGammaLightningTrainer
        return trainer_cls(
            runner=runner,
            freeze_encoder=True,
            loss_name=self.loss,
            focal_gamma=self.focal_gamma,
            lr_gamma=self.lr_gamma,
            lr_encoder=self.lr_encoder,
            weight_decay=self.weight_decay,
            alpha_entropy_weight=self.alpha_entropy_weight,
            alpha_train_only=self.alpha_train_only,
            alpha_mix_weight=self.alpha_mix_weight,
            alpha_mix_weight_warmup_steps=self.alpha_mix_weight_warmup_steps,
            query_diagonal_identity_weight=self.query_diagonal_identity_weight,
            query_diagonal_saturation_weight=self.query_diagonal_saturation_weight,
            margin=self.margin,
            contrastive_temperature=self.contrastive_temperature,
            soft_contrastive_temperature=self.soft_contrastive_temperature,
            grad_acc_steps=self.grad_acc_steps,
            optimizer_name=self.optimizer,
            max_negative_inverse_idf_recall=self.max_negative_inverse_idf_recall,
            contrastive_learnable_temperature=self.contrastive_learnable_temperature,
            contrastive_decoupled=self.contrastive_decoupled,
            contrastive_simcse_dropout_weight=self.contrastive_simcse_dropout_weight,
            contrastive_soft_fn_attract_weight=self.contrastive_soft_fn_attract_weight,
            contrastive_soft_fn_topk=self.contrastive_soft_fn_topk,
            contrastive_loss_alpha_gate=self.contrastive_loss_alpha_gate,
            contrastive_loss_alpha_gate_mode=self.contrastive_loss_alpha_gate_mode,
            memory_bank_size=self.memory_bank_size,
            memory_bank_warmup_steps=self.memory_bank_warmup_steps,
            memory_bank_mining_mode=self.memory_bank_mining_mode,
            memory_bank_hard_negatives=self.memory_bank_hard_negatives,
            memory_bank_random_negatives=self.memory_bank_random_negatives,
            memory_bank_hard_warmup_steps=self.memory_bank_hard_warmup_steps,
            memory_bank_hard_ramp_steps=self.memory_bank_hard_ramp_steps,
            memory_bank_too_hard_margin=self.memory_bank_too_hard_margin,
            min_negative_inverse_idf_recall=self.min_negative_inverse_idf_recall,
            negative_sampling_mode=self.negative_sampling_mode,
            hard_negative_top_k=self.hard_negative_top_k,
            lexical_text_by_content=lexical_text_by_content,
            save_dir=save_dir,
            metrics_path=metrics_path,
        )


class EncoderGammaTrainingJob(BaseTrainingJob):
    @property
    def training_mode(self) -> str:
        return "encoder+gamma"

    def build_lightning_module(
        self,
        *,
        processor,
        lexical_text_by_content,
        save_dir: Path,
        metrics_path: str | None,
    ):
        from justatom.modeling.mask import ILanguageModel
        from justatom.running.encoders import GammaHybridRunner
        from justatom.running.trainer import (
            BiGammaLightningTrainer,
            UniGammaLightningTrainer,
        )

        lm_model = ILanguageModel.load(model_name_or_path=self.model_name_or_path)
        device = maybe_cuda_or_mps()
        runner = GammaHybridRunner(
            model=lm_model,
            processor=processor,
            prediction_heads=[],
            device=device,
            include_semantic_gamma=self.include_semantic_gamma,
            include_keywords_gamma=self.include_keywords_gamma,
            gamma_joint=self.gamma_joint,
            activation_fn=self.activation_fn,
            query_diagonal_gate=self.query_diagonal_gate,
            query_diagonal_gate_scale=self.query_diagonal_gate_scale,
            query_diagonal_gate_mode=self.query_diagonal_gate_mode,
            alpha_residual=self.alpha_residual,
            alpha_prior=self.alpha_prior,
            alpha_residual_scale=self.alpha_residual_scale,
            alpha_head_hidden_dim=self.alpha_head_hidden_dim,
            alpha_head_layers=self.alpha_head_layers,
            alpha_head_dropout=self.alpha_head_dropout,
            alpha_head_activation=self.alpha_head_activation,
            alpha_include_doc=self.alpha_head_include_doc,
        )
        enabled_count = int(self.include_semantic_gamma) + int(self.include_keywords_gamma)
        trainer_cls = BiGammaLightningTrainer if enabled_count == 2 else UniGammaLightningTrainer
        return trainer_cls(
            runner=runner,
            freeze_encoder=False,
            loss_name=self.loss,
            focal_gamma=self.focal_gamma,
            lr_gamma=self.lr_gamma,
            lr_encoder=self.lr_encoder,
            weight_decay=self.weight_decay,
            alpha_entropy_weight=self.alpha_entropy_weight,
            alpha_train_only=self.alpha_train_only,
            alpha_mix_weight=self.alpha_mix_weight,
            alpha_mix_weight_warmup_steps=self.alpha_mix_weight_warmup_steps,
            query_diagonal_identity_weight=self.query_diagonal_identity_weight,
            query_diagonal_saturation_weight=self.query_diagonal_saturation_weight,
            margin=self.margin,
            contrastive_temperature=self.contrastive_temperature,
            soft_contrastive_temperature=self.soft_contrastive_temperature,
            grad_acc_steps=self.grad_acc_steps,
            optimizer_name=self.optimizer,
            max_negative_inverse_idf_recall=self.max_negative_inverse_idf_recall,
            contrastive_learnable_temperature=self.contrastive_learnable_temperature,
            contrastive_decoupled=self.contrastive_decoupled,
            contrastive_simcse_dropout_weight=self.contrastive_simcse_dropout_weight,
            contrastive_soft_fn_attract_weight=self.contrastive_soft_fn_attract_weight,
            contrastive_soft_fn_topk=self.contrastive_soft_fn_topk,
            contrastive_loss_alpha_gate=self.contrastive_loss_alpha_gate,
            contrastive_loss_alpha_gate_mode=self.contrastive_loss_alpha_gate_mode,
            memory_bank_size=self.memory_bank_size,
            memory_bank_warmup_steps=self.memory_bank_warmup_steps,
            memory_bank_mining_mode=self.memory_bank_mining_mode,
            memory_bank_hard_negatives=self.memory_bank_hard_negatives,
            memory_bank_random_negatives=self.memory_bank_random_negatives,
            memory_bank_hard_warmup_steps=self.memory_bank_hard_warmup_steps,
            memory_bank_hard_ramp_steps=self.memory_bank_hard_ramp_steps,
            memory_bank_too_hard_margin=self.memory_bank_too_hard_margin,
            min_negative_inverse_idf_recall=self.min_negative_inverse_idf_recall,
            negative_sampling_mode=self.negative_sampling_mode,
            hard_negative_top_k=self.hard_negative_top_k,
            lexical_text_by_content=lexical_text_by_content,
            save_dir=save_dir,
            metrics_path=metrics_path,
        )


class AtomGateTrainingJob(EncoderGammaTrainingJob):
    @property
    def training_mode(self) -> str:
        return "atom-gate"

    def __post_init__(self):
        self.recipe = "atom_gate"
        self.add_alpha_gate = True
        self.loss = "contrastive"
        self.freeze_encoder = False
        self.gamma_joint = True
        self.include_semantic_gamma = True
        self.include_keywords_gamma = True
        self.alpha_train_only = True
        self.optimizer = "adamw"
        self.contrastive_learnable_temperature = True
        self.contrastive_decoupled = True
        self.contrastive_soft_fn_attract_weight = 0.0
        self.contrastive_soft_fn_topk = 1
        self.contrastive_loss_alpha_gate = True
        self.contrastive_loss_alpha_gate_mode = "augment"
        if self.temperature is None and float(self.contrastive_temperature) == 0.03:
            self.contrastive_temperature = 0.05
        if float(self.contrastive_simcse_dropout_weight) == 0.0:
            self.contrastive_simcse_dropout_weight = 0.1
        super().__post_init__()


class EncoderOnlyTrainingJob(BaseTrainingJob):
    @property
    def training_mode(self) -> str:
        return "encoder-only"

    def build_lightning_module(
        self,
        *,
        processor,
        lexical_text_by_content,
        save_dir: Path,
        metrics_path: str | None,
    ):
        from justatom.modeling.mask import ILanguageModel
        from justatom.running.encoders import EncoderRunner
        from justatom.running.trainer import EncoderOnlyLightningTrainer

        lm_model = ILanguageModel.load(model_name_or_path=self.model_name_or_path)
        device = maybe_cuda_or_mps()
        runner = EncoderRunner(
            model=lm_model,
            processor=processor,
            prediction_heads=[],
            device=device,
        )
        return EncoderOnlyLightningTrainer(
            runner=runner,
            loss_name=self.loss,
            margin=self.margin,
            focal_gamma=self.focal_gamma,
            contrastive_temperature=self.contrastive_temperature,
            soft_contrastive_temperature=self.soft_contrastive_temperature,
            lr_encoder=self.lr_encoder,
            weight_decay=self.weight_decay,
            grad_acc_steps=self.grad_acc_steps,
            optimizer_name=self.optimizer,
            contrastive_learnable_temperature=self.contrastive_learnable_temperature,
            contrastive_decoupled=self.contrastive_decoupled,
            contrastive_simcse_dropout_weight=self.contrastive_simcse_dropout_weight,
            contrastive_soft_fn_attract_weight=self.contrastive_soft_fn_attract_weight,
            contrastive_soft_fn_topk=self.contrastive_soft_fn_topk,
            memory_bank_size=self.memory_bank_size,
            memory_bank_warmup_steps=self.memory_bank_warmup_steps,
            memory_bank_mining_mode=self.memory_bank_mining_mode,
            memory_bank_hard_negatives=self.memory_bank_hard_negatives,
            memory_bank_random_negatives=self.memory_bank_random_negatives,
            memory_bank_hard_warmup_steps=self.memory_bank_hard_warmup_steps,
            memory_bank_hard_ramp_steps=self.memory_bank_hard_ramp_steps,
            memory_bank_too_hard_margin=self.memory_bank_too_hard_margin,
            lexical_text_by_content=lexical_text_by_content,
            save_dir=save_dir,
            metrics_path=metrics_path,
        )


def create_training_job(**kwargs) -> BaseTrainingJob:
    recipe = _normalize_recipe_name(kwargs.get("recipe"))
    if recipe:
        if recipe != "atom_gate":
            raise ValueError(f"Unsupported training recipe={kwargs.get('recipe')!r}. Expected atom_gate")
        kwargs["recipe"] = recipe
        return AtomGateTrainingJob(**kwargs)

    mode = _resolve_training_mode(
        freeze_encoder=bool(kwargs.get("freeze_encoder", True)),
        include_semantic_gamma=bool(kwargs.get("include_semantic_gamma", True)),
        include_keywords_gamma=bool(kwargs.get("include_keywords_gamma", True)),
    )
    mapping = {
        "gamma-only": GammaOnlyTrainingJob,
        "encoder+gamma": EncoderGammaTrainingJob,
        "encoder-only": EncoderOnlyTrainingJob,
    }
    return mapping[mode](**kwargs)


__all__ = [
    "BaseTrainingJob",
    "GammaOnlyTrainingJob",
    "EncoderGammaTrainingJob",
    "AtomGateTrainingJob",
    "EncoderOnlyTrainingJob",
    "create_training_job",
]
