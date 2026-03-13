from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger


def maybe_cuda_or_mps() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.has_mps:
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
    model_name_or_path: str = "intfloat/multilingual-e5-small"
    loss: str = "contrastive"
    num_samples: int = 100
    batch_size: int = 4
    max_seq_len: int = 512
    freeze_encoder: bool = True
    include_semantic_gamma: bool = True
    include_keywords_gamma: bool = True
    activation_fn: str = "sigmoid"
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
        save_dir = Path(self.save_dir or Path(os.getcwd()) / "weights")
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def _resolved_metrics_path(self, save_dir: Path) -> str | None:
        default_metrics_path = save_dir / "gamma_metrics.csv"
        if self.metrics_path is not None:
            return str(self.roll_metrics_path_fn(Path(self.metrics_path)))
        if self.log_backend == "csv":
            return str(self.roll_metrics_path_fn(default_metrics_path))
        return None

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

        pl_logger = WandbLogger(project=self.wandb_project, name=self.wandb_run_name)
        wandb_config = {
            "dataset_name_or_path": str(self.dataset_name_or_path),
            "model_name_or_path": self.model_name_or_path,
            "loss": self.loss,
            "num_samples": self.num_samples,
            "batch_size": self.batch_size,
            "max_seq_len": self.max_seq_len,
            "n_epochs": self.n_epochs,
            "freeze_encoder": self.freeze_encoder,
            "include_semantic_gamma": self.include_semantic_gamma,
            "include_keywords_gamma": self.include_keywords_gamma,
            "activation_fn": self.activation_fn,
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
        }
        pl_logger.experiment.config.update(wandb_config, allow_val_change=True)
        return pl_logger

    def train(self) -> str:
        import pytorch_lightning as L

        save_dir = self._resolved_save_dir()
        metrics_path = self._resolved_metrics_path(save_dir)
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
        from justatom.running.trainer import BiGammaLightningTrainer, UniGammaLightningTrainer

        lm_model = ILanguageModel.load(model_name_or_path=self.model_name_or_path)
        device = maybe_cuda_or_mps()
        runner = GammaHybridRunner(
            model=lm_model,
            processor=processor,
            prediction_heads=[],
            device=device,
            include_semantic_gamma=self.include_semantic_gamma,
            include_keywords_gamma=self.include_keywords_gamma,
            activation_fn=self.activation_fn,
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
        from justatom.running.trainer import BiGammaLightningTrainer, UniGammaLightningTrainer

        lm_model = ILanguageModel.load(model_name_or_path=self.model_name_or_path)
        device = maybe_cuda_or_mps()
        runner = GammaHybridRunner(
            model=lm_model,
            processor=processor,
            prediction_heads=[],
            device=device,
            include_semantic_gamma=self.include_semantic_gamma,
            include_keywords_gamma=self.include_keywords_gamma,
            activation_fn=self.activation_fn,
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
            lexical_text_by_content=lexical_text_by_content,
            save_dir=save_dir,
            metrics_path=metrics_path,
        )


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

        del lexical_text_by_content
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
            focal_gamma=self.focal_gamma,
            lr_encoder=self.lr_encoder,
            weight_decay=self.weight_decay,
            save_dir=save_dir,
            metrics_path=metrics_path,
        )


def create_training_job(**kwargs) -> BaseTrainingJob:
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
    "EncoderOnlyTrainingJob",
    "create_training_job",
]
