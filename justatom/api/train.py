import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import dotenv
import polars as pl
import pytorch_lightning as L
import torch
from loguru import logger
from pytorch_lightning.loggers import WandbLogger

from justatom.modeling.mask import ILanguageModel
from justatom.processing import ITokenizer
from justatom.processing.loader import NamedDataLoader
from justatom.processing.prime import TrainWithContrastiveProcessor
from justatom.running.encoders import GammaHybridRunner
from justatom.running.trainer import BiGammaLightningTrainer, UniGammaLightningTrainer
from justatom.tooling.dataset import DatasetRecordAdapter


dotenv.load_dotenv()


logger.info(f"Enable MPS fallback = {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', -1)}")


def _parse_bool(raw: str | bool) -> bool:
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Unable to parse bool value: {raw}")


def _parse_args(argv: list[str] | None = None) -> dict:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="justatom|train",
        description="Gamma calibration training pipeline with encoder freeze/unfreeze modes",
    )

    parser.add_argument("--dataset-name-or-path", required=True)
    parser.add_argument("--model-name-or-path", default="intfloat/multilingual-e5-small")
    parser.add_argument("--loss", default="contrastive", choices=["contrastive", "focal-contrastive"])

    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=1)

    parser.add_argument("--content-col", default="content")
    parser.add_argument("--labels-col", default="queries")
    parser.add_argument("--id-col")
    parser.add_argument("--keywords-col")
    parser.add_argument("--keywords-nested-col")
    parser.add_argument("--explanation-nested-col")
    parser.add_argument("--filter-fields", nargs="+")

    parser.add_argument("--lr-gamma", type=float, default=1e-2)
    parser.add_argument("--lr-encoder", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    parser.add_argument("--freeze-encoder", type=_parse_bool, default=True)
    parser.add_argument("--include-semantic-gamma", type=_parse_bool, default=True)
    parser.add_argument("--include-keywords-gamma", type=_parse_bool, default=True)
    parser.add_argument("--activation-fn", default="sigmoid", choices=["sigmoid", "tanh", "relu", "identity"])
    parser.add_argument("--focal-gamma", type=float, default=2.0)

    parser.add_argument("--save-dir")
    parser.add_argument("--metrics-path")
    parser.add_argument("--log", default="csv", choices=["csv", "wandb"])
    parser.add_argument("--wandb-project", default="justatom-gamma")
    parser.add_argument("--wandb-run-name")

    args = parser.parse_args(argv)

    filters = {"fields": args.filter_fields} if args.filter_fields else None

    return {
        "dataset_name_or_path": args.dataset_name_or_path,
        "model_name_or_path": args.model_name_or_path,
        "loss": args.loss,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len,
        "n_epochs": args.n_epochs,
        "content_field": args.content_col,
        "labels_field": args.labels_col,
        "chunk_id_col": args.id_col,
        "keywords_or_phrases_field": args.keywords_col,
        "keywords_nested_col": args.keywords_nested_col,
        "explanation_nested_col": args.explanation_nested_col,
        "filters": filters,
        "lr_gamma": args.lr_gamma,
        "lr_encoder": args.lr_encoder,
        "weight_decay": args.weight_decay,
        "freeze_encoder": args.freeze_encoder,
        "include_semantic_gamma": args.include_semantic_gamma,
        "include_keywords_gamma": args.include_keywords_gamma,
        "activation_fn": args.activation_fn,
        "focal_gamma": args.focal_gamma,
        "log_backend": args.log,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
        "save_dir": args.save_dir,
        "metrics_path": args.metrics_path,
    }


def maybe_cuda_or_mps() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.has_mps:
        return "mps"
    return "cpu"


def _roll_metrics_path_if_exists(file_path: str | Path) -> Path:
    path = Path(file_path)
    if not path.exists():
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def _coerce_to_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return [value]


def _resolve_lexical_text(
    content: str,
    meta: dict,
    keywords_or_phrases_field: str | None,
    keywords_nested_col: str | None,
    explanation_nested_col: str | None,
) -> str:
    if keywords_or_phrases_field is None:
        return content

    items = _coerce_to_list(meta.get("keywords_or_phrases"))
    tokens: list[str] = []
    use_keywords = keywords_nested_col is not None or explanation_nested_col is None
    use_explanations = explanation_nested_col is not None

    for item in items:
        if isinstance(item, str):
            value = item.strip()
            if value and use_keywords:
                tokens.append(value)
            continue

        if not isinstance(item, dict):
            continue

        if use_keywords:
            keyword_value = item.get("keyword_or_phrase")
            if keyword_value is None and keywords_nested_col is not None:
                keyword_value = item.get(keywords_nested_col)
            if keyword_value is not None and str(keyword_value).strip():
                tokens.append(str(keyword_value).strip())

        if use_explanations:
            explanation_value = item.get("explanation")
            if explanation_value is None and explanation_nested_col is not None:
                explanation_value = item.get(explanation_nested_col)
            if explanation_value is not None and str(explanation_value).strip():
                tokens.append(str(explanation_value).strip())

    if len(tokens) == 0:
        return content
    return "\n".join(tokens)


def prepare_gamma_data(
    dataset_name_or_path: str | Path,
    num_samples: int = 100,
    content_field: str = "content",
    labels_field: str = "queries",
    chunk_id_col: str | None = None,
    keywords_or_phrases_field: str | None = "keywords_or_phrases",
    keywords_nested_col: str | None = None,
    explanation_nested_col: str | None = None,
    filters: dict | None = None,
) -> tuple[pl.DataFrame, list[dict], dict[str, str]]:
    if labels_field is None:
        msg = "labels_field must be provided for training"
        logger.error(msg)
        raise ValueError(msg)

    adapter = DatasetRecordAdapter.from_source(
        str(dataset_name_or_path),
        content_col=content_field,
        queries_col=labels_field,
        chunk_id_col=chunk_id_col,
        keywords_col=keywords_or_phrases_field,
        keywords_nested_col=keywords_nested_col,
        explanation_nested_col=explanation_nested_col,
        filter_fields=(filters or {}).get("fields", []),
        preserve_all_fields=False,
    )

    docs = list(adapter.iterator())
    rows: list[dict] = []
    lexical_by_content: dict[str, str] = {}

    for doc in docs:
        content = str(doc.get("content") or "")
        if not content.strip():
            continue

        meta = doc.get("meta") or {}
        labels = _coerce_to_list(meta.get("labels"))
        labels = [str(label).strip() for label in labels if str(label).strip()]
        if len(labels) == 0:
            continue

        lexical_text = _resolve_lexical_text(
            content=content,
            meta=meta,
            keywords_or_phrases_field=keywords_or_phrases_field,
            keywords_nested_col=keywords_nested_col,
            explanation_nested_col=explanation_nested_col,
        )
        lexical_by_content[content] = lexical_text

        keywords_payload = meta.get("keywords_or_phrases", [])
        for query in labels:
            rows.append(
                {
                    "content": content,
                    "queries": query,
                    "chunk_id": doc.get("id"),
                    "keywords_or_phrases": keywords_payload,
                    "lexical_text": lexical_text,
                }
            )

    if len(rows) == 0:
        msg = "No training rows were prepared from dataset"
        logger.error(msg)
        raise ValueError(msg)

    pl_data = (
        pl.from_dicts(rows)
        .sample(shuffle=True, fraction=1.0)
        .head(num_samples)
    )

    return pl_data, pl_data.to_dicts(), lexical_by_content


def train_gamma_calibration(
    dataset_name_or_path: str | Path,
    model_name_or_path: str = "intfloat/multilingual-e5-small",
    loss: str = "contrastive",
    num_samples: int = 100,
    batch_size: int = 4,
    max_seq_len: int = 512,
    freeze_encoder: bool = True,
    include_semantic_gamma: bool = True,
    include_keywords_gamma: bool = True,
    activation_fn: str = "sigmoid",
    focal_gamma: float = 2.0,
    log_backend: str = "csv",
    wandb_project: str = "justatom-gamma",
    wandb_run_name: str | None = None,
    n_epochs: int = 1,
    content_field: str = "content",
    labels_field: str = "queries",
    chunk_id_col: str | None = None,
    keywords_or_phrases_field: str | None = "keywords_or_phrases",
    keywords_nested_col: str | None = None,
    explanation_nested_col: str | None = None,
    filters: dict | None = None,
    lr_gamma: float = 1e-2,
    lr_encoder: float = 2e-5,
    weight_decay: float = 0.01,
    save_dir: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> str:
    save_dir = Path(save_dir or Path(os.getcwd()) / "weights")
    save_dir.mkdir(parents=True, exist_ok=True)

    pl_data, js_data, lexical_text_by_content = prepare_gamma_data(
        dataset_name_or_path=dataset_name_or_path,
        num_samples=num_samples,
        content_field=content_field,
        labels_field=labels_field,
        chunk_id_col=chunk_id_col,
        keywords_or_phrases_field=keywords_or_phrases_field,
        keywords_nested_col=keywords_nested_col,
        explanation_nested_col=explanation_nested_col,
        filters=filters,
    )
    logger.info(f"Prepared rows K=[{len(pl_data)}]")

    tokenizer = ITokenizer.from_pretrained(model_name_or_path)
    processor = TrainWithContrastiveProcessor(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        queries_field="queries",
    )
    lm_model = ILanguageModel.load(model_name_or_path=model_name_or_path)

    dataset, tensor_names, _, _ = processor.dataset_from_dicts(js_data, return_baskets=True)
    loader = NamedDataLoader(dataset=dataset, tensor_names=tensor_names, batch_size=batch_size)

    device = maybe_cuda_or_mps()
    if not include_semantic_gamma and not include_keywords_gamma:
        raise ValueError("At least one gamma must be enabled for gamma calibration training")

    lm_runner = GammaHybridRunner(
        model=lm_model,
        processor=processor,
        prediction_heads=[],
        device=device,
        include_semantic_gamma=include_semantic_gamma,
        include_keywords_gamma=include_keywords_gamma,
        activation_fn=activation_fn,
    )
    default_metrics_path = save_dir / "gamma_metrics.csv"
    if metrics_path is not None:
        metrics_path = _roll_metrics_path_if_exists(Path(metrics_path))
    elif log_backend == "csv":
        metrics_path = _roll_metrics_path_if_exists(default_metrics_path)
    else:
        metrics_path = None

    if metrics_path is not None:
        logger.info(f"Batch metrics CSV will be written to: {metrics_path}")

    enabled_count = int(include_semantic_gamma) + int(include_keywords_gamma)
    if enabled_count == 2:
        gamma_trainer = BiGammaLightningTrainer(
            runner=lm_runner,
            freeze_encoder=freeze_encoder,
            loss_name=loss,
            focal_gamma=focal_gamma,
            lr_gamma=lr_gamma,
            lr_encoder=lr_encoder,
            weight_decay=weight_decay,
            lexical_text_by_content=lexical_text_by_content,
            save_dir=save_dir,
            metrics_path=metrics_path,
        )
    elif enabled_count == 1:
        gamma_trainer = UniGammaLightningTrainer(
            runner=lm_runner,
            freeze_encoder=freeze_encoder,
            loss_name=loss,
            focal_gamma=focal_gamma,
            lr_gamma=lr_gamma,
            lr_encoder=lr_encoder,
            weight_decay=weight_decay,
            lexical_text_by_content=lexical_text_by_content,
            save_dir=save_dir,
            metrics_path=metrics_path,
        )
    else:
        raise ValueError("At least one gamma must be enabled for gamma calibration training")

    pl_logger = False
    if log_backend == "wandb":
        pl_logger = WandbLogger(project=wandb_project, name=wandb_run_name)
        wandb_config = {
            "dataset_name_or_path": str(dataset_name_or_path),
            "model_name_or_path": model_name_or_path,
            "loss": loss,
            "num_samples": num_samples,
            "batch_size": batch_size,
            "max_seq_len": max_seq_len,
            "n_epochs": n_epochs,
            "freeze_encoder": freeze_encoder,
            "include_semantic_gamma": include_semantic_gamma,
            "include_keywords_gamma": include_keywords_gamma,
            "activation_fn": activation_fn,
            "focal_gamma": focal_gamma,
            "lr_gamma": lr_gamma,
            "lr_encoder": lr_encoder,
            "weight_decay": weight_decay,
            "content_field": content_field,
            "labels_field": labels_field,
            "chunk_id_col": chunk_id_col,
            "keywords_or_phrases_field": keywords_or_phrases_field,
            "keywords_nested_col": keywords_nested_col,
            "explanation_nested_col": explanation_nested_col,
            "filters": filters,
            "save_dir": str(save_dir),
            "metrics_path": None if metrics_path is None else str(metrics_path),
            "log_backend": log_backend,
        }
        pl_logger.experiment.config.update(wandb_config, allow_val_change=True)

    pl_trainer = L.Trainer(
        max_epochs=n_epochs,
        accelerator="auto",
        devices="auto",
        logger=pl_logger,
        log_every_n_steps=1,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    pl_trainer.fit(
        model=gamma_trainer,
        train_dataloaders=loader,
    )

    return "" if metrics_path is None else str(metrics_path)


def main(**kwargs):
    metrics_path = train_gamma_calibration(**kwargs)
    if metrics_path:
        logger.info(f"Training completed. Metrics are saved to: {metrics_path}")
    else:
        logger.info("Training completed. CSV metrics path is not configured for this run.")


if __name__ == "__main__":
    main(**_parse_args())
