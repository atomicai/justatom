import argparse
import asyncio as asio
import inspect
import os
import sys
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Any

import dotenv
import numpy as np
import polars as pl
import torch
from loguru import logger
from torch.utils.data import ConcatDataset
from tqdm.auto import tqdm

from justatom.configuring.scenarios import deep_merge
from justatom.configuring.scenarios import load_scenario_config
from justatom.configuring.scenarios import parse_unknown_overrides
from justatom.running.evaluator import EvaluatorRunner
from justatom.running.mask import IEvaluatorRunner
from justatom.running.service import RunningService
from justatom.tooling.dataset import DatasetRecordAdapter
from justatom.tooling import stl

dotenv.load_dotenv()

logger.info(f"Enable MPS fallback = {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', -1)}")


def _legacy_cli_overlay(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if args.dataset_name_or_path is not None:
        cfg.setdefault("dataset", {})["name_or_path"] = args.dataset_name_or_path
    if args.collection_name is not None:
        cfg.setdefault("collection", {})["name"] = args.collection_name
    if args.save_results_to_dir is not None:
        cfg.setdefault("output", {})["save_results_to_dir"] = args.save_results_to_dir

    if args.search_pipeline is not None:
        cfg.setdefault("search", {})["pipeline"] = args.search_pipeline
    if args.top_k is not None:
        cfg.setdefault("search", {})["top_k"] = args.top_k
    if args.search_batch_size is not None:
        cfg.setdefault("search", {})["batch_size"] = args.search_batch_size

    if args.model_name_or_path is not None:
        cfg.setdefault("model", {})["name"] = args.model_name_or_path
    if args.query_prefix is not None:
        cfg.setdefault("model", {})["query_prefix"] = args.query_prefix
    if args.content_prefix is not None:
        cfg.setdefault("model", {})["content_prefix"] = args.content_prefix

    if args.index_batch_size is not None:
        cfg.setdefault("index", {})["batch_size"] = args.index_batch_size
    if args.flush_collection is not None:
        cfg.setdefault("index", {})["flush_collection"] = args.flush_collection

    if args.labels_col is not None:
        cfg.setdefault("dataset", {})["labels_field"] = args.labels_col
    if args.content_col is not None:
        cfg.setdefault("dataset", {})["content_field"] = args.content_col
    if args.dataset_split is not None:
        cfg.setdefault("dataset", {})["split"] = args.dataset_split
    if args.dataset_limit is not None:
        cfg.setdefault("dataset", {})["limit"] = args.dataset_limit
    if args.id_col is not None:
        cfg.setdefault("dataset", {})["chunk_id_col"] = args.id_col
    if args.keywords_col is not None:
        cfg.setdefault("dataset", {})["keywords_col"] = args.keywords_col
    if args.keywords_nested_col is not None:
        cfg.setdefault("dataset", {})["keywords_nested_col"] = args.keywords_nested_col
    if args.explanation_nested_col is not None:
        cfg.setdefault("dataset", {})["explanation_nested_col"] = args.explanation_nested_col
    if getattr(args, "drop_columns", None) is not None:
        cfg.setdefault("dataset", {})["drop_columns"] = args.drop_columns

    if args.metrics is not None:
        cfg.setdefault("metrics", {})["names"] = args.metrics
    if args.metrics_top_k is not None:
        cfg.setdefault("metrics", {})["top_k"] = args.metrics_top_k
    if args.eval_top_k is not None:
        cfg.setdefault("metrics", {})["eval_top_k"] = args.eval_top_k

    if args.filter_fields is not None:
        cfg.setdefault("filters", {})["fields"] = args.filter_fields

    if args.weaviate_host is not None:
        cfg.setdefault("weaviate", {})["host"] = args.weaviate_host
    if args.weaviate_port is not None:
        cfg.setdefault("weaviate", {})["port"] = args.weaviate_port

    if args.alpha is not None:
        cfg.setdefault("props", {})["alpha"] = args.alpha

    return cfg


def load_eval_config(
    *,
    config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return load_scenario_config(
        "evaluate",
        config=config,
        config_path=config_path,
        overrides=overrides,
    )


def _cfg_to_main_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    dataset = cfg.get("dataset") or {}
    collection = cfg.get("collection") or {}
    output = cfg.get("output") or {}
    search = cfg.get("search") or {}
    model = cfg.get("model") or {}
    index = cfg.get("index") or {}
    metrics = cfg.get("metrics") or {}
    filters_cfg = cfg.get("filters") or {}
    weaviate = cfg.get("weaviate") or {}
    props = cfg.get("props") or {}

    filter_fields = filters_cfg.get("fields")
    filters = {"fields": filter_fields} if filter_fields else None

    out = {
        "model_name_or_path": model.get("name"),
        "search_pipeline": search.get("pipeline", "embedding"),
        "query_prefix": model.get("query_prefix"),
        "content_prefix": model.get("content_prefix"),
        "collection_name": collection.get("name", "Document"),
        "flush_collection": bool(index.get("flush_collection", False)),
        "dataset_name_or_path": dataset.get("name_or_path"),
        "save_results_to_dir": output.get("save_results_to_dir"),
        "top_k": int(search.get("top_k", 20)),
        "index_batch_size": int(index.get("batch_size", 4)),
        "search_batch_size": int(search.get("batch_size", 32)),
        "filters": filters,
        "metrics": metrics.get("names"),
        "metrics_top_k": metrics.get("top_k") or ["HitRate"],
        "eval_top_k": metrics.get("eval_top_k"),
        "labels_field": dataset.get("labels_field"),
        "content_field": dataset.get("content_field", "content"),
        "split": dataset.get("split"),
        "limit": dataset.get("limit"),
        "chunk_id_col": dataset.get("chunk_id_col"),
        "keywords_or_phrases_field": dataset.get("keywords_col"),
        "keywords_nested_col": dataset.get("keywords_nested_col"),
        "explanation_nested_col": dataset.get("explanation_nested_col"),
        "drop_columns": dataset.get("drop_columns"),
        "weaviate_host": weaviate.get("host"),
        "weaviate_port": weaviate.get("port"),
    }
    out.update({k: v for k, v in props.items() if v is not None})
    return out


def resolve_eval_kwargs(
    *,
    config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = load_eval_config(
        config=config,
        config_path=config_path,
        overrides=overrides,
    )
    return _cfg_to_main_kwargs(cfg)


def _parse_args(argv: list[str] | None = None) -> dict:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="justatom|eval",
        description="Index and evaluate IR quality on a given dataset",
    )

    parser.add_argument("--config")
    parser.add_argument("--dataset-name-or-path")
    parser.add_argument("--collection-name")
    parser.add_argument("--save-results-to-dir")

    parser.add_argument(
        "--search-pipeline",
        choices=["embedding", "hybrid", "keywords", "atomicai"],
    )
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--query-prefix")
    parser.add_argument("--content-prefix")
    flush_group = parser.add_mutually_exclusive_group()
    flush_group.add_argument("--flush-collection", dest="flush_collection", action="store_true")
    flush_group.add_argument("--no-flush-collection", dest="flush_collection", action="store_false")
    parser.set_defaults(flush_collection=None)

    parser.add_argument("--top-k", type=int)
    parser.add_argument("--index-batch-size", type=int)
    parser.add_argument("--search-batch-size", type=int)
    parser.add_argument("--labels-col")
    parser.add_argument("--content-col")
    parser.add_argument("--dataset-split")
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--id-col")
    parser.add_argument("--keywords-col")
    parser.add_argument("--keywords-nested-col")
    parser.add_argument("--explanation-nested-col")
    parser.add_argument("--drop-columns", nargs="+")

    parser.add_argument("--metrics", nargs="+")
    parser.add_argument("--metrics-top-k", nargs="+")
    parser.add_argument("--eval-top-k", nargs="+", type=int)
    parser.add_argument("--filter-fields", nargs="+")

    parser.add_argument("--weaviate-host")
    parser.add_argument("--weaviate-port", type=int)

    parser.add_argument("--alpha", type=float)

    args, unknown = parser.parse_known_args(argv)
    overrides = _legacy_cli_overlay(args)
    dotted_overrides = parse_unknown_overrides(unknown)
    if dotted_overrides:
        overrides = deep_merge(overrides, dotted_overrides)

    return resolve_eval_kwargs(
        config_path=args.config,
        overrides=overrides,
    )


def to_numpy(container):
    try:
        return container.cpu().numpy()
    except AttributeError:
        return container


def random_split(ds: ConcatDataset, lengths: list[int]):
    if sum(lengths) != len(ds):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    try:
        idx_dataset = np.where(np.array(ds.cumulative_sizes) > lengths[0])[0][0]
    except IndexError as ex:
        raise Exception(
            "All dataset chunks are being assigned to train set leaving no samples for dev set. "
            "Either consider increasing dev_split or setting it to 0.0\n"
            f"Cumulative chunk sizes: {ds.cumulative_sizes}\n"
            f"train/dev split: {lengths}"
        ) from ex

    assert idx_dataset >= 1, "Dev_split ratio is too large, there is no data in train set."
    train = ConcatDataset(ds.datasets[:idx_dataset])
    test = ConcatDataset(ds.datasets[idx_dataset:])
    return train, test


async def main(
    model_name_or_path: str | None = None,
    search_pipeline: str = "embedding",
    query_prefix: str = None,
    content_prefix: str = None,
    collection_name: str = None,
    flush_collection: bool = False,
    dataset_name_or_path: str = None,
    save_results_to_dir: str = None,
    top_k: int = 20,
    index_batch_size: int = 4,
    search_batch_size: int = 32,
    filters: dict | None = None,
    metrics=None,
    metrics_top_k=["HitRate"],
    eval_top_k=None,
    labels_field: str | None = None,
    content_field: str = "content",
    split: str | None = None,
    limit: int | None = None,
    chunk_id_col: str | None = None,
    keywords_or_phrases_field: str = None,
    keywords_nested_col: str | None = None,
    explanation_nested_col: str | None = None,
    drop_columns: list[str] | None = None,
    weaviate_host: str = None,
    weaviate_port: int = None,
    **props,
):
    ir_runner = None
    collection_name = collection_name or "Document"
    if dataset_name_or_path is None:
        resolved_dataset_name_or_path = None
    else:
        raw_dataset_name_or_path = str(dataset_name_or_path)
        if "://" in raw_dataset_name_or_path or raw_dataset_name_or_path == "justatom":
            resolved_dataset_name_or_path = raw_dataset_name_or_path
        else:
            resolved_dataset_name_or_path = Path(raw_dataset_name_or_path)
    save_results_to_dir = Path(os.getcwd()) / "evals" if save_results_to_dir is None else Path(save_results_to_dir)

    docs_iter: Iterable[dict] = []
    if resolved_dataset_name_or_path is not None:
        docs_adapter = DatasetRecordAdapter.from_source(
            dataset_name_or_path=str(resolved_dataset_name_or_path),
            content_col=content_field,
            queries_col=labels_field,
            split=split,
            limit=limit,
            chunk_id_col=chunk_id_col,
            keywords_col=keywords_or_phrases_field,
            keywords_nested_col=keywords_nested_col,
            explanation_nested_col=explanation_nested_col,
            drop_columns=drop_columns,
            filter_fields=(filters or {}).get("fields", []),
            preserve_all_fields=False,
        )
        docs_iter = tqdm(docs_adapter.iterator())
        logger.info("Dataset is prepared in lazy/iterative mode for indexing.")
    else:
        logger.info("No dataset provided. Using existing index only.")

    try:
        ir_runner = await RunningService.do_index_and_prepare_for_search(
            collection_name=collection_name,
            documents=docs_iter,
            batch_size=index_batch_size,
            flush_collection=flush_collection,
            index_and_eval_by=search_pipeline,
            query_prefix=query_prefix,
            content_prefix=content_prefix,
            model_name_or_path=model_name_or_path,
            weaviate_host=weaviate_host,
            weaviate_port=weaviate_port,
            **props,
        )
        n_total_docs = await ir_runner.store.count_documents()
        logger.info(f"INDEX stage is ready. Total docs in index = {n_total_docs}")

        if labels_field is None:
            if n_total_docs == 0:
                logger.warning("labels-col is not provided and index is empty. Nothing to retrieve; stopping.")
                return
            logger.info("labels-col is not provided. Retrieval-only mode is selected; evaluation is skipped.")
            return

        if resolved_dataset_name_or_path is None:
            logger.warning("labels-col is provided but dataset-name-or-path is missing. Evaluation is skipped.")
            return

        labels_adapter = DatasetRecordAdapter.from_source(
            dataset_name_or_path=str(resolved_dataset_name_or_path),
            content_col=content_field,
            queries_col=labels_field,
            split=split,
            limit=limit,
            chunk_id_col=chunk_id_col,
            keywords_col=keywords_or_phrases_field,
            keywords_nested_col=keywords_nested_col,
            explanation_nested_col=explanation_nested_col,
            drop_columns=drop_columns,
            filter_fields=(filters or {}).get("fields", []),
            preserve_all_fields=False,
        )
        queries = DatasetRecordAdapter.extract_labels(labels_adapter.iterator())
        if len(queries) == 0:
            logger.warning("labels-col is provided, but no labels were found in dataset. Evaluation is skipped.")
            return

        el: IEvaluatorRunner = EvaluatorRunner(ir=ir_runner)
        eval_metrics = await el.evaluate_topk(
            queries=queries,
            top_k=top_k,
            metrics=metrics,
            metrics_top_k=metrics_top_k,
            eval_top_k=eval_top_k,
            batch_size=search_batch_size,
        )
        logger.info("EVALUATION stage is completed.")

        comp_eval_metrics = {k: list(v.compute()) for k, v in eval_metrics.items()}
        comp_eval_metrics = [
            {
                "name": k,
                "mean": v[0],
                "std": v[1],
                "dataset": str(resolved_dataset_name_or_path),
            }
            for k, v in comp_eval_metrics.items()
        ]
        pl_metrics = pl.from_dicts(comp_eval_metrics)

        snap_eval_metrics = [] if eval_metrics is None else eval_metrics
        model_name = "keywords" if model_name_or_path is None else Path(model_name_or_path).stem
        snap_name = stl.snapshot({"Evaluation": " ".join(snap_eval_metrics), "Model": model_name}, sep="|")
        snap_props = stl.snapshot(props, sep="|")
        snap_props = "|" if snap_props == "" else f"|{snap_props}|"

        save_results_to_dir.mkdir(exist_ok=True, parents=True)
        save_final_path = (save_results_to_dir / f"{search_pipeline}{snap_props}{snap_name}.csv").resolve()
        pl_metrics.write_csv(str(save_final_path))
        logger.info(
            "Evaluation metrics were saved to [{}]",
            save_final_path,
        )
    finally:
        try:
            if ir_runner is not None:
                maybe_close = getattr(ir_runner.store, "close", None)
                if callable(maybe_close):
                    maybe_result = maybe_close()
                    if inspect.isawaitable(maybe_result):
                        await maybe_result
        finally:
            await RunningService.close_embedding_clients()


async def run(
    *,
    config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    **overrides,
):
    resolved = resolve_eval_kwargs(
        config=config,
        config_path=config_path,
        overrides=overrides or None,
    )
    return await main(**resolved)


if __name__ == "__main__":
    asio.run(main(**_parse_args()))
