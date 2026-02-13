import argparse
import asyncio as asio
import itertools
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

from justatom.running.evaluator import EvaluatorRunner
from justatom.running.mask import IEvaluatorRunner
from justatom.running.service import RunningService
from justatom.tooling.dataset import DatasetRecordAdapter
from justatom.tooling import stl

dotenv.load_dotenv()

logger.info(
    f"Enable MPS fallback = {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', -1)}"
)


def _parse_args(argv: list[str] | None = None) -> dict:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="justatom|eval",
        description="Index and evaluate IR quality on a given dataset",
    )

    parser.add_argument("--dataset-name-or-path")
    parser.add_argument("--collection-name", default="Document")
    parser.add_argument("--save-results-to-dir")

    parser.add_argument(
        "--search-pipeline",
        default="embedding",
        choices=["embedding", "hybrid", "keywords", "atomicai"],
    )
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--query-prefix")
    parser.add_argument("--content-prefix")
    parser.add_argument("--flush-collection", action="store_true")

    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--index-batch-size", type=int, default=4)
    parser.add_argument("--search-batch-size", type=int, default=32)
    parser.add_argument("--labels-col")
    parser.add_argument("--content-col", default="content")
    parser.add_argument("--id-col")
    parser.add_argument("--keywords-col")
    parser.add_argument("--keywords-nested-col")
    parser.add_argument("--explanation-nested-col")

    parser.add_argument("--metrics", nargs="+")
    parser.add_argument("--metrics-top-k", nargs="+", default=["HitRate"])
    parser.add_argument("--eval-top-k", nargs="+", type=int)
    parser.add_argument("--filter-fields", nargs="+")

    parser.add_argument("--weaviate-host")
    parser.add_argument("--weaviate-port", type=int)

    parser.add_argument("--alpha", type=float)

    args = parser.parse_args(argv)

    filters = {"fields": args.filter_fields} if args.filter_fields else None
    props = {"alpha": args.alpha}
    props = {k: v for k, v in props.items() if v is not None}

    return {
        "model_name_or_path": args.model_name_or_path,
        "search_pipeline": args.search_pipeline,
        "query_prefix": args.query_prefix,
        "content_prefix": args.content_prefix,
        "collection_name": args.collection_name,
        "flush_collection": args.flush_collection,
        "dataset_name_or_path": args.dataset_name_or_path,
        "save_results_to_dir": args.save_results_to_dir,
        "top_k": args.top_k,
        "index_batch_size": args.index_batch_size,
        "search_batch_size": args.search_batch_size,
        "filters": filters,
        "metrics": args.metrics,
        "metrics_top_k": args.metrics_top_k,
        "eval_top_k": args.eval_top_k,
        "labels_field": args.labels_col,
        "content_field": args.content_col,
        "chunk_id_col": args.id_col,
        "keywords_or_phrases_field": args.keywords_col,
        "keywords_nested_col": args.keywords_nested_col,
        "explanation_nested_col": args.explanation_nested_col,
        "weaviate_host": args.weaviate_host,
        "weaviate_port": args.weaviate_port,
        **props,
    }


def to_numpy(container):
    try:
        return container.cpu().numpy()
    except AttributeError:
        return container


def random_split(ds: ConcatDataset, lengths: list[int]):
    if sum(lengths) != len(ds):
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    try:
        idx_dataset = np.where(np.array(ds.cumulative_sizes) > lengths[0])[0][0]
    except IndexError as ex:
        raise Exception(
            "All dataset chunks are being assigned to train set leaving no samples for dev set. "
            "Either consider increasing dev_split or setting it to 0.0\n"
            f"Cumulative chunk sizes: {ds.cumulative_sizes}\n"
            f"train/dev split: {lengths}"
        ) from ex

    assert (
        idx_dataset >= 1
    ), "Dev_split ratio is too large, there is no data in train set."
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
    chunk_id_col: str | None = None,
    keywords_or_phrases_field: str = None,
    keywords_nested_col: str | None = None,
    explanation_nested_col: str | None = None,
    weaviate_host: str = None,
    weaviate_port: int = None,
    **props,
):
    collection_name = collection_name or "Document"
    dataset_name_or_path = (
        None if dataset_name_or_path is None else Path(dataset_name_or_path)
    )
    save_results_to_dir = (
        Path(os.getcwd()) / "evals"
        if save_results_to_dir is None
        else Path(save_results_to_dir)
    )

    docs_iter: Iterable[dict] = []
    if dataset_name_or_path is not None:
        docs_adapter = DatasetRecordAdapter.from_source(
            dataset_name_or_path=str(dataset_name_or_path),
            content_col=content_field,
            queries_col=labels_field,
            chunk_id_col=chunk_id_col,
            keywords_col=keywords_or_phrases_field,
            keywords_nested_col=keywords_nested_col,
            explanation_nested_col=explanation_nested_col,
            filter_fields=(filters or {}).get("fields", []),
        )
        docs_iter = tqdm(docs_adapter.iter_documents())
        logger.info("Dataset is prepared in lazy/iterative mode for indexing.")
    else:
        logger.info("No dataset provided. Using existing index only.")

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
            logger.warning(
                "labels-col is not provided and index is empty. Nothing to retrieve; stopping."
            )
            return
        logger.info(
            "labels-col is not provided. Retrieval-only mode is selected; evaluation is skipped."
        )
        return

    if dataset_name_or_path is None:
        logger.warning(
            "labels-col is provided but dataset-name-or-path is missing. Evaluation is skipped."
        )
        return

    labels_adapter = DatasetRecordAdapter.from_source(
        dataset_name_or_path=str(dataset_name_or_path),
        content_col=content_field,
        queries_col=labels_field,
        chunk_id_col=chunk_id_col,
        keywords_col=keywords_or_phrases_field,
        keywords_nested_col=keywords_nested_col,
        explanation_nested_col=explanation_nested_col,
        filter_fields=(filters or {}).get("fields", []),
    )
    labels_iter = labels_adapter.iter_labels()
    first_label = next(labels_iter, None)
    if first_label is None:
        logger.warning(
            "labels-col is provided, but no labels were found in dataset. Evaluation is skipped."
        )
        return
    queries = itertools.chain([first_label], labels_iter)

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
        {"name": k, "mean": v[0], "std": v[1], "dataset": str(dataset_name_or_path)}
        for k, v in comp_eval_metrics.items()
    ]
    pl_metrics = pl.from_dicts(comp_eval_metrics)

    snap_eval_metrics = [] if eval_metrics is None else eval_metrics
    model_name = (
        "keywords" if model_name_or_path is None else Path(model_name_or_path).stem
    )
    snap_name = stl.snapshot(
        {"Evaluation": " ".join(snap_eval_metrics), "Model": model_name}, sep="|"
    )
    snap_props = stl.snapshot(props, sep="|")
    snap_props = "|" if snap_props == "" else f"|{snap_props}|"

    save_results_to_dir.mkdir(exist_ok=True, parents=False)
    save_final_path = (
        save_results_to_dir / f"{search_pipeline}{snap_props}{snap_name}.csv"
    )
    pl_metrics.write_csv(str(save_final_path))


if __name__ == "__main__":
    asio.run(main(**_parse_args()))
