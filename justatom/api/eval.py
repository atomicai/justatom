import os
from pathlib import Path
from typing import Dict, Iterable, List, Union, Optional, Tuple

import dotenv
import numpy as np
import polars as pl


from tqdm.auto import tqdm
from justatom.running.indexer import API as IndexerAPI
import torch
import asyncio as asio
from loguru import logger
from torch.utils.data import ConcatDataset

from more_itertools import chunked

from justatom.configuring import Config

# Model IO and Prediction Head Flow
from justatom.modeling.head import ANNHead
from justatom.modeling.mask import ILanguageModel
from justatom.processing import IProcessor, ITokenizer, igniset
from justatom.processing.loader import NamedDataLoader
from justatom.processing.prime import TRILMProcessor
from justatom.running.m1 import M1LMRunner
from justatom.etc.schema import Document
from justatom.tooling import stl
from justatom.storing.dataset import API as DatasetApi
from justatom.storing.weaviate import Finder as WeaviateApi, WeaviateDocStore
from justatom.running.retriever import API as RetrieverApi
from justatom.running.evaluator import EvaluatorRunner
from justatom.running.mask import IRetrieverRunner, IEvaluatorRunner
from justatom.modeling.mask import ILanguageModel
from justatom.tooling.stl import merge_in_order

dotenv.load_dotenv()


logger.info(f"Enable MPS fallback = {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', -1)}")


def to_numpy(container):
    try:
        return container.cpu().numpy()
    except AttributeError:
        return container


def maybe_cuda_or_mps(devices: List[str] = None):
    devices = {"cuda", "mps", "cpu"} if devices is None else set(devices)
    if torch.cuda.is_available() and "cuda" in devices:
        return "cuda:0"
    elif torch.has_mps and "mps" in devices:
        return "mps"
    else:
        return "cpu"


def random_split(ds: ConcatDataset, lengths: List[int]):
    """
    Roughly split a Concatdataset into non-overlapping new datasets of given lengths.
    Samples inside Concatdataset should already be shuffled.

    :param ds: Dataset to be split.
    :param lengths: Lengths of splits to be produced.
    """
    if sum(lengths) != len(ds):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    try:

        idx_dataset = np.where(np.array(ds.cumulative_sizes) > lengths[0])[0][0]
    except IndexError:
        raise Exception(
            "All dataset chunks are being assigned to train set leaving no samples for dev set. "
            "Either consider increasing dev_split or setting it to 0.0\n"
            f"Cumulative chunk sizes: {ds.cumulative_sizes}\n"
            f"train/dev split: {lengths}"
        )

    assert idx_dataset >= 1, (
        "Dev_split ratio is too large, there is no data in train set. Please lower split =" f" {str(lengths)}"
    )

    train = ConcatDataset(ds.datasets[:idx_dataset])  # type: Dataset
    test = ConcatDataset(ds.datasets[idx_dataset:])  # type: Dataset
    return train, test


def check_and_raise(
    fpath: Union[str, Path],
    name: str = None,
    allowed_suffixes: Iterable[str] = (".csv"),
) -> bool:
    if fpath is None:
        return False
    suffixes = set(iter(allowed_suffixes))
    fp = Path(fpath)
    assert fp.exists(), f"Provided {name} dataset path {str(fp)} does not exsits"
    assert (
        fp.suffix in suffixes
    ), f"{name} dataset path extension {fp.suffix} is not yet supported. Please provide one of {' | '.join(allowed_suffixes)}"
    return fp


def check_store_and_message(store: WeaviateDocStore, delete_if_not_empty: bool):
    if store.count_documents() <= 0:
        return store
    if delete_if_not_empty:
        store.delete_all_documents()
    else:
        count_docs = store.count_documents()
        collection_name = store._collection_settings.get("class")
        logger.warning(
            f"You're not deleting any documents. \
            Performing evaluation on old {count_docs} documents per [{collection_name}] collection name."
        )
    return store


def wrapper_docs_with_queries(
    pl_data: pl.DataFrame, search_field: str, content_field: str, batch_size: int = 128
) -> List[Dict]:
    pl_data = pl_data.group_by(content_field).agg(pl.col(search_field))
    queries, documents = (
        pl_data.select(search_field).to_series().to_list(),
        pl_data.select(content_field).to_series().to_list(),
    )
    docs = []
    for i, chunk in tqdm(enumerate(chunked(zip(queries, documents), n=batch_size))):
        _the_docs = [{"content": c[1], "meta": {"labels": c[0]}} for c in chunk]
        docs.extend(_the_docs)
    return docs


def igni_runners(store, index_by: str, model_name_or_path, device: str = "cpu"):
    ix_runner, ir_runner = None, None
    if index_by == "keywords":
        ix_runner = IndexerAPI.named(index_by, store=store)
        ir_runner = RetrieverApi.named(index_by, store=store)
    else:
        if model_name_or_path is None:
            msg = f"You have specified `index_by`=[{index_by}] but `model_name_or_path` is None."
            logger.error(msg)
            raise ValueError(msg)
        lm_model = ILanguageModel.load(model_name_or_path)
        processor = IProcessor.load(model_name_or_path)
        runner = M1LMRunner(model=lm_model, prediction_heads=[], device=device)
        ix_runner = IndexerAPI.named(index_by, store=store, runner=runner, processor=processor)
        ir_runner = RetrieverApi.named(index_by, store=store, runner=runner, processor=processor, device=device)

    return ix_runner, ir_runner


async def maybe_index_and_ir(
    collection_name: str,
    pl_data: pl.DataFrame,
    model_name_or_path: Optional[str] = None,
    index_by: str = "embedding",
    search_field: str = "query",
    content_field: str = "content",
    filters: Optional[Dict] = None,
    batch_size: int = 4,
    top_k: int = 5,
    delete_if_not_empty: bool = False,
    devices: List[str] = None,
) -> Tuple[IRetrieverRunner, List[str]]:
    delete_if_not_empty = delete_if_not_empty or Config.eval.delete_if_not_empty
    # Here we don't need any model to load. Only `DocumentStore`
    store: WeaviateDocStore = WeaviateApi.find(collection_name)
    store = check_store_and_message(store, delete_if_not_empty=delete_if_not_empty)

    device = maybe_cuda_or_mps(devices=devices)

    ix_runner, ir_runner = igni_runners(
        store=store, index_by=index_by, model_name_or_path=model_name_or_path, device=device
    )
    assert search_field in pl_data.columns, f"Search field [{search_field}] is not present within dataset."
    assert content_field in pl_data.columns, f"Content field [{content_field}] is not present within dataset."

    docs = wrapper_docs_with_queries(pl_data, search_field=search_field, content_field=content_field)
    print()
    await ix_runner.index(documents=docs, batch_size=batch_size, device=device)
    print()
    return ir_runner


async def main(
    model_name_or_path: Optional[str] = None,
    index_by: str = "embedding",
    collection_name: str = "EvaluationForKeywords",
    dataset_name_or_path: str = "justatom",
    top_k: int = 20,
    filters: Optional[Dict] = None,
    metrics=None,
    metrics_top_k=["HitRate"],
    eval_top_k=None,
    search_field: str = "query",
    content_field: str = "content",
    **props,
):

    dataset = list(DatasetApi.named(dataset_name_or_path).iterator())
    pl_data = pl.from_dicts(dataset)
    if filters is not None:
        fields = filters.get("fields", [])
        for field in fields:
            pl_data = pl_data.filter(pl.col(field).is_not_null())
    logger.info(f"Total {pl_data.shape[0]} dataset for test.")

    ir = await maybe_index_and_ir(
        collection_name=collection_name,
        pl_data=pl_data,
        index_by=index_by,
        model_name_or_path=model_name_or_path,
        filters=filters,
        search_field=search_field,
        content_field=content_field,
    )

    logger.info(f"Indexing stage is completed. Using evaluation per {ir.store.count_documents()}")

    el: IEvaluatorRunner = EvaluatorRunner(ir=ir)

    queries = pl_data.select(search_field).to_series().to_list()

    eval_metrics = el.evaluate_topk(
        queries=queries, top_k=top_k, metrics=metrics, metrics_top_k=metrics_top_k, eval_top_k=eval_top_k
    )
    print()
    comp_eval_metrics = {k: list(v.compute()) for k, v in eval_metrics.items()}
    comp_eval_metrics = [
        {"name": k, "mean": v[0], "std": v[1], "dataset": dataset_name_or_path} for k, v in comp_eval_metrics.items()
    ]
    pl_metrics = pl.from_dicts(comp_eval_metrics)
    snap_eval_metrics = [] if eval_metrics is None else eval_metrics
    snap_name = stl.snapshot({"evaluation": " ".join(snap_eval_metrics), "model": Path(model_name_or_path).stem})
    pl_metrics.write_csv(f"{snap_name}.csv")


if __name__ == "__main__":
    filters = {"fields": ["query"]}
    asio.run(
        *[
            main(
                eval_by="embedding",
                filters={"fields": ["query", "content"]},
                model_name_or_path=str(
                    Path(os.getcwd()) / "weights" / "bs=128_esm=DistanceAcc_dataset=polaroids.ai_margin=0.4"
                ),
                metrics_top_k=["HitRate"],
                eval_top_k=[2, 5, 10, 15, 20],
            )
        ]
    )