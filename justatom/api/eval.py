import asyncio as asio
import os
from collections.abc import Generator, Iterable
from pathlib import Path

import dotenv
import numpy as np
import polars as pl
import torch
from loguru import logger
from torch.utils.data import ConcatDataset
from tqdm.auto import tqdm

from justatom.etc.errors import DocumentStoreError

# Model IO and Prediction Head Flow
from justatom.modeling.mask import ILanguageModel
from justatom.processing import RuntimeProcessor, ITokenizer
from justatom.running.evaluator import EvaluatorRunner
from justatom.running.indexer import API as IndexerAPI
from justatom.running.encoders import EncoderRunner, BiEncoderRunner
from justatom.running.mask import IEvaluatorRunner, IRetrieverRunner
from justatom.running.retriever import API as RetrieverApi
from justatom.storing.dataset import API as DatasetApi
from justatom.storing.weaviate import Finder as WeaviateApi
from justatom.storing.weaviate import WeaviateDocStore
from justatom.tooling import stl

dotenv.load_dotenv()


logger.info(
    f"Enable MPS fallback = {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', -1)}"
)


def to_numpy(container):
    try:
        return container.cpu().numpy()
    except AttributeError:
        return container


def maybe_cuda_or_mps(devices: list[str] = None):  # type: ignore
    devices = {"cuda", "mps", "cpu"} if devices is None else set(devices)  # type: ignore
    if torch.cuda.is_available() and "cuda" in devices:
        return "cuda:0"
    elif torch.mps.is_available() and "mps" in devices:
        return "mps"
    else:
        return "cpu"


def random_split(ds: ConcatDataset, lengths: list[int]):
    """
    Roughly split a Concatdataset into non-overlapping new datasets of given lengths.
    Samples inside Concatdataset should already be shuffled.

    :param ds: Dataset to be split.
    :param lengths: Lengths of splits to be produced.
    """
    if sum(lengths) != len(ds):
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    try:
        idx_dataset = np.where(np.array(ds.cumulative_sizes) > lengths[0])[0][0]
    except IndexError:
        raise Exception(  # noqa: B904
            "All dataset chunks are being assigned to train set leaving no samples for dev set. "
            "Either consider increasing dev_split or setting it to 0.0\n"
            f"Cumulative chunk sizes: {ds.cumulative_sizes}\n"
            f"train/dev split: {lengths}"
        )

    assert idx_dataset >= 1, (
        "Dev_split ratio is too large, there is no data in train set. Please lower split ="
        f" {str(lengths)}"
    )

    train = ConcatDataset(ds.datasets[:idx_dataset])  # type: Dataset
    test = ConcatDataset(ds.datasets[idx_dataset:])  # type: Dataset
    return train, test


def check_and_raise(
    fpath: str | Path,
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


def source_from_dataset(dataset_name_or_path):
    import polars as pl

    maybe_df_or_iter = DatasetApi.named(dataset_name_or_path).iterator()
    if isinstance(maybe_df_or_iter, pl.DataFrame):
        pl_data = maybe_df_or_iter
    else:
        dataset = list(maybe_df_or_iter)
        pl_data = pl.from_dicts(dataset)
    return pl_data


async def check_store_and_message(store: WeaviateDocStore, delete_if_not_empty: bool):
    n_docs_count: int = await store.count_documents()
    collection_name = store.collection_name
    if delete_if_not_empty:
        status_message: bool = await store.delete_all_documents()
        if not status_message:
            raise DocumentStoreError(
                f"Documents per collection {collection_name} are not deleted. See logs for more details"
            )
    else:
        if n_docs_count > 0:
            logger.warning(
                f"You're not deleting any documents. \
                You may lose up to {n_docs_count} documents per collection {collection_name}"
            )
    return store, n_docs_count


def wrapper_for_docs(
    pl_data: pl.DataFrame,
    search_or_id_field: str,
    content_field: str,
    keywords_or_phrases_field: str = None,
    batch_size: int = 128,
    filters: dict | None = None,
) -> Generator[dict]:
    if keywords_or_phrases_field is None:
        pl_data = (
            pl_data.group_by(content_field)
            .agg(pl.col(search_or_id_field))
            .explode(pl.col(search_or_id_field))
        )
    else:
        pl_data = (
            pl_data.group_by(content_field)
            .agg(pl.col(search_or_id_field), pl.col(keywords_or_phrases_field))
            .explode(pl.col(search_or_id_field), pl.col(keywords_or_phrases_field))
        )
    js_data = pl_data.to_dicts()
    for js_chunk in tqdm(js_data):
        js_queries = [q for q in js_chunk["queries"] if q is not None]
        if keywords_or_phrases_field is None:
            yield dict(content=js_chunk[content_field], meta=dict(labels=js_queries))
        else:
            js_keywords_or_phrases = [
                kwp
                for kwp in js_chunk[keywords_or_phrases_field]
                if kwp["keyword_or_phrase"] is not None
                and kwp["explanation"] is not None
            ]
            yield dict(
                content=js_chunk[content_field],
                meta=dict(labels=js_queries),
                keywords_or_phrases=js_keywords_or_phrases,
            )


def igni_runners(
    store,
    search_pipeline: str,
    model_name_or_path,
    query_prefix: str = "",
    content_prefix: str = "",
    device: str = "cpu",
    **props,
):
    ix_runner, ir_runner = None, None
    if search_pipeline == "keywords":
        ix_runner = IndexerAPI.named(search_pipeline, store=store)
        ir_runner = RetrieverApi.named(search_pipeline, store=store)
    else:
        if model_name_or_path is None:
            msg = f"You have specified `runner_name`=[{search_pipeline}] but `model_name_or_path` is None."
            logger.error(msg)
            raise ValueError(msg)
        lm_model = ILanguageModel.load(model_name_or_path)
        ix_processor = RuntimeProcessor(
            ITokenizer.from_pretrained(model_name_or_path), prefix=content_prefix
        )
        ir_processor = RuntimeProcessor(
            ITokenizer.from_pretrained(model_name_or_path), prefix=query_prefix
        )
        runner = EncoderRunner(model=lm_model, prediction_heads=[], device=device)
        ix_runner = IndexerAPI.named(
            search_pipeline,
            store=store,
            runner=runner,
            processor=ix_processor,
            device=device,
        )
        ir_runner = RetrieverApi.named(
            search_pipeline,
            store=store,
            runner=runner,
            processor=ir_processor,
            device=device,
            **props,
        )

    return ix_runner, ir_runner


async def do_index_and_prepare_for_search(
    collection_name: str,
    pl_data: pl.DataFrame,
    model_name_or_path: str | None = None,
    index_and_eval_by: str = "embedding",
    search_or_id_field: str = "queries",
    content_field: str = "content",
    keywords_or_phrases_field: str = None,
    query_prefix: str = None,
    content_prefix: str = None,
    filters: dict | None = None,
    batch_size: int = 4,
    flush_collection: bool = False,
    devices: list[str] = None,
    weaviate_host: str = "localhost",
    weaviate_port: int = 2211,
    **props,
) -> tuple[IRetrieverRunner, list[str]]:
    # Here we don't need any model to load. Only `DocumentStore`
    store: WeaviateDocStore = await WeaviateApi.find(
        collection_name, WEAVIATE_HOST=weaviate_host, WEAVIATE_PORT=weaviate_port
    )
    store, n_total_docs = await check_store_and_message(
        store, delete_if_not_empty=flush_collection
    )

    device = maybe_cuda_or_mps(devices=devices)

    ix_runner, ir_runner = igni_runners(
        store=store,
        search_pipeline=index_and_eval_by,
        model_name_or_path=model_name_or_path,
        device=device,
        query_prefix=query_prefix,
        content_prefix=content_prefix,
        **props,
    )
    assert (
        search_or_id_field in pl_data.columns
    ), f"Search field [{search_or_id_field}] is not present within dataset."
    assert (
        content_field in pl_data.columns
    ), f"Content field [{content_field}] is not present within dataset."

    if not flush_collection and n_total_docs > 0:
        return ir_runner
    js_docs = list(
        wrapper_for_docs(
            pl_data,
            search_or_id_field=search_or_id_field,
            content_field=content_field,
            keywords_or_phrases_field=keywords_or_phrases_field,
        )
    )
    logger.info("Indexing in progress\n")
    n_total_docs = await ix_runner.index(
        documents=js_docs, batch_size=batch_size, device=device
    )
    logger.info(f"Total docs in index {n_total_docs}\n")
    return ir_runner


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
    metrics_top_k=["HitRate"],  # noqa: B006
    eval_top_k=None,
    search_field: str = "queries",
    content_field: str = "content",
    keywords_or_phrases_field: str = None,
    weaviate_host: str = None,
    weaviate_port: int = None,
    **props,
):
    collection_name = "Document" if collection_name is None else collection_name
    dataset_name_or_path = Path(
        dataset_name_or_path
    )  # pyright: ignore[reportAssignmentType]
    save_results_to_dir = (
        Path(os.getcwd()) / "evals"
        if save_results_to_dir is None
        else Path(save_results_to_dir)
    )  # pyright: ignore[reportAssignmentType]
    pl_data = source_from_dataset(str(dataset_name_or_path))
    if filters is not None:
        fields = filters.get("fields", [])
        for field in fields:
            pl_data = pl_data.filter(pl.col(field).is_not_null())
    logger.info(f"Total {pl_data.shape[0]} dataset for test.")

    ir_runner = await do_index_and_prepare_for_search(
        collection_name=collection_name,
        pl_data=pl_data,
        batch_size=index_batch_size,
        flush_collection=flush_collection,
        index_and_eval_by=search_pipeline,
        query_prefix=query_prefix,
        content_prefix=content_prefix,
        model_name_or_path=model_name_or_path,
        filters=filters,
        search_or_id_field=search_field,
        content_field=content_field,
        keywords_or_phrases_field=keywords_or_phrases_field,
        weaviate_host=weaviate_host,
        weaviate_port=weaviate_port,
        **props,
    )
    n_total_docs = await ir_runner.store.count_documents()
    logger.info(f"INDEXING stage is completed. Using evaluation per {n_total_docs}")

    el: IEvaluatorRunner = EvaluatorRunner(ir=ir_runner)

    queries = pl_data.select(search_field).explode(search_field).to_series().to_list()

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
    snap_name = stl.snapshot(
        {
            "Evaluation": " ".join(snap_eval_metrics),
            "Model": Path(model_name_or_path).stem,
        },
        sep="|",
    )
    snap_props = stl.snapshot(props, sep="|")
    snap_props = "|" if snap_props == "" else "|" + snap_props + "|"
    save_results_to_dir.mkdir(exist_ok=True, parents=False)
    save_final_path = (
        save_results_to_dir / f"{search_pipeline}{snap_props}{snap_name}.csv"
    )
    pl_metrics.write_csv(str(save_final_path))


if __name__ == "__main__":
    # --- EVAL pipeline ---
    # search_pipeline: str = embedding | hybrid | keywords | atomicai
    gamma1, gamma2 = 0.6225, 0.8176
    for alpha in [0.78, 0.8, 0.82]:
        asio.run(
            *[
                main(
                    collection_name="EvalVanillaLarge",
                    save_results_to_dir=Path(os.getcwd())
                    / "evals"
                    / "EvalVanilla"
                    / "large",  # pyright: ignore[reportArgumentType]
                    flush_collection=True,
                    search_field="queries",
                    content_field="content",
                    keywords_or_phrases_field="keywords_or_phrases",
                    filters={"fields": ["queries", "content"]},
                    search_pipeline="hybrid",
                    model_name_or_path="intfloat/multilingual-e5-large",  # intfloat/multilingual-e5-small | intfloat/multilingual-e5-base | intfloat/multilingual-e5-large #noqa
                    dataset_name_or_path=str(
                        Path(os.getcwd()) / ".data" / "polaroids.ai.data.json"
                    ),
                    query_prefix="query:",
                    content_prefix="passage:",
                    metrics_top_k=["HitRate"],
                    index_batch_size=32,
                    search_batch_size=64,
                    eval_top_k=[2, 5, 10, 15, 20],
                    weaviate_host="localhost",
                    weaviate_port=2211,
                    alpha=alpha,
                    include_keywords=False,
                    include_explanation=False,
                    include_content=True,
                )
            ]
        )
