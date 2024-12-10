import asyncio as asio
import os
from pathlib import Path

import polars as pl
import simplejson as json
from loguru import logger
from more_itertools import chunked
from tqdm import tqdm

from justatom.etc.io import io_snapshot
from justatom.running.igni import IGNIRunner
from justatom.storing.dataset import API as DatasetApi
from justatom.tooling.coro import _limit_concurrency
from justatom.tooling.reqs import openai_chat


def wrapper_for_props(d: dict, must_include_keys: list[str] = None) -> dict:
    """
    :param d: Source doc
    :param must_include_keys: List of keys to include
    :return: New doc with only specified `must_include_keys`
    """
    must_include_keys = d.keys() if must_include_keys is None else must_include_keys
    return {key: d[key] for key in must_include_keys if key in d}


async def pipeline(
    js_docs: list[dict],
    pr_runner,
    openai_model_name: str,
    batch_size: int = 16,
    coros_size: int = 2,
    save_snapshot_every: int = 5,
    snapshot_prefix: str = None,
    snapshot_where: str = None,
    timeout: int = 512,
    must_include_keys: list[str] | None = None,
    validate_json_response: bool = False,
):
    """
    We process `js_docs` by chunks where each chunk is of size `batch_size`.
    Each chunk is processed asynchronously via parallel `coros_size` coroutines.

    :param js_docs: documents to process
    :param pr_runner: One of the instance `IPromptRunner` to create specific prompt
    """
    pipes = []

    for i, batch in tqdm(enumerate(chunked(js_docs, n=batch_size))):
        _batch = batch
        cur_result = await asio.gather(
            *_limit_concurrency(
                [
                    openai_chat(
                        pr_runner.prompt(**d),
                        timeout=timeout,
                        model=openai_model_name,
                        props=wrapper_for_props(d, must_include_keys=must_include_keys),
                    )
                    for d in _batch
                ],
                concurrency=coros_size,
            )
        )
        if not validate_json_response:
            pipes.extend(cur_result)
        else:
            # NOTE: Order of execution is preserved.
            js_answer_docs = [
                pr_runner.finalize(
                    raw_response=js_res["response"], **wrapper_for_props(js_doc, must_include_keys=must_include_keys)
                )
                for js_doc, js_res in zip(batch, cur_result, strict=True)
            ]
            pipes.extend(js_answer_docs)

        if (i + 1) % save_snapshot_every == 0:
            io_snapshot(pipes, where=snapshot_where, snapshot_number=str(i + 1), snapshot_prefix=snapshot_prefix)
    return pipes


def source_from_dataset(dataset_name_or_path):
    maybe_df_or_iter = DatasetApi.named(dataset_name_or_path).iterator()
    if isinstance(maybe_df_or_iter, pl.DataFrame):
        pl_data = maybe_df_or_iter
    else:
        dataset = list(maybe_df_or_iter)
        pl_data = pl.from_dicts(dataset)
    return pl_data


async def main(
    dataset_name_or_path: str,
    dataset_save_path: str | None = None,
    snapshot_every_iters: int = 5,
    snapshot_prefix: str | None = None,
    snapshot_where: str | None = None,
    batch_size: int = 16,
    coros_size: int = 2,
    timeout: int = 256,
    content_field: str = "content",
    system_prompt: str | None = None,
    prompt_component: str | None = None,
    prompt_as_string: str | None = None,
    must_include_keys: list[str] | None = None,
    openai_model_name: str = None,
    **props,
):
    """
    :param dataset_name_or_path -
    :return: status message if all processing went good, or error if some trouble arise (e.g. ConnectionError)
    """

    PR_RUNNER_MAPPING = dict(KEYWORD=IGNIRunner.KEYWORDER, TRANSLATE=IGNIRunner.TRANLSATOR, REPHRASE=IGNIRunner.REPHRASER)
    if prompt_component is not None and prompt_as_string is not None:
        msg = f"""
        You've specified both `prompt_component`=[{prompt_component}] and `prompt_as_string`=[{prompt_as_string}]
        which is not compatable.
        """
        logger.error(msg)
        raise ValueError(msg)
    system_prompt = "You're helpful assistant." if system_prompt is None else system_prompt
    pl_data = source_from_dataset(dataset_name_or_path)
    pl_data = pl_data.select(must_include_keys + [content_field]).rename({content_field: "content"})
    logger.info(f"Total {pl_data.shape[0]} dataset for test.")
    js_docs = pl_data.to_dicts()
    if prompt_component is not None and prompt_component not in PR_RUNNER_MAPPING:
        msg = f"""
            You set `prompt_component`=[{prompt_component}] which is not yet available.
            Use one on {','.join(PR_RUNNER_MAPPING.keys())}
            """
        logger.error(msg)
        raise ValueError(msg)
    pr_runner = await PR_RUNNER_MAPPING[prompt_component](system_prompt=system_prompt, **props)
    js_pipe_answer = await pipeline(
        js_docs,
        openai_model_name=openai_model_name,
        pr_runner=pr_runner,
        batch_size=batch_size,
        coros_size=coros_size,
        timeout=timeout,
        save_snapshot_every=snapshot_every_iters,
        snapshot_prefix=snapshot_prefix,
        snapshot_where=snapshot_where,
        must_include_keys=must_include_keys,
    )
    dataset_save_path = (
        Path(os.getcwd()) / "outputs" / str(Path(__name__)) + ".json" if dataset_save_path is None else Path(dataset_save_path)
    )
    dataset_save_path.mkdir(exist_ok=True, parents=False)
    with open(dataset_save_path, "w+") as fp:
        json.dump(js_pipe_answer, fp, ensure_ascii=False)


if __name__ == "__main__":
    asio.run(
        *[
            main(
                dataset_name_or_path=str(Path(os.getcwd()) / ".data" / "polaroids.ai.data.json"),
                must_include_keys=["title", "chunk_id"],
                prompt_component="KEYWORD",
                snapshot_every_iters=1,
                batch_size=4,
                content_field="content",
                system_prompt="""
                Ты эксперт-лингвит. Твоя задача отвечать четко и понятно выполнять задачи по заданному тебе параграфу.
                Параграф может быть диалогом, текстом или повествованием из книги, игры или фильма.
                """,
                snapshot_prefix="RU_EN_KWARG_",
                snapshot_where="outputs",
                openai_model_name="gpt-4-turbo",
                source_language="английском",
            )
        ]
    )
