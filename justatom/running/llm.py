from __future__ import annotations

import asyncio as asio
import contextlib
import inspect
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from loguru import logger
from openai import AsyncOpenAI
from tqdm.auto import tqdm


@dataclass(slots=True)
class OpenAiTask:
    id: int | str
    text: str
    image_b64: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class OpenAIAsyncWrapper:
    """Async helper for OpenAI-compatible chat flows.

    Subclasses should implement `_call` to perform a single request.
    """

    def __init__(
        self,
        model: str = "",
        response_format: Optional[dict[str, Any]] = None,
        api_call_timeout: float = 220,
        max_concurrent: int = 5,
        snapshot_dir: str | Path | None = None,
        snapshot_prefix: str = "llm-results",
        snapshot_backend: str = "loguru",
        system_prompt_fpath: Optional[str] = None,
        system_prompt: Optional[str] | None = None,
        openai_token: Optional[str] = None,
        base_url: str | None = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 512,
        stream: bool = False,
        show_progress: bool = True,
        progress_desc: str | None = None,
    ) -> None:
        self.model = model
        self.response_format = response_format
        self.api_call_timeout = float(api_call_timeout)
        self.max_concurrent = max(1, int(max_concurrent))
        self._snapshot_offsets: dict[Path, int] = {}
        self.snapshot_dir = snapshot_dir
        self.snapshot_prefix = snapshot_prefix
        self.snapshot_backend = snapshot_backend
        self._snapshot_queue: asio.Queue[dict[str, Any] | None] | None = None
        self._snapshot_writer_task: asio.Task | None = None
        self._snapshot_sink_id: int | None = None

        try:
            logger.level("SNAPSHOT")
        except ValueError:
            logger.level("SNAPSHOT", no=5, color="<cyan>")

        # Model params
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)
        self.stream = bool(stream)
        self.show_progress = bool(show_progress)
        self.progress_desc = progress_desc

        api_key = openai_token or os.getenv("OPENAI_API_KEY") or os.getenv("JWT_TOKEN", None) or "sk-noauth"
        base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "").rstrip("/")

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or None,
            organization=organization,
            project=project,
        )

        if system_prompt is None:
            assert system_prompt_fpath is not None, logger.error(
                "OpenAIAsyncWrapper: either system_prompt or system_prompt_fpath must be set"
            )
            self.system_prompt = Path(system_prompt_fpath).read_text(encoding="utf-8").strip()
        else:
            self.system_prompt = system_prompt.strip()

    def run(self, tasks: Sequence[OpenAiTask]) -> list[dict[str, Any]]:
        if not tasks:
            return []
        return asio.run(self._arun(tasks))

    async def run_async(self, tasks: Sequence[OpenAiTask], progress_cb=None) -> Any:
        callback = self.logger_callback if progress_cb is None else progress_cb
        logger.info(f"run_async: starting {len(tasks)} tasks")
        results = await self._arun_with_progress(tasks, progress_cb=callback)
        logger.info(f"run_async: finished, got {len(results)} results")
        return results

    async def logger_callback(self, results: list, total: int):
        logger.info(f"OpenAIAsyncWrapper: completed {len(results)}/{total} tasks ({len(results)/total:.2%})")
        return True

    async def _arun(self, tasks: Sequence[OpenAiTask], progress_cb=None) -> list[dict[str, Any]]:
        sem = asio.Semaphore(self.max_concurrent)

        async def one(t: OpenAiTask) -> dict[str, Any]:
            async with sem:
                try:
                    return await asio.wait_for(self._call(t), timeout=self.api_call_timeout)
                except asio.TimeoutError:
                    return {
                        "id": t.id,
                        "metadata": t.metadata,
                        "model": self.model,
                        "error": "timeout",
                        "text": None,
                        "parsed_json": None,
                        "raw": None,
                    }

        return await asio.gather(*(one(t) for t in tasks))

    async def _arun_with_progress(self, tasks: Sequence[OpenAiTask], progress_cb=None) -> list[dict[str, Any]]:
        sem = asio.Semaphore(self.max_concurrent)

        async def one(t: OpenAiTask) -> dict[str, Any]:
            async with sem:
                try:
                    result = await asio.wait_for(self._call(t), timeout=self.api_call_timeout)
                except asio.TimeoutError:
                    result = {
                        "id": t.id,
                        "metadata": t.metadata,
                        "model": self.model,
                        "error": "timeout",
                        "text": None,
                        "parsed_json": None,
                        "raw": None,
                    }
                return result

        total = len(tasks)
        results = []
        coros = [one(t) for t in tasks]
        progress = (
            tqdm(total=total, desc=self.progress_desc or "OpenAI requests", leave=False)
            if self.show_progress and total > 0
            else None
        )

        try:
            for future in asio.as_completed(coros):
                resp = await future
                results.append(resp)
                if progress is not None:
                    progress.update(1)
                try:
                    await self._append_snapshot(resp)
                except Exception as exc:
                    logger.warning(f"snapshot write failed: {exc}")
                if progress_cb:
                    try:
                        maybe = progress_cb(results, total)
                        if inspect.isawaitable(maybe):
                            await maybe
                    except Exception as exc:
                        logger.error(f"progress_cb error: {exc}")
        finally:
            if progress is not None:
                progress.close()
        await self._finalize_snapshot_writer()
        return results

    async def _call(self, task: OpenAiTask) -> dict[str, Any]:
        raise NotImplementedError("Implement this in subclass: agent logic goes here")

    def _snapshot_path(
        self,
        snapshot_number: int | str | None = None,
        snapshot_suffix: str | None = None,
    ) -> Path:
        where = Path(os.getcwd()) if self.snapshot_dir is None else Path(self.snapshot_dir)
        snapshot_prefix = self.snapshot_prefix or ""
        snapshot_suffix = "" if snapshot_suffix is None else snapshot_suffix
        snapshot_number_str = "" if snapshot_number is None else str(snapshot_number)
        where.mkdir(parents=True, exist_ok=True)
        return where / f"{snapshot_prefix}{snapshot_number_str}{snapshot_suffix}.jsonl"

    @staticmethod
    def _write_lines_sync(where_path: Path, lines: list[str]) -> int:
        with where_path.open("a", encoding="utf-8") as fp:
            for line in lines:
                fp.write(line)
            fp.flush()
            return fp.tell()

    async def _ensure_snapshot_writer(self) -> None:
        if self.snapshot_backend == "loguru":
            if self._snapshot_sink_id is not None:
                return
            where_path = self._snapshot_path()
            self._snapshot_sink_id = logger.add(
                where_path,
                level="SNAPSHOT",
                format="{message}",
                filter=lambda record: record["level"].name == "SNAPSHOT",
                enqueue=True,
                backtrace=False,
                diagnose=False,
                catch=False,
            )
            return

        if self._snapshot_writer_task is not None:
            return
        self._snapshot_queue = asio.Queue()
        self._snapshot_writer_task = asio.create_task(self._snapshot_writer_loop())

    async def _snapshot_writer_loop(self) -> None:
        queue = self._snapshot_queue
        if queue is None:
            return

        where_path = self._snapshot_path()
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break

            # Batch ready records to reduce fs overhead under high concurrency.
            batch: list[dict[str, Any]] = [item]
            while True:
                try:
                    nxt = queue.get_nowait()
                except asio.QueueEmpty:
                    break
                if nxt is None:
                    queue.task_done()
                    lines = [json.dumps(it, ensure_ascii=False) + "\n" for it in batch]
                    self._snapshot_offsets[where_path] = await asio.to_thread(self._write_lines_sync, where_path, lines)
                    queue.task_done()
                    return
                batch.append(nxt)

            lines = [json.dumps(it, ensure_ascii=False) + "\n" for it in batch]
            self._snapshot_offsets[where_path] = await asio.to_thread(self._write_lines_sync, where_path, lines)
            for _ in batch:
                queue.task_done()

    async def _append_snapshot(
        self,
        item: dict[str, Any],
        snapshot_number: int | str | None = None,
        snapshot_suffix: str | None = None,
    ) -> None:
        if snapshot_number is not None or snapshot_suffix is not None:
            # Keep explicit API behavior for ad-hoc writes.
            where_path = self._snapshot_path(snapshot_number=snapshot_number, snapshot_suffix=snapshot_suffix)
            payload = json.dumps(item, ensure_ascii=False) + "\n"
            self._snapshot_offsets[where_path] = await asio.to_thread(self._write_lines_sync, where_path, [payload])
            return

        await self._ensure_snapshot_writer()
        if self.snapshot_backend == "loguru":
            logger.log("SNAPSHOT", json.dumps(item, ensure_ascii=False))
            return

        assert self._snapshot_queue is not None
        await self._snapshot_queue.put(item)

    async def _finalize_snapshot_writer(self) -> None:
        if self.snapshot_backend == "loguru":
            if self._snapshot_sink_id is not None:
                logger.remove(self._snapshot_sink_id)
                self._snapshot_sink_id = None
            return

        if self._snapshot_writer_task is None or self._snapshot_queue is None:
            return

        await self._snapshot_queue.put(None)
        await self._snapshot_queue.join()
        self._snapshot_writer_task.cancel()
        with contextlib.suppress(asio.CancelledError):
            await self._snapshot_writer_task
        self._snapshot_writer_task = None
        self._snapshot_queue = None

    async def _flush_progress(
        self,
        data,
        where=None,
        snapshot_number: int | str | None = None,
        snapshot_prefix: str | None = None,
        snapshot_suffix: str | None = None,
    ):
        where = Path(os.getcwd()) if not where else Path(where)
        snapshot_prefix = "" if snapshot_prefix is None else snapshot_prefix
        snapshot_suffix = "" if snapshot_suffix is None else snapshot_suffix
        snapshot_number_str = "" if snapshot_number is None else str(snapshot_number)
        where_path = where / f"{snapshot_prefix}{snapshot_number_str}{snapshot_suffix}.jsonl"
        where.mkdir(parents=True, exist_ok=True)

        is_ok_local: bool | None
        try:
            if isinstance(data, Sequence) and not isinstance(data, (dict, str, bytes)):
                lines = [json.dumps(item, ensure_ascii=False) + "\n" for item in data]
            else:
                lines = [json.dumps(data, ensure_ascii=False) + "\n"]

            self._snapshot_offsets[where_path] = await asio.to_thread(self._write_lines_sync, where_path, lines)
        except Exception as exc:
            logger.error(f"The data coming {data} is not JSONL compliant: {exc}")
            is_ok_local = False
        else:
            is_ok_local = True
        return is_ok_local


__all__ = ["OpenAiTask", "OpenAIAsyncWrapper"]
