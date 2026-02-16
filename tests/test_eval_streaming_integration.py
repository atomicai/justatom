import asyncio
import json
import random
import string
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import pytest

from justatom.running.service import RunningService
from justatom.storing.weaviate import Finder as WeaviateApi
from justatom.tooling.dataset import DatasetRecordAdapter


pytestmark = pytest.mark.integration


class EvalStreamingIntegrationTest(unittest.TestCase):
    @staticmethod
    def _random_collection_name(prefix: str = "EvalStream") -> str:
        suffix = "".join(random.choice(string.ascii_lowercase) for _ in range(8))
        return f"{prefix}{suffix}"

    @staticmethod
    def _ensure_weaviate_up() -> None:
        async def _ping() -> bool:
            try:
                store = await WeaviateApi.find(
                    "Healthcheck",
                    WEAVIATE_HOST="localhost",
                    WEAVIATE_PORT=2211,
                )
                await store.count_documents()
                await store.delete_collection()
                return True
            except Exception:
                return False

        if asyncio.run(_ping()):
            return

        project_root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            ["docker", "compose", "up", "-d", "weaviate"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Failed to start Weaviate via docker compose: {proc.stdout}\n{proc.stderr}"
            )

        deadline = time.time() + 60
        while time.time() < deadline:
            if asyncio.run(_ping()):
                return
            time.sleep(2)
        raise RuntimeError("Weaviate did not become ready in time")

    @staticmethod
    def _dummy_iterative_dataset(n_docs: int = 100) -> str:
        fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="eval_streaming_")
        Path(path).unlink(missing_ok=True)
        data_path = Path(path)
        with data_path.open("w", encoding="utf-8") as f:
            for i in range(n_docs):
                topic = f"topic-{i}"
                row = {
                    "chunk_id": f"chunk-{i}",
                    "content": f"This paragraph is about {topic}.",
                    "labels": [topic],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return str(data_path)

    def test_streaming_index_and_streaming_eval(self):
        try:
            self._ensure_weaviate_up()
        except Exception as ex:
            self.skipTest(f"Weaviate is not available and could not be started: {ex}")

        collection_name = self._random_collection_name()
        dataset_path = self._dummy_iterative_dataset(n_docs=100)
        try:
            async def _run_pipeline() -> tuple[int, int, int]:
                docs_adapter = DatasetRecordAdapter.from_source(
                    dataset_name_or_path=dataset_path,
                    content_col="content",
                    queries_col="labels",
                    chunk_id_col="chunk_id",
                )

                ir_runner = await RunningService.do_index_and_prepare_for_search(
                    collection_name=collection_name,
                    documents=docs_adapter.iterator(),
                    model_name_or_path=None,
                    index_and_eval_by="keywords",
                    batch_size=4,
                    flush_collection=True,
                    weaviate_host="localhost",
                    weaviate_port=2211,
                )

                n_docs = await ir_runner.store.count_documents()

                labels_adapter = DatasetRecordAdapter.from_source(
                    dataset_name_or_path=dataset_path,
                    content_col="content",
                    queries_col="labels",
                    chunk_id_col="chunk_id",
                )

                n_total, n_hit = 0, 0
                for q in DatasetRecordAdapter.extract_labels(labels_adapter.iterator()):
                    retrieved = await ir_runner.retrieve_topk(queries=q, top_k=5)
                    n_total += 1
                    if any(q in (doc.meta or {}).get("labels", []) for doc in retrieved):
                        n_hit += 1

                return n_docs, n_total, n_hit

            n_docs, n_total, n_hit = asyncio.run(_run_pipeline())
            self.assertGreaterEqual(n_docs, 100)
            self.assertGreater(n_total, 0)
            self.assertGreater(n_hit, 0)
        finally:
            try:
                async def _cleanup() -> None:
                    store = await WeaviateApi.find(
                        collection_name,
                        WEAVIATE_HOST="localhost",
                        WEAVIATE_PORT=2211,
                    )
                    await store.delete_collection()

                asyncio.run(_cleanup())
            except Exception:
                pass
            Path(dataset_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
