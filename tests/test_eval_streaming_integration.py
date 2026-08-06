import asyncio
import json
import os
import random
import string
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import pytest

from justatom.retrieval import build_runtime
from justatom.storing.weaviate import WeaviateDocumentStore
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
            store = None
            try:
                store = await WeaviateDocumentStore.connect(
                    "Healthcheck",
                    url="http://localhost:2211",
                    grpc_port=50051,
                )
                await store.count_documents()
                await store.delete_collection()
                return True
            except Exception:
                return False
            finally:
                if store is not None:
                    await store.close()

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
            raise RuntimeError(f"Failed to start Weaviate via docker compose: {proc.stdout}\n{proc.stderr}")

        deadline = time.time() + 60
        while time.time() < deadline:
            if asyncio.run(_ping()):
                return
            time.sleep(2)
        raise RuntimeError("Weaviate did not become ready in time")

    @staticmethod
    def _dummy_iterative_dataset(n_docs: int = 100) -> str:
        fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="eval_streaming_")
        os.close(fd)
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

                runtime = await build_runtime(
                    {
                        "mode": "keyword",
                        "store": {
                            "collection": collection_name,
                            "url": "http://localhost:2211",
                            "grpc_port": 50051,
                        },
                    }
                )
                try:
                    await runtime.store.clear()
                    await runtime.index(docs_adapter.iterator(), batch_size=4)
                    n_docs = await runtime.store.count_documents()

                    labels_adapter = DatasetRecordAdapter.from_source(
                        dataset_name_or_path=dataset_path,
                        content_col="content",
                        queries_col="labels",
                        chunk_id_col="chunk_id",
                    )

                    n_total, n_hit = 0, 0
                    for q in DatasetRecordAdapter.extract_labels(labels_adapter.iterator()):
                        retrieved = await runtime.retrieve(q, top_k=5)
                        n_total += 1
                        if any(q in (doc.meta or {}).get("labels", []) for doc in retrieved):
                            n_hit += 1
                finally:
                    await runtime.close()

                return n_docs, n_total, n_hit

            n_docs, n_total, n_hit = asyncio.run(_run_pipeline())
            self.assertGreaterEqual(n_docs, 100)
            self.assertGreater(n_total, 0)
            self.assertGreater(n_hit, 0)
        finally:
            try:

                async def _cleanup() -> None:
                    store = await WeaviateDocumentStore.connect(
                        collection_name,
                        url="http://localhost:2211",
                        grpc_port=50051,
                    )
                    try:
                        await store.delete_collection()
                    finally:
                        await store.close()

                asyncio.run(_cleanup())
            except Exception:
                pass
            Path(dataset_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
