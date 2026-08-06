import asyncio
import json
import queue
import threading

import httpx
import numpy as np
import pytest

from justatom.retrieval.embedders.openai_compatible import OpenAICompatibleEmbedder
from justatom.running.clusters import EmbeddingBackendAdapter


class FakeEmbedder:
    def __init__(self):
        self.loops = []
        self.close_calls = 0

    async def embed_documents(self, texts):
        self.loops.append(asyncio.get_running_loop())
        return [[float(index), 1.0] for index, _ in enumerate(texts)]

    async def close(self):
        self.close_calls += 1


def _close_if_supported(adapter):
    close = getattr(adapter, "close", None)
    if close is not None:
        close()


def test_bertopic_adapter_uses_document_embedding_role():
    adapter = EmbeddingBackendAdapter(FakeEmbedder())
    try:
        result = adapter.embed(["a", "b"])
        np.testing.assert_array_equal(result, np.asarray([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32))
    finally:
        _close_if_supported(adapter)


def test_repeated_calls_use_one_running_owner_loop_and_float32_results():
    embedder = FakeEmbedder()
    adapter = EmbeddingBackendAdapter(embedder)
    try:
        first = adapter.embed(["a"])
        second = adapter.embed(["a", "b"])

        assert embedder.loops[0] is embedder.loops[1]
        assert embedder.loops[0].is_running()
        assert first.dtype == np.float32
        np.testing.assert_array_equal(second, np.asarray([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32))
    finally:
        _close_if_supported(adapter)


def test_direct_embed_inside_another_running_loop_completes():
    adapter = EmbeddingBackendAdapter(FakeEmbedder())
    outcome = queue.Queue()

    async def invoke():
        return adapter.embed(["a"])

    def run():
        try:
            outcome.put(("result", asyncio.run(invoke())))
        except BaseException as error:
            outcome.put(("error", error))

    caller = threading.Thread(target=run, daemon=True)
    try:
        caller.start()
        caller.join(timeout=2)

        assert not caller.is_alive(), "adapter.embed deadlocked inside a running caller loop"
        kind, value = outcome.get_nowait()
        if kind == "error":
            raise value
        np.testing.assert_array_equal(value, np.asarray([[0.0, 1.0]], dtype=np.float32))
    finally:
        _close_if_supported(adapter)


def test_underlying_async_exception_propagates():
    class FailingEmbedder(FakeEmbedder):
        async def embed_documents(self, texts):
            raise ValueError(f"cannot embed {len(texts)} documents")

    adapter = EmbeddingBackendAdapter(FailingEmbedder())
    try:
        with pytest.raises(ValueError, match="cannot embed 1 documents"):
            adapter.embed(["a"])
    finally:
        _close_if_supported(adapter)


def test_close_stops_owned_thread_is_idempotent_and_does_not_close_embedder():
    embedder = FakeEmbedder()
    adapter = EmbeddingBackendAdapter(embedder)
    adapter.embed(["a"])
    owner_thread = adapter._thread

    adapter.close()
    adapter.close()

    assert not owner_thread.is_alive()
    assert embedder.close_calls == 0
    with pytest.raises(RuntimeError, match="closed"):
        adapter.embed(["b"])


def test_concurrent_close_waits_for_active_embed_without_abandoning_it():
    class BlockingEmbedder(FakeEmbedder):
        def __init__(self):
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()
            self.completed = threading.Event()

        async def embed_documents(self, texts):
            self.started.set()
            while not self.release.is_set():
                await asyncio.sleep(0.005)
            self.completed.set()
            return [[3.0, 4.0] for _ in texts]

    embedder = BlockingEmbedder()
    adapter = EmbeddingBackendAdapter(embedder)
    outcome = queue.Queue()

    def embed():
        try:
            outcome.put(("result", adapter.embed(["a"])))
        except BaseException as error:
            outcome.put(("error", error))

    embed_thread = threading.Thread(target=embed, daemon=True)
    close_thread = threading.Thread(target=adapter.close, daemon=True)
    try:
        embed_thread.start()
        assert embedder.started.wait(timeout=1)
        close_thread.start()
        close_thread.join(timeout=0.05)
        assert close_thread.is_alive(), "close returned before the active embed completed"

        embedder.release.set()
        embed_thread.join(timeout=2)
        close_thread.join(timeout=2)

        assert not embed_thread.is_alive()
        assert not close_thread.is_alive()
        assert embedder.completed.is_set()
        kind, value = outcome.get_nowait()
        if kind == "error":
            raise value
        np.testing.assert_array_equal(value, np.asarray([[3.0, 4.0]], dtype=np.float32))
        assert embedder.close_calls == 0
    finally:
        embedder.release.set()
        embed_thread.join(timeout=2)
        close_thread.join(timeout=2)
        _close_if_supported(adapter)


def test_openai_compatible_embedder_reuses_loop_sensitive_transport_without_network():
    class LoopSensitiveTransport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.loop = None
            self.close_calls = 0

        async def handle_async_request(self, request):
            loop = asyncio.get_running_loop()
            if self.loop is None:
                self.loop = loop
            elif self.loop is not loop:
                raise RuntimeError("Event loop is closed")
            texts = json.loads(request.content)["input"]
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": index, "embedding": [float(len(text)), 1.0]}
                        for index, text in enumerate(texts)
                    ]
                },
            )

        async def aclose(self):
            self.close_calls += 1

    transport = LoopSensitiveTransport()
    embedder = OpenAICompatibleEmbedder(
        base_url="http://embedding.test/v1",
        model="test-model",
        transport=transport,
    )
    adapter = EmbeddingBackendAdapter(embedder)
    try:
        np.testing.assert_array_equal(adapter.embed(["a"]), np.asarray([[1.0, 1.0]], dtype=np.float32))
        np.testing.assert_array_equal(adapter.embed(["bb"]), np.asarray([[2.0, 1.0]], dtype=np.float32))
        assert transport.close_calls == 0
    finally:
        _close_if_supported(adapter)
        asyncio.run(embedder.close())
