import asyncio
import threading

import pytest
import torch

from justatom.retrieval.contracts import EmbeddingProfile
from justatom.retrieval.embedders import huggingface as module
from justatom.retrieval.errors import ConfigurationError


class FakeEncoder:
    def __init__(self):
        self.calls = []
        self.closed = 0

    def encode(self, texts):
        self.calls.append(list(texts))
        return [[float(len(text)), 1.0] for text in texts]

    def close(self):
        self.closed += 1


class BlockingEncoder(FakeEncoder):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def encode(self, texts):
        self.started.set()
        assert self.release.wait(timeout=1)
        return super().encode(texts)


class SequencedBlockingEncoder(FakeEncoder):
    def __init__(self):
        super().__init__()
        self._state_lock = threading.Lock()
        self.entered = [threading.Event(), threading.Event()]
        self.release = [threading.Event(), threading.Event()]
        self.active = 0
        self.max_active = 0

    def encode(self, texts):
        with self._state_lock:
            call_index = sum(event.is_set() for event in self.entered)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.entered[call_index].set()
        try:
            assert self.release[call_index].wait(timeout=2)
            return super().encode(texts)
        finally:
            with self._state_lock:
                self.active -= 1


async def wait_for_active_encodes(embedder, expected):
    for _ in range(100):
        if embedder._active_encodes == expected:
            return
        await asyncio.sleep(0.01)
    pytest.fail(f"expected {expected} active encodes, observed {embedder._active_encodes}")


def test_local_encoder_converts_bfloat16_outputs_to_python_floats():
    class BFloat16Runner:
        def __call__(self, *, batch):
            assert set(batch) == {"input_ids"}
            return [torch.tensor([[1.5, -2.0]], dtype=torch.bfloat16)]

    encoder = module._LocalEncoder(
        torch=torch,
        processor=object(),
        runner=BFloat16Runner(),
        igniset=lambda **kwargs: (object(), ["input_ids"]),
        data_loader=lambda **kwargs: [{"input_ids": torch.tensor([[1]])}],
        device="cpu",
    )

    assert encoder.encode(["one"]) == [[1.5, -2.0]]


def test_local_embedder_builds_once_and_reuses_one_encoder(monkeypatch):
    built = []
    encoder = FakeEncoder()

    def fake_build(model, device, max_length):
        built.append((model, device, max_length))
        return encoder

    monkeypatch.setattr(module, "_build_local_encoder", fake_build)
    embedder = module.HuggingFaceEmbedder(
        model="local-model",
        device="cpu",
        profile=EmbeddingProfile(query_prefix="q: ", document_prefix="d: ", batch_size=2),
    )
    assert asyncio.run(embedder.embed_queries(["one", "two", "three"])) == [[6.0, 1.0], [6.0, 1.0], [8.0, 1.0]]
    assert asyncio.run(embedder.embed_documents(["one"])) == [[6.0, 1.0]]
    asyncio.run(embedder.close())
    asyncio.run(embedder.close())

    assert built == [("local-model", "cpu", 512)]
    assert encoder.calls == [["q: one", "q: two"], ["q: three"], ["d: one"]]
    assert encoder.closed == 1


def test_resolve_device_auto_prefers_cuda_then_mps_then_cpu(monkeypatch):
    monkeypatch.setattr(module, "_available_devices", lambda: (True, True))
    assert module.resolve_device("auto") == "cuda:0"
    monkeypatch.setattr(module, "_available_devices", lambda: (False, True))
    assert module.resolve_device("auto") == "mps"
    monkeypatch.setattr(module, "_available_devices", lambda: (False, False))
    assert module.resolve_device("auto") == "cpu"
    with pytest.raises(ConfigurationError, match="mps"):
        module.resolve_device("mps")


def test_resolve_device_preserves_available_cuda_indexes(monkeypatch):
    monkeypatch.setattr(module, "_available_devices", lambda: (True, False))
    monkeypatch.setattr(module, "_cuda_device_count", lambda: 2, raising=False)

    assert module.resolve_device("cuda") == "cuda:0"
    assert module.resolve_device("cuda:1") == "cuda:1"
    with pytest.raises(ConfigurationError, match="cuda:2"):
        module.resolve_device("cuda:2")


def test_close_waits_for_active_encode_and_only_closes_once(monkeypatch):
    encoder = BlockingEncoder()
    monkeypatch.setattr(module, "_build_local_encoder", lambda *args: encoder)

    async def exercise():
        embedder = module.HuggingFaceEmbedder(model="local-model", device="cpu")
        embed_task = asyncio.create_task(embedder.embed_documents(["one"]))
        assert await asyncio.to_thread(encoder.started.wait, 1)
        close_one = asyncio.create_task(embedder.close())
        close_two = asyncio.create_task(embedder.close())

        try:
            await asyncio.sleep(0)
            assert not close_one.done()
            assert not close_two.done()
            assert encoder.closed == 0
            with pytest.raises(RuntimeError, match="closed"):
                await embedder.embed_documents(["two"])
        finally:
            encoder.release.set()

        vectors, _, _ = await asyncio.gather(embed_task, close_one, close_two)
        assert vectors == [[3.0, 1.0]]
        assert encoder.closed == 1

    asyncio.run(exercise())


def test_cancelled_close_cannot_abandon_shutdown(monkeypatch):
    encoder = BlockingEncoder()
    monkeypatch.setattr(module, "_build_local_encoder", lambda *args: encoder)

    async def exercise():
        embedder = module.HuggingFaceEmbedder(model="local-model", device="cpu")
        embed_task = asyncio.create_task(embedder.embed_documents(["one"]))
        assert await asyncio.to_thread(encoder.started.wait, 1)
        close_task = asyncio.create_task(embedder.close())

        try:
            await asyncio.sleep(0)
            assert not close_task.done()
            close_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await close_task

            encoder.release.set()
            assert await embed_task == [[3.0, 1.0]]
            await asyncio.wait_for(embedder.close(), timeout=1)
            assert encoder.closed == 1
            with pytest.raises(RuntimeError, match="closed"):
                await embedder.embed_documents(["two"])
        finally:
            encoder.release.set()

    asyncio.run(exercise())


def test_concurrent_calls_serialize_encoder_work_and_close_waits_for_both(monkeypatch):
    encoder = SequencedBlockingEncoder()
    monkeypatch.setattr(module, "_build_local_encoder", lambda *args: encoder)

    async def exercise():
        embedder = module.HuggingFaceEmbedder(model="local-model", device="cpu")
        first = asyncio.create_task(embedder.embed_documents(["one"]))
        second = None
        close_task = None
        try:
            assert await asyncio.to_thread(encoder.entered[0].wait, 1)
            second = asyncio.create_task(embedder.embed_documents(["two"]))
            await wait_for_active_encodes(embedder, 2)
            close_task = asyncio.create_task(embedder.close())
            await asyncio.sleep(0)

            assert not encoder.entered[1].is_set()
            assert encoder.max_active == 1
            assert not close_task.done()
            with pytest.raises(RuntimeError, match="closed"):
                await embedder.embed_documents(["three"])

            encoder.release[0].set()
            assert await first == [[3.0, 1.0]]
            assert await asyncio.to_thread(encoder.entered[1].wait, 1)
            assert encoder.max_active == 1
            assert not close_task.done()
            assert encoder.closed == 0

            encoder.release[1].set()
            vectors, _ = await asyncio.gather(second, close_task)
            assert vectors == [[3.0, 1.0]]
            assert encoder.calls == [["one"], ["two"]]
            assert encoder.closed == 1
        finally:
            for release in encoder.release:
                release.set()
            pending = [task for task in (first, second, close_task) if task is not None]
            await asyncio.gather(*pending, return_exceptions=True)

    asyncio.run(exercise())


def test_cancelling_call_queued_for_encoder_releases_close_state(monkeypatch):
    encoder = SequencedBlockingEncoder()
    monkeypatch.setattr(module, "_build_local_encoder", lambda *args: encoder)

    async def exercise():
        embedder = module.HuggingFaceEmbedder(model="local-model", device="cpu")
        first = asyncio.create_task(embedder.embed_documents(["one"]))
        queued = None
        close_task = None
        try:
            assert await asyncio.to_thread(encoder.entered[0].wait, 1)
            queued = asyncio.create_task(embedder.embed_documents(["two"]))
            await wait_for_active_encodes(embedder, 2)

            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(queued, timeout=1)
            await wait_for_active_encodes(embedder, 1)
            assert not encoder.entered[1].is_set()

            close_task = asyncio.create_task(embedder.close())
            await asyncio.sleep(0)
            assert not close_task.done()
            assert encoder.closed == 0

            encoder.release[0].set()
            assert await first == [[3.0, 1.0]]
            await asyncio.wait_for(close_task, timeout=1)
            assert encoder.calls == [["one"]]
            assert encoder.closed == 1
        finally:
            for release in encoder.release:
                release.set()
            pending = [task for task in (first, queued, close_task) if task is not None]
            await asyncio.gather(*pending, return_exceptions=True)

    asyncio.run(exercise())
