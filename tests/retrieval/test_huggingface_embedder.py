import asyncio

import pytest

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
