# Retrieval Runtime Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the duplicated retrieval/indexing wrappers with one explicit runtime that uses either an in-process Hugging Face embedder or a remote OpenAI-compatible embedding endpoint.

**Architecture:** Add a focused `justatom.retrieval` package with structural `Embedder` and `DocumentStore` contracts, one indexer, three retrievers, and an owning runtime. Keep Weaviate as the only built-in store, remove global model/client caches and name-based factories, and migrate eval/server call sites before deleting the old modules.

**Tech Stack:** Python 3.10+, asyncio, PyTorch/Transformers through the existing local encoder stack, `httpx.AsyncClient`, Weaviate Python client, Quart, pytest/unittest, Black, isort, pylint, MkDocs.

## Global Constraints

- Python support remains `>=3.10`.
- Weaviate is the only built-in ANN/document store.
- Built-in embedding backends are exactly `local` and `openai-compatible`.
- llama.cpp, vLLM, and Triton are supported only through an OpenAI-compatible `/embeddings` endpoint.
- Never import or depend on `tritonclient`; ordinary installation and imports must work on macOS.
- Do not add a backend registry, service locator, dependency injection container, or compatibility wrapper.
- Do not add model, tokenizer, processor, runner, HTTP-client, query, document, or result LRUs.
- Keep the Hugging Face on-disk artifact cache enabled.
- Keyword mode must work without constructing or importing the local ML backend.
- `retrieve()` returns `list[Document]`; `retrieve_many()` returns `list[list[Document]]`.
- This is a hard API cut: `embedding` becomes `vector`, `keywords` becomes `keyword`, and `gamma-hybrid`/`atomicai` disappear from retrieval.
- Training encoders, losses, ATOMIC training, and checkpoint formats are out of scope.
- Run project commands in the `justatom` Conda environment.

---

## File Map

### New production files

- `justatom/retrieval/__init__.py`: stable public exports only.
- `justatom/retrieval/contracts.py`: protocols, `EmbeddingProfile`, `SearchMode`, prefix and vector validation.
- `justatom/retrieval/errors.py`: retrieval/embedding/configuration exception hierarchy.
- `justatom/retrieval/embedders/__init__.py`: lazy public embedder exports.
- `justatom/retrieval/embedders/openai_compatible.py`: remote async embedding transport.
- `justatom/retrieval/embedders/huggingface.py`: local model ownership and device resolution.
- `justatom/retrieval/indexer.py`: lazy batched document indexing.
- `justatom/retrieval/retriever.py`: keyword, vector, and hybrid retrievers.
- `justatom/retrieval/runtime.py`: resource-owning runtime and config composition.
- `configs/serve.yaml`: repository server runtime config.
- `justatom/builtins/configs/serve.default.yaml`: packaged server defaults.

### New tests

- `tests/retrieval/test_contracts.py`
- `tests/retrieval/test_openai_compatible_embedder.py`
- `tests/retrieval/test_huggingface_embedder.py`
- `tests/retrieval/test_indexer.py`
- `tests/retrieval/test_retrievers.py`
- `tests/retrieval/test_runtime.py`
- `tests/test_run_api.py`

### Existing files to modify

- `pyproject.toml`, `requirements.txt`: declare `httpx` directly.
- `justatom/storing/weaviate.py`: one async client, new class/method names, no `Finder`.
- `justatom/running/evaluator.py`: consume `Retriever.retrieve_many`.
- `justatom/running/clusters.py`: consume `Embedder.embed_documents`.
- `justatom/running/mask.py`: remove only old retrieval/indexer interfaces.
- `justatom/storing/mask.py`: remove only `INNDocStore`.
- `justatom/api/eval.py`: build and own the new runtime.
- `justatom/api/run.py`: one configured runtime per application lifecycle.
- `configs/evaluate.yaml`, `justatom/builtins/configs/evaluate.default.yaml`: new retrieval config shape.
- `scripts/run_pipeline.sh`, `scripts/run_benchmark.sh`, `scripts/run_eval_model_on_datasets.sh`: new mode/config names.
- `README.md`, `docs/architecture.md`, `docs/modules/runtime.md`, `docs/modules/storage.md`, `docs/launch-guide.md`: new public API.
- Existing eval, scenario, metrics, and Weaviate integration tests: migrate imports and assertions.

### Files to delete after migration

- `justatom/running/service.py`
- `justatom/running/indexer.py`
- `justatom/running/retriever.py`
- `justatom/running/embeddings/base.py`
- `justatom/running/embeddings/local.py`
- `justatom/running/embeddings/openai_compatible.py`
- `justatom/running/embeddings/__init__.py`
- `justatom/builtins/configs/embeddings.yaml`
- `tests/test_retriever_shape_and_factory.py`

---

### Task 1: Core Contracts, Profiles, and Validation

**Files:**
- Create: `justatom/retrieval/errors.py`
- Create: `justatom/retrieval/contracts.py`
- Create: `justatom/retrieval/__init__.py`
- Create: `tests/retrieval/__init__.py`
- Create: `tests/retrieval/test_contracts.py`

**Interfaces:**
- Produces: `EmbeddingProfile`, `Embedder`, `DocumentStore`, `Retriever`, `SearchMode`, `apply_prefix`, and `validate_embeddings`.
- Produces errors: `RetrievalError`, `ConfigurationError`, `EmbeddingError`, `EmbeddingBackendError`, `EmbeddingResponseError`.

- [ ] **Step 1: Write failing profile and vector validation tests**

```python
import math

import pytest

from justatom.retrieval.contracts import EmbeddingProfile, SearchMode, apply_prefix, validate_embeddings
from justatom.retrieval.errors import ConfigurationError, EmbeddingResponseError


def test_profile_validates_positive_limits_and_avoids_double_prefix():
    profile = EmbeddingProfile(query_prefix="query: ", document_prefix="passage: ")
    assert apply_prefix("cats", profile.query_prefix, skip_if_present=True) == "query: cats"
    assert apply_prefix("query: cats", profile.query_prefix, skip_if_present=True) == "query: cats"
    with pytest.raises(ConfigurationError, match="batch_size"):
        EmbeddingProfile(batch_size=0)


def test_validate_embeddings_checks_count_dimension_and_finite_values():
    assert validate_embeddings([[1, 2], [3.5, 4]], expected_count=2) == [[1.0, 2.0], [3.5, 4.0]]
    with pytest.raises(EmbeddingResponseError, match="Expected 2 vectors"):
        validate_embeddings([[1.0]], expected_count=2)
    with pytest.raises(EmbeddingResponseError, match="dimension"):
        validate_embeddings([[1.0], [1.0, 2.0]], expected_count=2)
    with pytest.raises(EmbeddingResponseError, match="finite"):
        validate_embeddings([[math.nan]], expected_count=1)


def test_search_modes_are_only_keyword_vector_and_hybrid():
    assert {mode.value for mode in SearchMode} == {"keyword", "vector", "hybrid"}
```

- [ ] **Step 2: Run the tests and verify the new package is missing**

Run: `conda run -n justatom python -m pytest tests/retrieval/test_contracts.py -v`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'justatom.retrieval'`.

- [ ] **Step 3: Add the exception hierarchy and contracts**

```python
# justatom/retrieval/errors.py
class RetrievalError(Exception):
    pass


class ConfigurationError(RetrievalError, ValueError):
    pass


class EmbeddingError(RetrievalError):
    pass


class EmbeddingBackendError(EmbeddingError):
    pass


class EmbeddingResponseError(EmbeddingError):
    pass
```

```python
# essential contents of justatom/retrieval/contracts.py
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from justatom.etc.schema import Document
from justatom.retrieval.errors import ConfigurationError, EmbeddingResponseError


@dataclass(frozen=True)
class EmbeddingProfile:
    query_prefix: str = ""
    document_prefix: str = ""
    max_length: int = 512
    batch_size: int = 64
    skip_prefix_if_present: bool = True

    def __post_init__(self) -> None:
        if self.max_length <= 0:
            raise ConfigurationError("max_length must be positive")
        if self.batch_size <= 0:
            raise ConfigurationError("batch_size must be positive")


class SearchMode(str, Enum):
    KEYWORD = "keyword"
    VECTOR = "vector"
    HYBRID = "hybrid"


def apply_prefix(text: str, prefix: str, *, skip_if_present: bool) -> str:
    if not prefix or (skip_if_present and text.startswith(prefix)):
        return text
    return f"{prefix}{text}"


def validate_embeddings(vectors: Sequence[Sequence[float]], *, expected_count: int) -> list[list[float]]:
    if len(vectors) != expected_count:
        raise EmbeddingResponseError(f"Expected {expected_count} vectors, received {len(vectors)}")
    normalized = [[float(value) for value in vector] for vector in vectors]
    dimensions = {len(vector) for vector in normalized}
    if expected_count and (0 in dimensions or len(dimensions) != 1):
        raise EmbeddingResponseError("Embedding vectors must have one non-zero dimension")
    if any(not math.isfinite(value) for vector in normalized for value in vector):
        raise EmbeddingResponseError("Embedding vectors must contain finite values")
    return normalized
```

Add the protocols directly after the validation helpers:

```python
@runtime_checkable
class Embedder(Protocol):
    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    async def close(self) -> None: ...


@runtime_checkable
class DocumentStore(Protocol):
    async def write_documents(self, documents: Sequence[Document]) -> int: ...
    async def search_vector(
        self,
        vectors: Sequence[Sequence[float]],
        *,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        include_vectors: bool = False,
    ) -> list[list[Document]]: ...
    async def search_keywords(
        self,
        queries: Sequence[str],
        *,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
    ) -> list[list[Document]]: ...
    async def search_hybrid(
        self,
        queries: Sequence[str],
        vectors: Sequence[Sequence[float]],
        *,
        alpha: float,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        include_vectors: bool = False,
    ) -> list[list[Document]]: ...
    async def count_documents(self) -> int: ...
    async def clear(self) -> None: ...
    async def close(self) -> None: ...


@runtime_checkable
class Retriever(Protocol):
    async def retrieve(self, query: str, *, top_k: int = 5, **kwargs: Any) -> list[Document]: ...
    async def retrieve_many(
        self,
        queries: Sequence[str],
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[list[Document]]: ...
```

Export only these stable public names from `justatom/retrieval/__init__.py`; do not import either concrete embedder there yet.

- [ ] **Step 4: Run the focused tests**

Run: `conda run -n justatom python -m pytest tests/retrieval/test_contracts.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit the contracts**

```bash
git add justatom/retrieval tests/retrieval
git commit -m "feat: add retrieval runtime contracts"
```

---

### Task 2: OpenAI-Compatible Remote Embedder

**Files:**
- Create: `justatom/retrieval/embedders/__init__.py`
- Create: `justatom/retrieval/embedders/openai_compatible.py`
- Create: `tests/retrieval/test_openai_compatible_embedder.py`
- Modify: `pyproject.toml`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: `EmbeddingProfile`, `apply_prefix`, `validate_embeddings`, `EmbeddingBackendError`, `EmbeddingResponseError`.
- Produces: `OpenAICompatibleEmbedder(base_url, model, api_key=None, timeout=30.0, profile=None, encoding_format=None, extra_body=None, transport=None)`.
- Produces methods: `embed_queries`, `embed_documents`, and idempotent `close`.

- [ ] **Step 1: Write failing transport, prefix, ordering, and error tests**

```python
import asyncio
import json

import httpx
import pytest

from justatom.retrieval.contracts import EmbeddingProfile
from justatom.retrieval.embedders.openai_compatible import OpenAICompatibleEmbedder
from justatom.retrieval.errors import EmbeddingBackendError, EmbeddingResponseError


def test_remote_embedder_applies_role_prefix_and_restores_index_order():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == "/v1/embeddings"
        body = json.loads(request.content)
        assert body["input"] == ["query: first", "query: second"]
        return httpx.Response(
            200,
            json={"data": [{"index": 1, "embedding": [2, 0]}, {"index": 0, "embedding": [1, 0]}]},
        )

    embedder = OpenAICompatibleEmbedder(
        base_url="http://embedding.test/v1",
        model="test-model",
        api_key=None,
        profile=EmbeddingProfile(query_prefix="query: "),
        transport=httpx.MockTransport(handler),
    )
    vectors = asyncio.run(embedder.embed_queries(["first", "second"]))
    asyncio.run(embedder.close())

    assert vectors == [[1.0, 0.0], [2.0, 0.0]]
    assert "authorization" not in requests[0].headers


def test_remote_embedder_splits_requests_by_profile_batch_size():
    batches = []

    def handler(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        batches.append(inputs)
        return httpx.Response(200, json={
            "data": [{"index": index, "embedding": [float(len(text))]} for index, text in enumerate(inputs)]
        })

    embedder = OpenAICompatibleEmbedder(
        base_url="http://embedding.test/v1",
        model="test-model",
        profile=EmbeddingProfile(batch_size=2),
        transport=httpx.MockTransport(handler),
    )
    assert len(asyncio.run(embedder.embed_documents(["a", "bb", "ccc"]))) == 3
    assert batches == [["a", "bb"], ["ccc"]]
    asyncio.run(embedder.close())


def test_remote_embedder_sanitizes_http_failures():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="provider failed: secret document body")

    embedder = OpenAICompatibleEmbedder(
        base_url="http://embedding.test/v1",
        model="test-model",
        api_key="top-secret",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(EmbeddingBackendError) as exc_info:
        asyncio.run(embedder.embed_documents(["private input text"]))
    message = str(exc_info.value)
    assert "503" in message and "test-model" in message
    assert "top-secret" not in message and "private input text" not in message
    asyncio.run(embedder.close())


def test_remote_embedder_rejects_missing_or_duplicate_response_indexes():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1]}, {"index": 0, "embedding": [2]}]})

    embedder = OpenAICompatibleEmbedder(
        base_url="http://embedding.test/v1",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(EmbeddingResponseError, match="indexes"):
        asyncio.run(embedder.embed_queries(["a", "b"]))
    asyncio.run(embedder.close())
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `conda run -n justatom python -m pytest tests/retrieval/test_openai_compatible_embedder.py -v`

Expected: FAIL because `OpenAICompatibleEmbedder` does not exist in the new package.

- [ ] **Step 3: Declare and implement the async remote transport**

Add `httpx>=0.27,<1` to `[project].dependencies`, `project.optional-dependencies.test`, and `requirements.txt`.

Implement a client-owned `httpx.AsyncClient` and a single role-aware private method:

```python
async def _embed(self, texts: Sequence[str], *, prefix: str) -> list[list[float]]:
    if not texts:
        return []
    normalized = [
        apply_prefix(text, prefix, skip_if_present=self.profile.skip_prefix_if_present)
        for text in texts
    ]
    vectors: list[list[float]] = []
    for chunk in chunked(normalized, self.profile.batch_size):
        payload = {**self.extra_body, "model": self.model, "input": list(chunk)}
        if self.encoding_format is not None:
            payload["encoding_format"] = self.encoding_format
        try:
            response = await self._client.post("embeddings", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise EmbeddingBackendError(
                f"Embedding endpoint {self.base_url!r} failed for model {self.model!r} "
                f"with HTTP {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise EmbeddingBackendError(
                f"Embedding endpoint {self.base_url!r} failed for model {self.model!r}: {type(exc).__name__}"
            ) from exc
        vectors.extend(self._parse_response(response.json(), expected_count=len(chunk)))
    return validate_embeddings(vectors, expected_count=len(texts))
```

`_parse_response` must require integer indexes exactly equal to `range(expected_count)`. Headers contain `Authorization: Bearer ...` only when `api_key` is non-empty. `extra_body` is merged before authoritative `model` and `input`, so it cannot override them. `close()` guards a `_closed` flag.

Normalize the HTTP client's base URL as `f"{base_url.rstrip('/')}/"`; this preserves a supplied `/v1` prefix when posting the relative `embeddings` path.

- [ ] **Step 4: Run remote embedder tests and an import isolation check**

Run: `conda run -n justatom python -m pytest tests/retrieval/test_openai_compatible_embedder.py -v`

Run: `conda run -n justatom python -c "import sys; import justatom.retrieval.embedders.openai_compatible; assert 'tritonclient' not in sys.modules"`

Expected: tests PASS and the import command exits 0.

- [ ] **Step 5: Commit the remote backend**

```bash
git add pyproject.toml requirements.txt justatom/retrieval/embedders tests/retrieval/test_openai_compatible_embedder.py
git commit -m "feat: add OpenAI-compatible embedder"
```

---

### Task 3: In-Process Hugging Face Embedder

**Files:**
- Create: `justatom/retrieval/embedders/huggingface.py`
- Modify: `justatom/retrieval/embedders/__init__.py`
- Create: `tests/retrieval/test_huggingface_embedder.py`

**Interfaces:**
- Consumes: `EmbeddingProfile`, `apply_prefix`, `validate_embeddings`, `ConfigurationError`.
- Produces: `resolve_device(requested: str) -> str`.
- Produces: `HuggingFaceEmbedder(model, device="auto", profile=None)` with the same three async methods as the remote embedder.

- [ ] **Step 1: Write failing lifecycle, prefix, batching, and device tests**

```python
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
    assert asyncio.run(embedder.embed_queries(["one", "two", "three"])) == [
        [6.0, 1.0], [6.0, 1.0], [8.0, 1.0]
    ]
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
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `conda run -n justatom python -m pytest tests/retrieval/test_huggingface_embedder.py -v`

Expected: FAIL because the local module is missing.

- [ ] **Step 3: Implement lazy local stack ownership**

Keep torch and existing encoder imports inside `_build_local_encoder` and `_available_devices`, not at module import time. `_build_local_encoder` creates exactly one `ILanguageModel`, tokenizer, prefix-free `RuntimeProcessor`, and `EncoderRunner`. Explicit `cuda`/`mps` requests raise `ConfigurationError` when unavailable; only `auto` chooses the best available device.

```python
class HuggingFaceEmbedder:
    def __init__(self, model: str, device: str = "auto", profile: EmbeddingProfile | None = None):
        if not model.strip():
            raise ConfigurationError("model must be non-empty")
        self.model = model
        self.profile = profile or EmbeddingProfile()
        self.device = resolve_device(device)
        self._encoder = _build_local_encoder(model, self.device, self.profile.max_length)
        self._closed = False

    async def _embed(self, texts: Sequence[str], *, prefix: str) -> list[list[float]]:
        if not texts:
            return []
        normalized = [apply_prefix(text, prefix, skip_if_present=self.profile.skip_prefix_if_present) for text in texts]
        vectors = []
        for batch in chunked(normalized, self.profile.batch_size):
            vectors.extend(await asyncio.to_thread(self._encoder.encode, list(batch)))
        return validate_embeddings(vectors, expected_count=len(texts))
```

The private encoder uses `torch.inference_mode()` inside `encode`. `close()` calls the private encoder's `close`, clears its references, and is idempotent. Missing local dependencies raise `ConfigurationError("Local embeddings require pip install 'justatom[torch]'")` with the original import error as cause.

- [ ] **Step 4: Run local and remote import tests**

Run: `conda run -n justatom python -m pytest tests/retrieval/test_huggingface_embedder.py tests/retrieval/test_openai_compatible_embedder.py -v`

Run: `conda run -n justatom python -c "import sys; from justatom.retrieval.embedders.openai_compatible import OpenAICompatibleEmbedder; assert 'torch' not in sys.modules"`

Expected: all tests PASS; importing the remote module does not initialize torch.

- [ ] **Step 5: Commit the local backend**

```bash
git add justatom/retrieval/embedders tests/retrieval/test_huggingface_embedder.py
git commit -m "feat: add local Hugging Face embedder"
```

---

### Task 4: One Streaming Indexer

**Files:**
- Create: `justatom/retrieval/indexer.py`
- Create: `tests/retrieval/test_indexer.py`
- Modify: `justatom/retrieval/__init__.py`

**Interfaces:**
- Consumes: `DocumentStore`, optional `Embedder`, `Document`.
- Produces: `Indexer(store, embedder=None)` and `index(documents, batch_size=64, max_parallel_writes=1) -> int`.

- [ ] **Step 1: Write failing keyword, vector, lazy-input, and failure tests**

```python
import asyncio

import pytest

from justatom.etc.schema import Document
from justatom.retrieval.indexer import Indexer


class FakeStore:
    def __init__(self):
        self.batches = []

    async def write_documents(self, documents):
        self.batches.append(list(documents))
        return len(documents)


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    async def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


def test_indexer_streams_batches_and_skips_embedding_in_keyword_mode():
    seen = []

    def documents():
        for index in range(5):
            seen.append(index)
            yield {"content": f"doc-{index}"}

    store = FakeStore()
    written = asyncio.run(Indexer(store).index(documents(), batch_size=2))
    assert written == 5
    assert seen == [0, 1, 2, 3, 4]
    assert [len(batch) for batch in store.batches] == [2, 2, 1]
    assert all(doc.embedding is None for batch in store.batches for doc in batch)


def test_indexer_attaches_one_embedding_per_document():
    store = FakeStore()
    embedder = FakeEmbedder()
    written = asyncio.run(Indexer(store, embedder).index([Document(content="a"), Document(content="b")]))
    assert written == 2
    assert embedder.calls == [["a", "b"]]
    assert [doc.embedding for doc in store.batches[0]] == [[0.0, 1.0], [1.0, 1.0]]
```

Add the failure case to the same test module:

```python
class FailingStore(FakeStore):
    async def write_documents(self, documents):
        raise DocumentStoreError("write failed")


def test_indexer_preserves_write_failure_as_cause():
    with pytest.raises(DocumentStoreError, match="batch 0") as exc_info:
        asyncio.run(Indexer(FailingStore()).index([{"content": "a"}]))
    assert isinstance(exc_info.value.__cause__, DocumentStoreError)
```

- [ ] **Step 2: Run the indexer tests and verify failure**

Run: `conda run -n justatom python -m pytest tests/retrieval/test_indexer.py -v`

Expected: FAIL because `justatom.retrieval.indexer` is missing.

- [ ] **Step 3: Implement one batched indexer**

Normalize each input exactly once:

```python
def _to_document(value: Document | dict[str, Any]) -> Document:
    if isinstance(value, Document):
        return Document.from_dict(value.to_dict())
    if isinstance(value, dict):
        return Document.from_dict(value)
    raise TypeError(f"Expected Document or dict, received {type(value).__name__}")
```

For each `more_itertools.chunked` batch, call `embed_documents` only when an embedder exists, set `document.embedding`, then write the same batch. Use an `asyncio.Semaphore` when `max_parallel_writes > 1`, cancel pending writes after the first failure, and return the sum of successful writes. Validate both numeric arguments as positive integers.

- [ ] **Step 4: Run focused tests**

Run: `conda run -n justatom python -m pytest tests/retrieval/test_indexer.py tests/retrieval/test_contracts.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit the indexer**

```bash
git add justatom/retrieval/indexer.py justatom/retrieval/__init__.py tests/retrieval/test_indexer.py
git commit -m "feat: add explicit retrieval indexer"
```

---

### Task 5: Three Retrievers and Evaluator Migration

**Files:**
- Create: `justatom/retrieval/retriever.py`
- Create: `tests/retrieval/test_retrievers.py`
- Modify: `justatom/retrieval/__init__.py`
- Modify: `justatom/running/evaluator.py:1-119`
- Modify: `tests/test_eval_metrics.py`

**Interfaces:**
- Consumes: `DocumentStore`, `Embedder`, `Retriever` protocol.
- Produces: `KeywordRetriever`, `VectorRetriever`, `HybridRetriever` with `retrieve` and `retrieve_many`.
- Changes evaluator dependency from `IRetrieverRunner.retrieve_topk` to `Retriever.retrieve_many`.

- [ ] **Step 1: Write failing stable-shape and forwarding tests**

```python
import asyncio

import pytest

from justatom.etc.schema import Document
from justatom.retrieval.retriever import HybridRetriever, KeywordRetriever, VectorRetriever


class FakeStore:
    def __init__(self):
        self.calls = []

    async def search_keywords(self, queries, **kwargs):
        self.calls.append(("keyword", list(queries), kwargs))
        return [[Document(content=f"kw:{query}")] for query in queries]

    async def search_vector(self, vectors, **kwargs):
        self.calls.append(("vector", list(vectors), kwargs))
        return [[Document(content=f"vec:{index}")] for index, _ in enumerate(vectors)]

    async def search_hybrid(self, queries, vectors, **kwargs):
        self.calls.append(("hybrid", list(queries), list(vectors), kwargs))
        return [[Document(content=f"hy:{query}")] for query in queries]


class FakeEmbedder:
    async def embed_queries(self, texts):
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


def test_keyword_retriever_has_explicit_single_and_many_shapes():
    retriever = KeywordRetriever(FakeStore())
    assert [doc.content for doc in asyncio.run(retriever.retrieve("q", top_k=3))] == ["kw:q"]
    many = asyncio.run(retriever.retrieve_many(["q1", "q2"], top_k=3))
    assert [[doc.content for doc in row] for row in many] == [["kw:q1"], ["kw:q2"]]


def test_vector_and_hybrid_forward_vectors_filters_and_alpha():
    store = FakeStore()
    embedder = FakeEmbedder()
    asyncio.run(VectorRetriever(store, embedder).retrieve_many(["q"], top_k=4, filters={"lang": "ru"}))
    asyncio.run(HybridRetriever(store, embedder, alpha=0.3).retrieve_many(["q"], top_k=5))
    assert store.calls[0] == ("vector", [[0.0, 1.0]], {"top_k": 4, "filters": {"lang": "ru"}, "include_vectors": False})
    assert store.calls[1][0] == "hybrid"
    assert store.calls[1][3]["alpha"] == 0.3


def test_retrievers_reject_invalid_top_k_and_alpha():
    with pytest.raises(ValueError, match="alpha"):
        HybridRetriever(FakeStore(), FakeEmbedder(), alpha=1.1)
    with pytest.raises(ValueError, match="top_k"):
        asyncio.run(KeywordRetriever(FakeStore()).retrieve("q", top_k=0))
```

- [ ] **Step 2: Run retriever/evaluator tests and verify failure**

Run: `conda run -n justatom python -m pytest tests/retrieval/test_retrievers.py tests/test_eval_metrics.py -v`

Expected: new retriever tests FAIL because the module is missing.

- [ ] **Step 3: Implement the retrievers and update EvaluatorRunner**

Each concrete retriever implements `retrieve_many`; use this exact single-query wrapper:

```python
async def retrieve(self, query: str, *, top_k: int = 5, **kwargs) -> list[Document]:
    results = await self.retrieve_many([query], top_k=top_k, **kwargs)
    return results[0]
```

Return `[]` for empty query sequences without calling the store/embedder. Reject `top_k <= 0`. Vector/hybrid call `embed_queries` exactly once per `retrieve_many` input.

Change `EvaluatorRunner.__init__(self, ir: Retriever)` and line 81 to:

```python
res_topk = await self.ir.retrieve_many(js_batch_queries, top_k=retrieval_top_k)
```

Update `tests/test_eval_metrics.py` to import the new `KeywordRetriever`, and rename the fake store method to `search_keywords`.

- [ ] **Step 4: Run retriever and metrics tests**

Run: `conda run -n justatom python -m pytest tests/retrieval/test_retrievers.py tests/test_eval_metrics.py -v`

Expected: all tests PASS with unchanged metric values.

- [ ] **Step 5: Commit retrievers and evaluator migration**

```bash
git add justatom/retrieval justatom/running/evaluator.py tests/retrieval/test_retrievers.py tests/test_eval_metrics.py
git commit -m "feat: add explicit retrieval strategies"
```

---

### Task 6: Owning Retrieval Runtime

**Files:**
- Create: `justatom/retrieval/runtime.py`
- Create: `tests/retrieval/test_runtime.py`
- Modify: `justatom/retrieval/__init__.py`

**Interfaces:**
- Consumes: `Indexer`, three retrievers, `SearchMode`.
- Produces: `RetrievalRuntime(store, embedder, mode, alpha=0.5)`.
- Runtime properties used later: `store`, `embedder`, `indexer`, `retriever`, `mode`.

- [ ] **Step 1: Write failing mode, lifecycle, keyword, and delegation tests**

```python
import asyncio

import pytest

from justatom.retrieval.contracts import SearchMode
from justatom.retrieval.errors import ConfigurationError
from justatom.retrieval.runtime import RetrievalRuntime


class CloseableStore:
    def __init__(self):
        self.closed = 0

    async def close(self):
        self.closed += 1

    async def search_keywords(self, queries, **kwargs):
        return [[] for _ in queries]


class CloseableEmbedder:
    def __init__(self):
        self.closed = 0

    async def close(self):
        self.closed += 1


def test_runtime_selects_mode_and_closes_resources_once():
    store = CloseableStore()
    embedder = CloseableEmbedder()
    runtime = RetrievalRuntime(store=store, embedder=embedder, mode=SearchMode.VECTOR)
    asyncio.run(runtime.close())
    asyncio.run(runtime.close())
    assert store.closed == 1
    assert embedder.closed == 1


def test_keyword_runtime_requires_no_embedder():
    runtime = RetrievalRuntime(store=CloseableStore(), embedder=None, mode=SearchMode.KEYWORD)
    assert asyncio.run(runtime.retrieve("q")) == []
    asyncio.run(runtime.close())


def test_vector_runtime_requires_embedder():
    with pytest.raises(ConfigurationError, match="embedder"):
        RetrievalRuntime(store=CloseableStore(), embedder=None, mode=SearchMode.VECTOR)
```

Add the close-failure case:

```python
class FailingCloseEmbedder(CloseableEmbedder):
    async def close(self):
        self.closed += 1
        raise RuntimeError("embedder close failed")


def test_runtime_closes_store_when_embedder_close_fails():
    store = CloseableStore()
    embedder = FailingCloseEmbedder()
    runtime = RetrievalRuntime(store=store, embedder=embedder, mode=SearchMode.VECTOR)
    with pytest.raises(RuntimeError, match="embedder close failed"):
        asyncio.run(runtime.close())
    assert embedder.closed == 1
    assert store.closed == 1
```

- [ ] **Step 2: Run runtime tests and verify failure**

Run: `conda run -n justatom python -m pytest tests/retrieval/test_runtime.py -v`

Expected: FAIL because runtime composition does not exist.

- [ ] **Step 3: Implement runtime selection and ownership**

```python
class RetrievalRuntime:
    def __init__(self, store: DocumentStore, embedder: Embedder | None, mode: SearchMode | str, alpha: float = 0.5):
        try:
            self.mode = mode if isinstance(mode, SearchMode) else SearchMode(mode)
        except ValueError as exc:
            raise ConfigurationError(f"Unsupported retrieval mode: {mode!r}") from exc
        if self.mode is not SearchMode.KEYWORD and embedder is None:
            raise ConfigurationError(f"{self.mode.value} retrieval requires an embedder")
        self.store = store
        self.embedder = embedder
        self.indexer = Indexer(store, embedder)
        match self.mode:
            case SearchMode.KEYWORD:
                self.retriever = KeywordRetriever(store)
            case SearchMode.VECTOR:
                self.retriever = VectorRetriever(store, embedder)
            case SearchMode.HYBRID:
                self.retriever = HybridRetriever(store, embedder, alpha=alpha)
        self._closed = False
```

Delegate `index`, `retrieve`, and `retrieve_many`. Implement `__aenter__`/`__aexit__`. Close embedder first, then store, continue closing the store if embedder close fails, and re-raise the first error.

- [ ] **Step 4: Run runtime and component tests**

Run: `conda run -n justatom python -m pytest tests/retrieval -v`

Expected: all retrieval package tests PASS.

- [ ] **Step 5: Commit runtime composition**

```bash
git add justatom/retrieval tests/retrieval/test_runtime.py
git commit -m "feat: compose retrieval runtime explicitly"
```

---

### Task 7: Simplify and Rename the Weaviate Store

**Files:**
- Modify: `justatom/storing/weaviate.py:71-833`
- Modify: `tests/test_scenario_configs.py:10-12,355-390`
- Create: `tests/retrieval/test_weaviate_store_api.py`

**Interfaces:**
- Produces: `WeaviateDocumentStore.connect(collection, url=None, grpc_port=50051, grpc_secure=False, **client_options)`.
- Produces protocol methods: `search_vector`, `search_keywords`, `search_hybrid`, `clear`, `close`.
- Removes: sync client, sync keyword search, `IFinder`, `Finder`, and `WeaviateDocStore` name.

- [ ] **Step 1: Write failing public API and single-client tests**

```python
import inspect

from justatom.retrieval.contracts import DocumentStore
from justatom.storing.weaviate import WeaviateDocumentStore


def test_weaviate_store_exposes_protocol_vocabulary():
    assert inspect.iscoroutinefunction(WeaviateDocumentStore.search_vector)
    assert inspect.iscoroutinefunction(WeaviateDocumentStore.search_keywords)
    assert inspect.iscoroutinefunction(WeaviateDocumentStore.search_hybrid)
    assert inspect.iscoroutinefunction(WeaviateDocumentStore.clear)
    assert not hasattr(WeaviateDocumentStore, "search_by_keywords_sync")


def test_weaviate_store_satisfies_document_store_protocol_structurally():
    store = object.__new__(WeaviateDocumentStore)
    assert isinstance(store, DocumentStore)
```

- [ ] **Step 2: Run store API tests and verify failure**

Run: `conda run -n justatom python -m pytest tests/retrieval/test_weaviate_store_api.py tests/test_scenario_configs.py -v`

Expected: FAIL because the new class/method names do not exist.

- [ ] **Step 3: Rename the store and remove the second client**

Rename `WeaviateDocStore` to `WeaviateDocumentStore`. Construct and retain only `weaviate.WeaviateAsyncClient`; delete `_sync_client`, `_ensure_sync_connection`, and `search_by_keywords_sync`.

Use these method mappings:

```text
search_by_embedding -> search_vector
search_by_keywords  -> search_keywords
search              -> search_hybrid
delete_all_documents -> clear
```

`clear()` raises `DocumentStoreError` on deletion failure instead of returning `False`. `connect` accepts the final URL contract and derives it from `WEAVIATE_HOST`/`WEAVIATE_PORT` only when `url` is absent. Delete `IFinder` and `Finder`; export only `WeaviateDocumentStore`.

Keep Weaviate-specific methods used by the server (`delete_documents`, `get_all_documents_by_ids`, collection deletion) outside the protocol.

- [ ] **Step 4: Run focused store tests**

Run: `conda run -n justatom python -m pytest tests/retrieval/test_weaviate_store_api.py tests/test_scenario_configs.py -v`

Expected: all focused tests PASS.

- [ ] **Step 5: Commit the store simplification**

```bash
git add justatom/storing/weaviate.py tests/retrieval/test_weaviate_store_api.py tests/test_scenario_configs.py
git commit -m "refactor: simplify Weaviate document store"
```

---

### Task 8: Strict Runtime Config Builder

**Files:**
- Modify: `justatom/retrieval/runtime.py`
- Modify: `justatom/retrieval/__init__.py`
- Modify: `tests/retrieval/test_runtime.py`

**Interfaces:**
- Consumes: `RetrievalRuntime`, `WeaviateDocumentStore`, both concrete embedders, `EmbeddingProfile`.
- Produces: `async build_runtime(config: Mapping[str, Any]) -> RetrievalRuntime`.
- Accepts root keys: `mode`, `alpha`, `embedding`, `store`.

- [ ] **Step 1: Write failing strict builder and cleanup tests**

```python
def test_builder_rejects_unknown_keys_before_opening_resources(monkeypatch):
    opened = []

    async def fake_connect(*args, **kwargs):
        opened.append((args, kwargs))

    monkeypatch.setattr(runtime_module.WeaviateDocumentStore, "connect", fake_connect)
    with pytest.raises(ConfigurationError, match="unknown retrieval keys"):
        asyncio.run(build_runtime({"mode": "keyword", "unknown": True, "store": {"collection": "Docs"}}))
    assert opened == []


def test_keyword_builder_never_constructs_an_embedder(monkeypatch):
    store = CloseableStore()

    async def fake_connect(*args, **kwargs):
        return store

    monkeypatch.setattr(runtime_module.WeaviateDocumentStore, "connect", fake_connect)
    monkeypatch.setattr(runtime_module, "HuggingFaceEmbedder", lambda **kwargs: pytest.fail("local embedder constructed"))
    monkeypatch.setattr(runtime_module, "OpenAICompatibleEmbedder", lambda **kwargs: pytest.fail("remote embedder constructed"))
    runtime = asyncio.run(build_runtime({"mode": "keyword", "store": {"collection": "Docs"}}))
    assert runtime.embedder is None
    asyncio.run(runtime.close())


def test_builder_closes_embedder_when_store_connection_fails(monkeypatch):
    embedder = CloseableEmbedder()
    monkeypatch.setattr(runtime_module, "HuggingFaceEmbedder", lambda **kwargs: embedder)

    async def failed_connect(*args, **kwargs):
        raise RuntimeError("weaviate unavailable")

    monkeypatch.setattr(runtime_module.WeaviateDocumentStore, "connect", failed_connect)
    config = {
        "mode": "vector",
        "embedding": {"backend": "local", "model": "local-model"},
        "store": {"collection": "Docs"},
    }
    with pytest.raises(RuntimeError, match="unavailable"):
        asyncio.run(build_runtime(config))
    assert embedder.closed == 1
```

Add constructor-capture coverage:

```python
def test_builder_maps_local_and_remote_embedding_config(monkeypatch):
    captures = []

    def fake_local(**kwargs):
        captures.append(("local", kwargs))
        return CloseableEmbedder()

    def fake_remote(**kwargs):
        captures.append(("remote", kwargs))
        return CloseableEmbedder()

    async def fake_connect(*args, **kwargs):
        return CloseableStore()

    monkeypatch.setattr(runtime_module, "HuggingFaceEmbedder", fake_local)
    monkeypatch.setattr(runtime_module, "OpenAICompatibleEmbedder", fake_remote)
    monkeypatch.setattr(runtime_module.WeaviateDocumentStore, "connect", fake_connect)

    local = asyncio.run(build_runtime({
        "mode": "vector",
        "embedding": {
            "backend": "local", "model": "local-model", "device": "mps",
            "query_prefix": "q: ", "document_prefix": "d: ",
        },
        "store": {"collection": "LocalDocs"},
    }))
    remote = asyncio.run(build_runtime({
        "mode": "hybrid",
        "alpha": 0.3,
        "embedding": {
            "backend": "openai-compatible", "base_url": "http://encoder/v1",
            "model": "remote-model", "api_key": "key", "timeout": 12,
            "encoding_format": "float", "extra_body": {"pooling": "mean"},
        },
        "store": {"collection": "RemoteDocs"},
    }))

    assert captures[0][0] == "local"
    assert captures[0][1]["model"] == "local-model"
    assert captures[0][1]["device"] == "mps"
    assert captures[0][1]["profile"].query_prefix == "q: "
    assert captures[1][0] == "remote"
    assert captures[1][1]["base_url"] == "http://encoder/v1"
    assert captures[1][1]["extra_body"] == {"pooling": "mean"}
    asyncio.run(local.close())
    asyncio.run(remote.close())
```

- [ ] **Step 2: Run builder tests and verify failure**

Run: `conda run -n justatom python -m pytest tests/retrieval/test_runtime.py -v`

Expected: FAIL because `build_runtime` is not defined.

- [ ] **Step 3: Implement exact-key validation and resource-safe construction**

Use fixed key sets:

```python
_ROOT_KEYS = {"mode", "alpha", "embedding", "store"}
_STORE_KEYS = {"collection", "url", "grpc_port", "grpc_secure"}
_LOCAL_KEYS = {
    "backend", "model", "device", "batch_size", "max_length",
    "query_prefix", "document_prefix", "skip_prefix_if_present",
}
_REMOTE_KEYS = {
    "backend", "base_url", "model", "api_key", "timeout", "batch_size",
    "max_length", "query_prefix", "document_prefix", "skip_prefix_if_present",
    "encoding_format", "extra_body",
}


def _reject_unknown(values: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ConfigurationError(f"unknown {section} keys: {', '.join(unknown)}")
```

Validate root/store/backend sections and all scalar ranges before constructing an embedder or connecting to Weaviate. Build no embedder for keyword mode. For vector/hybrid, create one `EmbeddingProfile`, branch explicitly on `local` or `openai-compatible`, then connect the store:

```python
embedder = None
try:
    embedder = _build_embedder(mode, embedding_config)
    store = await WeaviateDocumentStore.connect(
        store_config["collection"],
        url=store_config.get("url"),
        grpc_port=int(store_config.get("grpc_port", 50051)),
        grpc_secure=bool(store_config.get("grpc_secure", False)),
    )
    return RetrievalRuntime(store=store, embedder=embedder, mode=mode, alpha=alpha)
except BaseException:
    if embedder is not None:
        await embedder.close()
    raise
```

Export `build_runtime` from `justatom.retrieval`.

- [ ] **Step 4: Run all retrieval package tests**

Run: `conda run -n justatom python -m pytest tests/retrieval -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit the builder**

```bash
git add justatom/retrieval/runtime.py justatom/retrieval/__init__.py tests/retrieval/test_runtime.py
git commit -m "feat: build retrieval runtime from strict config"
```

---

### Task 9: Migrate Evaluation and Scenario Configuration

**Files:**
- Modify: `justatom/api/eval.py:1-470`
- Modify: `configs/evaluate.yaml`
- Modify: `justatom/builtins/configs/evaluate.default.yaml`
- Modify: `tests/test_eval_dataset_flow.py`
- Modify: `tests/test_scenario_configs.py`
- Modify: `tests/test_eval_streaming_integration.py`

**Interfaces:**
- Consumes: `build_runtime(retrieval_config)`, `RetrievalRuntime`, migrated `EvaluatorRunner`.
- Produces eval config keys: `retrieval.mode`, `retrieval.alpha`, `retrieval.embedding`, `retrieval.store`.
- Keeps eval-only keys: dataset, output, index batch/flush, search top-k/batch, metrics, and filters.

- [ ] **Step 1: Rewrite config and dataset-flow assertions first**

```python
def test_eval_uses_packaged_retrieval_defaults_without_repo_config():
    kwargs = resolve_eval_kwargs(config={"dataset": {"name_or_path": "demo.jsonl"}})
    retrieval = kwargs["retrieval_config"]
    assert retrieval["mode"] == "vector"
    assert retrieval["embedding"]["backend"] == "local"
    assert retrieval["store"]["collection"] == "Document"


def test_eval_remote_config_reaches_runtime_builder():
    kwargs = resolve_eval_kwargs(
        config={
            "retrieval": {
                "mode": "vector",
                "embedding": {
                    "backend": "openai-compatible",
                    "base_url": "http://encoder:8000/v1",
                    "model": "remote-model",
                },
                "store": {"collection": "RemoteDocs", "url": "http://weaviate:8080"},
            }
        }
    )
    assert kwargs["retrieval_config"]["embedding"]["base_url"] == "http://encoder:8000/v1"


def test_eval_consumes_collection_tag_before_runtime_builder():
    kwargs = resolve_eval_kwargs(config={
        "dataset": {"id": "justatom"},
        "retrieval": {
            "mode": "vector",
            "embedding": {"backend": "local", "model": "intfloat/multilingual-e5-small"},
            "store": {"collection": "Document", "tag": "ablation-lr-1e5"},
        },
    })
    assert kwargs["collection_tag"] == "AblationLr1e5"
    assert kwargs["retrieval_config"]["store"]["collection"].endswith("SEPTagAblationLr1e5")
    assert "tag" not in kwargs["retrieval_config"]["store"]
```

In `tests/test_eval_dataset_flow.py`, replace `RunningService` patches with an async `fake_build_runtime` returning a fake runtime whose `index`, `store.count_documents`, `retriever`, and `close` calls are recorded. Preserve the existing assertions that the dataset is opened once and labels are exhausted when an existing index skips writes.

- [ ] **Step 2: Run eval/scenario tests and verify failures**

Run: `conda run -n justatom python -m pytest tests/test_scenario_configs.py tests/test_eval_dataset_flow.py -v`

Expected: FAIL because config still emits `model_name_or_path`/`search_pipeline` and eval still calls `RunningService`.

- [ ] **Step 3: Migrate config shape and eval lifecycle**

Use this packaged/repository config structure:

```yaml
retrieval:
  mode: vector
  alpha: 0.5
  embedding:
    backend: local
    model: null
    device: auto
    batch_size: 64
    max_length: 512
    query_prefix: null
    document_prefix: null
  store:
    collection: Document
    tag: null
    url: null
    grpc_port: 50051
```

Keep `search.top_k`, `search.batch_size`, `index.batch_size`, and `index.flush_collection`. Remove top-level `model`, `weaviate`, and `collection` sections.

`_cfg_to_main_kwargs` returns one `retrieval_config` mapping. It consumes and removes `retrieval.store.tag`, returns the normalized value separately as `collection_tag`, and uses it when resolving the collection name. Auto collection naming uses `retrieval.embedding.model` and writes the resolved name back to `retrieval.store.collection`. The strict builder therefore never receives the eval-only `tag` key. E5 auto-prefix detection updates a copied embedding config before calling the builder.

Replace the service call with:

```python
runtime = await build_runtime(retrieval_config)
try:
    existing = await runtime.store.count_documents()
    if flush_collection:
        await runtime.store.clear()
        existing = 0
    if existing == 0:
        await runtime.index(docs_iter, batch_size=index_batch_size)
    # preserve label exhaustion and metric calculation
    evaluator = EvaluatorRunner(ir=runtime.retriever)
finally:
    await runtime.close()
```

Rename CLI `--search-pipeline` to `--search-mode`; use choices `keyword`, `vector`, `hybrid`. Add explicit retrieval options `--embedding-backend`, `--embedding-base-url`, `--embedding-api-key`, `--embedding-model`, `--query-prefix`, `--document-prefix`, `--collection-name`, `--weaviate-url`, and `--weaviate-grpc-port`. Overlay them into `retrieval` without accepting old names.

- [ ] **Step 4: Run eval unit and integration tests**

Run: `conda run -n justatom python -m pytest tests/test_scenario_configs.py tests/test_eval_dataset_flow.py tests/test_eval_metrics.py -v`

Run: `conda run -n justatom python -m pytest tests/test_eval_streaming_integration.py -m integration -v`

Expected: all unit tests PASS; Weaviate integration PASS.

- [ ] **Step 5: Commit eval migration**

```bash
git add justatom/api/eval.py configs/evaluate.yaml justatom/builtins/configs/evaluate.default.yaml tests/test_scenario_configs.py tests/test_eval_dataset_flow.py tests/test_eval_streaming_integration.py
git commit -m "refactor: run evaluation through retrieval runtime"
```

---

### Task 10: One Runtime per HTTP Application

**Files:**
- Modify: `justatom/api/run.py:1-240`
- Create: `configs/serve.yaml`
- Create: `justatom/builtins/configs/serve.default.yaml`
- Create: `tests/test_run_api.py`

**Interfaces:**
- Consumes: `build_runtime`, `RetrievalRuntime`.
- Produces: `create_app(config=None, runtime=None, start_mq=True) -> Quart` for dependency-injected tests and production startup.
- Search request accepts: `text`, `top_k`, optional `filter_by`.
- Index request accepts: `dataset_name_or_docs`, optional `batch_size`.

- [ ] **Step 1: Write failing application lifecycle and endpoint tests**

```python
import asyncio

from justatom.api.run import create_app
from justatom.etc.schema import Document


class FakeStore:
    def __init__(self):
        self.cleared = 0

    async def count_documents(self):
        return 7

    async def clear(self):
        self.cleared += 1


class FakeRuntime:
    def __init__(self):
        self.store = FakeStore()
        self.closed = 0
        self.indexed = []

    async def retrieve(self, query, **kwargs):
        return [Document(content=f"result:{query}")]

    async def index(self, documents, **kwargs):
        self.indexed.extend(list(documents))
        return len(self.indexed)

    async def close(self):
        self.closed += 1


def test_app_reuses_one_runtime_for_search_index_and_shutdown():
    async def scenario():
        runtime = FakeRuntime()
        app = create_app(runtime=runtime, start_mq=False)
        async with app.test_app() as test_app:
            client = test_app.test_client()
            search = await client.post("/searching", json={"text": "cats", "top_k": 3})
            assert (await search.get_json())["docs"][0]["content"] == "result:cats"
            indexed = await client.post("/indexing", json={"dataset_name_or_docs": [{"content": "doc"}]})
            assert (await indexed.get_json())["total_docs"] == 7
        assert runtime.closed == 1

    asyncio.run(scenario())
```

Add an API validation helper and assertions for forbidden composition fields:

```python
def _reject_unknown_fields(payload: dict, allowed: set[str]):
    unknown = sorted(set(payload) - allowed)
    if unknown:
        return {"error": f"unsupported fields: {', '.join(unknown)}"}, 400
    return None


def test_search_rejects_runtime_composition_fields():
    async def scenario():
        app = create_app(runtime=FakeRuntime(), start_mq=False)
        async with app.test_app() as test_app:
            response = await test_app.test_client().post(
                "/searching",
                json={"text": "q", "collection_name": "other", "search_by": "hybrid", "top_p": 0.9},
            )
            assert response.status_code == 400
            assert "collection_name" in (await response.get_json())["error"]

    asyncio.run(scenario())
```

- [ ] **Step 2: Run API tests and verify failure**

Run: `conda run -n justatom python -m pytest tests/test_run_api.py -v`

Expected: FAIL because `create_app` and lifecycle-owned runtime do not exist.

- [ ] **Step 3: Refactor the Quart application**

Build exactly one runtime in `before_serving` when no runtime was injected, store it at `app.extensions["retrieval_runtime"]`, and close it in `after_serving`. Do not close the event loop manually.

Handlers retrieve the runtime with:

```python
def _runtime() -> RetrievalRuntime:
    return app.extensions["retrieval_runtime"]
```

`/searching` allows only `text`, `top_k`, and `filter_by`; `/indexing` allows only `dataset_name_or_docs` and `batch_size`. `/searching` calls `runtime.retrieve`, `/indexing` calls `runtime.index`, and `/delete` calls `runtime.store.clear`. Remove `/deletebyids` and `/patching`; document-id administration and cross-collection copying are outside retrieval runtime. Remove the unused Redis-backed session and its write from search. Keep the RabbitMQ consumer lifecycle behind `start_mq`, and await cancellation during shutdown.

Load `configs/serve.yaml` through `load_scenario_config("serve")`; its `retrieval` section uses the same schema as evaluate config.

- [ ] **Step 4: Run API and retrieval tests**

Run: `conda run -n justatom python -m pytest tests/test_run_api.py tests/retrieval -v`

Expected: all tests PASS and the fake runtime closes once.

- [ ] **Step 5: Commit the server lifecycle migration**

```bash
git add justatom/api/run.py configs/serve.yaml justatom/builtins/configs/serve.default.yaml tests/test_run_api.py
git commit -m "refactor: own retrieval runtime in API lifecycle"
```

---

### Task 11: Migrate Clustering and Delete the Legacy Retrieval Stack

**Files:**
- Modify: `justatom/running/clusters.py:1-170`
- Modify: `justatom/running/mask.py:155-175`
- Modify: `justatom/storing/mask.py:82-213`
- Modify: `justatom/configuring/prime.py:52-145`
- Create: `tests/test_clustering_embedding_adapter.py`
- Delete: `justatom/running/service.py`
- Delete: `justatom/running/indexer.py`
- Delete: `justatom/running/retriever.py`
- Delete: `justatom/running/embeddings/`
- Delete: `justatom/builtins/configs/embeddings.yaml`
- Delete: `tests/test_retriever_shape_and_factory.py`
- Modify: remaining tests/imports found by the removal scan.

**Interfaces:**
- Consumes: new `Embedder` and `embed_documents`.
- Produces: `EmbeddingBackendAdapter` for BERTopic's synchronous `BaseEmbedder` interface.
- Removes: `IEmbeddingClient`, `IRetrieverRunner`, `IIndexerRunner`, `INNDocStore`, all caches/factories/services.

- [ ] **Step 1: Add a failing clustering adapter test**

Create `tests/test_clustering_embedding_adapter.py` with:

```python
import numpy as np

from justatom.running.clusters import EmbeddingBackendAdapter


class FakeEmbedder:
    async def embed_documents(self, texts):
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


def test_bertopic_adapter_uses_document_embedding_role():
    result = EmbeddingBackendAdapter(FakeEmbedder()).embed(["a", "b"])
    np.testing.assert_array_equal(result, np.asarray([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32))
```

- [ ] **Step 2: Run the adapter test and removal scan before changes**

Run: `conda run -n justatom python -m pytest tests -k "bertopic_adapter" -v`

Run: `rg -n "RunningService|IndexerAPI|RetrieverAPI|RetrieverApi|IEmbeddingClient|INNDocStore|IRetrieverRunner|IIndexerRunner|running\.embeddings|Finder as Weaviate" justatom tests`

Expected: adapter test FAIL; scan reports the legacy modules and imports.

- [ ] **Step 3: Migrate adapter and delete old modules**

Rename `IEmbeddingClientBackend` to `EmbeddingBackendAdapter`, type it with the new `Embedder`, and call:

```python
vectors = self._run_async(self.embedder.embed_documents(documents))
```

Retain the existing safe sync-to-async bridge because BERTopic owns a synchronous callback. Remove only the old retrieval/indexer interfaces from `running.mask` and only `INNDocStore` from `storing.mask`.

After `api/run.py` no longer imports `Config`, remove the dead `api` block from `_default_config_data`, its compatibility normalization branch, and the `IConfig.api` annotation in `configuring/prime.py`. Keep loguru and training configuration unchanged.

Delete all listed legacy files. Remove cache environment variables and dead imports. Do not add re-export modules or aliases.

- [ ] **Step 4: Prove the old stack is gone and run non-integration tests**

Run: `test -z "$(rg -l "RunningService|IndexerAPI|RetrieverAPI|RetrieverApi|IEmbeddingClient|INNDocStore|IRetrieverRunner|IIndexerRunner|running\.embeddings|Finder as Weaviate" justatom tests || true)"`

Run: `test -z "$(rg -l "RUNNING_(MODEL|TOKENIZER|PROCESSOR|ENCODER|EMBED)_CACHE_SIZE" . --glob '!docs/superpowers/**' || true)"`

Run: `conda run -n justatom python -m pytest tests -m "not integration and not network"`

Expected: both scans exit 0 and all non-integration tests PASS.

- [ ] **Step 5: Commit legacy removal**

```bash
git add -A justatom tests
git commit -m "refactor: remove legacy retrieval wrappers and caches"
```

---

### Task 12: Migrate Scripts and Public Documentation

**Files:**
- Modify: `scripts/run_pipeline.sh`
- Modify: `scripts/run_benchmark.sh`
- Modify: `scripts/run_eval_model_on_datasets.sh`
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/modules/runtime.md`
- Modify: `docs/modules/storage.md`
- Modify: `docs/launch-guide.md`

**Interfaces:**
- Documents: direct components, local runtime, remote runtime, context-managed cleanup, and new config shape.
- Scripts expose `--search-mode` with `keyword|vector|hybrid`.

- [ ] **Step 1: Add shell validation and stale-documentation scans**

Run these commands before editing and record that they fail/find old content:

```bash
rg -n "IndexerAPI|RetrieverApi|RunningService|search-pipeline|search\.pipeline|pipeline: embedding|pipeline: keywords" README.md docs scripts
bash -n scripts/run_pipeline.sh
bash -n scripts/run_benchmark.sh
bash -n scripts/run_eval_model_on_datasets.sh
```

Expected: shell syntax is valid; `rg` finds legacy API/mode names.

- [ ] **Step 2: Update all shell interfaces atomically**

Rename shell state and flags:

```text
SEARCH_PIPELINE -> SEARCH_MODE
--search-pipeline -> --search-mode
embedding -> vector
keywords -> keyword
```

Pipeline eval commands pass `--search-mode`, `--embedding-model`, `--query-prefix`, `--document-prefix`, `--weaviate-url`, and `--collection-name`. Benchmark forwarding accepts only `keyword`, `vector`, or `hybrid` and exits 2 for other values.

- [ ] **Step 3: Replace README and module examples with copyable new APIs**

Include both deployment forms:

```python
store = await WeaviateDocumentStore.connect("JustAtom", url="http://localhost:2211")
embedder = HuggingFaceEmbedder(
    "intfloat/multilingual-e5-small",
    device="auto",
    profile=EmbeddingProfile(query_prefix="query: ", document_prefix="passage: "),
)
async with RetrievalRuntime(store, embedder, mode="hybrid", alpha=0.5) as runtime:
    await runtime.index(documents)
    results = await runtime.retrieve("как устроен индекс", top_k=10)
```

```python
embedder = OpenAICompatibleEmbedder(
    base_url="http://ubuntu-box:8000/v1",
    model="deployed-embedding-model",
    api_key=os.getenv("EMBEDDING_API_KEY"),
)
```

State explicitly that an OpenAI-compatible Triton deployment runs on its supported host, while the justatom client has no Triton dependency and works on macOS.

- [ ] **Step 4: Validate scripts, docs, and stale references**

Run: `bash -n scripts/run_pipeline.sh scripts/run_benchmark.sh scripts/run_eval_model_on_datasets.sh`

Run: `test -z "$(rg -l "IndexerAPI|RetrieverApi|RunningService|search-pipeline|search\.pipeline|pipeline: embedding|pipeline: keywords" README.md docs scripts --glob '!docs/superpowers/**' || true)"`

Run: `conda run -n justatom python -m mkdocs build --strict`

Expected: shell syntax and docs build PASS; stale-reference scan exits 0.

- [ ] **Step 5: Commit scripts and documentation**

```bash
git add scripts README.md docs
git commit -m "docs: publish simplified retrieval runtime"
```

---

### Task 13: Full Platform-Oriented Verification

**Files:**
- Modify only files changed by deterministic formatters or defects exposed by verification.

**Interfaces:**
- Verifies all acceptance criteria from the design spec.

- [ ] **Step 1: Format the complete tracked Python set**

Run: `conda run -n justatom make fix-format`

Expected: Black and isort complete without error.

- [ ] **Step 2: Run formatting and lint gates**

Run: `conda run -n justatom make format-check`

Run: `conda run -n justatom pylint justatom --errors-only --disable=import-error,not-callable`

Expected: both commands exit 0.

- [ ] **Step 3: Run the full local unit suite**

Run: `conda run -n justatom python -m pytest tests -m "not integration and not network"`

Expected: all tests PASS.

- [ ] **Step 4: Run Weaviate integration and docs**

Run: `docker compose up -d weaviate`

Run: `conda run -n justatom python -m pytest tests -m integration`

Run: `conda run -n justatom python -m mkdocs build --strict`

Expected: integration tests and docs build PASS.

- [ ] **Step 5: Verify import graph and removed architecture**

```bash
conda run -n justatom python -c "import sys; from justatom.retrieval.embedders.openai_compatible import OpenAICompatibleEmbedder; assert 'torch' not in sys.modules; assert 'tritonclient' not in sys.modules"
test -z "$(rg -l "RunningService|IndexerAPI|RetrieverAPI|RetrieverApi|IEmbeddingClient|INNDocStore|IRetrieverRunner|IIndexerRunner|RUNNING_.*CACHE_SIZE" justatom tests || true)"
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 6: Review and commit verification-only changes**

If formatting changed tracked files, commit exactly those changes:

```bash
git add -A
git commit -m "chore: finalize retrieval runtime refactor"
```

If `git status --short` is empty, do not create an empty commit.
