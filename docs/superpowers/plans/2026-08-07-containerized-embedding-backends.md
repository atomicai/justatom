# Containerized API and Embedding Backends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a model-free JustAtom retrieval API image plus CPU and CUDA OpenAI-compatible embedding images while preserving native macOS MPS inference.

**Architecture:** `justatom-api` owns retrieval orchestration and Weaviate connections but calls embeddings over HTTP. CPU, CUDA, native MPS, and third-party inference servers all satisfy the same `/v1/embeddings` contract. Each embedding process owns exactly one `HuggingFaceEmbedder` and one model load.

**Tech Stack:** Python 3.12, Quart, Hypercorn, httpx, PyTorch 2.8, Transformers, Weaviate 1.34.2, Docker BuildKit, Docker Compose profiles, pytest.

## Global Constraints

- Dockerfiles are named exactly `Dockerfile.api`, `Dockerfile.embedder.cpu`, and `Dockerfile.embedder.cuda`.
- Images are named `justatom-api`, `justatom-embedder-cpu`, and `justatom-embedder-cuda`.
- `justatom-api` must install neither Torch nor model weights.
- CPU and CUDA services expose the same `GET /health`, `GET /v1/models`, and `POST /v1/embeddings` behavior.
- The default model is exactly `Qwen/Qwen3-Embedding-0.6B`.
- The default embedding batch size is `8`; the default maximum sequence length is `512`.
- The API listens on port `5555`; built-in embedding services listen on port `8000`; Weaviate uses HTTP `2211` and gRPC `50051`.
- Hypercorn runs exactly one worker for both API and embedding services.
- Native MPS remains outside Docker and uses the same embedding service module with `EMBEDDING_DEVICE=mps`.
- CPU and CUDA Compose profiles are mutually exclusive.
- Docker Compose `2.20` or newer is required for optional health-checked profile dependencies.
- Model weights and credentials must not be copied into image layers.
- Model cache data is persisted at `/cache/huggingface` through a named volume.
- API JSON remains UTF-8 and upstream/backend failures remain sanitized.
- Every behavior change follows red-green TDD and ends in a focused commit.

---

## File Structure

### New production modules

- `justatom/api/hypercorn_server.py`: construct one-worker Hypercorn configuration and run a Quart ASGI app.
- `justatom/api/serve.py`: load the retrieval config from environment and start the production retrieval API.
- `justatom/api/embedding_server.py`: validate embedding server settings, own one embedder, and expose OpenAI-compatible routes.
- `justatom/api/serve_embeddings.py`: start the built-in embedding service on CPU, CUDA, or native MPS.

### New deployment artifacts

- `Dockerfile.api`: model-free retrieval API image.
- `Dockerfile.embedder.cpu`: CPU PyTorch embedding image.
- `Dockerfile.embedder.cuda`: CUDA PyTorch embedding image.
- `.dockerignore`: prevent caches, credentials, model weights, worktrees, and research outputs from entering build context.
- `configs/serve.docker.yaml`: strict OpenAI-compatible retrieval configuration for Compose.

### Modified deployment artifacts

- `docker-compose.yaml`: add API, CPU profile, CUDA profile, health checks, network alias, and Hugging Face cache volume.
- `pyproject.toml`: define focused `serve` and `embedder` runtime extras without changing the existing training extra.

### Tests and documentation

- `tests/test_hypercorn_server.py`: production server configuration and delegation.
- `tests/test_api_server_entrypoint.py`: config path, environment validation, MQ default, and bind behavior.
- `tests/test_embedding_server.py`: settings, OpenAI schema, ordering, lifecycle, UTF-8, and errors.
- `tests/test_embedding_server_entrypoint.py`: environment-driven embedding process startup.
- `tests/test_runtime_extras.py`: package dependency boundary checks.
- `tests/test_docker_assets.py`: Dockerfile and Compose structure checks.
- `tests/integration/test_embedding_server_contract.py`: real HTTP contract against the built-in service with an injected deterministic embedder.
- `tests/fixtures/openai_embedding_stub.py`: deterministic external OpenAI-compatible endpoint for API-image smoke testing.
- `scripts/smoke_api_external_backend.sh`: model-free API image against the fake endpoint and real Weaviate.
- `scripts/smoke_containerized_retrieval.sh`: end-to-end CPU profile smoke test.
- `scripts/smoke_native_embedding.sh`: native Apple Silicon MPS lifecycle and contract smoke test.
- `docs/launch-guide.md`: native MPS, CPU Compose, CUDA Compose, and external backend commands.
- `docs/architecture.md`: process boundary and model ownership.

---

### Task 1: Production Hypercorn Runner and Retrieval API Entrypoint

**Files:**
- Create: `justatom/api/hypercorn_server.py`
- Create: `justatom/api/serve.py`
- Modify: `justatom/api/run.py:122-128`
- Test: `tests/test_hypercorn_server.py`
- Test: `tests/test_api_server_entrypoint.py`
- Test: `tests/test_run_api.py`

**Interfaces:**
- Consumes: `justatom.api.run.create_app(config_path: str | Path | None, config: dict[str, Any] | None, runtime: RetrievalRuntime | None, start_mq: bool) -> Quart`.
- Produces: `build_hypercorn_config(host: str, port: int) -> hypercorn.config.Config`.
- Produces: `serve_app(app: Quart, *, host: str, port: int) -> None`.
- Produces: `build_retrieval_app(env: Mapping[str, str] | None = None) -> Quart`.

- [ ] **Step 1: Write failing tests for config-path injection and unresolved variables**

Add to `tests/test_run_api.py`:

```python
def test_create_app_loads_explicit_config_path(monkeypatch, tmp_path):
    path = tmp_path / "serve.yaml"
    path.write_text(
        "retrieval:\n"
        "  mode: keyword\n"
        "  store:\n"
        "    collection: ExplicitConfig\n",
        encoding="utf-8",
    )
    app = create_app(config_path=path, runtime=FakeRuntime(), start_mq=False)
    assert app.extensions["retrieval_config"]["store"]["collection"] == "ExplicitConfig"


def test_create_app_rejects_unresolved_environment_placeholders(tmp_path):
    path = tmp_path / "serve.yaml"
    path.write_text(
        "retrieval:\n"
        "  mode: vector\n"
        "  embedding:\n"
        "    backend: openai-compatible\n"
        "    base_url: ${EMBEDDING_BASE_URL}\n"
        "    model: model\n"
        "  store:\n"
        "    collection: Docs\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="EMBEDDING_BASE_URL"):
        create_app(config_path=path, runtime=FakeRuntime(), start_mq=False)


def test_create_app_sanitizes_embedding_backend_failures():
    class FailingRuntime(FakeRuntime):
        async def retrieve(self, query, **kwargs):
            raise EmbeddingBackendError(f"upstream secret for {query}")

    async def scenario():
        app = create_app(runtime=FailingRuntime(), start_mq=False)
        async with app.test_app() as test_app:
            response = await test_app.test_client().post(
                "/searching", json={"text": "private query"}
            )
            body = await response.get_data()
            assert response.status_code == 502
            assert await response.get_json() == {"error": "embedding backend unavailable"}
            assert b"upstream secret" not in body
            assert b"private query" not in body

    asyncio.run(scenario())
```

Add these test imports:

```python
from justatom.retrieval.errors import ConfigurationError, EmbeddingBackendError
```

- [ ] **Step 2: Run the config tests and verify RED**

Run:

```bash
conda run -n justatom python -m pytest \
  tests/test_run_api.py::test_create_app_loads_explicit_config_path \
  tests/test_run_api.py::test_create_app_rejects_unresolved_environment_placeholders \
  tests/test_run_api.py::test_create_app_sanitizes_embedding_backend_failures -q
```

Expected: FAIL because `create_app` has no `config_path` argument and no embedding error mapping.

- [ ] **Step 3: Add config-path support and strict placeholder rejection**

In `justatom/api/run.py`, add:

```python
import re
from pathlib import Path

from loguru import logger

from justatom.retrieval.errors import ConfigurationError, EmbeddingBackendError, EmbeddingResponseError

_ENV_PLACEHOLDER = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _unresolved_environment(node: object) -> set[str]:
    if isinstance(node, dict):
        return set().union(*(_unresolved_environment(value) for value in node.values()), set())
    if isinstance(node, list):
        return set().union(*(_unresolved_environment(value) for value in node), set())
    if isinstance(node, str):
        return set(_ENV_PLACEHOLDER.findall(node))
    return set()
```

Change the factory signature and loading block to:

```python
def create_app(
    config_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
    runtime: RetrievalRuntime | None = None,
    start_mq: bool = True,
) -> Quart:
    scenario_config = load_scenario_config("serve", config_path=config_path, config=config)
    unresolved = sorted(_unresolved_environment(scenario_config))
    if unresolved:
        raise ConfigurationError(f"unresolved environment variables: {', '.join(unresolved)}")
```

After constructing `app`, register one sanitized mapping for both remote HTTP failures and malformed embedding responses:

```python
    @app.errorhandler(EmbeddingBackendError)
    @app.errorhandler(EmbeddingResponseError)
    async def embedding_failure(error):
        logger.error("embedding request failed [{}]", type(error).__name__)
        return {"error": "embedding backend unavailable"}, 502
```

The handler must not log `str(error)`, request payloads, query text, API keys, or upstream response bodies.

- [ ] **Step 4: Run existing and new API tests**

Run: `conda run -n justatom python -m pytest tests/test_run_api.py -q`

Expected: all tests PASS.

- [ ] **Step 5: Write failing Hypercorn and retrieval entrypoint tests**

Create `tests/test_hypercorn_server.py`:

```python
import asyncio

from justatom.api import hypercorn_server


def test_build_hypercorn_config_uses_one_worker_and_explicit_bind():
    config = hypercorn_server.build_hypercorn_config("0.0.0.0", 5555)
    assert config.bind == ["0.0.0.0:5555"]
    assert config.workers == 1


def test_serve_app_delegates_to_hypercorn(monkeypatch):
    calls = []

    async def fake_serve(app, config):
        calls.append((app, config.bind, config.workers))

    monkeypatch.setattr(hypercorn_server, "serve", fake_serve)
    app = object()
    asyncio.run(hypercorn_server.serve_app(app, host="127.0.0.1", port=7777))
    assert calls == [(app, ["127.0.0.1:7777"], 1)]
```

Create `tests/test_api_server_entrypoint.py`:

```python
from justatom.api import serve as module


def test_build_retrieval_app_uses_container_defaults(monkeypatch):
    calls = []

    def fake_create_app(**kwargs):
        calls.append(kwargs)
        return "app"

    monkeypatch.setattr(module, "create_app", fake_create_app)
    assert module.build_retrieval_app({}) == "app"
    assert calls == [{"config_path": "/etc/justatom/serve.yaml", "start_mq": False}]


def test_build_retrieval_app_allows_explicit_mq_boolean(monkeypatch):
    calls = []
    monkeypatch.setattr(module, "create_app", lambda **kwargs: calls.append(kwargs) or "app")
    module.build_retrieval_app({"JUSTATOM_CONFIG": "/cfg/serve.yaml", "JUSTATOM_START_MQ": "true"})
    assert calls == [{"config_path": "/cfg/serve.yaml", "start_mq": True}]
```

- [ ] **Step 6: Run the entrypoint tests and verify RED**

Run:

```bash
conda run -n justatom python -m pytest \
  tests/test_hypercorn_server.py tests/test_api_server_entrypoint.py -q
```

Expected: collection errors because the modules do not exist.

- [ ] **Step 7: Implement the production server modules**

Create `justatom/api/hypercorn_server.py`:

```python
from __future__ import annotations

from hypercorn.asyncio import serve
from hypercorn.config import Config


def build_hypercorn_config(host: str, port: int) -> Config:
    config = Config()
    config.bind = [f"{host}:{port}"]
    config.workers = 1
    config.accesslog = "-"
    config.errorlog = "-"
    return config


async def serve_app(app, *, host: str, port: int) -> None:
    await serve(app, build_hypercorn_config(host, port))
```

Create `justatom/api/serve.py`:

```python
from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping

from justatom.api.hypercorn_server import serve_app
from justatom.api.run import create_app
from justatom.retrieval.errors import ConfigurationError


def _boolean(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be a boolean")


def build_retrieval_app(env: Mapping[str, str] | None = None):
    values = os.environ if env is None else env
    config_path = values.get("JUSTATOM_CONFIG", "/etc/justatom/serve.yaml")
    start_mq = _boolean(values.get("JUSTATOM_START_MQ", "false"), "JUSTATOM_START_MQ")
    return create_app(config_path=config_path, start_mq=start_mq)


def main() -> None:
    app = build_retrieval_app()
    asyncio.run(serve_app(app, host="0.0.0.0", port=5555))


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Run Task 1 tests and commit**

Run:

```bash
conda run -n justatom python -m pytest \
  tests/test_run_api.py tests/test_hypercorn_server.py tests/test_api_server_entrypoint.py -q
```

Expected: PASS.

Commit:

```bash
git add justatom/api/run.py justatom/api/hypercorn_server.py justatom/api/serve.py \
  tests/test_run_api.py tests/test_hypercorn_server.py tests/test_api_server_entrypoint.py
git commit -m "feat: add production retrieval API entrypoint"
```

---

### Task 2: Embedding Server Settings and One-time Model Construction

**Files:**
- Create: `justatom/api/embedding_server.py`
- Test: `tests/test_embedding_server.py`

**Interfaces:**
- Consumes: `EmbeddingProfile`, `Embedder`, and `HuggingFaceEmbedder`.
- Produces: immutable `EmbeddingServerSettings`.
- Produces: `EmbeddingServerSettings.from_env(env: Mapping[str, str] | None = None) -> EmbeddingServerSettings`.
- Produces: `build_local_embedder(settings: EmbeddingServerSettings) -> Embedder`.

- [ ] **Step 1: Write failing settings tests**

Create `tests/test_embedding_server.py` with:

```python
import pytest

from justatom.api import embedding_server as module
from justatom.retrieval.errors import ConfigurationError


def test_settings_use_qwen_defaults():
    settings = module.EmbeddingServerSettings.from_env({})
    assert settings.model == "Qwen/Qwen3-Embedding-0.6B"
    assert settings.device == "cpu"
    assert settings.batch_size == 8
    assert settings.max_length == 512


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"EMBEDDING_MODEL": " "}, "EMBEDDING_MODEL"),
        ({"EMBEDDING_BATCH_SIZE": "0"}, "EMBEDDING_BATCH_SIZE"),
        ({"EMBEDDING_BATCH_SIZE": "true"}, "EMBEDDING_BATCH_SIZE"),
        ({"EMBEDDING_MAX_LENGTH": "-1"}, "EMBEDDING_MAX_LENGTH"),
    ],
)
def test_settings_reject_invalid_environment(env, message):
    with pytest.raises(ConfigurationError, match=message):
        module.EmbeddingServerSettings.from_env(env)


def test_build_local_embedder_uses_one_empty_prefix_profile(monkeypatch):
    calls = []

    class FakeEmbedder:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(module, "HuggingFaceEmbedder", FakeEmbedder)
    settings = module.EmbeddingServerSettings.from_env(
        {"EMBEDDING_MODEL": "model", "EMBEDDING_DEVICE": "mps", "EMBEDDING_BATCH_SIZE": "4"}
    )
    embedder = module.build_local_embedder(settings)
    assert isinstance(embedder, FakeEmbedder)
    assert calls[0]["model"] == "model"
    assert calls[0]["device"] == "mps"
    assert calls[0]["profile"].query_prefix == ""
    assert calls[0]["profile"].document_prefix == ""
    assert calls[0]["profile"].batch_size == 4
```

- [ ] **Step 2: Run the settings tests and verify RED**

Run: `conda run -n justatom python -m pytest tests/test_embedding_server.py -q`

Expected: import error because `embedding_server` does not exist.

- [ ] **Step 3: Implement strict settings and construction**

Create the initial `justatom/api/embedding_server.py`:

```python
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from justatom.retrieval.contracts import Embedder, EmbeddingProfile
from justatom.retrieval.embedders.huggingface import HuggingFaceEmbedder
from justatom.retrieval.errors import ConfigurationError


def _positive_integer(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{name} must be a positive integer") from error
    if parsed <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return parsed


@dataclass(frozen=True)
class EmbeddingServerSettings:
    model: str
    device: str
    batch_size: int
    max_length: int

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "EmbeddingServerSettings":
        values = os.environ if env is None else env
        model = values.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B").strip()
        if not model:
            raise ConfigurationError("EMBEDDING_MODEL must be non-empty")
        device = values.get("EMBEDDING_DEVICE", "cpu").strip()
        if not device:
            raise ConfigurationError("EMBEDDING_DEVICE must be non-empty")
        return cls(
            model=model,
            device=device,
            batch_size=_positive_integer(values.get("EMBEDDING_BATCH_SIZE", "8"), "EMBEDDING_BATCH_SIZE"),
            max_length=_positive_integer(values.get("EMBEDDING_MAX_LENGTH", "512"), "EMBEDDING_MAX_LENGTH"),
        )


def build_local_embedder(settings: EmbeddingServerSettings) -> Embedder:
    return HuggingFaceEmbedder(
        model=settings.model,
        device=settings.device,
        profile=EmbeddingProfile(batch_size=settings.batch_size, max_length=settings.max_length),
    )
```

- [ ] **Step 4: Run tests and commit**

Run: `conda run -n justatom python -m pytest tests/test_embedding_server.py -q`

Expected: PASS.

Commit:

```bash
git add justatom/api/embedding_server.py tests/test_embedding_server.py
git commit -m "feat: configure local embedding service"
```

---

### Task 3: OpenAI-compatible Embedding HTTP Contract

**Files:**
- Modify: `justatom/api/embedding_server.py`
- Modify: `tests/test_embedding_server.py`

**Interfaces:**
- Consumes: `EmbeddingServerSettings` and an optional injected `Embedder`.
- Produces: `create_embedding_app(settings: EmbeddingServerSettings | None = None, embedder: Embedder | None = None) -> Quart`.
- Produces HTTP: `GET /health`, `GET /v1/models`, and `POST /v1/embeddings`.

- [ ] **Step 1: Add a deterministic fake embedder and failing happy-path tests**

Append to `tests/test_embedding_server.py`:

```python
import asyncio


class FakeEmbedder:
    def __init__(self, error=None):
        self.calls = []
        self.closed = 0
        self.error = error

    async def embed_documents(self, texts):
        self.calls.append(list(texts))
        if self.error is not None:
            raise self.error
        return [[float(index), float(len(text))] for index, text in enumerate(texts)]

    async def embed_queries(self, texts):
        return await self.embed_documents(texts)

    async def close(self):
        self.closed += 1


def test_embedding_endpoint_returns_ordered_openai_response_and_utf8():
    async def scenario():
        embedder = FakeEmbedder()
        settings = module.EmbeddingServerSettings("модель", "cpu", 8, 512)
        app = module.create_embedding_app(settings=settings, embedder=embedder)
        async with app.test_app() as test_app:
            response = await test_app.test_client().post(
                "/v1/embeddings",
                json={"model": "модель", "input": ["первый", "second"], "encoding_format": "float"},
            )
            body = await response.get_data()
            payload = await response.get_json()
        assert response.status_code == 200
        assert "модель".encode() in body
        assert b"\\u043c" not in body
        assert [item["index"] for item in payload["data"]] == [0, 1]
        assert [item["embedding"] for item in payload["data"]] == [[0.0, 6.0], [1.0, 6.0]]
        assert embedder.calls == [["первый", "second"]]
        assert embedder.closed == 1

    asyncio.run(scenario())


def test_models_and_health_report_configured_model():
    async def scenario():
        settings = module.EmbeddingServerSettings("model", "cpu", 8, 512)
        app = module.create_embedding_app(settings=settings, embedder=FakeEmbedder())
        async with app.test_app() as test_app:
            client = test_app.test_client()
            health = await client.get("/health")
            models = await client.get("/v1/models")
            assert await health.get_json() == {"status": "ok", "model": "model"}
            assert (await models.get_json())["data"][0]["id"] == "model"

    asyncio.run(scenario())
```

- [ ] **Step 2: Run happy-path tests and verify RED**

Run:

```bash
conda run -n justatom python -m pytest \
  tests/test_embedding_server.py::test_embedding_endpoint_returns_ordered_openai_response_and_utf8 \
  tests/test_embedding_server.py::test_models_and_health_report_configured_model -q
```

Expected: FAIL because `create_embedding_app` does not exist.

- [ ] **Step 3: Implement lifecycle and successful routes**

Add to `justatom/api/embedding_server.py`:

```python
from typing import Any

from quart import Quart, request

from justatom.retrieval.contracts import validate_embeddings


def create_embedding_app(
    settings: EmbeddingServerSettings | None = None,
    embedder: Embedder | None = None,
) -> Quart:
    resolved = settings or EmbeddingServerSettings.from_env()
    app = Quart(__name__, static_folder=None)
    app.json.ensure_ascii = False
    app.extensions["embedding_settings"] = resolved
    if embedder is not None:
        app.extensions["embedder"] = embedder

    @app.before_serving
    async def start() -> None:
        if "embedder" not in app.extensions:
            app.extensions["embedder"] = build_local_embedder(resolved)

    @app.after_serving
    async def stop() -> None:
        owned = app.extensions.pop("embedder", None)
        if owned is not None:
            await owned.close()

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": resolved.model}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": resolved.model, "object": "model", "owned_by": "justatom"}]}

    @app.post("/v1/embeddings")
    async def embeddings():
        payload: Any = await request.get_json(silent=True)
        error, texts = _validate_embedding_request(payload, resolved)
        if error is not None:
            return error
        vectors = await app.extensions["embedder"].embed_documents(texts)
        return {
            "object": "list",
            "model": resolved.model,
            "data": [
                {"object": "embedding", "index": index, "embedding": vector}
                for index, vector in enumerate(vectors)
            ],
        }

    return app
```

Implement `_validate_embedding_request` as a pure function:

```python
def _error(message: str, status: int, error_type: str):
    return {"error": {"message": message, "type": error_type}}, status


def _validate_embedding_request(payload: Any, settings: EmbeddingServerSettings):
    if not isinstance(payload, dict):
        return (_error("request body must be a JSON object", 400, "invalid_request_error"), None)
    unknown = sorted(set(payload) - {"model", "input", "encoding_format"})
    if unknown:
        return (_error(f"unsupported fields: {', '.join(unknown)}", 400, "invalid_request_error"), None)
    if payload.get("model") != settings.model:
        return (_error("requested model is not available", 404, "model_not_found"), None)
    encoding = payload.get("encoding_format", "float")
    if encoding != "float":
        return (_error("encoding_format must be 'float'", 400, "invalid_request_error"), None)

    source = payload.get("input")
    if isinstance(source, str):
        texts = [source]
    elif isinstance(source, list):
        texts = source
    else:
        return (_error("input must be a string or list of strings", 400, "invalid_request_error"), None)
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        return (_error("input strings must be non-empty", 400, "invalid_request_error"), None)
    if len(texts) > settings.batch_size:
        return (_error("input exceeds configured batch size", 413, "request_too_large"), None)
    return None, texts
```

- [ ] **Step 4: Run happy-path tests**

Run: `conda run -n justatom python -m pytest tests/test_embedding_server.py -q`

Expected: current tests PASS.

- [ ] **Step 5: Add failing validation and sanitization tests**

Append parameterized cases:

```python
@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"model": "other", "input": ["x"]},
        {"model": "model", "input": []},
        {"model": "model", "input": ["x", "y", "z"]},
        {"model": "model", "input": ["x", 7]},
        {"model": "model", "input": ["x"], "encoding_format": "base64"},
        {"model": "model", "input": ["x"], "dimensions": 128},
    ],
)
def test_embedding_endpoint_rejects_invalid_requests(payload):
    async def scenario():
        settings = module.EmbeddingServerSettings("model", "cpu", 2, 512)
        app = module.create_embedding_app(settings=settings, embedder=FakeEmbedder())
        async with app.test_app() as test_app:
            response = await test_app.test_client().post("/v1/embeddings", json=payload)
            assert 400 <= response.status_code < 500
            assert "error" in await response.get_json()

    asyncio.run(scenario())


def test_embedding_endpoint_sanitizes_backend_failure():
    async def scenario():
        secret = "secret-backend-detail"
        settings = module.EmbeddingServerSettings("model", "cpu", 8, 512)
        app = module.create_embedding_app(settings=settings, embedder=FakeEmbedder(RuntimeError(secret)))
        app.config["PROPAGATE_EXCEPTIONS"] = False
        async with app.test_app() as test_app:
            response = await test_app.test_client().post(
                "/v1/embeddings", json={"model": "model", "input": ["русский"]}
            )
            body = await response.get_data()
            assert response.status_code == 500
            assert secret.encode() not in body
            assert "embedding backend failed".encode() in body
            assert "русский".encode() not in body

    asyncio.run(scenario())


class FixedVectorsEmbedder(FakeEmbedder):
    def __init__(self, vectors):
        super().__init__()
        self.vectors = vectors

    async def embed_documents(self, texts):
        self.calls.append(list(texts))
        return self.vectors


@pytest.mark.parametrize(
    ("texts", "vectors"),
    [
        (["x"], []),
        (["x", "y"], [[1.0], [1.0, 2.0]]),
        (["x"], [[float("nan")]]),
    ],
)
def test_embedding_endpoint_rejects_invalid_backend_vectors(texts, vectors):
    async def scenario():
        settings = module.EmbeddingServerSettings("model", "cpu", 8, 512)
        app = module.create_embedding_app(settings=settings, embedder=FixedVectorsEmbedder(vectors))
        async with app.test_app() as test_app:
            response = await test_app.test_client().post(
                "/v1/embeddings", json={"model": "model", "input": texts}
            )
            assert response.status_code == 500
            assert await response.get_json() == {
                "error": {"message": "embedding backend failed", "type": "server_error"}
            }

    asyncio.run(scenario())
```

- [ ] **Step 6: Run validation tests and verify RED**

Run: `conda run -n justatom python -m pytest tests/test_embedding_server.py -q`

Expected: one or more validation cases or the sanitized 500 assertion FAIL.

- [ ] **Step 7: Complete strict validation and the 500 handler**

Register a handler for otherwise unhandled HTTP 500 failures:

```python
@app.errorhandler(500)
async def internal_error(error):
    logger.error("unhandled embedding server error [{}]", type(error).__name__)
    return _error("embedding backend failed", 500, "server_error")
```

Wrap the backend call directly so its exception message cannot become a framework response:

```python
try:
    vectors = validate_embeddings(
        await app.extensions["embedder"].embed_documents(texts),
        expected_count=len(texts),
    )
except Exception as error:
    logger.error("embedding backend failed [{}]", type(error).__name__)
    return _error("embedding backend failed", 500, "server_error")
```

Never log the payload, API key, input text, or exception message.

- [ ] **Step 8: Verify all embedding server tests and commit**

Run: `conda run -n justatom python -m pytest tests/test_embedding_server.py -q`

Expected: PASS.

Commit:

```bash
git add justatom/api/embedding_server.py tests/test_embedding_server.py
git commit -m "feat: expose OpenAI-compatible embedding service"
```

---

### Task 4: Cancellation-safe Shutdown and Embedding Entrypoint

**Files:**
- Modify: `justatom/api/embedding_server.py`
- Create: `justatom/api/serve_embeddings.py`
- Modify: `tests/test_embedding_server.py`
- Create: `tests/test_embedding_server_entrypoint.py`

**Interfaces:**
- Consumes: `create_embedding_app`, `EmbeddingServerSettings`, and `serve_app`.
- Produces: `build_embedding_app(env: Mapping[str, str] | None = None) -> Quart`.
- Produces: executable module `python -m justatom.api.serve_embeddings`.

- [ ] **Step 1: Add failing one-build and cancellation-safe close tests**

Add a factory injection parameter to the intended app interface and test it:

```python
def test_embedding_app_builds_model_once_and_closes_once(monkeypatch):
    async def scenario():
        built = []
        embedder = FakeEmbedder()

        def build(settings):
            built.append(settings)
            return embedder

        settings = module.EmbeddingServerSettings("model", "cpu", 8, 512)
        app = module.create_embedding_app(settings=settings, embedder_factory=build)
        async with app.test_app() as test_app:
            client = test_app.test_client()
            await client.post("/v1/embeddings", json={"model": "model", "input": ["one"]})
            await client.post("/v1/embeddings", json={"model": "model", "input": ["two"]})
        assert built == [settings]
        assert embedder.calls == [["one"], ["two"]]
        assert embedder.closed == 1

    asyncio.run(scenario())
```

Add a blocking close fake and cancellation test:

```python
class BlockingCloseEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__()
        self.close_started = asyncio.Event()
        self.close_release = asyncio.Event()

    async def close(self):
        self.closed += 1
        self.close_started.set()
        await self.close_release.wait()


def test_cancelled_embedding_shutdown_finishes_one_close():
    async def scenario():
        embedder = BlockingCloseEmbedder()
        settings = module.EmbeddingServerSettings("model", "cpu", 8, 512)
        app = module.create_embedding_app(settings=settings, embedder=embedder)
        stop = app.after_serving_funcs[0]
        stop_task = asyncio.create_task(stop())
        await embedder.close_started.wait()
        stop_task.cancel()
        embedder.close_release.set()
        with pytest.raises(asyncio.CancelledError):
            await stop_task
        await stop()
        assert embedder.closed == 1

    asyncio.run(scenario())
```

- [ ] **Step 2: Run lifecycle tests and verify RED**

Run: `conda run -n justatom python -m pytest tests/test_embedding_server.py -q`

Expected: FAIL because `embedder_factory` and shared close-task ownership do not exist.

- [ ] **Step 3: Implement one factory call and shielded shutdown**

Change the factory signature to:

```python
EmbedderFactory = Callable[[EmbeddingServerSettings], Embedder]


def create_embedding_app(
    settings: EmbeddingServerSettings | None = None,
    embedder: Embedder | None = None,
    embedder_factory: EmbedderFactory = build_local_embedder,
) -> Quart:
```

Store and await one close task:

```python
async def _finish_close(app: Quart) -> None:
    close_task = app.extensions.get("embedding_close_task")
    if close_task is None:
        embedder = app.extensions.pop("embedder", None)
        if embedder is None:
            return
        close_task = asyncio.create_task(embedder.close())
        app.extensions["embedding_close_task"] = close_task
    try:
        await asyncio.shield(close_task)
    except asyncio.CancelledError:
        await asyncio.shield(close_task)
        raise
```

The `after_serving` hook calls only `await _finish_close(app)`.

- [ ] **Step 4: Add failing entrypoint tests**

Create `tests/test_embedding_server_entrypoint.py`:

```python
from justatom.api import serve_embeddings as module


def test_build_embedding_app_passes_environment_settings(monkeypatch):
    calls = []
    monkeypatch.setattr(module, "create_embedding_app", lambda settings: calls.append(settings) or "app")
    app = module.build_embedding_app(
        {
            "EMBEDDING_MODEL": "model",
            "EMBEDDING_DEVICE": "cuda:0",
            "EMBEDDING_BATCH_SIZE": "4",
            "EMBEDDING_MAX_LENGTH": "256",
        }
    )
    assert app == "app"
    assert calls[0].model == "model"
    assert calls[0].device == "cuda:0"


def test_embedding_server_port_is_8000(monkeypatch):
    calls = []
    monkeypatch.setattr(module, "build_embedding_app", lambda: "app")

    async def fake_serve(app, *, host, port):
        calls.append((app, host, port))

    monkeypatch.setattr(module, "serve_app", fake_serve)
    module.main()
    assert calls == [("app", "0.0.0.0", 8000)]
```

- [ ] **Step 5: Run entrypoint tests and verify RED**

Run: `conda run -n justatom python -m pytest tests/test_embedding_server_entrypoint.py -q`

Expected: import error because `serve_embeddings` does not exist.

- [ ] **Step 6: Implement the embedding entrypoint**

Create `justatom/api/serve_embeddings.py`:

```python
from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping

from justatom.api.embedding_server import EmbeddingServerSettings, create_embedding_app
from justatom.api.hypercorn_server import serve_app


def build_embedding_app(env: Mapping[str, str] | None = None):
    values = os.environ if env is None else env
    return create_embedding_app(EmbeddingServerSettings.from_env(values))


def main() -> None:
    asyncio.run(serve_app(build_embedding_app(), host="0.0.0.0", port=8000))


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run Task 4 tests and commit**

Run:

```bash
conda run -n justatom python -m pytest \
  tests/test_embedding_server.py tests/test_embedding_server_entrypoint.py tests/test_hypercorn_server.py -q
```

Expected: PASS.

Commit:

```bash
git add justatom/api/embedding_server.py justatom/api/serve_embeddings.py \
  tests/test_embedding_server.py tests/test_embedding_server_entrypoint.py
git commit -m "feat: own embedding server lifecycle"
```

---

### Task 5: Focused Runtime Dependency Extras

**Files:**
- Modify: `pyproject.toml:13-42`
- Create: `tests/test_runtime_extras.py`

**Interfaces:**
- Produces package extra `justatom[serve]` for the model-free retrieval API.
- Produces package extra `justatom[embedder]` for local model service code excluding the platform-specific Torch wheel.
- Preserves existing `justatom[torch]` and `justatom[test]` behavior.

- [ ] **Step 1: Write failing dependency-boundary tests**

Create `tests/test_runtime_extras.py`:

```python
import tomllib
from pathlib import Path


def _extras():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]


def test_serve_extra_has_http_storage_and_data_dependencies_without_torch():
    serve = "\n".join(_extras()["serve"]).lower()
    assert "quart" in serve
    assert "hypercorn" in serve
    assert "weaviate-client" in serve
    assert "polars" in serve
    assert "torch" not in serve
    assert "transformers" not in serve


def test_embedder_extra_has_local_runner_dependencies_without_torch_wheel():
    embedder = "\n".join(_extras()["embedder"]).lower()
    assert "transformers" in embedder
    assert "pytorch-lightning" in embedder
    assert not any(line.startswith("torch==") for line in embedder.splitlines())
```

- [ ] **Step 2: Run dependency tests and verify RED**

Run: `conda run -n justatom python -m pytest tests/test_runtime_extras.py -q`

Expected: FAIL with missing `serve` and `embedder` keys.

- [ ] **Step 3: Add explicit extras**

Add to `[project.optional-dependencies]`:

```toml
serve = [
  "quart==0.19.6",
  "hypercorn>=0.17,<1",
  "weaviate-client==4.11.3",
  "PyYAML>=6,<7",
  "loguru==0.7.2",
  "python-dotenv==1.0.0",
  "polars==1.38.1",
  "datasets==2.18.0",
  "pyarrow>=12,<24",
  "smart-open==6.4.0",
  "more-itertools==10.1.0",
]
embedder = [
  "quart==0.19.6",
  "hypercorn>=0.17,<1",
  "transformers>=4.44,<6",
  "pytorch-lightning==2.2.1",
  "torchmetrics==1.3.2",
  "setuptools<81",
  "PyYAML>=6,<7",
  "loguru==0.7.2",
  "polars==1.38.1",
  "more-itertools==10.1.0",
]
```

Do not add `torch` to either new extra. Dockerfiles install the CPU or CUDA wheel explicitly before installing `justatom[embedder]`.

- [ ] **Step 4: Verify clean-install import closures**

Run in disposable virtual environments:

```bash
python -m venv .tmp_runs/venv-serve
.tmp_runs/venv-serve/bin/pip install -e '.[serve]'
.tmp_runs/venv-serve/bin/python -c 'from justatom.api.serve import build_retrieval_app'
.tmp_runs/venv-serve/bin/pip check

python -m venv .tmp_runs/venv-embedder
.tmp_runs/venv-embedder/bin/pip install torch==2.8.0 -e '.[embedder]'
.tmp_runs/venv-embedder/bin/python -c 'from justatom.api.embedding_server import create_embedding_app'
.tmp_runs/venv-embedder/bin/pip check
```

Expected: imports and `pip check` succeed. If an import reports a missing runtime package, add that exact package with the version constraint already used in `requirements.txt`, rerun from a fresh venv, and record it in the same commit.

- [ ] **Step 5: Run tests and commit**

Run: `conda run -n justatom python -m pytest tests/test_runtime_extras.py -q`

Expected: PASS.

Commit:

```bash
git add pyproject.toml tests/test_runtime_extras.py
git commit -m "build: split API and embedder dependencies"
```

---

### Task 6: Three Docker Images and Build-context Safety

**Files:**
- Create: `.dockerignore`
- Create: `Dockerfile.api`
- Create: `Dockerfile.embedder.cpu`
- Create: `Dockerfile.embedder.cuda`
- Create: `tests/test_docker_assets.py`

**Interfaces:**
- Consumes: executable modules `justatom.api.serve` and `justatom.api.serve_embeddings`.
- Produces: three images with the names and responsibilities defined in Global Constraints.

- [ ] **Step 1: Write failing static Docker asset tests**

Create `tests/test_docker_assets.py`:

```python
from pathlib import Path


def _read(path):
    return Path(path).read_text(encoding="utf-8")


def test_api_image_is_model_free_and_runs_production_entrypoint():
    dockerfile = _read("Dockerfile.api")
    assert ".[serve]" in dockerfile
    assert "justatom.api.serve" in dockerfile
    assert "torch" not in dockerfile.lower()
    assert "Qwen" not in dockerfile


def test_embedding_images_share_endpoint_and_select_platform_torch():
    cpu = _read("Dockerfile.embedder.cpu")
    cuda = _read("Dockerfile.embedder.cuda")
    assert "justatom.api.serve_embeddings" in cpu
    assert "justatom.api.serve_embeddings" in cuda
    assert "download.pytorch.org/whl/cpu" in cpu
    assert "download.pytorch.org/whl/cu128" in cuda
    assert "EMBEDDING_DEVICE=cpu" in cpu
    assert "EMBEDDING_DEVICE=cuda:0" in cuda


def test_dockerignore_excludes_credentials_weights_and_worktrees():
    ignored = _read(".dockerignore").splitlines()
    assert ".env" in ignored
    assert ".cache/" in ignored
    assert "weights/" in ignored
    assert "models/" in ignored
    assert "*.safetensors" in ignored
    assert "*.gguf" in ignored
    assert ".worktrees/" in ignored
    assert ".tmp_runs/" in ignored
    assert "phd.paper/" in ignored
```

- [ ] **Step 2: Run static tests and verify RED**

Run: `conda run -n justatom python -m pytest tests/test_docker_assets.py -q`

Expected: file-not-found failures.

- [ ] **Step 3: Add `.dockerignore`**

Create:

```text
.git/
.github/
.worktrees/
.tmp_runs/
.data/
.cache/
.env
.env.*
weights/
models/
*.safetensors
*.gguf
*.pt
*.pth
phd.paper/
site/
tmp/
__pycache__/
*.py[cod]
*.log
```

- [ ] **Step 4: Add the model-free API Dockerfile**

Create `Dockerfile.api`:

```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    JUSTATOM_CONFIG=/etc/justatom/serve.yaml \
    JUSTATOM_START_MQ=false

WORKDIR /app
COPY pyproject.toml README.md ./
COPY justatom ./justatom
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir '.[serve]' \
    && useradd --create-home --uid 10001 justatom

USER 10001:10001
EXPOSE 5555
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=6 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5555/', timeout=2)"]
CMD ["python", "-m", "justatom.api.serve"]
```

The file must not install from `requirements.txt`, install a Torch extra, set a model identifier, or copy `.env`.

- [ ] **Step 5: Add CPU and CUDA embedding Dockerfiles**

Create `Dockerfile.embedder.cpu`:

```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B \
    EMBEDDING_DEVICE=cpu \
    EMBEDDING_BATCH_SIZE=8 \
    EMBEDDING_MAX_LENGTH=512

WORKDIR /app
COPY pyproject.toml README.md ./
COPY justatom ./justatom
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
       --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 \
    && python -m pip install --no-cache-dir '.[embedder]' \
    && useradd --create-home --uid 10001 justatom \
    && mkdir -p /cache/huggingface \
    && chown -R 10001:10001 /cache/huggingface

USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=120s --retries=30 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]
CMD ["python", "-m", "justatom.api.serve_embeddings"]
```

Create `Dockerfile.embedder.cuda`:

```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B \
    EMBEDDING_DEVICE=cuda:0 \
    EMBEDDING_BATCH_SIZE=8 \
    EMBEDDING_MAX_LENGTH=512

WORKDIR /app
COPY pyproject.toml README.md ./
COPY justatom ./justatom
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
       --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 \
    && python -m pip install --no-cache-dir '.[embedder]' \
    && useradd --create-home --uid 10001 justatom \
    && mkdir -p /cache/huggingface \
    && chown -R 10001:10001 /cache/huggingface

USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=120s --retries=30 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]
CMD ["python", "-m", "justatom.api.serve_embeddings"]
```

- [ ] **Step 6: Run static tests and build the API image**

Run:

```bash
conda run -n justatom python -m pytest tests/test_docker_assets.py -q
docker build -f Dockerfile.api -t justatom-api:test .
docker run --rm justatom-api:test python -c \
  'import importlib.util; assert importlib.util.find_spec("torch") is None'
```

Expected: tests PASS, image builds, and Torch assertion passes.

- [ ] **Step 7: Build and inspect the CPU image**

Run:

```bash
docker build -f Dockerfile.embedder.cpu -t justatom-embedder-cpu:test .
docker run --rm justatom-embedder-cpu:test python -c \
  'import torch; from justatom.api.embedding_server import create_embedding_app; assert not torch.cuda.is_available()'
```

Expected: image builds and imports succeed.

- [ ] **Step 8: Validate the CUDA Dockerfile without requiring a Mac GPU**

Run:

```bash
docker build --platform linux/amd64 -f Dockerfile.embedder.cuda \
  -t justatom-embedder-cuda:test .
```

Expected: image builds. Do not run CUDA inference on macOS.

- [ ] **Step 9: Commit Docker image artifacts**

```bash
git add .dockerignore Dockerfile.api Dockerfile.embedder.cpu Dockerfile.embedder.cuda tests/test_docker_assets.py
git commit -m "build: add API CPU and CUDA images"
```

---

### Task 7: Docker Compose Topology and Runtime Configuration

**Files:**
- Create: `configs/serve.docker.yaml`
- Modify: `docker-compose.yaml`
- Modify: `tests/test_docker_assets.py`

**Interfaces:**
- Consumes: three Dockerfiles and named executables from Tasks 1-6.
- Produces services `api`, `embedder-cpu`, `embedder-cuda`, and existing `weaviate`.
- Produces network alias `embedder` for exactly one active managed backend.

- [ ] **Step 1: Add failing Compose structure tests**

Append to `tests/test_docker_assets.py`:

```python
import yaml


def test_compose_defines_api_and_mutually_exclusive_embedding_profiles():
    compose = yaml.safe_load(_read("docker-compose.yaml"))
    services = compose["services"]
    assert services["api"]["build"]["dockerfile"] == "Dockerfile.api"
    assert services["api"]["ports"] == ["${JUSTATOM_API_PORT:-5555}:5555"]
    assert services["api"]["extra_hosts"] == ["host.docker.internal:host-gateway"]
    assert services["weaviate"]["ports"] == [
        "${WEAVIATE_HTTP_PORT:-2211}:2211",
        "${WEAVIATE_GRPC_PORT:-50051}:50051",
    ]
    assert services["embedder-cpu"]["profiles"] == ["cpu"]
    assert services["embedder-cuda"]["profiles"] == ["cuda"]
    assert services["api"]["depends_on"]["embedder-cpu"]["required"] is False
    assert services["api"]["depends_on"]["embedder-cuda"]["required"] is False
    assert services["embedder-cuda"]["deploy"]["resources"]["reservations"]["devices"][0]["capabilities"] == ["gpu"]
    assert "huggingface-cache" in compose["volumes"]


def test_docker_serve_config_uses_internal_weaviate_and_embedding_alias():
    config = yaml.safe_load(_read("configs/serve.docker.yaml"))
    retrieval = config["retrieval"]
    assert retrieval["embedding"]["backend"] == "openai-compatible"
    assert retrieval["embedding"]["base_url"] == "${EMBEDDING_BASE_URL}"
    assert retrieval["store"]["url"] == "http://weaviate:2211"
```

- [ ] **Step 2: Run Compose tests and verify RED**

Run: `conda run -n justatom python -m pytest tests/test_docker_assets.py -q`

Expected: missing services and config failures.

- [ ] **Step 3: Add strict Docker retrieval config**

Create `configs/serve.docker.yaml`:

```yaml
retrieval:
  mode: vector
  alpha: 0.5
  embedding:
    backend: openai-compatible
    base_url: ${EMBEDDING_BASE_URL}
    model: ${EMBEDDING_MODEL}
    timeout: 60.0
    batch_size: 8
    max_length: 512
    query_prefix: "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
    skip_prefix_if_present: true
  store:
    collection: Document
    url: http://weaviate:2211
    grpc_port: 50051
    grpc_secure: false
```

Authenticated deployments mount an override config containing `api_key: ${EMBEDDING_API_KEY}`. The repository default intentionally omits the optional key.

- [ ] **Step 4: Add Compose services and health checks**

Add these service mappings while preserving the existing Redis service:

```yaml
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    image: justatom-api
    ports:
      - "${JUSTATOM_API_PORT:-5555}:5555"
    environment:
      JUSTATOM_CONFIG: /etc/justatom/serve.yaml
      JUSTATOM_START_MQ: "false"
      JUSTATOM_EMBEDDING_PROFILES: ${COMPOSE_PROFILES:-external}
      EMBEDDING_BASE_URL: ${EMBEDDING_BASE_URL:-http://embedder:8000/v1}
      EMBEDDING_MODEL: ${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}
    volumes:
      - ./configs/serve.docker.yaml:/etc/justatom/serve.yaml:ro
    extra_hosts:
      - "host.docker.internal:host-gateway"
    depends_on:
      weaviate:
        condition: service_healthy
      embedder-cpu:
        condition: service_healthy
        required: false
      embedder-cuda:
        condition: service_healthy
        required: false
    restart: unless-stopped

  embedder-cpu:
    profiles: ["cpu"]
    build:
      context: .
      dockerfile: Dockerfile.embedder.cpu
    image: justatom-embedder-cpu
    ports:
      - "${EMBEDDING_PORT:-8000}:8000"
    environment:
      EMBEDDING_MODEL: ${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}
      EMBEDDING_DEVICE: cpu
      HF_HOME: /cache/huggingface
      HF_TOKEN: ${HF_TOKEN:-}
    volumes:
      - huggingface-cache:/cache/huggingface
    networks:
      default:
        aliases: ["embedder"]
    restart: unless-stopped

  embedder-cuda:
    profiles: ["cuda"]
    build:
      context: .
      dockerfile: Dockerfile.embedder.cuda
    image: justatom-embedder-cuda
    ports:
      - "${EMBEDDING_PORT:-8000}:8000"
    environment:
      EMBEDDING_MODEL: ${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}
      EMBEDDING_DEVICE: cuda:0
      HF_HOME: /cache/huggingface
      HF_TOKEN: ${HF_TOKEN:-}
    volumes:
      - huggingface-cache:/cache/huggingface
    networks:
      default:
        aliases: ["embedder"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: ["gpu"]
    restart: unless-stopped
```

Add this healthcheck to `weaviate`:

```yaml
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://127.0.0.1:2211/v1/.well-known/ready"]
      interval: 5s
      timeout: 3s
      retries: 20
```

Replace the existing Weaviate host-port mapping with overrideable ports so the smoke project can coexist with a developer's running stack:

```yaml
    ports:
      - "${WEAVIATE_HTTP_PORT:-2211}:2211"
      - "${WEAVIATE_GRPC_PORT:-50051}:50051"
```

Add `huggingface-cache:` to the top-level `volumes` mapping. The embedder Dockerfiles already provide their own `/health` image healthcheck. Exposing port `8000` from both managed services also makes accidental simultaneous profile startup fail at host-port allocation after the explicit preflight guard.

- [ ] **Step 5: Add an explicit profile preflight test and command**

Add this to `justatom/api/serve.py`:

```python
from collections.abc import Mapping, Sequence


def validate_embedding_profiles(profiles: Sequence[str]) -> None:
    selected = {profile.strip().lower() for profile in profiles if profile.strip()}
    if {"cpu", "cuda"}.issubset(selected):
        raise ConfigurationError("cpu and cuda embedding profiles are mutually exclusive")
```

At the start of `build_retrieval_app`, parse and validate:

```python
profiles = values.get("JUSTATOM_EMBEDDING_PROFILES", "external").split(",")
validate_embedding_profiles(profiles)
```

Compose passes:

```yaml
JUSTATOM_EMBEDDING_PROFILES: ${COMPOSE_PROFILES:-external}
```

Document profile launches with `COMPOSE_PROFILES=cpu` or `COMPOSE_PROFILES=cuda`, which both selects the service and makes validation visible inside the API container.

Test:

```python
def test_retrieval_entrypoint_rejects_cpu_and_cuda_profiles_together():
    with pytest.raises(ConfigurationError, match="mutually exclusive"):
        validate_embedding_profiles(["cpu", "cuda"])
```

- [ ] **Step 6: Validate all Compose modes**

Run:

```bash
EMBEDDING_BASE_URL=http://host.docker.internal:8000/v1 docker compose config --quiet
COMPOSE_PROFILES=cpu docker compose config --quiet
COMPOSE_PROFILES=cuda docker compose config --quiet
```

Expected: all three valid single-backend configurations pass.

Run:

```bash
JUSTATOM_EMBEDDING_PROFILES=cpu,cuda conda run -n justatom python -c \
  'from justatom.api.serve import build_retrieval_app; build_retrieval_app()'
```

Expected: non-zero exit with `cpu and cuda embedding profiles are mutually exclusive` before network startup.

- [ ] **Step 7: Run tests and commit**

Run:

```bash
conda run -n justatom python -m pytest \
  tests/test_docker_assets.py tests/test_api_server_entrypoint.py tests/test_scenario_configs.py -q
```

Expected: PASS.

Commit:

```bash
git add configs/serve.docker.yaml docker-compose.yaml justatom/api/serve.py \
  tests/test_docker_assets.py tests/test_api_server_entrypoint.py
git commit -m "feat: compose retrieval and embedding services"
```

---

### Task 8: HTTP Contract Integration and CPU End-to-end Smoke Test

**Files:**
- Create: `tests/integration/test_embedding_server_contract.py`
- Create: `scripts/smoke_containerized_retrieval.sh`
- Modify: `tests/test_docker_assets.py`

**Interfaces:**
- Consumes: built-in embedding app, `OpenAICompatibleEmbedder`, Compose CPU profile, retrieval API, and Weaviate.
- Produces: repeatable contract and deployment verification.

- [ ] **Step 1: Add the in-process HTTP contract test**

Create `tests/integration/test_embedding_server_contract.py`. Use `httpx.ASGITransport` so the test exercises the real HTTP request/response boundary without allocating a flaky test port:

```python
from __future__ import annotations

import asyncio
from collections.abc import Sequence

import httpx
import pytest

from justatom.api.embedding_server import EmbeddingServerSettings, create_embedding_app
from justatom.retrieval.embedders.openai_compatible import OpenAICompatibleEmbedder


class DeterministicEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.closed = 0

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        batch = list(texts)
        self.calls.append(batch)
        return [[float(len(text)), 1.0] for text in batch]

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return await self.embed_documents(texts)

    async def close(self) -> None:
        self.closed += 1


@pytest.mark.integration
def test_builtin_server_satisfies_openai_compatible_embedder_contract():
    async def scenario() -> None:
        backend = DeterministicEmbedder()
        settings = EmbeddingServerSettings("model", "cpu", 8, 512)
        app = create_embedding_app(settings=settings, embedder=backend)
        transport = httpx.ASGITransport(app=app)

        async with app.test_app():
            client = OpenAICompatibleEmbedder(
                base_url="http://embedder.test/v1",
                model="model",
                transport=transport,
            )
            try:
                first = await client.embed_documents(["one", "two"])
                second = await client.embed_queries(["three"])
            finally:
                await client.close()

        assert first == [[3.0, 1.0], [3.0, 1.0]]
        assert second == [[5.0, 1.0]]
        assert backend.calls == [["one", "two"], ["three"]]
        assert backend.closed == 1

    asyncio.run(scenario())
```

- [ ] **Step 2: Run the integration contract test**

Run:

```bash
conda run -n justatom python -m pytest \
  tests/integration/test_embedding_server_contract.py -q
```

Expected: PASS with the exact vectors, calls, and close count shown above. A failure blocks the container work and must be fixed in the already-owned `create_embedding_app` or `OpenAICompatibleEmbedder`; do not add another client implementation.

- [ ] **Step 3: Write the CPU Compose smoke script**

Create `scripts/smoke_containerized_retrieval.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROJECT="justatom-smoke-$(date +%s)-$$"
export COMPOSE_PROJECT_NAME="$PROJECT"
export COMPOSE_PROFILES=cpu
export JUSTATOM_API_PORT="${JUSTATOM_API_PORT:-15555}"
export EMBEDDING_PORT="${EMBEDDING_PORT:-18000}"
export WEAVIATE_HTTP_PORT="${WEAVIATE_HTTP_PORT:-13211}"
export WEAVIATE_GRPC_PORT="${WEAVIATE_GRPC_PORT:-15051}"

COMPOSE=(docker compose -p "$PROJECT")

cleanup() {
  "${COMPOSE[@]}" down -v --remove-orphans >/dev/null 2>&1 || true
}

fail() {
  printf 'smoke failure: %s\n' "$1" >&2
  "${COMPOSE[@]}" logs --no-color >&2 || true
  exit 1
}

wait_http() {
  local name="$1"
  local url="$2"
  local attempt
  for attempt in $(seq 1 300); do
    if curl --fail --silent --show-error "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  fail "$name did not become ready at $url"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

command -v docker >/dev/null || fail "docker is required"
command -v curl >/dev/null || fail "curl is required"
command -v jq >/dev/null || fail "jq is required"

"${COMPOSE[@]}" up -d --build weaviate embedder-cpu api
wait_http "embedding health" "http://127.0.0.1:${EMBEDDING_PORT}/health"
wait_http "embedding models" "http://127.0.0.1:${EMBEDDING_PORT}/v1/models"
wait_http "retrieval API" "http://127.0.0.1:${JUSTATOM_API_PORT}/"

index_response="$(
  curl --fail --silent --show-error \
    -H 'Content-Type: application/json' \
    -d '{"dataset_name_or_docs":[{"content":"Memory bank расширяет множество негативных примеров при контрастном обучении информационного поиска.","meta":{"topic":"retrieval"}},{"content":"Qwen3 Embedding преобразует запросы и документы в плотные векторные представления.","meta":{"topic":"embeddings"}},{"content":"Weaviate хранит векторы документов и выполняет поиск ближайших соседей.","meta":{"topic":"storage"}}]}' \
    "http://127.0.0.1:${JUSTATOM_API_PORT}/indexing"
)" || fail "indexing request failed"
printf '%s' "$index_response" | jq -e '.total_docs == 3' >/dev/null || fail "expected three indexed documents"

search_response="$(
  curl --fail --silent --show-error \
    -H 'Content-Type: application/json' \
    -d '{"text":"Что даёт банк негативов при обучении поиска?","top_k":2}' \
    "http://127.0.0.1:${JUSTATOM_API_PORT}/searching"
)" || fail "first search request failed"
printf '%s' "$search_response" | jq -e '.docs[0].meta.topic == "retrieval"' >/dev/null \
  || fail "retrieval document was not ranked first"
if printf '%s' "$search_response" | grep -Eq '\\u[0-9a-fA-F]{4}'; then
  fail "API response escaped readable UTF-8 text"
fi
printf '%s' "$search_response" | grep -Fq 'банк негативов' \
  || fail "Russian result text is missing"

second_response="$(
  curl --fail --silent --show-error \
    -H 'Content-Type: application/json' \
    -d '{"text":"Где хранятся векторы документов?","top_k":2}' \
    "http://127.0.0.1:${JUSTATOM_API_PORT}/searching"
)" || fail "second search request failed"
printf '%s' "$second_response" | jq -e '.docs | length > 0' >/dev/null \
  || fail "second search returned no documents"

model_loads="$(
  "${COMPOSE[@]}" logs --no-color embedder-cpu \
    | grep -F -c 'Loading from huggingface hub via' || true
)"
[[ "$model_loads" == "1" ]] || fail "expected one model load, observed $model_loads"

printf 'containerized retrieval smoke passed: project=%s\n' "$PROJECT"
```

Make it executable:

```bash
chmod +x scripts/smoke_containerized_retrieval.sh
```

- [ ] **Step 4: Add a static smoke-script assertion and run focused tests**

Add to `tests/test_docker_assets.py`:

```python
def test_cpu_smoke_script_has_cleanup_and_utf8_assertions():
    script = _read("scripts/smoke_containerized_retrieval.sh")
    assert "set -euo pipefail" in script
    assert "down -v --remove-orphans" in script
    assert "COMPOSE_PROFILES=cpu" in script
    assert "JUSTATOM_API_PORT" in script
    assert "WEAVIATE_HTTP_PORT" in script
    assert "Loading from huggingface hub via" in script
    assert "\\\\u" in script
```

Run:

```bash
conda run -n justatom python -m pytest \
  tests/test_docker_assets.py tests/integration/test_embedding_server_contract.py -q
```

Expected: PASS.

- [ ] **Step 5: Run the end-to-end CPU smoke test**

Run: `bash scripts/smoke_containerized_retrieval.sh`

Expected: one Qwen model load, three indexed documents, two successful searches, readable Russian JSON, and clean container removal.

- [ ] **Step 6: Commit integration assets**

```bash
git add tests/integration/test_embedding_server_contract.py tests/test_docker_assets.py \
  scripts/smoke_containerized_retrieval.sh
git commit -m "test: verify containerized embedding contract"
```

---

### Task 9: Model-free API Image Smoke Test

**Files:**
- Create: `tests/fixtures/openai_embedding_stub.py`
- Create: `scripts/smoke_api_external_backend.sh`
- Modify: `tests/test_docker_assets.py`

**Interfaces:**
- Consumes: `justatom-api`, external `EMBEDDING_BASE_URL`, and real Compose Weaviate.
- Produces: deterministic proof that the API image indexes and searches without Torch or local model ownership.

- [ ] **Step 1: Add the failing static asset test**

Append to `tests/test_docker_assets.py`:

```python
def test_external_backend_smoke_uses_real_api_image_without_torch():
    script = _read("scripts/smoke_api_external_backend.sh")
    fixture = _read("tests/fixtures/openai_embedding_stub.py")
    assert "host.docker.internal" in script
    assert "up -d --build weaviate api" in script
    assert 'find_spec("torch") is None' in script
    assert "down -v --remove-orphans" in script
    assert "/v1/embeddings" in fixture
    assert "fixture-embedding-model" in fixture
```

- [ ] **Step 2: Run the asset test and verify RED**

Run:

```bash
conda run -n justatom python -m pytest \
  tests/test_docker_assets.py::test_external_backend_smoke_uses_real_api_image_without_torch -q
```

Expected: FAIL because the stub and script do not exist.

- [ ] **Step 3: Create the deterministic external embedding stub**

Create `tests/fixtures/openai_embedding_stub.py`:

```python
from __future__ import annotations

import asyncio
import os
from typing import Any

from quart import Quart, request

from justatom.api.hypercorn_server import serve_app


MODEL = os.getenv("FAKE_EMBEDDING_MODEL", "fixture-embedding-model")
PORT = int(os.getenv("FAKE_EMBEDDING_PORT", "18001"))
app = Quart(__name__, static_folder=None)
app.json.ensure_ascii = False


def _vector(text: str) -> list[float]:
    normalized = text.lower()
    return [
        float("банк" in normalized or "негатив" in normalized),
        float("qwen" in normalized or "эмбед" in normalized),
        float("weaviate" in normalized or "вектор" in normalized),
    ]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL}


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": MODEL, "object": "model"}]}


@app.post("/v1/embeddings")
async def embeddings():
    payload: Any = await request.get_json(silent=True)
    if not isinstance(payload, dict) or payload.get("model") != MODEL:
        return {"error": {"message": "model not found", "type": "model_not_found"}}, 404
    source = payload.get("input")
    texts = [source] if isinstance(source, str) else source
    if not isinstance(texts, list) or not texts or any(not isinstance(text, str) for text in texts):
        return {"error": {"message": "invalid input", "type": "invalid_request_error"}}, 400
    return {
        "object": "list",
        "model": MODEL,
        "data": [
            {"object": "embedding", "index": index, "embedding": _vector(text)}
            for index, text in enumerate(texts)
        ],
    }


if __name__ == "__main__":
    asyncio.run(serve_app(app, host="0.0.0.0", port=PORT))
```

- [ ] **Step 4: Create the external-backend smoke script**

Create `scripts/smoke_api_external_backend.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROJECT="justatom-api-smoke-$(date +%s)-$$"
export COMPOSE_PROJECT_NAME="$PROJECT"
unset COMPOSE_PROFILES
export JUSTATOM_API_PORT="${JUSTATOM_API_PORT:-15556}"
export WEAVIATE_HTTP_PORT="${WEAVIATE_HTTP_PORT:-13212}"
export WEAVIATE_GRPC_PORT="${WEAVIATE_GRPC_PORT:-15052}"
export FAKE_EMBEDDING_PORT="${FAKE_EMBEDDING_PORT:-18001}"
export EMBEDDING_MODEL=fixture-embedding-model
export EMBEDDING_BASE_URL="http://host.docker.internal:${FAKE_EMBEDDING_PORT}/v1"

COMPOSE=(docker compose -p "$PROJECT")
FAKE_LOG="${TMPDIR:-/tmp}/${PROJECT}-embedder.log"
FAKE_PID=""

cleanup() {
  "${COMPOSE[@]}" down -v --remove-orphans >/dev/null 2>&1 || true
  if [[ -n "$FAKE_PID" ]]; then
    kill "$FAKE_PID" >/dev/null 2>&1 || true
    wait "$FAKE_PID" >/dev/null 2>&1 || true
  fi
  rm -f "$FAKE_LOG"
}

fail() {
  printf 'external-backend smoke failure: %s\n' "$1" >&2
  "${COMPOSE[@]}" logs --no-color >&2 || true
  [[ ! -f "$FAKE_LOG" ]] || cat "$FAKE_LOG" >&2
  exit 1
}

wait_http() {
  local name="$1"
  local url="$2"
  local attempt
  for attempt in $(seq 1 150); do
    if curl --fail --silent --show-error "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  fail "$name did not become ready at $url"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

command -v python >/dev/null || fail "python is required"
command -v docker >/dev/null || fail "docker is required"
command -v curl >/dev/null || fail "curl is required"
command -v jq >/dev/null || fail "jq is required"
python -c 'import hypercorn, quart' || fail "activate the justatom Python environment"

python tests/fixtures/openai_embedding_stub.py >"$FAKE_LOG" 2>&1 &
FAKE_PID=$!
wait_http "fake embedding endpoint" "http://127.0.0.1:${FAKE_EMBEDDING_PORT}/health"

"${COMPOSE[@]}" up -d --build weaviate api
wait_http "retrieval API" "http://127.0.0.1:${JUSTATOM_API_PORT}/"

index_response="$(
  curl --fail --silent --show-error \
    -H 'Content-Type: application/json' \
    -d '{"dataset_name_or_docs":[{"content":"Memory bank расширяет множество негативных примеров.","meta":{"topic":"retrieval"}},{"content":"Qwen создаёт эмбеддинги документов.","meta":{"topic":"embeddings"}},{"content":"Weaviate хранит векторы документов.","meta":{"topic":"storage"}}]}' \
    "http://127.0.0.1:${JUSTATOM_API_PORT}/indexing"
)" || fail "indexing through external endpoint failed"
printf '%s' "$index_response" | jq -e '.total_docs == 3' >/dev/null \
  || fail "expected three indexed documents"

search_response="$(
  curl --fail --silent --show-error \
    -H 'Content-Type: application/json' \
    -d '{"text":"Зачем нужен банк негативов?","top_k":1}' \
    "http://127.0.0.1:${JUSTATOM_API_PORT}/searching"
)" || fail "search through external endpoint failed"
printf '%s' "$search_response" | jq -e '.docs[0].meta.topic == "retrieval"' >/dev/null \
  || fail "deterministic retrieval result is incorrect"
if printf '%s' "$search_response" | grep -Eq '\\u[0-9a-fA-F]{4}'; then
  fail "API response escaped readable UTF-8 text"
fi

"${COMPOSE[@]}" exec -T api python -c \
  'import importlib.util; assert importlib.util.find_spec("torch") is None' \
  || fail "Torch is installed in the model-free API image"

printf 'model-free API smoke passed: project=%s\n' "$PROJECT"
```

Make it executable:

```bash
chmod +x scripts/smoke_api_external_backend.sh
```

- [ ] **Step 5: Make the script executable and verify GREEN**

Run:

```bash
chmod +x scripts/smoke_api_external_backend.sh
conda run -n justatom python -m pytest \
  tests/test_docker_assets.py::test_external_backend_smoke_uses_real_api_image_without_torch -q
```

Expected: PASS.

- [ ] **Step 6: Run the model-free API smoke test**

Run:

```bash
conda run -n justatom bash scripts/smoke_api_external_backend.sh
```

Expected: the API image indexes three documents through the host stub, retrieves the deterministic Russian result from real Weaviate, emits raw UTF-8 JSON, and proves Torch is absent.

- [ ] **Step 7: Commit the external-backend verification**

```bash
git add tests/fixtures/openai_embedding_stub.py scripts/smoke_api_external_backend.sh \
  tests/test_docker_assets.py
git commit -m "test: smoke model-free retrieval API"
```

---

### Task 10: User Documentation and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/launch-guide.md`
- Modify: `docs/architecture.md`
- Modify: `docs/modules/runtime.md`
- Create: `scripts/smoke_native_embedding.sh`
- Modify: `tests/test_docker_assets.py`

**Interfaces:**
- Documents all executable commands and the stable API/embedder boundary.
- Does not change production interfaces.

- [ ] **Step 1: Add documentation assertions before editing prose**

Extend `tests/test_docker_assets.py`:

```python
def test_launch_guide_documents_all_embedding_deployments():
    guide = _read("docs/launch-guide.md")
    assert "COMPOSE_PROFILES=cpu" in guide
    assert "COMPOSE_PROFILES=cuda" in guide
    assert "host.docker.internal:8000/v1" in guide
    assert "EMBEDDING_DEVICE=mps" in guide
    assert "Dockerfile.embedder.cpu" in guide
    assert "Dockerfile.embedder.cuda" in guide


def test_native_mps_smoke_checks_reuse_and_contract():
    script = _read("scripts/smoke_native_embedding.sh")
    assert "EMBEDDING_DEVICE=mps" in script
    assert "/v1/embeddings" in script
    assert "torch.backends.mps.is_available()" in script
    assert "Loading from huggingface hub via" in script
```

- [ ] **Step 2: Run the documentation assertion and verify RED**

Run:

```bash
conda run -n justatom python -m pytest \
  tests/test_docker_assets.py::test_launch_guide_documents_all_embedding_deployments \
  tests/test_docker_assets.py::test_native_mps_smoke_checks_reuse_and_contract -q
```

Expected: FAIL because the launch guide and native MPS smoke script do not exist yet.

- [ ] **Step 3: Document exact launch flows**

Add copyable sections for:

```bash
# Native MPS embedder
EMBEDDING_DEVICE=mps \
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B \
conda run -n justatom python -m justatom.api.serve_embeddings

# API + Weaviate, using native MPS on the host
EMBEDDING_BASE_URL=http://host.docker.internal:8000/v1 \
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B \
docker compose up --build api weaviate

# Managed CPU model
COMPOSE_PROFILES=cpu docker compose up --build

# Managed CUDA model on Linux/NVIDIA
COMPOSE_PROFILES=cuda docker compose up --build
```

Document that ordinary Docker Desktop containers cannot use MPS, one embedding worker equals one model copy, model cache persistence prevents redownloads but not RAM reloads, and external vLLM/Triton/llama.cpp servers need only the same OpenAI-compatible config.

- [ ] **Step 4: Add the native MPS smoke script**

Create `scripts/smoke_native_embedding.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  printf 'native MPS smoke skipped: Apple Silicon macOS is required\n'
  exit 0
fi

export EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
export EMBEDDING_DEVICE=mps
LOG_FILE="${TMPDIR:-/tmp}/justatom-mps-embedding-$$.log"
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
  rm -f "$LOG_FILE"
}

fail() {
  printf 'native MPS smoke failure: %s\n' "$1" >&2
  [[ ! -f "$LOG_FILE" ]] || cat "$LOG_FILE" >&2
  exit 1
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

command -v python >/dev/null || fail "python is required"
command -v curl >/dev/null || fail "curl is required"
command -v jq >/dev/null || fail "jq is required"
python -c 'import torch; assert torch.backends.mps.is_available()' \
  || fail "PyTorch MPS is unavailable"
if lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null; then
  fail "port 8000 is already in use"
fi

python -m justatom.api.serve_embeddings >"$LOG_FILE" 2>&1 &
SERVER_PID=$!
for attempt in $(seq 1 600); do
  if curl --fail --silent --show-error http://127.0.0.1:8000/health >/dev/null 2>&1; then
    break
  fi
  [[ "$attempt" != "600" ]] || fail "embedding server did not become ready"
  sleep 1
done

first="$(
  curl --fail --silent --show-error \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${EMBEDDING_MODEL}\",\"input\":[\"русский запрос\",\"English passage\"],\"encoding_format\":\"float\"}" \
    http://127.0.0.1:8000/v1/embeddings
)" || fail "first MPS embedding request failed"
printf '%s' "$first" | jq -e \
  '.data as $data | ($data | length) == 2 and ($data[0].embedding | length) > 0 and (($data[0].embedding | length) == ($data[1].embedding | length))' \
  >/dev/null || fail "first MPS response has invalid dimensions"

second="$(
  curl --fail --silent --show-error \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${EMBEDDING_MODEL}\",\"input\":\"повторный запрос\"}" \
    http://127.0.0.1:8000/v1/embeddings
)" || fail "second MPS embedding request failed"
printf '%s' "$second" | jq -e '.data | length == 1 and (.[0].embedding | length > 0)' \
  >/dev/null || fail "second MPS response is invalid"

model_loads="$(grep -F -c 'Loading from huggingface hub via' "$LOG_FILE" || true)"
[[ "$model_loads" == "1" ]] || fail "expected one model load, observed $model_loads"

printf 'native MPS embedding smoke passed: model=%s\n' "$EMBEDDING_MODEL"
```

Make it executable and run the static assertion:

```bash
chmod +x scripts/smoke_native_embedding.sh
conda run -n justatom python -m pytest \
  tests/test_docker_assets.py::test_native_mps_smoke_checks_reuse_and_contract -q
```

Expected: PASS.

- [ ] **Step 5: Update architecture diagrams and module ownership**

Show `client -> justatom-api -> OpenAICompatibleEmbedder -> embedding server` and the independent `justatom-api -> WeaviateDocumentStore -> Weaviate` edge. Explicitly state that API images contain no model and that query/document prefixes are applied before HTTP inference.

- [ ] **Step 6: Run the full verification matrix**

Run:

```bash
conda run -n justatom python -m pytest tests -q
conda run -n justatom make format-check
conda run -n justatom mkdocs build --strict
git diff --check
docker compose config --quiet
COMPOSE_PROFILES=cpu docker compose config --quiet
COMPOSE_PROFILES=cuda docker compose config --quiet
docker build -f Dockerfile.api -t justatom-api:verify .
docker build -f Dockerfile.embedder.cpu -t justatom-embedder-cpu:verify .
docker buildx build --platform linux/amd64,linux/arm64 -f Dockerfile.api .
docker build --platform linux/amd64 -f Dockerfile.embedder.cuda -t justatom-embedder-cuda:verify .
```

Expected: all commands exit zero. CUDA image build is required on an amd64-capable builder; CUDA inference is required only on a Linux/NVIDIA runner.

- [ ] **Step 7: Run security and image-content checks**

Run:

```bash
docker history --no-trunc justatom-api:verify
docker history --no-trunc justatom-embedder-cpu:verify
docker run --rm justatom-api:verify python -c \
  'import importlib.util; assert importlib.util.find_spec("torch") is None'
docker run --rm justatom-embedder-cpu:verify python -m pip check
docker scout cves --only-severity critical justatom-api:verify
docker scout cves --only-severity critical justatom-embedder-cpu:verify
```

Inspect history output and fail the release if it contains an API key, Hugging Face token, `.env` content, local absolute path, or model weight layer.

- [ ] **Step 8: Run native MPS verification on Apple Silicon**

Run:

```bash
conda run -n justatom bash scripts/smoke_native_embedding.sh
```

Expected on Apple Silicon macOS: two successful embedding requests, equal non-zero dimensions, and exactly one Qwen load. Other platforms print the explicit skip reason; record the skip rather than claiming MPS coverage.

- [ ] **Step 9: Commit documentation and final verification state**

```bash
git add README.md docs/launch-guide.md docs/architecture.md docs/modules/runtime.md \
  scripts/smoke_native_embedding.sh tests/test_docker_assets.py
git commit -m "docs: explain containerized retrieval deployment"
```

- [ ] **Step 10: Record final evidence in the PR description**

Include exact test counts, image sizes, platforms built, CPU smoke latency, Qwen model-load count, and whether CUDA inference was executed or only statically built. Do not claim CUDA or MPS verification that was not actually run.
