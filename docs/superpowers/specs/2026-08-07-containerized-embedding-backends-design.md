# Containerized API and Embedding Backends

Date: 2026-08-07
Status: Approved design

## Goal

Package the retrieval API independently from model inference while preserving all supported deployment modes:

- a lightweight JustAtom API container that uses an OpenAI-compatible embedding endpoint;
- a self-contained CPU embedding container for local and portable testing;
- a CUDA embedding container for Linux hosts with NVIDIA GPUs;
- a native macOS embedding process that keeps PyTorch MPS acceleration outside Docker.

The API and every embedding backend use one HTTP contract. Switching hardware must not require changing indexing or retrieval code.

## Non-goals

- Passing an Apple GPU or the PyTorch MPS device into a Linux container.
- Bundling model weights into an image.
- Making Redis or RabbitMQ mandatory for retrieval API startup.
- Replacing external OpenAI-compatible servers such as vLLM, Triton, or llama.cpp.
- Adding orchestration beyond Docker Compose.

## Image Names and Dockerfiles

The repository provides three explicit build artifacts:

| Dockerfile | Image | Responsibility |
| --- | --- | --- |
| `Dockerfile.api` | `justatom-api` | Retrieval HTTP API, runtime orchestration, and Weaviate client. No Torch or model weights. |
| `Dockerfile.embedder.cpu` | `justatom-embedder-cpu` | OpenAI-compatible embedding service backed by local Hugging Face inference on CPU. |
| `Dockerfile.embedder.cuda` | `justatom-embedder-cuda` | The same embedding service backed by CUDA on Linux/NVIDIA. |

The `embedder` qualifier is required. Names such as `Dockerfile.cpu` and `Dockerfile.cuda` are rejected because they do not identify whether the hardware variant belongs to the retrieval API or to model inference.

## Process Boundaries

```text
client
  |
  v
justatom-api
  |- RetrievalRuntime
  |- Indexer / Retriever
  |- WeaviateDocumentStore --------> weaviate
  `- OpenAICompatibleEmbedder -----> embedding endpoint
                                        |- native MPS process, or
                                        |- justatom-embedder-cpu, or
                                        |- justatom-embedder-cuda, or
                                        `- vLLM / Triton / llama.cpp
```

`justatom-api` never imports Torch and never owns model weights. It owns one `RetrievalRuntime` for the API process lifetime and one reusable HTTP client to the embedding endpoint.

Each JustAtom embedding service owns one `HuggingFaceEmbedder`. The model is loaded during service startup, reused for every request, and released during graceful shutdown.

## OpenAI-compatible Embedding Service

The built-in CPU and CUDA services expose the smallest contract needed by `OpenAICompatibleEmbedder`:

- `GET /health` returns liveness and readiness after the model has loaded;
- `GET /v1/models` reports the configured model identifier;
- `POST /v1/embeddings` accepts an OpenAI-compatible embedding request.

Request:

```json
{
  "model": "Qwen/Qwen3-Embedding-0.6B",
  "input": ["first text", "second text"],
  "encoding_format": "float"
}
```

Response:

```json
{
  "object": "list",
  "model": "Qwen/Qwen3-Embedding-0.6B",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
    {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]}
  ]
}
```

The service preserves input ordering through explicit indexes. It accepts one string or a non-empty list of strings, rejects unknown models, malformed inputs, unsupported encodings, and oversized batches with HTTP 4xx responses, and returns sanitized HTTP 5xx errors for backend failures. Float embeddings are the required format for the first release.

The query/document distinction remains in `justatom-api`: its embedding profile adds query and document prefixes before calling the shared endpoint. The embedding service only encodes the supplied text and therefore remains domain-agnostic.

## Configuration

The API container reads a YAML file from `JUSTATOM_CONFIG`, defaulting to `/etc/justatom/serve.yaml`. Compose mounts that file read-only. Environment placeholders are resolved before the existing strict retrieval validation runs; a missing required variable fails startup with a configuration error.

The API continues to use the strict retrieval configuration:

```yaml
retrieval:
  mode: vector
  embedding:
    backend: openai-compatible
    base_url: ${EMBEDDING_BASE_URL}
    model: ${EMBEDDING_MODEL}
    batch_size: 8
    max_length: 512
    query_prefix: "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
  store:
    collection: Document
    url: http://weaviate:2211
    grpc_port: 50051
    grpc_secure: false
```

Deployments that require authentication add `api_key: ${EMBEDDING_API_KEY}`. The key is omitted from the rendered mapping when `EMBEDDING_API_KEY` is absent; it is never replaced with an empty string or an unresolved placeholder. Secrets are injected at runtime and are never copied into an image. An absent API key is valid for trusted local backends.

The embedding service is configured through environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Hugging Face model identifier or mounted local path. |
| `EMBEDDING_DEVICE` | Image-specific | `cpu` in the CPU image, `cuda:0` in the CUDA image, and `mps` for native macOS. |
| `EMBEDDING_BATCH_SIZE` | `8` | Maximum inference batch size. |
| `EMBEDDING_MAX_LENGTH` | `512` | Tokenization limit. |
| `HF_HOME` | `/cache/huggingface` | Persistent model cache location. |
| `HF_TOKEN` | unset | Optional Hugging Face access token. |

Invalid device/model configuration fails startup before the readiness endpoint becomes healthy.

## Docker Compose Modes

Managed profiles require Docker Compose 2.20 or newer because the API uses optional health-checked dependencies for profile-owned embedding services.

Compose always supports `api` and `weaviate`. The supported operational entrypoint is:

```text
scripts/services.sh <external|cpu|cuda> <up|down|config|build|ps|logs> [compose args...]
```

The launcher is deliberately bounded. The command must be the token immediately after the single `external`, `cpu`, or `cuda` mode; leading/global Compose options and commands outside `up`, `down`, `config`, `build`, `ps`, and `logs` are rejected before Docker is invoked. Options and argument values after a supported command are forwarded unchanged except that `--profile` and `--profile=...` are rejected anywhere. The launcher overwrites inherited `COMPOSE_PROFILES` with exactly the selected mode so Compose service selection and API startup validation receive the same value.

`up` is the only supported command that starts workloads and is idempotent for initial startup and subsequent reconciliation. Operators use `up` instead of `create`, `start`, `restart`, `scale`, or `watch`. `run`, `exec`, global Compose options, and direct raw `docker compose` remain low-level/unsupported interfaces. The in-API profile validator remains as defense in depth.

### External or native backend

Set `EMBEDDING_BASE_URL` to an existing OpenAI-compatible server. On Docker Desktop for macOS, a native MPS server is reached through `host.docker.internal`. The launcher requires a non-empty URL for external `up`; `down`, `config`, `build`, `ps`, and `logs` do not start workloads and do not require the URL.

```bash
EMBEDDING_BASE_URL=http://host.docker.internal:8000/v1 \
  scripts/services.sh external up -d
```

### CPU profile

```bash
scripts/services.sh cpu up -d --build
```

The `embedder-cpu` service receives the shared network alias `embedder`, and the API uses `http://embedder:8000/v1`.

### CUDA profile

```bash
scripts/services.sh cuda up -d --build
```

The `embedder-cuda` service receives the same network alias and reserves an NVIDIA GPU through the Compose device specification. Before `cuda up`, the launcher requires a Linux host and a successful `nvidia-smi` probe. It fails clearly without invoking Compose when either precondition is missing; CUDA never falls back to CPU. Artifact validation remains available on macOS with `scripts/services.sh cuda config` and `scripts/services.sh cuda build`, but those commands do not demonstrate CUDA inference. CPU and CUDA profiles are mutually exclusive; launcher validation rejects enabling both before Compose and API startup validation rejects both as defense in depth.

Both model services mount a named Hugging Face cache volume. Rebuilding or recreating a container does not redownload an unchanged model, although each new process still loads weights from disk into RAM or accelerator memory.

## Platform Support

| Mode | Platform | Acceleration |
| --- | --- | --- |
| Native local backend | macOS Apple Silicon | MPS |
| `justatom-embedder-cpu` | Linux amd64/arm64, Docker Desktop | CPU |
| `justatom-embedder-cuda` | Linux with NVIDIA Container Toolkit | CUDA |
| External OpenAI-compatible backend | Any reachable host | Backend-defined |

The API image targets both amd64 and arm64. The CUDA image initially targets Linux amd64. CUDA `config` and `build` validation on macOS do not establish runtime support. No claim is made that CUDA or MPS inference is available inside Docker Desktop containers.

## Runtime and Health

- Production HTTP serving uses Hypercorn rather than Quart's development runner.
- One API worker is the default so lifecycle ownership remains obvious and connection pools are not duplicated.
- One embedding worker is mandatory because every worker would load another model copy.
- The API liveness endpoint remains `GET /`.
- The embedding readiness endpoint returns success only after model loading completes.
- Compose waits for Weaviate readiness and, when a managed embedding profile is active, embedding readiness before starting the API.
- RabbitMQ is disabled for the retrieval API container unless explicitly enabled by a separate deployment configuration.
- SIGTERM triggers graceful closure of the API runtime, HTTP clients, Weaviate client, and model resources.

## Dependency Boundaries

Runtime dependencies are expressed as package extras rather than by installing the repository's monolithic development `requirements.txt`:

- an API/serve extra contains Quart, Hypercorn, configuration, HTTP, and Weaviate dependencies;
- the local embedding extra adds Torch, Transformers, tokenizer, and local runner dependencies;
- CUDA-specific Torch packages are installed only by `Dockerfile.embedder.cuda` from the appropriate PyTorch/CUDA source or base image.

This keeps `justatom-api` small and prevents accidental Torch installation in remote-only deployments.

## Failure Handling

- The API reports an unavailable embedding endpoint as a sanitized embedding backend error; it does not leak API keys or upstream response bodies.
- The embedding service validates request cardinality and vector dimensions before returning data.
- Model load failure terminates the embedding process with a non-zero exit code.
- Weaviate connection failure terminates API startup and closes any resources already created.
- Requests arriving during shutdown are rejected by the existing `RetrievalRuntime` lifecycle guard.
- Compose restart policies may restart failed services, but application code does not hide persistent configuration errors with infinite retries.

## Verification

The implementation must include:

1. Unit tests for OpenAI-compatible request validation, ordering, UTF-8 JSON, backend failure sanitization, and repeated shutdown.
2. Existing `OpenAICompatibleEmbedder` contract tests against the built-in embedding service response shape.
3. API image smoke test using a fake OpenAI-compatible endpoint and a real Weaviate container.
4. CPU image smoke test that verifies one model load and multiple embedding requests. A tiny fixture model may be used in regular CI; Qwen is covered by a network-marked test.
5. Docker Compose config validation for external, CPU, and CUDA modes, including the mutually exclusive profile rule.
6. CUDA runtime verification on a Linux/NVIDIA runner when one is available; normal macOS CI only builds or statically validates the CUDA artifact.
7. Native MPS smoke test outside Docker on Apple Silicon.
8. Full existing test suite, formatting, documentation build, and image vulnerability/dependency checks.

## Acceptance Criteria

- `justatom-api` starts without Torch installed and can index and retrieve through an OpenAI-compatible embedding endpoint.
- CPU and CUDA embedding images expose the same `/v1/embeddings` behavior.
- Qwen weights are loaded once per embedding process, not once per request.
- Native MPS inference remains supported without Docker.
- Changing embedding hardware requires configuration changes only; retrieval API code and clients remain unchanged.
- No credentials or model weights are embedded in image layers.
