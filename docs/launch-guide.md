# Launch Guide

Evaluation scenarios combine dataset settings with a strict retrieval runtime
configuration. The default file is `configs/evaluate.yaml`.

## Retrieval Config

```yaml
retrieval:
  mode: hybrid
  alpha: 0.5
  embedding:
    backend: local
    model: intfloat/multilingual-e5-small
    device: auto
    batch_size: 64
    max_length: 512
    query_prefix: "query: "
    document_prefix: "passage: "
  store:
    collection: Document
    url: http://localhost:2211
    grpc_port: 50051

dataset:
  id: demo-eval

search:
  top_k: 20
  batch_size: 32
```

`retrieval.mode` must be `keyword`, `vector`, or `hybrid`. Keyword retrieval
does not need an embedding section; vector and hybrid retrieval do. The runtime
rejects unknown retrieval, embedding, and store keys rather than silently
ignoring a misspelling.

## Evaluate Locally

```bash
python -m justatom.api.eval \
  --config configs/evaluate.yaml \
  --dataset.id demo-eval \
  --search-mode hybrid \
  --embedding-model intfloat/multilingual-e5-small \
  --query-prefix "query: " \
  --document-prefix "passage: " \
  --collection-name JustAtomEval \
  --weaviate-url http://localhost:2211 \
  --weaviate-grpc-port 50051
```

## Evaluate Against a Remote Embedder

```bash
python -m justatom.api.eval \
  --config configs/evaluate.yaml \
  --dataset.id demo-eval \
  --search-mode vector \
  --embedding-backend openai-compatible \
  --embedding-base-url http://ubuntu-box:8000/v1 \
  --embedding-api-key "$EMBEDDING_API_KEY" \
  --embedding-model deployed-embedding-model \
  --collection-name JustAtomRemoteEval \
  --weaviate-url http://localhost:2211
```

The evaluator accepts explicit `--search-mode`, `--embedding-model`,
`--query-prefix`, `--document-prefix`, `--weaviate-url`, and
`--collection-name` flags. Values are passed to the strict runtime schema; old
retrieval flag names are rejected.

## Retrieval Service Deployment

Use `scripts/services.sh` for every container deployment. It chooses exactly
one embedding mode and forwards the supported lifecycle command. The API image
is built from `Dockerfile.api` and contains no Torch or model weights. The
`Dockerfile.embedder.cpu` image is a portable CPU service. The
`Dockerfile.embedder.cuda` image is Linux/NVIDIA only. Docker Desktop on macOS
cannot expose MPS to containers, so native MPS stays a host process.

### Native MPS on Apple Silicon

Run the embedding service directly on an Apple Silicon macOS host:

```bash
EMBEDDING_DEVICE=mps \
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B \
conda run -n justatom python -m justatom.api.serve_embeddings
```

The service listens on `8000`. Point the API container at it with the external
mode:

```bash
EMBEDDING_BASE_URL=http://host.docker.internal:8000/v1 \
  scripts/services.sh external up -d --build api weaviate
```

### Managed CPU Service

The portable CPU backend runs alongside the API and Weaviate:

```bash
scripts/services.sh cpu up -d --build
```

### Managed CUDA Service

The CUDA backend requires a Linux/NVIDIA host with a working `nvidia-smi`.
Starting it elsewhere fails before any workload starts. On non-CUDA hosts, use
the launcher’s CUDA configuration and platform-build validation only; neither
establishes CUDA inference.

CUDA mode never falls back to CPU.

```bash
scripts/services.sh cuda up -d --build
```

### External OpenAI-compatible Service

Any service implementing the same `/v1/embeddings` contract can back the API,
including vLLM, Triton adapters, llama.cpp servers, and the built-in host
process. Configure its URL and start only the API and Weaviate:

```bash
EMBEDDING_BASE_URL=http://host.docker.internal:8000/v1 \
  scripts/services.sh external up -d --build api weaviate
```

One embedding service process owns one model instance. Repeated requests do not reload it.
The persistent Hugging Face cache avoids a redownload for unchanged weights,
but does not preserve the model in RAM after a process restart.

## Shell Wrappers

The pipeline wrapper preserves the `vanilla`, `atom_gate`, and `atomic` training
variants while passing evaluator settings through the current CLI:

```bash
bash scripts/run_pipeline.sh \
  --dataset-ids demo-eval \
  --method atomic \
  --search-mode hybrid \
  --model intfloat/multilingual-e5-small \
  --weaviate-url http://localhost:2211
```

`scripts/run_benchmark.sh` records shell-escaped commands in `COMMANDS.md`, so
each `vanilla`, `atom_gate`, and `atomic` benchmark invocation is reproducible.

## Dataset Presets

Set `dataset.id` to use a configured preset. Resolution checks
`configs/dataset/<id>.yaml` and then packaged defaults.

- `demo-eval`: small packaged evaluation dataset.
- `demo-train`: small packaged training dataset.
- `justatom`: repository-local dataset preset.

You can instead set `dataset.name_or_path` to a JSON, JSONL, Parquet, CSV, XLSX,
or Hugging Face dataset source. See [Getting Started](getting-started.md) for
dataset adapter examples.
