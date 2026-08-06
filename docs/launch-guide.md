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
