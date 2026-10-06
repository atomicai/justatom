# justatom.api

Each HTTP service lives in one module with its settings, application factory,
routes, lifecycle, and Hypercorn startup. There are two independently deployable
services: retrieval and embeddings.

| Service | Application factory | Launcher | Default port |
|---|---|---|---:|
| Retrieval | `serve.create_app()` | `python -m justatom.api.serve` | 5555 |
| Embeddings | `serve_embeddings.create_embedding_app()` | `python -m justatom.api.serve_embeddings` | 8000 |

Each module's `main()` runs Hypercorn. The application factories can also be used
directly in tests or with another ASGI server; constructing an app does not start
an HTTP listener.

## Retrieval service

`serve.py` exposes `/searching`, `/searching/agentic`, `/indexing`,
`/delete`, and the health route `/`. It owns the retrieval runtime and, when
enabled, the agent runtime. Retrieval connects to the configured document store.

```bash
JUSTATOM_CONFIG=/path/to/serve.yaml python -m justatom.api.serve
```

Without `JUSTATOM_CONFIG`, the standard scenario loader uses packaged defaults
and overlays `configs/serve.yaml` from the current directory when present. The
Docker image explicitly sets `JUSTATOM_CONFIG=/etc/justatom/serve.yaml`.

RabbitMQ is opt-in: use `JUSTATOM_START_MQ=true` for the launcher, or
`create_app(start_mq=True)` when constructing an application. Both default to
disabled. Enabling the legacy consumer requires the RabbitMQ dependencies and
configuration separately.

## Embedding service

`serve_embeddings.py` exposes `/v1/embeddings`, `/v1/models`, and `/health`.
It loads one model during application startup and closes it during shutdown.
Requests select that configured model; they do not load or switch models.

```bash
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B \
EMBEDDING_DEVICE=cpu \
python -m justatom.api.serve_embeddings
```

The model process reads `EMBEDDING_MODEL`, `EMBEDDING_DEVICE`,
`EMBEDDING_BATCH_SIZE`, and `EMBEDDING_MAX_LENGTH`. It computes vectors only;
document search, indexing, and agent planning belong to the retrieval service.

## Choosing where embeddings run

With `retrieval.embedding.backend: local`, the retrieval process loads the
embedding model itself; no embedding HTTP service is needed. With
`backend: openai-compatible`, it sends requests to the configured `base_url`.
That endpoint can be the built-in embedding service or an external compatible
implementation.

For HTTP embeddings, query/document prefixes are applied by the client. The
server owns tokenization and truncation: the client's `max_length` setting is
not sent over the embeddings API. Keep the client batch size at or below the
built-in server's `EMBEDDING_BATCH_SIZE` request limit.

The Docker modes `cpu`, `cuda`, and `external` choose which embedding process
is deployed; they do not change the retrieval backend configuration. See the
[Launch Guide](../launch-guide.md) for container commands.

## Other commands

- `train.py`, `eval.py`, and `datasets.py` are command-line tools, not additional
  HTTP services. `dataset_input.py` handles dataset input for the retrieval API.
- `welcome.py` is a legacy terminal greeting utility, not part of server startup.

Import LLM utilities such as `OpenAiTask` and `OpenAIAsyncWrapper` directly from
`justatom.running.llm`. The `justatom.api` package does not re-export them or load
their dependencies on import.

Source and configuration files use UTF-8 and LF line endings, enforced by
`.editorconfig`, `.gitattributes`, and the pre-commit line-ending hook.
