# Retrieval Runtime Simplification

> Design spec. Approved on 2026-08-05.
> Branch: `feature/retrieval-runtime` (off `master`).
> Scope: replace the current retrieval/indexing composition with an explicit,
> resource-owned runtime that supports an in-process Hugging Face embedder and
> remote OpenAI-compatible embedding endpoints.

---

## 1. Context

The current retrieval path has two overlapping implementations of embedding:

1. `justatom.running.embeddings` defines local and OpenAI-compatible clients.
2. `justatom.running.indexer` and `justatom.running.retriever` bypass those
   clients and each build their own processor/runner execution path.

`RunningService` then layers several process-global caches and string factories
over both implementations:

- cached language models;
- cached tokenizers;
- cached processors;
- cached encoder runners;
- a separate LRU of embedding clients;
- `IndexerAPI.named(...)` and `RetrieverAPI.named(...)` factories;
- `igni_runners(...)`, which assembles a different graph based on a string.

The result has duplicate preprocessing and inference code, hidden model
ownership, ambiguous cleanup, and a remote embedding abstraction that the main
retrieval flow does not use. The public API also returns a different shape for
one query and many queries through the same method.

The storage side has a broad `INNDocStore` abstract class, but
`WeaviateDocStore` does not implement it nominally. A global `Finder` object opens
new Weaviate clients without making ownership visible at the call site.

## 2. Goals

1. Use one embedding contract for indexing, vector retrieval, hybrid retrieval,
   clustering adapters, local inference, and remote inference.
2. Keep Weaviate as the only built-in ANN/document store.
3. Let indexers and retrievers accept an abstract document store so tests and
   user integrations do not need to inherit from a justatom base class.
4. Support two built-in embedding paths:
   - an in-process Hugging Face model;
   - an OpenAI-compatible `/embeddings` endpoint.
5. Treat llama.cpp, vLLM, and Triton deployments as remote embedding servers,
   not as separate retrieval implementations.
6. Make resource ownership explicit and remove process-global runtime caches.
7. Keep the direct Python API small while allowing config-driven CLI/eval usage.
8. Make response shapes, validation, and failures predictable.

## 3. Non-goals

- Adding Qdrant, FAISS, Elasticsearch, or another built-in store.
- Adding a Triton Python/gRPC client or a `tritonclient` dependency.
- Supporting a remote service that performs retrieval and returns final
  documents. The remote boundary in this iteration returns embeddings only.
- Building a plugin registry, entry-point system, dependency injection
  container, or generic backend discovery mechanism.
- Adding query/result embedding caches.
- Changing training encoders, losses, ATOMIC training, or saved model formats.
- Preserving the old retrieval API with compatibility wrappers.

## 4. Architecture

The new public implementation lives under `justatom.retrieval`:

```text
justatom/retrieval/
  __init__.py
  contracts.py
  errors.py
  config.py
  indexer.py
  retriever.py
  runtime.py
  embedders/
    __init__.py
    huggingface.py
    openai_compatible.py
```

Weaviate remains in `justatom/storing/weaviate.py` and satisfies the new store
protocol structurally.

The runtime data flow is:

```text
                         +---------------------------+
query/document text ---> | Embedder                  |
                         | - HuggingFaceEmbedder     |
                         | - OpenAICompatibleEmbedder|
                         +-------------+-------------+
                                       |
                                       | vectors
                                       v
documents ---> Indexer ---> DocumentStore <--- Retriever <--- queries
                             |
                             +--- WeaviateDocumentStore
```

Local versus remote inference ends at the `Embedder` boundary. Indexing,
retrieval, evaluation, and storage do not branch on server technology.

## 5. Core contracts

### 5.1 Embedder

`Embedder` is a runtime-checkable structural protocol:

```python
class Embedder(Protocol):
    async def embed_queries(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]: ...

    async def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]: ...

    async def close(self) -> None: ...
```

The role-specific methods replace `embed(..., input_type=..., **props)`. They
make asymmetric query/document preprocessing visible in the type-level API and
prevent prefixes from being passed through unrelated layers.

Both built-in implementations share a frozen profile:

```python
@dataclass(frozen=True)
class EmbeddingProfile:
    query_prefix: str = ""
    document_prefix: str = ""
    max_length: int = 512
    batch_size: int = 64
    skip_prefix_if_present: bool = True
```

Prefix application is client-side for both local and remote inference. This
ensures that the same model receives the same text regardless of deployment
topology. Empty input returns `[]` without loading a batch or making a request.

Every implementation validates that:

- one vector is returned for every input text;
- all vectors in one response have the same non-zero dimension;
- values can be converted to finite floats.

Validation failures raise `EmbeddingResponseError`.

### 5.2 DocumentStore

`DocumentStore` is also a structural protocol. Implementations do not need to
inherit from a justatom class.

```python
class DocumentStore(Protocol):
    async def write_documents(
        self,
        documents: Sequence[Document],
    ) -> int: ...

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
```

The existing Weaviate methods are renamed/adapted to this vocabulary. The
implementation may retain additional Weaviate-specific operations outside the
protocol.

The `INNDocStore` retrieval contract is removed. Unrelated event/dataframe store
interfaces are not part of this refactor.

### 5.3 Indexer

There is one `Indexer`:

```python
indexer = Indexer(store=store, embedder=embedder)
written = await indexer.index(documents, batch_size=64)
```

- With `embedder=None`, documents are written without vectors and remain usable
  for keyword/BM25 retrieval.
- With an embedder, each batch uses `embed_documents`, attaches one vector per
  document, and writes the batch.
- Input remains a lazy `Iterable[Document | dict]`; the full dataset is never
  materialized.
- Indexing batch size controls both embedding and write flow. Store-specific
  request subdivision remains private to the store implementation.
- Bounded write concurrency is allowed, but defaults to one in-flight write.
- Manual `torch.mps.empty_cache()` and `torch.cuda.empty_cache()` calls are not
  part of the indexer.

There is no separate keyword indexer because keyword indexing is ordinary
document writing without vectors. Hybrid indexing uses the same vectors as
vector indexing.

### 5.4 Retrievers

The built-in retrievers are:

- `KeywordRetriever(store)`;
- `VectorRetriever(store, embedder)`;
- `HybridRetriever(store, embedder, alpha=0.5)`.

They expose two methods with stable shapes:

```python
await retriever.retrieve("one query", top_k=10)
# list[Document]

await retriever.retrieve_many(["q1", "q2"], top_k=10)
# list[list[Document]]
```

`retrieve` delegates to `retrieve_many` and unwraps exactly one result. The
shape-coercion helper and overloaded `str | list[str]` return type are removed.

Only vector and hybrid retrievers call `embed_queries`. Keyword retrieval never
loads or calls an embedder.

The supported modes are a string enum:

```python
class SearchMode(str, Enum):
    KEYWORD = "keyword"
    VECTOR = "vector"
    HYBRID = "hybrid"
```

Legacy input names `keywords` and `embedding` are replaced by `keyword` and
`vector` in configs, CLI arguments, scripts, and documentation. There is no
runtime alias table. `gamma-hybrid` and the invalid eval choice `atomicai` are
removed. The ATOMIC name remains reserved for the training method.

### 5.5 RetrievalRuntime

`RetrievalRuntime` is a convenience composition root, not a service locator:

```python
runtime = RetrievalRuntime(
    store=store,
    embedder=embedder,
    mode=SearchMode.HYBRID,
    alpha=0.5,
)

async with runtime:
    await runtime.index(documents)
    results = await runtime.retrieve(query, top_k=10)
```

The constructor contains one explicit `match` over the three supported modes.
There is no registry or dynamic class lookup. It creates one `Indexer` and one
concrete retriever and delegates public operations to them.

The runtime owns the supplied store and embedder. `close()` is idempotent and
closes each owned resource exactly once. Components used directly do not close
their dependencies; their caller owns those resources.

## 6. Embedding implementations

### 6.1 HuggingFaceEmbedder

`HuggingFaceEmbedder` loads exactly one model, tokenizer, and processor per
instance. It does not use module-level caches.

```python
embedder = HuggingFaceEmbedder(
    model="intfloat/multilingual-e5-small",
    device="auto",
    profile=EmbeddingProfile(
        query_prefix="query: ",
        document_prefix="passage: ",
    ),
)
```

The processor itself has no fixed role prefix. The embedder applies the selected
prefix to text and reuses the same processor for queries and documents. Model
execution is performed under inference/no-grad mode. A blocking local forward
pass is moved off the async event loop.

`device="auto"` selects CUDA, then MPS, then CPU and logs the resolved device
once. There is no silent fallback after a model execution failure.

Torch/model imports must be lazy enough that keyword-only and remote-only use do
not require loading the local ML stack at package import time. Selecting the
local backend without the optional dependencies raises an actionable error that
mentions `justatom[torch]`.

### 6.2 OpenAICompatibleEmbedder

The remote implementation sends OpenAI-compatible embedding requests:

```json
{
  "model": "model-name",
  "input": ["text one", "text two"]
}
```

It accepts responses in the ordered OpenAI data form:

```json
{
  "data": [
    {"index": 0, "embedding": [0.1, 0.2]},
    {"index": 1, "embedding": [0.3, 0.4]}
  ]
}
```

The client sorts by `index`, validates uniqueness/completeness, then validates
vector count and dimensions. It uses one persistent `httpx.AsyncClient` per
embedder and closes it in `close()`. The lightweight async HTTP dependency is
declared explicitly instead of relying on transitive installation.

Configuration supports `base_url`, `model`, optional `api_key`, timeout,
`encoding_format`, and a narrowly scoped `extra_body` escape hatch for
OpenAI-compatible server extensions. The Authorization header is omitted when
no API key is configured, which supports trusted local endpoints.

llama.cpp, vLLM, and Triton deployments do not receive dedicated classes or
configuration names. Any of them work only through an OpenAI-compatible
embedding endpoint. In particular:

- justatom never imports `tritonclient`;
- no Triton package is installed on macOS or any other platform by justatom;
- platform-specific deployment concerns remain outside the Python package.

Remote non-success responses raise `EmbeddingBackendError` with the endpoint,
model, HTTP status, and a bounded/sanitized server message. API keys and complete
input texts are never included in exception messages or logs.

## 7. Configuration

Config is translated to runtime objects by one `build_runtime(config)` function.
It uses explicit branches for `local` and `openai-compatible`; there is no
backend registry.

Local example:

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
    url: http://localhost:2211
    grpc_port: 50051
    collection: justatom
```

Remote example:

```yaml
retrieval:
  mode: vector

  embedding:
    backend: openai-compatible
    base_url: http://ubuntu-box:8000/v1
    model: deployed-embedding-model
    api_key: ${EMBEDDING_API_KEY}
    timeout: 60
    batch_size: 64
    query_prefix: "query: "
    document_prefix: "passage: "

  store:
    url: http://localhost:2211
    grpc_port: 50051
    collection: justatom
```

Keyword mode permits the entire `embedding` section to be absent.

Configuration errors are raised before opening store/network/model resources.
Unknown keys under the retrieval section should be rejected rather than silently
forwarded through `**props`.

## 8. Resource and cache policy

The following are removed:

- `_cached_lm_model`;
- `_cached_tokenizer`;
- `_cached_processor`;
- `_cached_encoder_runner`;
- `RunningService._embedding_clients`;
- `_embedding_openai_defaults` caching;
- all `get_or_create_*` and `clear_runner_caches` methods;
- `EmbeddingClientFactory`;
- `IndexerAPI`, `RetrieverAPI`, and both `ByName` implementations;
- the global `Finder` object;
- `RunningService.igni_runners` and
  `RunningService.do_index_and_prepare_for_search`.

No query, document, result, model, tokenizer, processor, runner, or HTTP-client
LRU is maintained by justatom.

Reuse is object reuse: one runtime keeps one model or one remote HTTP session for
its lifetime. A second runtime is an explicit second resource graph.

The normal Hugging Face on-disk download cache remains enabled. It is an artifact
cache managed by Hugging Face, not hidden runtime object state.

## 9. Migration and removals

This is a hard API cut. The repository is still at `0.1.x`, and compatibility
wrappers would preserve the architecture being removed.

Remove after internal call sites are migrated:

- `justatom/running/service.py`;
- `justatom/running/indexer.py`;
- `justatom/running/retriever.py`;
- `justatom/running/embeddings/`.

Also remove `justatom/builtins/configs/embeddings.yaml` and the
`RUNNING_*_CACHE_SIZE` environment knobs because their only consumer is the old
service/cache layer.

Move the small device resolver into the local embedder module. Remove only
`IRetrieverRunner` and `IIndexerRunner` from `justatom/running/mask.py`; unrelated
runner interfaces remain.

Migrate these call sites:

- `justatom/api/eval.py` builds one runtime, evaluates through its retriever, and
  closes the runtime in `finally`/`async with`;
- `justatom/running/evaluator.py` depends on the new `Retriever` protocol and
  calls `retrieve_many` for each evaluation batch;
- `justatom/api/run.py` creates exactly one configured runtime during application
  startup, stores it in application state, and closes it during application
  shutdown. Requests no longer choose model, backend, or collection; deploying
  another configured search stack means starting another application instance.
  This avoids both per-request model loading and an implicit runtime cache;
- `justatom/running/clusters.py` accepts the new `Embedder` protocol and uses
  `embed_documents`;
- retrieval unit/integration tests use the new package;
- README examples and module documentation use the new API.

`WeaviateDocStore` is renamed to `WeaviateDocumentStore`. Its async construction
is exposed directly as `await WeaviateDocumentStore.connect(...)`; the `Finder`
singleton is deleted. The store retains only the async Weaviate client; the sync
client and `search_by_keywords_sync` are removed.

Old/new migration example:

```python
# removed
indexer = IndexerAPI.named(
    "embedding",
    store=store,
    runner=runner,
    processor=processor,
    device=device,
)

# replacement
embedder = HuggingFaceEmbedder(model=model_name, device=device)
indexer = Indexer(store=store, embedder=embedder)
```

## 10. Error model

The retrieval package defines focused public errors:

- `RetrievalError`;
- `ConfigurationError`;
- `EmbeddingError`;
- `EmbeddingBackendError`;
- `EmbeddingResponseError`.

Input validation includes:

- `top_k > 0`;
- `0.0 <= alpha <= 1.0`;
- non-empty model identifier for embedding modes;
- valid positive timeout, batch size, and max length;
- an embedder present for vector/hybrid modes;
- no required embedder for keyword mode.

Store connection/search exceptions retain their original cause. There is no
automatic fallback from vector/hybrid search to BM25, from remote to local
embedding, or from a requested accelerator to CPU.

## 11. Testing strategy

### Unit tests

- `EmbeddingProfile` query/document prefix behavior and duplicate-prefix guard;
- empty embedding input is a no-op;
- vector count, dimension, finite-value, and response-index validation;
- OpenAI-compatible request shape, order restoration, optional authorization,
  timeout, malformed response, and sanitized HTTP error;
- local embedder loads model/tokenizer/processor once per instance;
- keyword indexing without an embedder;
- vector indexing with a fake embedder and lazy iterable;
- bounded indexing/write batches and write failure propagation;
- keyword, vector, and hybrid retrieval arguments;
- `retrieve` and `retrieve_many` stable shapes;
- runtime mode selection and alpha validation;
- runtime closes store/embedder exactly once even after an operation fails;
- importing remote/keyword paths does not import or require Triton;
- importing remote/keyword paths does not initialize the torch runtime.

### Integration tests

- migrate the existing Weaviate streaming index/eval test to
  `RetrievalRuntime`;
- add one OpenAI-compatible fake HTTP server integration covering multiple
  batches;
- keep an optional/network-marked small Hugging Face embedding smoke test;
- run the full existing test suite on Linux, macOS, and Windows.

## 12. Acceptance criteria

The refactor is complete when:

1. Indexing and all three retrieval modes use the same `Embedder` contract.
2. Switching local Hugging Face inference to a remote OpenAI-compatible endpoint
   changes only embedder configuration.
3. Keyword mode works without constructing an embedder.
4. Weaviate is the only built-in store, while fake/custom stores satisfy the
   structural protocol without inheritance.
5. There are no process-global retrieval/model/client caches or name-based
   factories.
6. Runtime resource ownership and cleanup are covered by tests.
7. Triton is absent from imports and dependencies on every platform.
8. The old retrieval service/indexer/retriever/embedding modules are deleted.
9. `api/eval.py`, `api/run.py`, clustering, docs, and tests use the new public
   package.
10. The full unit suite and Weaviate integration suite pass.

## 13. Implementation order

1. Add contracts, errors, profile, and fake-based tests.
2. Add OpenAI-compatible and local embedders.
3. Add indexer and retrievers against fake stores.
4. Add runtime/config composition and lifecycle tests.
5. Adapt and rename Weaviate store methods/class.
6. Migrate eval, HTTP API, clustering, integration tests, and docs.
7. Delete old modules, factories, caches, and dead configuration.
8. Run platform/unit/integration verification and inspect the final import graph.
