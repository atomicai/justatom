# Retrieval Runtime

The retrieval runtime is built from direct components rather than a process-wide
service or string factory. `RetrievalRuntime` receives a `DocumentStore`, an
optional `Embedder`, a search mode, and a hybrid alpha. It exposes `index`,
`retrieve`, and `retrieve_many`, while owning the `Indexer`, mode-specific
`Retriever`, and asynchronous cleanup lifecycle.

## Local Runtime

```python
from justatom.etc.schema import Document
from justatom.retrieval import EmbeddingProfile, RetrievalRuntime
from justatom.retrieval.embedders.huggingface import HuggingFaceEmbedder
from justatom.storing.weaviate import WeaviateDocumentStore

documents = [Document(content="Документ для локального поиска.")]
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

`device="auto"` resolves to CUDA, MPS, or CPU as available. Leaving the
`async with` block closes the runtime's store and embedder.

## Remote Runtime

```python
import os

from justatom.etc.schema import Document
from justatom.retrieval import EmbeddingProfile, RetrievalRuntime
from justatom.retrieval.embedders.openai_compatible import OpenAICompatibleEmbedder
from justatom.storing.weaviate import WeaviateDocumentStore

store = await WeaviateDocumentStore.connect("JustAtom", url="http://localhost:2211")
embedder = OpenAICompatibleEmbedder(
    base_url="http://ubuntu-box:8000/v1",
    model="deployed-embedding-model",
    api_key=os.getenv("EMBEDDING_API_KEY"),
    profile=EmbeddingProfile(query_prefix="query: ", document_prefix="passage: "),
)

async with RetrievalRuntime(store, embedder, mode="vector") as runtime:
    await runtime.index([Document(content="Документ для удаленного поиска.")])
    results = await runtime.retrieve("где работает модель", top_k=10)
```

An OpenAI-compatible Triton deployment is used only through its HTTP API. The
client does not import or require `tritonclient`, so it can run on macOS while
the Triton server runs on its supported host.

## Strict Config Builder

For config-driven applications, `build_runtime` accepts exactly the retrieval
keys `mode`, `alpha`, `embedding`, and `store`. Unknown keys, invalid value
types, missing stores, missing vector embedders, and unsupported modes fail with
`ConfigurationError` before the runtime is built.

```python
from justatom.retrieval import build_runtime

runtime = await build_runtime(
    {
        "mode": "vector",
        "embedding": {
            "backend": "openai-compatible",
            "base_url": "http://ubuntu-box:8000/v1",
            "model": "deployed-embedding-model",
        },
        "store": {
            "collection": "JustAtom",
            "url": "http://localhost:2211",
            "grpc_port": 50051,
        },
    }
)
try:
    results = await runtime.retrieve("проверочный запрос", top_k=5)
finally:
    await runtime.close()
```

Use the context-manager form when possible. The explicit `try`/`finally` form
is useful when a framework owns the surrounding lifecycle.
