# justatom

`justatom` is a Python toolkit for indexing documents and evaluating keyword,
vector, and hybrid retrieval. The retrieval runtime is explicit: applications
construct a document store and, when needed, an embedder, then give both to one
`RetrievalRuntime`.

## Install

```bash
pip install -e .
pip install -r requirements-docs.txt
```

Run the standard non-integration suite with:

```bash
pytest tests -m "not integration"
```

## Retrieval Runtime

The public composition is deliberately small:

- `DocumentStore` owns document persistence and search operations.
- `Embedder` creates query and document vectors.
- `Indexer` writes documents through the store and embedder.
- `Retriever` performs keyword, vector, or hybrid search.
- `RetrievalRuntime` owns the indexer, retriever, and cleanup lifecycle.

`RetrievalRuntime` creates its `Indexer` and mode-specific `Retriever` from the
supplied components. Use it as an async context manager so the store and
embedder close even when indexing or retrieval raises.

### Local Hugging Face embeddings

```python
from justatom.etc.schema import Document
from justatom.retrieval import EmbeddingProfile, RetrievalRuntime
from justatom.retrieval.embedders.huggingface import HuggingFaceEmbedder
from justatom.storing.weaviate import WeaviateDocumentStore

documents = [Document(content="Индекс хранит документы для поиска.")]

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

The context manager is the cleanup step; do not separately close its owned
store or embedder.

### Remote OpenAI-compatible embeddings

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
    await runtime.index([Document(content="Удаленный эмбеддер обслуживает индекс.")])
    results = await runtime.retrieve("как работает удаленный эмбеддер", top_k=5)
```

For an embedding deployment backed by Triton, expose its supported server over
OpenAI-compatible HTTP and use `OpenAICompatibleEmbedder`. The `justatom`
client has no `tritonclient` dependency and is safe to run on macOS; the Triton
server remains on its supported host.

## Containerized Deployment

The production retrieval API and model inference are separate processes. The
`Dockerfile.api` image contains no Torch or model weights; it calls an
OpenAI-compatible embedding endpoint. `Dockerfile.embedder.cpu` provides a
portable CPU service, while `Dockerfile.embedder.cuda` is for Linux/NVIDIA
hosts. Docker Desktop on macOS cannot expose MPS to containers, so Apple
Silicon inference remains a native host process.

Use `scripts/services.sh` as the supported deployment launcher. It selects one
of the external, CPU, or CUDA modes and keeps the API, embedder, and Weaviate
configuration aligned. The [Launch Guide](docs/launch-guide.md) has copyable
native MPS, CPU, CUDA, and external-backend commands.

## Strict Retrieval Configuration

`build_runtime` accepts a strict `retrieval` mapping. It rejects unknown keys,
requires `store`, requires an `embedding` section for `vector` and `hybrid`, and
only accepts `keyword`, `vector`, or `hybrid` modes.

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
    collection: JustAtom
    url: http://localhost:2211
    grpc_port: 50051
```

For a remote deployment, set `embedding.backend` to `openai-compatible` and
provide `base_url`, `model`, and an optional `api_key`. See the [Launch
Guide](docs/launch-guide.md) for evaluator commands and [Runtime
details](docs/modules/runtime.md) for the component boundaries.

## Training and Benchmarks

The training wrappers retain three variants: `vanilla`, `atom_gate`, and
`atomic`. Retrieval settings use `--search-mode keyword|vector|hybrid`; legacy
retrieval flags are not accepted.

```bash
bash scripts/run_pipeline.sh \
  --dataset-ids demo-eval \
  --method atomic \
  --search-mode hybrid \
  --model intfloat/multilingual-e5-small \
  --weaviate-url http://localhost:2211

bash scripts/run_benchmark.sh \
  --dataset-ids demo-eval \
  --search-mode vector \
  --dry-run
```

The benchmark wrapper writes shell-escaped command arrays to `COMMANDS.md` in
the benchmark directory, alongside result summaries.

## Documentation

```bash
make docs-serve
make docs-build
```

The MkDocs site includes the [architecture](docs/architecture.md), [runtime
module](docs/modules/runtime.md), [storage module](docs/modules/storage.md),
and [launch guide](docs/launch-guide.md).
