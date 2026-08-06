# Storage

`WeaviateDocumentStore` is the async document-store implementation used by the
retrieval runtime. It owns one async Weaviate client, creates the collection
when necessary, and implements document writes, deletes, keyword search, vector
search, hybrid search, and shutdown.

## Connect Explicitly

```python
from justatom.storing.weaviate import WeaviateDocumentStore

store = await WeaviateDocumentStore.connect(
    "JustAtom",
    url="http://localhost:2211",
    grpc_port=50051,
)
try:
    count = await store.count_documents()
finally:
    await store.close()
```

When the store is passed to `RetrievalRuntime`, let the runtime close it through
`async with RetrievalRuntime(...)` instead of closing it separately.

## Runtime Role

The store is the `DocumentStore` component in the public composition:

```text
WeaviateDocumentStore + Embedder -> RetrievalRuntime -> Indexer + Retriever
```

Keyword mode needs only the store. Vector and hybrid modes also require an
embedder. Store URLs must be absolute `http` or `https` origins, and the
retrieval config validates collection names, gRPC ports, and unknown keys before
opening a connection.
