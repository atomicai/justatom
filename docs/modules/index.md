# Modules

This section breaks the repository down by working area so you can navigate `justatom` without reading the whole tree first.

## Package Map

- [`justatom.api`](api-cli.md): user-facing command entrypoints such as train, eval, and runtime launch helpers.
- [`justatom.running`](runtime.md): orchestration for embeddings, retrievers, evaluators, services, and training jobs.
- [`justatom.processing`](data-processing.md): data loading, sampling, silo management, and tokenization.
- [`justatom.storing`](storage.md): dataset persistence and Weaviate-facing integrations.
- [`justatom.modeling`](modeling.md): model components, shared numeric helpers, and metrics.

## Recommended Reading Order

1. Start with `justatom.api` to understand how users enter the system.
2. Continue with `justatom.running` to see how experiments and services are orchestrated.
3. Read `justatom.processing` and `justatom.storing` to understand data movement.
4. End with `justatom.modeling` when you need the lower-level model internals.
