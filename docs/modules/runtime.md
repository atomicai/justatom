# justatom.running

The `justatom.running` package is where orchestration lives. It connects encoders, embeddings, retrievers, evaluators, indexing flows, and service logic.

## Files

- `encoders.py`: encoding orchestration
- `embeddings/`: embedding-related runtime helpers
- `retriever.py`: retrieval flow composition
- `indexer.py`: indexing pipeline logic
- `evaluator.py`: evaluation runtime layer
- `trainer.py`: training runtime layer
- `trainer_jobs.py`: job-oriented training helpers
- `service.py`: service wrappers and runtime glue
- `llm.py`: LLM-facing runtime helpers
- `clusters.py`: clustering-oriented runtime flows

## Mental Model

Think of `justatom.running` as the coordination layer between:

- package entrypoints in `justatom.api`
- raw data structures from `justatom.processing`
- persistence backends from `justatom.storing`
- reusable model code in `justatom.modeling`

## Typical Flow

1. A CLI command enters through `justatom.api`.
2. Runtime objects inside `justatom.running` resolve encoders, retrievers, and scenario settings.
3. Data is loaded or normalized.
4. Results are evaluated, indexed, or served.
