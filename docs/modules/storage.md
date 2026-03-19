# justatom.storing

The `justatom.storing` package focuses on persistence and backend-facing storage logic.

## Files

- `dataset.py`: dataset storage and adapter-facing persistence helpers
- `weaviate.py`: Weaviate integration points
- `mask.py`: storage-related support layer

## When You Read This Package

Open this part of the codebase when you need to answer questions like:

- how documents are persisted
- how dataset objects are represented for downstream retrieval work
- how Weaviate is wired into the rest of the project

## Why It Is Separate

Keeping storage concerns out of `justatom.running` and `justatom.modeling` makes the system easier to swap, test, and reason about.
