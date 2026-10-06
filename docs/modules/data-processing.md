# justatom.processing

The `justatom.processing` package handles the data path before model execution or indexing.

## Files

- `loader.py`: dataset and source loading helpers
- `sample.py`: sampling utilities
- `silo.py`: data grouping and batching abstractions
- `tokenizer.py`: tokenization helpers
- `prime.py`: shared preprocessing primitives

## Responsibilities

- load input records from local or configured sources
- shape records into the internal format expected by runtime code
- tokenize or batch data for downstream components
- support repeatable training and evaluation setup

## Related Packages

- use [`justatom.storing`](storage.md) when the question is persistence or vector backend integration
- use [`justatom.running`](runtime.md) when the question is orchestration
