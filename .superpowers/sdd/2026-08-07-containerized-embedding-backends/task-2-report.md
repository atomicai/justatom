# Task 2 Report: Embedding Server Settings and One-time Model Construction

## RED evidence

Added `tests/test_embedding_server.py` with the required settings, validation, and factory tests.

Command:

```text
conda run -n justatom python -m pytest tests/test_embedding_server.py -q
```

Result: collection failed as expected because `justatom.api.embedding_server` did not exist:

```text
ImportError: cannot import name 'embedding_server' from 'justatom.api'
1 error in 0.31s
```

## GREEN evidence

Added `justatom/api/embedding_server.py` with immutable `EmbeddingServerSettings`, strict positive-integer parsing, Qwen defaults, and `build_local_embedder()` using the existing `Embedder`, `HuggingFaceEmbedder`, and empty-prefix `EmbeddingProfile` boundary.

Focused command:

```text
conda run -n justatom python -m pytest tests/test_embedding_server.py -q
```

Result: `6 passed in 0.32s`.

Full-suite command:

```text
conda run -n justatom python -m pytest -q
```

Result: `372 passed, 9 warnings in 18.35s`.

## Changed files

- `justatom/api/embedding_server.py`
- `tests/test_embedding_server.py`
- `.superpowers/sdd/2026-08-07-containerized-embedding-backends/task-2-report.md`

## Self-review

- Settings are frozen and read from an explicit mapping or `os.environ`.
- Model and device values reject blank input.
- Batch size and max length reject zero, negative, and non-integer values with `ConfigurationError`.
- The factory creates one `HuggingFaceEmbedder` with one empty-prefix profile and forwards configured batch size and max length.
- No HTTP routes, server lifecycle ownership, caching, or unrelated refactors were added.
- `git diff --check` passed.

## Concerns

The full suite retains the 9 pre-existing dependency warnings documented by the task context. No new warnings or test failures were observed.

## Fix Round 1

Updated the existing factory test to pass the explicit non-default `EMBEDDING_MAX_LENGTH=256` and assert that the constructed profile has `max_length == 256`. Production code was unchanged because it already forwards `settings.max_length`.

Command:

```text
conda run -n justatom python -m pytest tests/test_embedding_server.py -q
```

Result: `6 passed in 0.31s`.
