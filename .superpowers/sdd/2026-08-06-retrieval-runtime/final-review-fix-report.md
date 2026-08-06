# Final Review Fix Report

Workspace: `/Users/thebat/IProject/justatom/.worktrees/ci-pipelines`

## Focused RED

`conda run -n justatom python -m pytest tests/retrieval/test_weaviate_store_api.py tests/retrieval/test_indexer.py -q` exited 1 before production changes.

- `test_batch_write_rejects_weaviate_object_errors_without_leaking_provider_details`: failed because `_batch_write()` returned a reduced count instead of raising for the installed Weaviate `BatchObjectReturn.errors` mapping.
- `test_batch_write_wraps_client_exception_and_preserves_cause`: failed because a thrown client `RuntimeError` escaped `_batch_write()` directly.
- `test_indexer_keyword_mode_clears_embeddings_from_cloned_streamed_inputs`: failed because keyword-mode normalized clones retained the three supplied embeddings.

Result: `3 failed, 48 passed`.

## Fixes

- `WeaviateDocumentStore._batch_write()` now raises `DocumentStoreError` for every non-empty installed Weaviate `BatchObjectReturn.errors` mapping. Its message contains only bounded counts, object indexes, and error type names; it excludes document content and provider error bodies. Thrown client exceptions are wrapped with their original exception as `__cause__`.
- `Indexer.index()` now clears `embedding` on every normalized clone in keyword mode. The original `Document` and dict inputs remain unchanged.

No final-review Minor was changed: validation wrapping, server-message diagnostics, the JSON 500 handler, device logging, and deferred test gaps remain out of scope.

## Verification

1. Focused GREEN: `conda run -n justatom python -m pytest tests/retrieval/test_weaviate_store_api.py tests/retrieval/test_indexer.py -q` -- exit 0, `51 passed in 0.37s`.
2. `conda run -n justatom make format-check` -- exit 0, `157 files would be left unchanged`.
3. `conda run -n justatom pylint justatom --errors-only --disable=import-error,not-callable` -- exit 0, no diagnostics.
4. `conda run -n justatom python -m pytest tests -m "not integration and not network"` -- exit 0, `355 passed, 3 deselected, 6 warnings in 9.87s`.
5. Weaviate: `docker compose up -d weaviate` encountered an existing stale detached container; recreating it found ports `2211` and `50051` already owned by the independently running `production-training-weaviate-1`. The integration suite verified that live endpoint: `conda run -n justatom python -m pytest tests -m integration` -- exit 0, `1 passed, 357 deselected, 6 warnings in 9.66s`.
6. `conda run -n justatom python -m mkdocs build --strict` -- exit 0, built in `0.12s`.
7. Import isolation command for `OpenAICompatibleEmbedder` -- exit 0; neither `torch` nor `tritonclient` was imported.
8. Removed-architecture legacy scan -- exit 0 with no matches.

Commit: `fix: enforce retrieval indexing integrity`.
