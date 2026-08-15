# Task 3 Report: Detach head input and remove lexical plumbing

## Implementation Summary

- Detached query embeddings before the alpha head and passed `alpha_gate.supervision_weight` into the objective.
- Added alpha-target distribution, mean absolute alpha error, and fixed low/medium/high target-confidence retrieval buckets.
- Removed lexical lookup construction, lexical row payloads, pair-score plumbing, retired mix-weight scheduling, and the third loader/factory value.
- Simplified training data functions to return sampled rows and `(frame, rows)`; training loaders now return `(loader, processor)`.

## TDD Evidence

Tests were changed before production code.

### RED

Command:

```bash
conda run -n justatom python -m pytest tests/test_training_module.py tests/test_train_data_preparation.py tests/test_training_job.py -q
```

Result: `14 failed, 13 passed`.

The failures demonstrated the stale production handoff: `effective_alpha_mix_weight` accessed removed config fields, confidence-bucket telemetry was absent, data functions still returned lexical lookup values, and `TrainingJob` still unpacked a three-value loader result. The resulting blocked alpha-path failures preceded the detached-input assertion.

Mutation check after GREEN: temporarily changing `self.alpha_gate(queries.detach())` back to `self.alpha_gate(queries)` made `test_atom_gate_bce_updates_only_head_parameters` fail because encoder parameters received gradients (`1 failed`). Detachment was restored immediately after the check.

### GREEN

```bash
conda run -n justatom python -m pytest tests/test_training_module.py tests/test_train_data_preparation.py tests/test_training_job.py -q
```

Result: `27 passed, 26 warnings in 6.17s`. The MPS finite-two-step test ran (no skip reported).

Focused task-file lint:

```bash
conda run -n justatom ruff check justatom/training/module.py justatom/training/telemetry.py justatom/training/data.py justatom/training/job.py tests/test_training_module.py tests/test_train_data_preparation.py tests/test_training_job.py
```

Result: `All checks passed!`

Additional checks:

- `git diff --check` passed.
- Retired production-symbol scan found no production references. Its sole result is `tests/test_training_config.py`, which intentionally verifies removed config fields are rejected.

## Files Changed

- `justatom/training/module.py`
- `justatom/training/telemetry.py`
- `justatom/training/data.py`
- `justatom/training/job.py`
- `tests/test_training_module.py`
- `tests/test_train_data_preparation.py`
- `tests/test_training_job.py`

## Self-Review

- The BCE-only test uses the real alpha head and encoder and proves the intended gradient boundary; it also proves a live head input fails.
- Bucket metrics always emit all three bucket keys, preserving stable CSV columns when a bucket is empty.
- Repository-wide caller search found no remaining users of the old data, loader, or checkpoint-construction signatures.
- Existing `ObjectiveOutput` fakes now provide the Task 2 alpha fields.

## Concerns

- The exact broad lint command from the brief still exits nonzero with 20 pre-existing findings in untouched files: `config.py`, `core.py`, `diagnostics.py`, `gradient_projection.py`, `loss.py`, and `memory_bank.py`. Task-owned files lint cleanly.
- Test output contains 26 third-party deprecation and Lightning data-loader warnings; no test failures or task-owned warnings were reported.
