# Final Alpha Fix Round 1 Report

## Scope and Result

Repaired alpha supervision to consume live gate logits with
`binary_cross_entropy_with_logits`. `QueryAlphaGate.forward()` still returns
probabilities, while the training module now obtains logits through
`QueryAlphaGate.logits()` and keeps the encoder-side gate input detached.

The confidence target remains
`softmax((stop_gradient(q) @ stop_gradient(p).T) / stop_gradient(tau)).diagonal()`.
SimCSE continues to use `1 - stop_gradient(sigmoid(alpha_logits))`. Alpha-only
backpropagation reaches the head/logits and not query embeddings, positive
embeddings, or learnable temperature. The gate network structure, checkpoint
parameter names/shapes, checkpoint schema version, and bank behavior were not
changed.

## RED

Command:

```bash
conda run -n justatom pytest tests/test_alpha_gate.py tests/test_training_objective.py tests/test_training_module.py tests/test_training_telemetry.py -q
```

Result: expected RED, `4 failed, 35 passed, 1 skipped`.

Failures were exactly the missing `QueryAlphaGate.logits()` method and missing
`ObjectiveInputs.alpha_logits` interface. The new target, gradient-isolation,
telemetry, and CUDA tests were otherwise collectable, so the failures isolated
the required production change.

## GREEN

Command:

```bash
conda run -n justatom pytest tests/test_alpha_gate.py tests/test_training_objective.py tests/test_training_module.py tests/test_training_telemetry.py -q
```

Result: GREEN, `39 passed, 1 skipped`.

The focused suite covers:

- probability compatibility: `gate(q) == sigmoid(gate.logits(q))`;
- non-unit learnable-temperature confidence targets and logits BCE;
- supervision-only gradient isolation for queries, positives, and `log_tau`,
  with finite logits/head gradients;
- low, medium, and high telemetry buckets plus NaN metrics for empty buckets;
- FP16 CUDA-autocast forward/backward through the real training module.

## CUDA Autocast and Mutation Coverage

Command:

```bash
conda run -n justatom pytest tests/test_training_module.py::test_atom_gate_alpha_supervision_runs_under_cuda_fp16_autocast -rs -q
```

Result: `1 skipped` with `CUDA unavailable`.

Environment check reported `cuda_available=False` and `mps_available=True`.
The test is skip-guarded with `torch.cuda.is_available()` and, on a CUDA host,
runs the real module under `torch.autocast(device_type="cuda", dtype=torch.float16)`
through alpha-supervision backward. Replacing the production
`binary_cross_entropy_with_logits` call with probability-form
`binary_cross_entropy` causes that CUDA-autocast path to hit PyTorch's unsafe
BCE error; that mutation could not be executed on this MPS Mac and remains
honestly recorded as skipped here.

## Verification

Scoped Ruff and whitespace check:

```bash
git diff --check
conda run --no-capture-output -n justatom ruff check justatom/training/alpha_gate.py justatom/training/objective.py justatom/training/module.py tests/test_alpha_gate.py tests/test_training_objective.py tests/test_training_module.py
```

Result: both passed.

Complete offline, non-network suite:

```bash
conda run --no-capture-output -n justatom python -m pytest tests -m "not integration and not network" -q
```

Result: completed with no failures; 575 selected tests reached 100%, including
one expected CUDA skip. Collection confirmed `575/595 tests collected (20 deselected)`.
The suite emits pre-existing third-party deprecation warnings from Lightning,
Matplotlib, and PyParsing.

## Files Changed

- `justatom/training/alpha_gate.py`
- `justatom/training/objective.py`
- `justatom/training/module.py`
- `tests/test_alpha_gate.py`
- `tests/test_training_objective.py`
- `tests/test_training_module.py`
- `.superpowers/sdd/2026-08-14-difficulty-supervised-alpha/final-fix-1-report.md`

## Self-Review and Concerns

Reviewed the alpha data flow end to end: detached encoder outputs feed gate
logits; detached query/positive embeddings and detached temperature form the
target; detached alpha probabilities weight SimCSE; only logits feed the
supervision loss. No stateful layer, configuration field, checkpoint schema,
or memory-bank path changed, so gate parameter names and tensor shapes remain
identical.

Concern: CUDA hardware is unavailable in this environment, so the new smoke
test and its probability-BCE mutation sensitivity are CI/CUDA-host checks,
not locally executed evidence. All CPU/MPS-compatible focused and offline
verification completed.
