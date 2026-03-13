# justatom.api

The `justatom.api` package is the thin command-line layer of the project. It gives developers and CI jobs a clean way to launch training, evaluation, and runtime flows.

## Files

- `eval.py`: evaluation entrypoint
- `train.py`: training entrypoint
- `run.py`: runtime launcher and service-oriented startup logic

## Why It Matters

This package keeps operational commands separate from the deeper implementation layers. That separation is useful because:

- CI and shell users need stable entrypoints.
- notebooks can call into the same lower-level logic without duplicating CLI code.
- runtime startup remains easy to trace from one small surface area.

## Read With

Pair this page with:

- [`Launch Guide`](../launch-guide.md)
- [`justatom.running`](runtime.md)
