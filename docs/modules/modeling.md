# justatom.modeling

The `justatom.modeling` package contains lower-level model and metric building blocks.

## Files

- `core.py`: foundational model abstractions
- `head.py`: model heads and output-facing pieces
- `metrics.py`: metrics helpers
- `numeric.py`: numeric utilities
- `prime.py`: core shared model helpers
- `div.py`: supporting model utilities

## Typical Use

You usually reach `justatom.modeling` from higher layers such as `justatom.running.trainer` or `justatom.running.encoders`, but it is the right place when you need to inspect:

- how model components are composed
- where task-specific heads live
- where shared numeric or evaluation logic is defined

## Rule Of Thumb

If the question is "how do we run the system?" start with `justatom.api` or `justatom.running`.
If the question is "what is the model made of?" start here.
