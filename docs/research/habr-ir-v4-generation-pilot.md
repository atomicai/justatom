# Habr IR v4 Generation Pilot

Date: 2026-07-14

## Purpose

This run replaces the historical chunker-v3 generation pilot with a fully
bound chunker-v4 pilot. The v4 corpus prevents overlap from crossing Markdown
section boundaries and pins both the source dataset and tokenizer revisions.
No 10k generation run is authorized by this record.

## Corpus Identity

Source: private `justatom/habr-ds`, revision
`2c6fddf3812055062ce7cd5b1d00e24a6fe5f427`. Tokenizer and dense model
revision: `614241f622f53c4eeff9890bdc4f31cfecc418b3`.

| Property | Value |
| --- | ---: |
| Prepared passages | 100,000 |
| Represented articles | 24,259 |
| Maximum source articles scanned | 25,000 |
| Chunker version | 4 |
| Maximum passage tokens | 504 |
| Globally duplicated content rows | 102 |
| Globally duplicated content values | 25 |

Globally duplicated content remains available as retrieval distractors but is
excluded from target selection.

| Artifact | SHA-256 |
| --- | --- |
| Corpus manifest fingerprint | `9c98a176e4bf6869742cc9a59379ea7375c029d7eb36c3b7f3d37e0f62b253c6` |
| Passages | `55c3757f1dd1cd26a6cf735d2313c38cbe47ff33cca7ae67e91d3f5449866bbd` |
| BM25 index | `b0be1059c988bd6ecdc855ac3dc60dbfb1131c1a6bb0458aa73e0232b37610b1` |
| Dense index | `c78e760ab9e166adb7380f1dda1790a044236be87a595f0ddab7e48dfafb967f` |

The retrieval diagnostic contains 200 deterministic passage queries and 6,032
union-neighbor rows: 3,520 BM25 contributions, 3,669 dense contributions, and
934 structural contributions.

## Pilot Contract

The pilot selects 100 unique target passages from 50 articles, with two
distinct requested intents per article and article-disjoint 80/10/10
train/dev/test slots. It attaches three collision candidates to every target,
for 300 context rows total.

| Artifact | SHA-256 |
| --- | --- |
| Targets | `bd57dbdc6e3881a943f893eb44f2a7d5e28c3b6ac8dbaa31490bc1ee1679c749` |
| Generation context | `66e4647446138f30ad8bb8e4f57fae7e49e3f2fbcf6052d029ca4623be8f48f3` |
| Batch request shard | `0e22b922759f2d5d090e304b1317ec9ad6ad97082bc2e4597ad80a9105f67d88` |
| Generation fingerprint | `7628151eb1a4233d2ed5174bf4a30ea2cbfaf4c5d4268ab9e26b639e8781beb3` |

The single 100-request shard uses `gpt-5.6-terra` with low reasoning effort.
It was submitted as OpenAI batch
`batch_6a556faaa13c81908fbfe8ec96712b3b`; its status was `validating` when this
record was written.

## Scale Gate

The checked-in config keeps `generation.scale_authorized: false`. A 10k run
must not be prepared or submitted until this exact v4 pilot is collected and
meets both preregistered thresholds:

1. Usable generation rate at least 70%.
2. Deterministic validation pass rate at least 60%.

The earlier v2 pilot achieved 96% usable and 94% deterministic pass rates after
the fixed finalizer, but it is tied to the old chunker-v3 corpus and therefore
cannot authorize v4 scale generation.

## Reproduction

```bash
conda activate justatom
python -m justatom.api.ir_dataset prepare --config configs/datasets/habr-ir.yaml
python -m justatom.api.ir_dataset embed --config configs/datasets/habr-ir.yaml
python -m justatom.api.ir_dataset neighbors --config configs/datasets/habr-ir.yaml
python -m justatom.api.ir_dataset select-targets --config configs/datasets/habr-ir.yaml
python -m justatom.api.ir_dataset prepare-generation --config configs/datasets/habr-ir.yaml
python -m justatom.api.ir_dataset submit-generation --config configs/datasets/habr-ir.yaml
python -m justatom.api.ir_dataset generation-status --config configs/datasets/habr-ir.yaml
```
