# Habr IR Local 100k Foundation

Date: 2026-07-12

## Scope

This run validates the local corpus and retrieval foundation only. It does not
yet contain generated queries, qrels, train/dev/test splits, or human labels.

Source: private `justatom/habr-ds`, Russian articles, revision `main`.

## Reproduction

```bash
conda activate justatom
python -m justatom.api.ir_dataset \
  --config configs/datasets/habr-ir.yaml \
  --preparation.max_articles 25000 \
  --preparation.max_passages 100000 \
  --retrieval.query_passages 200 \
  --output.root .tmp_runs/datasets/habr-ir/local-100k \
  run
```

Preparation fingerprint:
`15b15dc36245fc16b4ded82cc7d94d8d38932b0aa3a0dc8b4da028567f52652c`.
A repeated `prepare` invocation returned `reused: true`.

## Corpus

| Property | Value |
| --- | ---: |
| Passages | 100,000 |
| Represented articles | 23,968 |
| Duplicate passage IDs | 0 |
| Character range | 600-1,800 |
| Character median / p95 | 1,301 / 1,780 |
| Token maximum | 504 |
| Token median / p95 | 382 / 493 |
| Section character maximum | 240 |

The pipeline took about 26 minutes on an M5 Max. Observed process RSS stayed
below approximately 5.2 GB. Dense embeddings are normalized float32 vectors
with shape `100000 x 384` (about 146.5 MiB).

## Retrieval Diagnostic

The diagnostic artifact contains 200 deterministic passage queries and 5,966
RRF-union neighbors.

| Contribution | Rows |
| --- | ---: |
| BM25 only | 2,288 |
| Dense only | 2,451 |
| Both | 1,227 |
| Same article | 966 |
| Self-neighbors | 0 |

Only 20.6% of union rows occur in both source rankings. BM25 and dense retrieval
therefore contribute meaningfully different candidate neighborhoods. Same-article
neighbors account for 16.2% of rows and are useful collision candidates for
testing whether a generated query is one-to-one with its target passage.

## Findings

1. Markdown-aware preparation, persistent BM25, dense E5 indexing, and exact
   top-k retrieval are ready as reusable local building blocks.
2. Random passages are not suitable query-generation targets. Manual inspection
   found useful prose alongside code-only, encoded, table-of-contents, and other
   low-questionability passages.
3. Code-heavy and same-article passages should remain in the retrieval corpus as
   distractors, but target passages need a separate deterministic quality gate.
4. The hybrid neighborhood should be passed to the labeling stage so the model
   can avoid queries that also match a near-duplicate or sibling passage.

## Next Stage

Build target selection and labeling on top of these immutable artifacts:

1. Compute passage quality features and reject low-questionability targets.
2. Sample targets by topic and query intent while keeping the 100k corpus fixed.
3. Generate grounded one-to-one queries using the target plus BM25/dense
   collision candidates.
4. Apply deterministic answerability, leakage, collision, and deduplication
   gates before human audit.
5. Produce 10k accepted pairs and qrels, then derive nested 250k/500k/1M corpus
   variants without changing query IDs.
