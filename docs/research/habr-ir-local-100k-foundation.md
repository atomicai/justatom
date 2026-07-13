# Habr IR Local 100k Foundation

Date: 2026-07-12

## Scope

This run validates the local corpus and retrieval foundation only. It does not
yet contain generated queries, qrels, train/dev/test splits, or human labels.

Source: private `justatom/habr-ds`, Russian articles, revision
`2c6fddf3812055062ce7cd5b1d00e24a6fe5f427`. Dense model revision:
`614241f622f53c4eeff9890bdc4f31cfecc418b3`.

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
`bb6ad903b82c337a61cce2b1cd5bf5dd7e3303b6b3263258979372c00e40c3c9`.
A repeated `prepare` invocation returned `reused: true`.

## Corpus

| Property | Value |
| --- | ---: |
| Passages | 100,000 |
| Represented articles | 24,226 |
| Duplicate passage IDs | 0 |
| Character range | 600-1,800 |
| Character median / p95 | 1,302 / 1,780 |
| Token maximum | 504 |
| Token median / p95 | 384 / 494 |
| Section character maximum | 240 |

The hardened pipeline took about 30 minutes on an M5 Max. Observed process RSS stayed
below approximately 5.2 GB. Dense embeddings are normalized float32 vectors
with shape `100000 x 384` (about 146.5 MiB).

Artifact identities:

| Artifact | SHA-256 |
| --- | --- |
| Passages | `603e1e89dc5bfa8966d6182eb2694278be35de69f856f23c9e866c5489ca5435` |
| BM25 index | `6544610feb9d56f8d369e58becce804ce4b85e474a74c69909f4768e9e26fbe1` |
| Dense index | `da31be52e382e9ec39dc9b58ee640ff4b747accebe5dd7999a502edd68c32ca4` |

## Retrieval Diagnostic

The diagnostic artifact contains 200 deterministic passage queries and 6,063
RRF-union neighbors.

| Contribution | Rows |
| --- | ---: |
| BM25 only | 2,314 |
| Dense only | 2,481 |
| Both | 1,163 |
| Same article / structural | 964 |
| Adjacent | 162 |
| Structural only | 105 |
| Self-neighbors | 0 |

Only 19.2% of diagnostic rows occur in both source rankings. BM25 and dense retrieval
therefore contribute meaningfully different candidate neighborhoods. Same-article
neighbors account for 15.9% of rows and are useful collision candidates for
testing whether a generated query is one-to-one with its target passage.
All 200 diagnostic targets have at least one same-article corpus sibling.

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
5. Passage-level corpus capping can leave singleton articles in the distractor
   corpus. The target sampler excludes them while retaining their passages as
   legitimate retrieval distractors.

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
