# Habr IR Release Artifacts Design

Date: 2026-07-14

## Decision

Habr IR uses two artifact layers:

1. JSONL preserves the generation and validation audit trail.
2. Parquet is the canonical release, training, evaluation, and Hugging Face
   Dataset format.

`evidence` is an exact continuous quote from the target passage. It is audit
metadata and never replaces the full positive passage. Retriever training uses
`positive_passage`, which is the exact serialized target passage referenced by
`positive_passage_id`.

## Artifact Boundary

Provider response JSONL is immutable and checksummed. The derived
`generation_collected.jsonl` remains a checksummed staging artifact that may be
reproduced when the validator or finalizer version changes. It preserves one
record per submitted generation request, including accepted and rejected
responses, reason codes, and the provider binding through `custom_id`. It is
not consumed directly by model training.

Finalization joins each accepted generation to the immutable target and corpus
artifacts, validates the join, and writes release Parquet files. Rejected rows
remain in the audit trail and never enter release pairs or qrels.

## Hugging Face Layout

The dataset repository uses explicit Hugging Face configurations because pair,
corpus, and qrel rows have different schemas:

```text
README.md
data/
  pairs/
    train.parquet
    validation.parquet
    test.parquet
  corpus-100k/
    corpus.parquet
  qrels/
    train.parquet
    validation.parquet
    test.parquet
  manifests/
    release-manifest.json
```

The dataset card declares three initial configurations:

- `pairs`: `train`, `validation`, and `test` splits;
- `corpus-100k`: one `train` split containing the complete retrieval corpus;
- `qrels`: `train`, `validation`, and `test` splits.

The corpus configuration uses the conventional Hugging Face `train` split
name only as a transport convention. Corpus rows are not training pairs.
Additional nested corpus sizes become separate configurations such as
`corpus-250k`, without changing query IDs or qrels.

Consumers load the release without a custom dataset script:

```python
from datasets import load_dataset

pairs = load_dataset("justatom/habr-ir", "pairs")
corpus = load_dataset("justatom/habr-ir", "corpus-100k", split="train")
qrels = load_dataset("justatom/habr-ir", "qrels")
```

## Pair Contract

Every accepted pair row contains:

```text
pair_id, query_id, article_id, positive_passage_id, split,
query, answer, evidence, positive_passage,
requested_intent, actual_intent,
title, section, url, topic_flows, topic_hubs, tags,
generator_model, generator_prompt_hash, generation_attempt,
generation_custom_id, generation_batch_id
```

Required invariants:

- `query`, `answer`, `evidence`, and `positive_passage` are non-empty;
- `evidence` is an exact substring of `positive_passage` and is not supported
  only by an overlap prefix;
- `positive_passage_id` resolves to exactly one corpus row;
- `positive_passage` exactly equals that corpus row's serialized passage;
- every `query_id`, `pair_id`, and normalized query is unique;
- split is inherited from deterministic article-level target selection;
- no article appears in more than one pair split.

Release IDs are independent of row order and generated wording:

```text
query_id = "q-" + sha256("habr-ir-query-v1\0" + generation_custom_id)
pair_id = "pair-" + sha256(
    "habr-ir-pair-v1\0" + query_id + "\0" + positive_passage_id
)
```

The full hexadecimal digest is retained. A bounded retry keeps the same
`generation_custom_id`, and therefore the same release IDs, while recording
the accepted `generation_attempt` and `generation_batch_id`.

`answer` and `evidence` support audit and future answer-aware experiments. The
minimal bi-encoder training view is `(query, positive_passage)`.

## Corpus Contract

Corpus rows contain:

```text
passage_id, article_id, title, section, content, serialized_passage,
url, flows, hubs, tags, char_count, token_count,
corpus_rank, is_positive, source_hash
```

`serialized_passage` is copied into pair rows as `positive_passage`. The corpus
keeps non-target and duplicate-content passages as legitimate distractors,
while target selection continues to exclude globally duplicated content.

## Qrel Contract

Qrel rows contain:

```text
query_id, passage_id, relevance
```

Every accepted query has exactly one qrel with `relevance=1`, and its
`passage_id` equals the pair row's `positive_passage_id`. A qrel split must
contain exactly the query IDs from the corresponding pair split.

## Finalization Flow

1. Verify generation, targets, context, and corpus fingerprints and checksums.
2. Read accepted rows from the collected generation audit.
3. Resolve each immutable `custom_id` to its target slot and passage.
4. Materialize pair, corpus, and qrel rows in deterministic order.
5. Run all pair, split, corpus, qrel, and evidence invariants.
6. Write each artifact to a temporary file, fsync, checksum, and atomically
   rename it into the release directory.
7. Write a manifest containing schemas, row counts, checksums, source and model
   revisions, prompt hashes, generation fingerprint, and producing git SHA.

Any missing or ambiguous binding fails finalization. The process never guesses
by row order and never silently drops an accepted generation.

## Publication

Publication uploads the checked release directory and dataset card to a private
`justatom/habr-ir` dataset repository by default. The raw JSONL audit remains
local unless a separate release policy explicitly includes it; provider raw
payloads and rejected generations are not part of the default Dataset Viewer.

The upload command must verify local checksums before upload and then validate
the Hub configurations, splits, row counts, and first rows through the Hugging
Face Dataset Viewer API.

## Testing

Unit tests use small local fixtures and no network. They cover:

- exact `custom_id` to target and corpus joins;
- exact evidence containment in `positive_passage`;
- rejected-row exclusion without accepted-row loss;
- stable IDs and deterministic output order;
- article-disjoint splits and one qrel per accepted query;
- missing positives, duplicate IDs, duplicate normalized queries, and checksum
  mismatches as hard failures;
- idempotent finalization with unchanged inputs;
- Hugging Face dataset-card configuration paths matching produced files.

An integration test may upload a tiny private fixture repository, load all
three configurations with `datasets.load_dataset`, and remove the fixture after
verification. It is opt-in and never runs in the unit suite.

## Out of Scope

- Publishing the current 100-row pilot as the final benchmark.
- Enabling the 10k generation scale gate.
- LLM validation and human-audit UI.
- Nested 250k, 500k, and 1M corpus construction.
- Public release or license approval.
