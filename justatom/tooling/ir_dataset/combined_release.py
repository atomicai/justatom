from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from justatom.tooling.ir_dataset.artifacts import sha256_file
from justatom.tooling.ir_dataset.release import (
    CORPUS_COLUMNS,
    PAIR_SCHEMA,
    QREL_SCHEMA,
    RELEASE_ARTIFACT_PATHS,
)


COMBINED_RELEASE_CONTRACT = "habr-ir-combined-release-v1"
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class CombinedReleaseSummary:
    root: Path
    manifest_path: Path
    pair_count: int
    corpus_count: int
    qrel_count: int
    review_count: int
    fingerprint: str
    reused: bool


@dataclass(frozen=True, slots=True)
class _ValidatedRelease:
    root: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    pairs: dict[str, pl.DataFrame]
    qrels: dict[str, pl.DataFrame]
    corpus: pl.DataFrame
    audit: pl.DataFrame


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_manifest(root: Path) -> tuple[dict[str, Any], Path]:
    path = root / "data/manifests/release-manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"release manifest does not exist: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid release manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid release manifest: {path}")
    return value, path


def _validate_artifacts(root: Path, manifest: Mapping[str, Any]) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError(f"release manifest has invalid artifacts: {root}")
    paths = [item.get("path") for item in artifacts if isinstance(item, Mapping)]
    if len(paths) != len(artifacts) or len(paths) != len(set(paths)) or set(paths) != RELEASE_ARTIFACT_PATHS:
        raise ValueError(f"release artifact set mismatch: {root}")
    resolved_root = root.resolve()
    for item in artifacts:
        relative = item["path"]
        path = (root / relative).resolve()
        try:
            path.relative_to(resolved_root)
        except ValueError:
            raise ValueError(f"release artifact path escapes its root: {relative}") from None
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise ValueError(f"release artifact checksum mismatch: {path}")
        if path.stat().st_size != item.get("bytes"):
            raise ValueError(f"release artifact byte count mismatch: {path}")


def _require_schema(frame: pl.DataFrame, schema: Mapping[str, pl.DataType], label: str) -> None:
    if frame.columns != list(schema) or any(frame.schema[name] != dtype for name, dtype in schema.items()):
        raise ValueError(f"{label} schema mismatch")


def _validate_release(root: Path) -> _ValidatedRelease:
    manifest, manifest_path = _read_manifest(root)
    if manifest.get("version") != 1 or not isinstance(manifest.get("fingerprint"), str):
        raise ValueError(f"unsupported release manifest: {manifest_path}")
    _validate_artifacts(root, manifest)

    pairs: dict[str, pl.DataFrame] = {}
    qrels: dict[str, pl.DataFrame] = {}
    for split in SPLITS:
        pair_frame = pl.read_parquet(root / f"data/pairs/{split}.parquet")
        qrel_frame = pl.read_parquet(root / f"data/qrels/{split}.parquet")
        _require_schema(pair_frame, PAIR_SCHEMA, f"{root.name} {split} pairs")
        _require_schema(qrel_frame, QREL_SCHEMA, f"{root.name} {split} qrels")
        if pair_frame.filter(pl.col("split") != split).height:
            raise ValueError(f"{root.name} {split} pair file contains another split")
        expected_qrels = pair_frame.select(
            "query_id",
            pl.col("positive_passage_id").alias("passage_id"),
            pl.lit(1, dtype=pl.Int64).alias("relevance"),
        ).sort("query_id")
        if not qrel_frame.sort("query_id").equals(expected_qrels):
            raise ValueError(f"{root.name} {split} pairs and qrels disagree")
        pairs[split] = pair_frame
        qrels[split] = qrel_frame

    corpus = pl.read_parquet(root / "data/corpus-100k/corpus.parquet")
    expected_corpus_columns = [*CORPUS_COLUMNS, "is_positive"]
    if corpus.columns != expected_corpus_columns or corpus.schema["is_positive"] != pl.Boolean:
        raise ValueError(f"{root.name} corpus schema mismatch")
    if corpus["passage_id"].n_unique() != corpus.height or corpus["corpus_rank"].n_unique() != corpus.height:
        raise ValueError(f"{root.name} corpus identities are not unique")

    audit = pl.read_csv(root / "audit/pilot-review.csv", infer_schema_length=0)
    required_audit_columns = {"generation_custom_id", "automatic_status"}
    if not required_audit_columns.issubset(audit.columns):
        raise ValueError(f"{root.name} audit schema mismatch")
    if audit["generation_custom_id"].n_unique() != audit.height:
        raise ValueError(f"{root.name} audit contains duplicate generation_custom_id")
    statuses = set(audit["automatic_status"].drop_nulls().to_list())
    if not statuses.issubset({"accepted", "rejected"}):
        raise ValueError(f"{root.name} audit contains an invalid automatic_status")

    combined_pairs = pl.concat(list(pairs.values()), how="vertical")
    accepted_ids = set(audit.filter(pl.col("automatic_status") == "accepted")["generation_custom_id"].to_list())
    if accepted_ids != set(combined_pairs["generation_custom_id"].to_list()):
        raise ValueError(f"{root.name} accepted audit rows and pairs disagree")

    counts = manifest.get("counts")
    actual_split_counts = {split: pairs[split].height for split in SPLITS}
    actual_counts = {
        "pairs": sum(actual_split_counts.values()),
        "pair_splits": actual_split_counts,
        "corpus": corpus.height,
        "qrels": sum(frame.height for frame in qrels.values()),
        "audit_rows": audit.height,
    }
    if counts != actual_counts:
        raise ValueError(f"{root.name} release counts mismatch")
    return _ValidatedRelease(
        root=root,
        manifest=manifest,
        manifest_sha256=sha256_file(manifest_path),
        pairs=pairs,
        qrels=qrels,
        corpus=corpus,
        audit=audit,
    )


def _require_unique(frame: pl.DataFrame, column: str, label: str) -> None:
    duplicate = frame.group_by(column).len().filter(pl.col("len") > 1).select(column).head(1)
    if duplicate.height:
        raise ValueError(f"combined releases contain duplicate {label}: {duplicate.item()}")


def _release_descriptor(release: _ValidatedRelease) -> dict[str, Any]:
    manifest = release.manifest
    return {
        "name": release.root.name,
        "fingerprint": manifest["fingerprint"],
        "manifest_sha256": release.manifest_sha256,
        "generation": manifest.get("generation"),
        "git": manifest.get("git"),
        "counts": manifest.get("counts"),
    }


def _summary(root: Path, manifest: Mapping[str, Any], *, reused: bool) -> CombinedReleaseSummary:
    counts = manifest["counts"]
    return CombinedReleaseSummary(
        root=root,
        manifest_path=root / "data/manifests/release-manifest.json",
        pair_count=int(counts["pairs"]),
        corpus_count=int(counts["corpus"]),
        qrel_count=int(counts["qrels"]),
        review_count=int(counts["audit_rows"]),
        fingerprint=str(manifest["fingerprint"]),
        reused=reused,
    )


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _hf_arrow_type(dtype: pa.DataType) -> pa.DataType:
    if pa.types.is_large_string(dtype):
        return pa.string()
    if pa.types.is_large_binary(dtype):
        return pa.binary()
    if pa.types.is_large_list(dtype) or pa.types.is_list(dtype):
        return pa.list_(_hf_arrow_type(dtype.value_type))
    if pa.types.is_struct(dtype):
        return pa.struct([pa.field(field.name, _hf_arrow_type(field.type), nullable=field.nullable) for field in dtype])
    return dtype


def _write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = frame.to_arrow()
    schema = pa.schema([pa.field(field.name, _hf_arrow_type(field.type), nullable=field.nullable) for field in table.schema])
    pq.write_table(table.cast(schema), path, compression="zstd", row_group_size=50_000, write_page_index=True)
    _fsync_file(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def _dataset_card(pair_count: int, corpus_count: int) -> str:
    return f"""---
pretty_name: Habr IR
task_categories:
- information-retrieval
language:
- ru
size_categories:
- 10K<n<100K
tags:
- retrieval
- russian
- synthetic-queries
configs:
- config_name: pairs
  data_files:
  - split: train
    path: data/pairs/train.parquet
  - split: validation
    path: data/pairs/validation.parquet
  - split: test
    path: data/pairs/test.parquet
- config_name: corpus-100k
  data_files:
  - split: train
    path: data/corpus-100k/corpus.parquet
- config_name: qrels
  data_files:
  - split: train
    path: data/qrels/train.parquet
  - split: validation
    path: data/qrels/validation.parquet
  - split: test
    path: data/qrels/test.parquet
---

# Habr IR

Russian information-retrieval dataset built from technical Habr articles. It contains {pair_count:,} accepted
query-passage pairs and a shared corpus of {corpus_count:,} passages. Splits are assigned at article level to
prevent passages from one article leaking across train, validation, and test.

Queries and short answers were generated from passages and passed deterministic evidence, ambiguity, intent,
and competitor checks. The generation process is synthetic: these examples are not a substitute for fully
human-authored relevance judgments. The local pilot was manually reviewed before scale generation.

## Configurations

- `pairs`: queries, answers, positive passages, intent labels, and generation provenance.
- `corpus-100k`: the complete retrieval corpus; `is_positive` marks passages referenced by accepted pairs.
- `qrels`: one positive relevance judgment per query.

```python
from datasets import load_dataset

pairs = load_dataset("justatom/habr-ir", "pairs")
corpus = load_dataset("justatom/habr-ir", "corpus-100k", split="train")
qrels = load_dataset("justatom/habr-ir", "qrels")
```

## Source and limitations

The underlying article data comes from the `justatom/habr-ds` snapshot of Habr. Passages may contain names,
URLs, code, and other text present in the source articles. Generated questions can still contain factual or
linguistic errors despite automatic filtering. Inspect the release manifest for exact source fingerprints,
generator metadata, code revisions, counts, and SHA-256 checksums.
"""


def _validate_existing(root: Path, fingerprint: str) -> CombinedReleaseSummary:
    manifest, _ = _read_manifest(root)
    if manifest.get("contract") != COMBINED_RELEASE_CONTRACT or manifest.get("fingerprint") != fingerprint:
        raise ValueError("refusing to overwrite a combined release with a different fingerprint")
    _validate_artifacts(root, manifest)
    pair_splits = {split: pq.ParquetFile(root / f"data/pairs/{split}.parquet").metadata.num_rows for split in SPLITS}
    actual_counts = {
        "pairs": sum(pair_splits.values()),
        "pair_splits": pair_splits,
        "corpus": pq.ParquetFile(root / "data/corpus-100k/corpus.parquet").metadata.num_rows,
        "qrels": sum(pq.ParquetFile(root / f"data/qrels/{split}.parquet").metadata.num_rows for split in SPLITS),
        "audit_rows": pl.read_csv(root / "audit/pilot-review.csv", infer_schema_length=0).height,
    }
    if manifest.get("counts") != actual_counts:
        raise ValueError("combined release counts mismatch")
    return _summary(root, manifest, reused=True)


def combine_releases(
    release_roots: Sequence[str | Path],
    output_root: str | Path,
    *,
    git_sha: str,
    git_dirty: bool,
) -> CombinedReleaseSummary:
    """Validate and atomically combine disjoint Habr IR releases."""
    roots = [Path(root) for root in release_roots]
    if len(roots) < 2:
        raise ValueError("at least two release roots are required")
    if len({root.resolve() for root in roots}) != len(roots):
        raise ValueError("release roots must be unique")
    if not isinstance(git_sha, str) or not git_sha:
        raise ValueError("git_sha must be a non-empty string")

    releases = [_validate_release(root) for root in roots]
    releases.sort(key=lambda item: str(item.manifest["fingerprint"]))
    source = releases[0].manifest.get("source")
    if not isinstance(source, dict) or any(release.manifest.get("source") != source for release in releases[1:]):
        raise ValueError("source release corpus provenance mismatch")

    base_corpus = releases[0].corpus.drop("is_positive").sort("corpus_rank")
    for release in releases[1:]:
        if not release.corpus.drop("is_positive").sort("corpus_rank").equals(base_corpus):
            raise ValueError("source release corpora differ")

    pairs = pl.concat([release.pairs[split] for release in releases for split in SPLITS], how="vertical")
    qrels = pl.concat([release.qrels[split] for release in releases for split in SPLITS], how="vertical")
    audits = pl.concat([release.audit for release in releases], how="vertical")
    for column, label in (
        ("pair_id", "pair_id"),
        ("query_id", "query_id"),
        ("positive_passage_id", "positive_passage_id"),
        ("generation_custom_id", "generation_custom_id"),
    ):
        _require_unique(pairs, column, label)
    _require_unique(audits, "generation_custom_id", "audit generation_custom_id")
    normalized = pairs.with_columns(
        pl.col("query").str.to_lowercase().str.replace_all(r"\s+", " ").str.strip_chars().alias("_normalized_query")
    )
    _require_unique(normalized, "_normalized_query", "normalized query")
    split_conflicts = pairs.group_by("article_id").agg(pl.col("split").n_unique().alias("count")).filter(pl.col("count") > 1)
    if split_conflicts.height:
        raise ValueError(f"article appears in multiple splits: {split_conflicts['article_id'][0]}")

    expected_qrels = pairs.select(
        "query_id",
        pl.col("positive_passage_id").alias("passage_id"),
        pl.lit(1, dtype=pl.Int64).alias("relevance"),
    ).sort("query_id")
    if not qrels.sort("query_id").equals(expected_qrels):
        raise ValueError("combined pairs and qrels disagree")
    positive_ids = pairs["positive_passage_id"].to_list()
    corpus_ids = set(base_corpus["passage_id"].to_list())
    if not set(positive_ids).issubset(corpus_ids):
        raise ValueError("combined pairs reference passages missing from the corpus")
    corpus = base_corpus.with_columns(pl.col("passage_id").is_in(positive_ids).alias("is_positive"))

    descriptors = [_release_descriptor(release) for release in releases]
    fingerprint = _canonical_hash(
        {
            "contract": COMBINED_RELEASE_CONTRACT,
            "source": source,
            "releases": descriptors,
            "git": {"sha": git_sha, "dirty": bool(git_dirty)},
        }
    )
    output = Path(output_root)
    if output.exists():
        return _validate_existing(output, fingerprint)

    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        for split in SPLITS:
            split_pairs = pairs.filter(pl.col("split") == split).sort("query_id")
            split_qrels = qrels.join(split_pairs.select("query_id"), on="query_id", how="semi").sort("query_id")
            _write_parquet(split_pairs, temporary / f"data/pairs/{split}.parquet")
            _write_parquet(split_qrels, temporary / f"data/qrels/{split}.parquet")
        _write_parquet(corpus, temporary / "data/corpus-100k/corpus.parquet")
        audit_path = temporary / "audit/pilot-review.csv"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audits.sort("generation_custom_id").write_csv(audit_path, include_bom=True)
        _fsync_file(audit_path)
        _write_text(temporary / "README.md", _dataset_card(pairs.height, corpus.height))

        artifacts = []
        for path in sorted(item for item in temporary.rglob("*") if item.is_file()):
            artifacts.append(
                {
                    "path": path.relative_to(temporary).as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
        split_counts = {split: pairs.filter(pl.col("split") == split).height for split in SPLITS}
        manifest = {
            "version": 1,
            "contract": COMBINED_RELEASE_CONTRACT,
            "fingerprint": fingerprint,
            "source": source,
            "releases": descriptors,
            "git": {"sha": git_sha, "dirty": bool(git_dirty)},
            "counts": {
                "pairs": pairs.height,
                "pair_splits": split_counts,
                "corpus": corpus.height,
                "qrels": qrels.height,
                "audit_rows": audits.height,
            },
            "schemas": {
                "pairs": {name: str(dtype) for name, dtype in pairs.schema.items()},
                "corpus": {name: str(dtype) for name, dtype in corpus.schema.items()},
                "qrels": {name: str(dtype) for name, dtype in qrels.schema.items()},
                "audit": {name: str(dtype) for name, dtype in audits.schema.items()},
            },
            "artifacts": artifacts,
        }
        _write_text(
            temporary / "data/manifests/release-manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output)
        return _summary(output, manifest, reused=False)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


__all__ = ["CombinedReleaseSummary", "combine_releases"]
