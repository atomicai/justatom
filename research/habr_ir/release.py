from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from huggingface_hub import HfApi

from justatom.tooling.ir_dataset.artifacts import sha256_file
from justatom.tooling.ir_dataset.release import (
    CORPUS_COLUMNS,
    PAIR_SCHEMA,
    QREL_SCHEMA,
    RELEASE_ARTIFACT_PATHS,
)


RELEASE_CONTRACT = "habr-ir-retrieval-release-v1"
SOURCE_SPLITS = ("train", "validation", "test")
DEFAULT_SPLIT_MAP = {"train": "train", "validation": "dev", "test": "test", "corpus": "corpus"}

PUBLIC_SCHEMA = {
    "pair_id": pl.String,
    "query_id": pl.String,
    "positive_doc_id": pl.String,
    "negative_doc_id": pl.String,
    "query": pl.String,
    "positive": pl.String,
    "doc_id": pl.String,
    "content": pl.String,
    "source": pl.String,
    "bucket": pl.String,
    "answer": pl.String,
    "evidence": pl.String,
    "article_id": pl.String,
    "title": pl.String,
    "section": pl.String,
    "url": pl.String,
    "requested_intent": pl.String,
    "actual_intent": pl.String,
    "topic_flows": pl.List(pl.String),
    "topic_hubs": pl.List(pl.String),
    "tags": pl.List(pl.String),
    "generator_model": pl.String,
    "generator_prompt_hash": pl.String,
    "generation_attempt": pl.Int64,
    "generation_custom_id": pl.String,
    "generation_batch_id": pl.String,
    "is_positive": pl.Boolean,
}


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
class ReleaseConfig:
    repo_id: str
    private: bool
    config_name: str
    layout: str
    source_releases: tuple[Path, ...]
    output_root: Path
    split_map: dict[str, str]
    include_audit: bool
    include_qrels_artifacts: bool

    def __post_init__(self) -> None:
        if "/" not in self.repo_id:
            raise ValueError("release.repo_id must use namespace/name format")
        if not self.config_name:
            raise ValueError("release.config_name must be non-empty")
        if self.layout != "retrieval":
            raise ValueError("release.layout must be 'retrieval'")
        if len(self.source_releases) < 2:
            raise ValueError("release.source_releases must contain at least two releases")
        if set(self.split_map) != set(DEFAULT_SPLIT_MAP) or len(set(self.split_map.values())) != 4:
            raise ValueError("release.split_map must map train, validation, test, and corpus to unique split names")


@dataclass(frozen=True, slots=True)
class _ValidatedRelease:
    root: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    pairs: dict[str, pl.DataFrame]
    qrels: dict[str, pl.DataFrame]
    corpus: pl.DataFrame
    audit: pl.DataFrame


def load_release_config(path: str | Path) -> ReleaseConfig:
    config_path = Path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        raise FileNotFoundError(f"release config does not exist: {config_path}") from None
    release = payload.get("release")
    if not isinstance(release, Mapping):
        raise ValueError("release config must contain a 'release' mapping")
    allowed = {
        "repo_id",
        "private",
        "config_name",
        "layout",
        "source_releases",
        "output_root",
        "split_map",
        "include_audit",
        "include_qrels_artifacts",
    }
    unknown = sorted(set(release) - allowed)
    if unknown:
        raise ValueError(f"unknown release config keys: {', '.join(unknown)}")
    try:
        return ReleaseConfig(
            repo_id=str(release["repo_id"]),
            private=bool(release.get("private", True)),
            config_name=str(release.get("config_name", "default")),
            layout=str(release.get("layout", "retrieval")),
            source_releases=tuple(Path(value) for value in release["source_releases"]),
            output_root=Path(release["output_root"]),
            split_map={str(key): str(value) for key, value in (release.get("split_map") or DEFAULT_SPLIT_MAP).items()},
            include_audit=bool(release.get("include_audit", True)),
            include_qrels_artifacts=bool(release.get("include_qrels_artifacts", True)),
        )
    except KeyError as exc:
        raise ValueError(f"release config is missing required key: {exc.args[0]}") from exc


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
    for split in SOURCE_SPLITS:
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
    actual_split_counts = {split: pairs[split].height for split in SOURCE_SPLITS}
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
        manifest_path=root / "manifest.json",
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


def _public_pair_frame(frame: pl.DataFrame, *, repo_id: str, split_name: str) -> pl.DataFrame:
    return frame.select(
        "pair_id",
        "query_id",
        pl.col("positive_passage_id").alias("positive_doc_id"),
        pl.lit("", dtype=pl.String).alias("negative_doc_id"),
        "query",
        pl.col("positive_passage").alias("positive"),
        pl.lit("", dtype=pl.String).alias("doc_id"),
        pl.lit("", dtype=pl.String).alias("content"),
        pl.lit(f"{repo_id}/{split_name}", dtype=pl.String).alias("source"),
        pl.lit("qrels", dtype=pl.String).alias("bucket"),
        "answer",
        "evidence",
        "article_id",
        "title",
        "section",
        "url",
        "requested_intent",
        "actual_intent",
        "topic_flows",
        "topic_hubs",
        "tags",
        "generator_model",
        "generator_prompt_hash",
        "generation_attempt",
        "generation_custom_id",
        "generation_batch_id",
        pl.lit(True, dtype=pl.Boolean).alias("is_positive"),
    )


def _public_corpus_frame(frame: pl.DataFrame, *, repo_id: str, split_name: str) -> pl.DataFrame:
    return frame.select(
        pl.lit("", dtype=pl.String).alias("pair_id"),
        pl.lit("", dtype=pl.String).alias("query_id"),
        pl.lit("", dtype=pl.String).alias("positive_doc_id"),
        pl.lit("", dtype=pl.String).alias("negative_doc_id"),
        pl.lit("", dtype=pl.String).alias("query"),
        pl.lit("", dtype=pl.String).alias("positive"),
        pl.col("passage_id").alias("doc_id"),
        pl.col("serialized_passage").alias("content"),
        pl.lit(f"{repo_id}/{split_name}", dtype=pl.String).alias("source"),
        pl.lit("", dtype=pl.String).alias("bucket"),
        pl.lit("", dtype=pl.String).alias("answer"),
        pl.lit("", dtype=pl.String).alias("evidence"),
        "article_id",
        "title",
        "section",
        "url",
        pl.lit("", dtype=pl.String).alias("requested_intent"),
        pl.lit("", dtype=pl.String).alias("actual_intent"),
        pl.col("flows").alias("topic_flows"),
        pl.col("hubs").alias("topic_hubs"),
        "tags",
        pl.lit("", dtype=pl.String).alias("generator_model"),
        pl.lit("", dtype=pl.String).alias("generator_prompt_hash"),
        pl.lit(0, dtype=pl.Int64).alias("generation_attempt"),
        pl.lit("", dtype=pl.String).alias("generation_custom_id"),
        pl.lit("", dtype=pl.String).alias("generation_batch_id"),
        "is_positive",
    )


def _dataset_card(
    pair_count: int,
    corpus_count: int,
    *,
    repo_id: str,
    config_name: str,
    split_map: Mapping[str, str],
) -> str:
    return f"""---
pretty_name: Habr IR
task_categories:
- text-retrieval
language:
- ru
size_categories:
- 10K<n<100K
tags:
- retrieval
- russian
- synthetic-queries
configs:
- config_name: {config_name}
  data_files:
  - split: {split_map['train']}
    path: data/{split_map['train']}-*.parquet
  - split: {split_map['validation']}
    path: data/{split_map['validation']}-*.parquet
  - split: {split_map['test']}
    path: data/{split_map['test']}-*.parquet
  - split: {split_map['corpus']}
    path: data/{split_map['corpus']}-*.parquet
---

# Habr IR

Russian information-retrieval dataset built from technical Habr articles. It contains {pair_count:,} accepted
query-passage pairs and a shared corpus of {corpus_count:,} passages. Splits are assigned at article level to
prevent passages from one article leaking across train, validation, and test.

Queries and short answers were generated from passages and passed deterministic evidence, ambiguity, intent,
and competitor checks. The generation process is synthetic: these examples are not a substitute for fully
human-authored relevance judgments. The local pilot was manually reviewed before scale generation.

## Splits

- `{split_map['train']}`, `{split_map['validation']}`, and `{split_map['test']}` contain query-positive rows.
- `{split_map['corpus']}` contains candidate documents as `doc_id, content`.
- All splits share one schema and one `{config_name}` Dataset Viewer configuration.
- Raw qrels, the audit sheet, and release provenance remain downloadable repository artifacts.

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")
print(dataset["{split_map['train']}"][0]["query"])
print(dataset["{split_map['corpus']}"][0]["content"])
```

## Source and limitations

The underlying article data comes from the `justatom/habr-ds` snapshot of Habr. Passages may contain names,
URLs, code, and other text present in the source articles. Generated questions can still contain factual or
linguistic errors despite automatic filtering. Inspect the release manifest for exact source fingerprints,
generator metadata, code revisions, counts, and SHA-256 checksums.
"""


def _read_output_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"combined release manifest does not exist: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid combined release manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid combined release manifest: {path}")
    return value


def _validate_output_artifacts(root: Path, manifest: Mapping[str, Any]) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("combined release artifact list is invalid")
    paths = [item.get("path") for item in artifacts if isinstance(item, Mapping)]
    actual_paths = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.name != "manifest.json"
    }
    if len(paths) != len(artifacts) or len(paths) != len(set(paths)) or set(paths) != actual_paths:
        raise ValueError("combined release artifact set mismatch")
    resolved_root = root.resolve()
    for item in artifacts:
        path = (root / item["path"]).resolve()
        try:
            path.relative_to(resolved_root)
        except ValueError:
            raise ValueError(f"combined release artifact path escapes its root: {item['path']}") from None
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise ValueError(f"combined release artifact checksum mismatch: {path}")
        if path.stat().st_size != item.get("bytes"):
            raise ValueError(f"combined release artifact byte count mismatch: {path}")


def _validate_existing(root: Path, fingerprint: str) -> CombinedReleaseSummary:
    manifest = _read_output_manifest(root)
    if manifest.get("contract") != RELEASE_CONTRACT or manifest.get("fingerprint") != fingerprint:
        raise ValueError("refusing to overwrite a combined release with a different fingerprint")
    _validate_output_artifacts(root, manifest)
    release = manifest["release"]
    split_map = release["split_map"]
    split_counts = {
        name: pq.ParquetFile(root / f"data/{name}-00000-of-00001.parquet").metadata.num_rows for name in split_map.values()
    }
    if manifest["counts"]["splits"] != split_counts:
        raise ValueError("combined release split counts mismatch")
    schemas = [pq.read_schema(root / f"data/{name}-00000-of-00001.parquet") for name in split_map.values()]
    if any(schema != schemas[0] for schema in schemas[1:]):
        raise ValueError("combined release split schemas differ")
    return _summary(root, manifest, reused=True)


def combine_releases(
    release_roots: Sequence[str | Path],
    output_root: str | Path,
    *,
    git_sha: str,
    git_dirty: bool,
    repo_id: str = "justatom/habr-ir",
    config_name: str = "default",
    split_map: Mapping[str, str] | None = None,
    include_audit: bool = True,
    include_qrels_artifacts: bool = True,
) -> CombinedReleaseSummary:
    """Validate and atomically combine disjoint Habr IR releases."""
    roots = [Path(root) for root in release_roots]
    if len(roots) < 2:
        raise ValueError("at least two release roots are required")
    if len({root.resolve() for root in roots}) != len(roots):
        raise ValueError("release roots must be unique")
    if not isinstance(git_sha, str) or not git_sha:
        raise ValueError("git_sha must be a non-empty string")
    active_split_map = dict(split_map or DEFAULT_SPLIT_MAP)
    if set(active_split_map) != set(DEFAULT_SPLIT_MAP) or len(set(active_split_map.values())) != 4:
        raise ValueError("split_map must map train, validation, test, and corpus to unique split names")

    releases = [_validate_release(root) for root in roots]
    releases.sort(key=lambda item: str(item.manifest["fingerprint"]))
    source = releases[0].manifest.get("source")
    if not isinstance(source, dict) or any(release.manifest.get("source") != source for release in releases[1:]):
        raise ValueError("source release corpus provenance mismatch")

    base_corpus = releases[0].corpus.drop("is_positive").sort("corpus_rank")
    for release in releases[1:]:
        if not release.corpus.drop("is_positive").sort("corpus_rank").equals(base_corpus):
            raise ValueError("source release corpora differ")

    pairs = pl.concat([release.pairs[split] for release in releases for split in SOURCE_SPLITS], how="vertical")
    qrels = pl.concat([release.qrels[split] for release in releases for split in SOURCE_SPLITS], how="vertical")
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

    public_splits = {
        active_split_map[source_split]: _public_pair_frame(
            pairs.filter(pl.col("split") == source_split).sort("query_id"),
            repo_id=repo_id,
            split_name=active_split_map[source_split],
        )
        for source_split in SOURCE_SPLITS
    }
    public_splits[active_split_map["corpus"]] = _public_corpus_frame(
        corpus,
        repo_id=repo_id,
        split_name=active_split_map["corpus"],
    )

    descriptors = [_release_descriptor(release) for release in releases]
    fingerprint = _canonical_hash(
        {
            "contract": RELEASE_CONTRACT,
            "source": source,
            "releases": descriptors,
            "git": {"sha": git_sha, "dirty": bool(git_dirty)},
            "release": {
                "repo_id": repo_id,
                "config_name": config_name,
                "layout": "retrieval",
                "split_map": active_split_map,
                "include_audit": bool(include_audit),
                "include_qrels_artifacts": bool(include_qrels_artifacts),
            },
        }
    )
    output = Path(output_root)
    if output.exists():
        return _validate_existing(output, fingerprint)

    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        for split_name, frame in public_splits.items():
            _write_parquet(frame, temporary / f"data/{split_name}-00000-of-00001.parquet")
        if include_qrels_artifacts:
            for source_split in SOURCE_SPLITS:
                split_name = active_split_map[source_split]
                split_pairs = pairs.filter(pl.col("split") == source_split).select("query_id")
                split_qrels = qrels.join(split_pairs, on="query_id", how="semi").sort("query_id")
                _write_parquet(split_qrels, temporary / f"artifacts/qrels/{split_name}.parquet")
        if include_audit:
            audit_path = temporary / "audit/pilot-review.csv"
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audits.sort("generation_custom_id").write_csv(audit_path, include_bom=True)
            _fsync_file(audit_path)
        _write_text(
            temporary / "README.md",
            _dataset_card(
                pairs.height,
                corpus.height,
                repo_id=repo_id,
                config_name=config_name,
                split_map=active_split_map,
            ),
        )

        artifacts = []
        for path in sorted(item for item in temporary.rglob("*") if item.is_file()):
            artifacts.append(
                {
                    "path": path.relative_to(temporary).as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
        split_counts = {name: frame.height for name, frame in public_splits.items()}
        manifest = {
            "version": 1,
            "contract": RELEASE_CONTRACT,
            "fingerprint": fingerprint,
            "source": source,
            "releases": descriptors,
            "git": {"sha": git_sha, "dirty": bool(git_dirty)},
            "release": {
                "repo_id": repo_id,
                "config_name": config_name,
                "layout": "retrieval",
                "split_map": active_split_map,
                "include_audit": bool(include_audit),
                "include_qrels_artifacts": bool(include_qrels_artifacts),
            },
            "counts": {
                "pairs": pairs.height,
                "splits": split_counts,
                "corpus": corpus.height,
                "qrels": qrels.height,
                "audit_rows": audits.height,
            },
            "schemas": {
                "public": {name: str(dtype) for name, dtype in PUBLIC_SCHEMA.items()},
                "qrels": {name: str(dtype) for name, dtype in qrels.schema.items()},
                "audit": {name: str(dtype) for name, dtype in audits.schema.items()},
            },
            "artifacts": artifacts,
        }
        _write_text(
            temporary / "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output)
        return _summary(output, manifest, reused=False)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _git_identity() -> tuple[str, bool]:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True).stdout.strip())
    return sha, dirty


def build_release(config: ReleaseConfig) -> CombinedReleaseSummary:
    git_sha, git_dirty = _git_identity()
    return combine_releases(
        config.source_releases,
        config.output_root,
        git_sha=git_sha,
        git_dirty=git_dirty,
        repo_id=config.repo_id,
        config_name=config.config_name,
        split_map=config.split_map,
        include_audit=config.include_audit,
        include_qrels_artifacts=config.include_qrels_artifacts,
    )


def publish_release(config: ReleaseConfig, *, api: HfApi | None = None) -> str:
    active_api = api or HfApi()
    manifest = _read_output_manifest(config.output_root)
    _validate_existing(config.output_root, str(manifest.get("fingerprint")))
    expected = {
        "repo_id": config.repo_id,
        "config_name": config.config_name,
        "layout": config.layout,
        "split_map": config.split_map,
        "include_audit": config.include_audit,
        "include_qrels_artifacts": config.include_qrels_artifacts,
    }
    if manifest.get("release") != expected:
        raise ValueError("built release does not match publication config")
    active_api.create_repo(config.repo_id, repo_type="dataset", private=config.private, exist_ok=True)
    active_api.update_repo_settings(config.repo_id, repo_type="dataset", private=config.private)
    local_files = {path.relative_to(config.output_root).as_posix() for path in config.output_root.rglob("*") if path.is_file()}
    remote_files = set(active_api.list_repo_files(config.repo_id, repo_type="dataset"))
    stale_files = sorted(remote_files - local_files - {".gitattributes"})
    commit = active_api.upload_folder(
        repo_id=config.repo_id,
        repo_type="dataset",
        folder_path=config.output_root,
        path_in_repo=".",
        delete_patterns=stale_files or None,
        commit_message="Publish Habr IR retrieval release",
        commit_description=f"Release fingerprint: {manifest['fingerprint']}",
    )
    return str(commit.commit_url)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or publish the research-only Habr IR release.")
    parser.add_argument("action", choices=("build", "publish"))
    parser.add_argument("--config", default="configs/datasets/habr-ir-release.yaml")
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = load_release_config(args.config)
    result: Any = build_release(config) if args.action == "build" else {"url": publish_release(config)}
    payload = asdict(result) if isinstance(result, CombinedReleaseSummary) else result
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


__all__ = [
    "CombinedReleaseSummary",
    "PUBLIC_SCHEMA",
    "ReleaseConfig",
    "build_release",
    "combine_releases",
    "load_release_config",
    "publish_release",
]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
