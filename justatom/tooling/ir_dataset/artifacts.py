from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl

from justatom.tooling.ir_dataset.chunking import CHUNKER_VERSION, MarkdownPassageChunker, Passage


@dataclass(frozen=True, slots=True)
class PrepareConfig:
    seed: int = 42
    max_articles: int | None = None
    max_passages: int | None = None
    max_passages_per_article: int = 8
    part_rows: int = 10_000

    def __post_init__(self) -> None:
        for name in ("max_articles", "max_passages"):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"preparation.{name} must be > 0 when set")
        if self.max_passages_per_article <= 0:
            raise ValueError("preparation.max_passages_per_article must be > 0")
        if self.part_rows <= 0:
            raise ValueError("preparation.part_rows must be > 0")


@dataclass(frozen=True, slots=True)
class PrepareSummary:
    passages_path: Path
    manifest_path: Path
    article_count: int
    passage_count: int
    fingerprint: str
    reused: bool


def _canonical_hash(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _preparation_fingerprint(
    *,
    source_fingerprint: str,
    chunker: MarkdownPassageChunker,
    config: PrepareConfig,
) -> str:
    return _canonical_hash(
        {
            "version": 1,
            "chunker_version": CHUNKER_VERSION,
            "source_fingerprint": source_fingerprint,
            "chunking": asdict(chunker.config),
            "preparation": asdict(config),
        }
    )


def _selection_key(seed: int, passage: Passage) -> str:
    return hashlib.sha256(f"{seed}:{passage.article_id}:{passage.passage_id}".encode("utf-8")).hexdigest()


def _passage_record(passage: Passage, *, selection_key: str) -> dict[str, Any]:
    return {
        "passage_id": passage.passage_id,
        "article_id": passage.article_id,
        "title": passage.title,
        "section": passage.section,
        "content": passage.content,
        "serialized_passage": passage.serialized_passage,
        "url": passage.url,
        "flows": list(passage.flows),
        "hubs": list(passage.hubs),
        "tags": list(passage.tags),
        "char_count": passage.char_count,
        "token_count": passage.token_count,
        "overlap_prefix_chars": passage.overlap_prefix_chars,
        "source_hash": passage.source_hash,
        "start_unit": passage.start_unit,
        "end_unit": passage.end_unit,
        "selection_key": selection_key,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _reuse_summary(output_dir: Path, fingerprint: str) -> PrepareSummary | None:
    passages_path = output_dir / "passages.parquet"
    manifest_path = output_dir / "manifest.json"
    if not passages_path.exists() or not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("fingerprint") != fingerprint:
        return None
    counts = manifest.get("counts") or {}
    return PrepareSummary(
        passages_path=passages_path,
        manifest_path=manifest_path,
        article_count=int(counts.get("articles", 0)),
        passage_count=int(counts.get("passages", 0)),
        fingerprint=fingerprint,
        reused=True,
    )


def _write_part(parts_dir: Path, index: int, records: list[dict[str, Any]]) -> Path:
    path = parts_dir / f"part-{index:06d}.parquet"
    pl.DataFrame(records).write_parquet(path, compression="zstd")
    return path


def prepare_passages(
    rows: Iterable[dict[str, Any]],
    output_dir: str | Path,
    chunker: MarkdownPassageChunker,
    config: PrepareConfig | None = None,
    source_fingerprint: str = "unknown",
) -> PrepareSummary:
    active_config = config or PrepareConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _preparation_fingerprint(
        source_fingerprint=source_fingerprint,
        chunker=chunker,
        config=active_config,
    )
    reusable = _reuse_summary(output_dir, fingerprint)
    if reusable is not None:
        return reusable

    work_dir = output_dir / f".prepare-{uuid.uuid4().hex}"
    parts_dir = work_dir / "parts"
    parts_dir.mkdir(parents=True)
    buffer: list[dict[str, Any]] = []
    part_paths: list[Path] = []
    accepted_articles = 0
    seen_article_ids: set[str] = set()

    try:
        for row in rows:
            if str(row.get("language", "")).strip().casefold() != "ru":
                continue
            if str(row.get("type", "")).strip().casefold() != "article":
                continue
            if not str(row.get("title", "")).strip() or not str(row.get("text_markdown", "")).strip():
                continue

            article_id = str(row.get("id", "")).strip()
            if not article_id or article_id in seen_article_ids:
                continue
            passages = chunker.chunk_article(row)
            if not passages:
                continue
            seen_article_ids.add(article_id)
            accepted_articles += 1

            ranked = sorted(passages, key=lambda passage: (_selection_key(active_config.seed, passage), passage.passage_id))
            for passage in ranked[: active_config.max_passages_per_article]:
                buffer.append(
                    _passage_record(
                        passage,
                        selection_key=_selection_key(active_config.seed, passage),
                    )
                )
                if len(buffer) >= active_config.part_rows:
                    part_paths.append(_write_part(parts_dir, len(part_paths), buffer))
                    buffer = []

            if active_config.max_articles is not None and accepted_articles >= active_config.max_articles:
                break

        if buffer:
            part_paths.append(_write_part(parts_dir, len(part_paths), buffer))
        if not part_paths:
            raise RuntimeError("No eligible passages were produced from the source rows.")

        final_temp = output_dir / f".passages.{uuid.uuid4().hex}.parquet"
        lazy = pl.scan_parquet(part_paths).sort(["selection_key", "passage_id"])
        if active_config.max_passages is not None:
            lazy = lazy.limit(active_config.max_passages)
        lazy.with_row_index("corpus_rank").sink_parquet(final_temp, compression="zstd")

        selected = (
            pl.scan_parquet(final_temp)
            .select(
                pl.len().alias("passages"),
                pl.col("article_id").n_unique().alias("articles"),
            )
            .collect()
        )
        passage_count = int(selected.item(0, "passages"))
        article_count = int(selected.item(0, "articles"))
        if passage_count <= 0:
            raise RuntimeError("Prepared passage artifact is empty.")

        final_path = output_dir / "passages.parquet"
        os.replace(final_temp, final_path)
        flow_counts: Counter[str] = Counter()
        for values in pl.scan_parquet(final_path).select("flows").collect()["flows"]:
            normalized_values = values.to_list() if isinstance(values, pl.Series) else list(values or [])
            flow_counts.update(normalized_values or ["other"])
        manifest = {
            "version": 1,
            "chunker_version": CHUNKER_VERSION,
            "fingerprint": fingerprint,
            "source_fingerprint": source_fingerprint,
            "chunking": asdict(chunker.config),
            "preparation": asdict(active_config),
            "counts": {"articles": article_count, "passages": passage_count},
            "flow_counts": dict(sorted(flow_counts.items())),
            "passages_file": final_path.name,
        }
        manifest_path = output_dir / "manifest.json"
        _write_json_atomic(manifest_path, manifest)
        return PrepareSummary(
            passages_path=final_path,
            manifest_path=manifest_path,
            article_count=article_count,
            passage_count=passage_count,
            fingerprint=fingerprint,
            reused=False,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


__all__ = ["PrepareConfig", "PrepareSummary", "prepare_passages"]
