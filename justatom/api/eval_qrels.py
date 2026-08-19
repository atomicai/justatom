from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from tqdm.auto import tqdm

from justatom.configuring.environment import load_runtime_environment
from justatom.configuring.scenarios import deep_merge, load_scenario_config, parse_unknown_overrides
from justatom.retrieval.contracts import Embedder, EmbeddingProfile
from justatom.retrieval.embedders.huggingface import HuggingFaceEmbedder, resolve_device
from justatom.running.qrels import exact_single_positive_ranks
from justatom.storing.datasets import DatasetLoader

load_runtime_environment()


@dataclass(frozen=True)
class BenchmarkRecords:
    query_ids: tuple[str, ...]
    queries: tuple[str, ...]
    positive_document_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    document_ids: tuple[str, ...]
    documents: tuple[str, ...]


@dataclass(frozen=True)
class DatasetContract:
    name_or_path: str
    config: str | None
    revision: str | None
    eval_split: str
    query_id_col: str
    query_field: str
    relevant_id_col: str
    group_id_col: str | None
    corpus_split: str
    document_id_col: str
    content_field: str


def load_eval_qrels_config(
    *,
    config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return load_scenario_config(
        "eval_qrels",
        config=config,
        config_path=config_path,
        overrides=overrides,
    )


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _required_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, path)


def resolve_dataset_contract(config: Mapping[str, Any]) -> DatasetContract:
    dataset = _mapping(config.get("dataset"), "dataset")
    eval_config = _mapping(dataset.get("eval"), "dataset.eval")
    corpus_config = _mapping(dataset.get("corpus"), "dataset.corpus")
    default_source = dataset.get("name_or_path")
    source = eval_config.get("name_or_path", default_source)
    corpus_source = corpus_config.get("name_or_path", default_source)
    source = _required_text(source, "dataset.name_or_path")
    corpus_source = _required_text(corpus_source, "dataset.corpus.name_or_path")
    if source != corpus_source:
        raise ValueError("eval and corpus must currently use the same dataset source")

    return DatasetContract(
        name_or_path=source,
        config=_optional_text(eval_config.get("config", dataset.get("config")), "dataset.eval.config"),
        revision=_optional_text(eval_config.get("revision", dataset.get("revision")), "dataset.eval.revision"),
        eval_split=_required_text(eval_config.get("split"), "dataset.eval.split"),
        query_id_col=_required_text(eval_config.get("query_id_col"), "dataset.eval.query_id_col"),
        query_field=_required_text(eval_config.get("query_field"), "dataset.eval.query_field"),
        relevant_id_col=_required_text(eval_config.get("relevant_id_col"), "dataset.eval.relevant_id_col"),
        group_id_col=_optional_text(eval_config.get("group_id_col"), "dataset.eval.group_id_col"),
        corpus_split=_required_text(corpus_config.get("split"), "dataset.corpus.split"),
        document_id_col=_required_text(corpus_config.get("document_id_col"), "dataset.corpus.document_id_col"),
        content_field=_required_text(corpus_config.get("content_field"), "dataset.corpus.content_field"),
    )


def _row_text(row: Mapping[str, Any], field: str, *, split: str) -> str:
    value = row.get(field)
    if value is None or not str(value).strip():
        raise ValueError(f"{split} contains an empty {field!r}")
    return str(value)


def _rows(
    loader: Callable[..., Iterable[dict[str, Any]]],
    contract: DatasetContract,
    *,
    split: str,
) -> Iterable[dict[str, Any]]:
    result = loader(
        contract.name_or_path,
        lazy=True,
        split=split,
        config=contract.config,
        revision=contract.revision,
    )
    return result


def load_benchmark_records(
    config: Mapping[str, Any],
    *,
    loader: Callable[..., Iterable[dict[str, Any]]] = DatasetLoader.read,
) -> BenchmarkRecords:
    contract = resolve_dataset_contract(config)
    query_ids: list[str] = []
    queries: list[str] = []
    positives: list[str] = []
    groups: list[str] = []
    for row in _rows(loader, contract, split=contract.eval_split):
        query_id = _row_text(row, contract.query_id_col, split=contract.eval_split)
        query_ids.append(query_id)
        queries.append(_row_text(row, contract.query_field, split=contract.eval_split))
        positives.append(_row_text(row, contract.relevant_id_col, split=contract.eval_split))
        groups.append(
            query_id if contract.group_id_col is None else _row_text(row, contract.group_id_col, split=contract.eval_split)
        )

    document_ids: list[str] = []
    documents: list[str] = []
    for row in _rows(loader, contract, split=contract.corpus_split):
        document_ids.append(_row_text(row, contract.document_id_col, split=contract.corpus_split))
        documents.append(_row_text(row, contract.content_field, split=contract.corpus_split))

    if not query_ids:
        raise ValueError(f"eval split {contract.eval_split!r} is empty")
    if not document_ids:
        raise ValueError(f"corpus split {contract.corpus_split!r} is empty")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("query IDs must be unique for single-positive evaluation")
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("corpus document IDs must be unique")
    missing = sorted(set(positives) - set(document_ids))
    if missing:
        preview = ", ".join(missing[:3])
        raise ValueError(f"{len(missing)} relevant documents are missing from corpus: {preview}")

    return BenchmarkRecords(
        query_ids=tuple(query_ids),
        queries=tuple(queries),
        positive_document_ids=tuple(positives),
        group_ids=tuple(groups),
        document_ids=tuple(document_ids),
        documents=tuple(documents),
    )


def _fingerprint(ids: Sequence[str], texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for identifier, text in zip(ids, texts, strict=True):
        for value in (identifier, text):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _local_model_fingerprint(model: str) -> str | None:
    root = Path(model)
    if not root.exists():
        return None
    files = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        relative = path.name if root.is_file() else path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _cached_hf_revision(model: str) -> str | None:
    if Path(model).exists():
        return None
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(model, "config.json")
    except (ImportError, OSError, ValueError):
        return None
    if not isinstance(cached, str):
        return None
    parts = Path(cached).parts
    try:
        snapshot_index = parts.index("snapshots")
        return parts[snapshot_index + 1]
    except (ValueError, IndexError):
        return None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    temporary.replace(path)


async def _encode_texts(
    embedder: Embedder,
    texts: Sequence[str],
    *,
    query: bool,
    outer_batch_size: int,
    label: str,
) -> np.ndarray:
    if outer_batch_size <= 0:
        raise ValueError("embedding.outer_batch_size must be positive")
    result: np.ndarray | None = None
    for start in tqdm(range(0, len(texts), outer_batch_size), desc=f"embed {label}"):
        stop = min(start + outer_batch_size, len(texts))
        batch = texts[start:stop]
        vectors = await (embedder.embed_queries(batch) if query else embedder.embed_documents(batch))
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != len(batch):
            raise ValueError(f"embedder returned an invalid {label} matrix")
        if result is None:
            result = np.empty((len(texts), array.shape[1]), dtype=np.float32)
        elif array.shape[1] != result.shape[1]:
            raise ValueError("embedding dimension changed between batches")
        result[start:stop] = array
    if result is None:
        raise ValueError(f"cannot embed empty {label}")
    return result


async def _load_or_encode(
    *,
    embedder: Embedder,
    ids: Sequence[str],
    texts: Sequence[str],
    query: bool,
    outer_batch_size: int,
    cache_path: Path,
    signature: Mapping[str, Any],
    reuse: bool,
    label: str,
) -> np.ndarray:
    metadata_path = cache_path.with_suffix(".json")
    expected = {**signature, "rows": len(texts), "text_fingerprint": _fingerprint(ids, texts)}
    if reuse and cache_path.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            cached = np.load(cache_path, mmap_mode="r", allow_pickle=False)
            if metadata == expected and cached.ndim == 2 and cached.shape[0] == len(texts):
                logger.info("Reusing {} embeddings from {}", label, cache_path)
                return cached
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    encoded = await _encode_texts(
        embedder,
        texts,
        query=query,
        outer_batch_size=outer_batch_size,
        label=label,
    )
    _save_array(cache_path, encoded)
    _write_json(metadata_path, expected)
    return np.load(cache_path, mmap_mode="r", allow_pickle=False)


def _positive_indices(records: BenchmarkRecords) -> np.ndarray:
    index = {document_id: position for position, document_id in enumerate(records.document_ids)}
    return np.asarray([index[document_id] for document_id in records.positive_document_ids], dtype=np.int64)


def _positive_int(config: Mapping[str, Any], key: str, default: int, *, section: str) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{section}.{key} must be a positive integer")
    return value


async def evaluate_qrels(
    *,
    config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    embedder: Embedder | None = None,
    loader: Callable[..., Iterable[dict[str, Any]]] = DatasetLoader.read,
) -> dict[str, Any]:
    resolved = load_eval_qrels_config(config=config, config_path=config_path, overrides=overrides)
    contract = resolve_dataset_contract(resolved)
    records = load_benchmark_records(resolved, loader=loader)
    embedding = _mapping(resolved.get("embedding"), "embedding")
    ranking = _mapping(resolved.get("ranking"), "ranking")
    output = _mapping(resolved.get("output"), "output")
    model = _required_text(embedding.get("model"), "embedding.model")
    output_dir = Path(_required_text(output.get("dir"), "output.dir")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_batch_size = _positive_int(embedding, "batch_size", 16, section="embedding")
    outer_batch_size = _positive_int(embedding, "outer_batch_size", 256, section="embedding")
    max_length = _positive_int(embedding, "max_length", 512, section="embedding")
    query_prefix = str(embedding.get("query_prefix") or "")
    document_prefix = str(embedding.get("document_prefix") or "")
    skip_prefix_if_present = bool(embedding.get("skip_prefix_if_present", True))
    requested_device = _required_text(embedding.get("device", "auto"), "embedding.device")
    embedding_device = resolve_device(requested_device)
    reuse_embeddings = bool(output.get("reuse_embeddings", True))

    owns_embedder = embedder is None
    if embedder is None:
        embedder = HuggingFaceEmbedder(
            model,
            device=embedding_device,
            profile=EmbeddingProfile(
                query_prefix=query_prefix,
                document_prefix=document_prefix,
                max_length=max_length,
                batch_size=embedding_batch_size,
                skip_prefix_if_present=skip_prefix_if_present,
            ),
        )

    model_revision = _cached_hf_revision(model)
    model_fingerprint = _local_model_fingerprint(model)
    signature = {
        "schema_version": 1,
        "model": str(Path(model).resolve()) if Path(model).exists() else model,
        "dataset": contract.name_or_path,
        "revision": contract.revision,
        "max_length": max_length,
        "embedding_batch_size": embedding_batch_size,
        "embedding_device": embedding_device,
        "query_prefix": query_prefix,
        "document_prefix": document_prefix,
        "skip_prefix_if_present": skip_prefix_if_present,
    }
    if model_fingerprint is not None:
        signature["model_fingerprint"] = model_fingerprint
    started = time.perf_counter()
    try:
        corpus_started = time.perf_counter()
        corpus_embeddings = await _load_or_encode(
            embedder=embedder,
            ids=records.document_ids,
            texts=records.documents,
            query=False,
            outer_batch_size=outer_batch_size,
            cache_path=output_dir / "corpus_embeddings.npy",
            signature={**signature, "role": "corpus", "split": contract.corpus_split},
            reuse=reuse_embeddings,
            label="corpus",
        )
        corpus_seconds = time.perf_counter() - corpus_started

        query_started = time.perf_counter()
        query_embeddings = await _load_or_encode(
            embedder=embedder,
            ids=records.query_ids,
            texts=records.queries,
            query=True,
            outer_batch_size=outer_batch_size,
            cache_path=output_dir / f"{contract.eval_split}_query_embeddings.npy",
            signature={**signature, "role": "query", "split": contract.eval_split},
            reuse=reuse_embeddings,
            label=contract.eval_split,
        )
        query_seconds = time.perf_counter() - query_started
    finally:
        if owns_embedder:
            await embedder.close()

    ranking_device = _required_text(ranking.get("device", "auto"), "ranking.device")
    ranking_device = embedding_device if ranking_device == "auto" else resolve_device(ranking_device)
    rank_started = time.perf_counter()
    rank_result = exact_single_positive_ranks(
        query_embeddings,
        corpus_embeddings,
        _positive_indices(records),
        device=ranking_device,
        query_batch_size=_positive_int(ranking, "query_batch_size", 64, section="ranking"),
        corpus_block_size=_positive_int(ranking, "corpus_block_size", 8192, section="ranking"),
    )
    rank_seconds = time.perf_counter() - rank_started

    ranks_path = output_dir / f"{contract.eval_split}.ranks.npz"
    np.savez_compressed(
        ranks_path,
        ranks=rank_result.ranks,
        positive_indices=rank_result.positive_indices,
        query_ids=np.asarray(records.query_ids),
        group_ids=np.asarray(records.group_ids),
        positive_document_ids=np.asarray(records.positive_document_ids),
    )
    metrics = rank_result.metrics()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name_or_path": contract.name_or_path,
            "config": contract.config,
            "revision": contract.revision,
            "eval_split": contract.eval_split,
            "corpus_split": contract.corpus_split,
            "queries": len(records.queries),
            "corpus": len(records.documents),
            "query_fingerprint": _fingerprint(records.query_ids, records.queries),
            "qrels_fingerprint": _fingerprint(records.query_ids, records.positive_document_ids),
            "group_fingerprint": _fingerprint(records.query_ids, records.group_ids),
            "corpus_fingerprint": _fingerprint(records.document_ids, records.documents),
        },
        "embedding": {
            "model": model,
            "resolved_revision": model_revision,
            "local_fingerprint": model_fingerprint,
            "device": embedding_device,
            "batch_size": embedding_batch_size,
            "max_length": max_length,
            "query_prefix": query_prefix,
            "document_prefix": document_prefix,
            "dimensions": int(query_embeddings.shape[1]),
        },
        "ranking": {
            "device": ranking_device,
            "similarity": "cosine",
            "dtype": "float32",
            "exact": True,
            "tie_policy": rank_result.tie_policy,
        },
        "metrics": metrics,
        "timing_seconds": {
            "corpus_embedding": corpus_seconds,
            "query_embedding": query_seconds,
            "ranking": rank_seconds,
            "total": time.perf_counter() - started,
        },
        "artifacts": {"ranks": str(ranks_path)},
    }
    results_path = output_dir / f"{contract.eval_split}.results.json"
    _write_json(results_path, payload)
    logger.info("Exact qrels evaluation written to {}", results_path)
    logger.info("Metrics: {}", metrics)
    return payload


def _parse_args(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        prog="justatom|eval-qrels",
        description="Exact dense evaluation with independent query, qrels and corpus splits.",
    )
    parser.add_argument("--config")
    parser.add_argument("--split", choices=("dev", "test"))
    args, unknown = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    overrides = parse_unknown_overrides(unknown)
    if args.split is not None:
        overrides = deep_merge(overrides, {"dataset": {"eval": {"split": args.split}}})
    return {"config_path": args.config, "overrides": overrides or None}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    return asyncio.run(evaluate_qrels(**_parse_args(argv)))


if __name__ == "__main__":
    main()


__all__ = [
    "BenchmarkRecords",
    "DatasetContract",
    "evaluate_qrels",
    "load_benchmark_records",
    "load_eval_qrels_config",
    "main",
    "resolve_dataset_contract",
]
