from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import polars as pl
import yaml
from dotenv import find_dotenv, load_dotenv

from justatom.configuring.scenarios import deep_merge, parse_unknown_overrides
from justatom.tooling.ir_dataset.artifacts import PrepareConfig, PrepareSummary, prepare_passages
from justatom.tooling.ir_dataset.batch import (
    REQUIRED_SOURCE_CORPUS_FINGERPRINT,
    collect_completed_shards,
    prepare_generation_batches,
    refresh_batch_status,
    submit_pending_shards,
)
from justatom.tooling.ir_dataset.chunking import ChunkingConfig, MarkdownPassageChunker
from justatom.tooling.ir_dataset.dense import DenseIndex, E5TextEncoder, TextEncoder
from justatom.tooling.ir_dataset.generation import GeneratorConfig
from justatom.tooling.ir_dataset.generation_context import GenerationContextConfig, build_generation_context
from justatom.tooling.ir_dataset.neighbors import (
    NeighborBuildConfig,
    NeighborSummary,
    build_neighbor_artifact,
    merge_neighbors,
)
from justatom.tooling.ir_dataset.source import HabrSource, promote_hf_token_env
from justatom.tooling.ir_dataset.sparse import BM25_INDEX_VERSION, BM25Index, TECHNICAL_TOKEN_PATTERN
from justatom.tooling.ir_dataset.targets import TargetSelectionConfig, select_target_slots


@dataclass(frozen=True, slots=True)
class SourceConfig:
    repo_id: str = "justatom/habr-ds"
    config: str = "default"
    split: str = "train"
    revision: str = "main"
    cache_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    model_name: str = "intfloat/multilingual-e5-small"
    model_revision: str = "main"
    device: str = "mps"
    batch_size: int = 64
    dense_block_size: int = 65_536
    bm25_k: int = 20
    dense_k: int = 20
    union_k: int = 30
    rrf_k: int = 60
    query_passages: int = 200

    def __post_init__(self) -> None:
        for field_name in (
            "batch_size",
            "dense_block_size",
            "bm25_k",
            "dense_k",
            "union_k",
            "rrf_k",
            "query_passages",
        ):
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"retrieval.{field_name} must be > 0")


@dataclass(frozen=True, slots=True)
class OutputConfig:
    root: Path = Path(".tmp_runs/datasets/habr-ir/local-100k")
    generation_root: Path = Path(".tmp_runs/datasets/habr-ir/generation-v1")


@dataclass(frozen=True, slots=True)
class IRDatasetConfig:
    source: SourceConfig
    chunking: ChunkingConfig
    preparation: PrepareConfig
    retrieval: RetrievalConfig
    generation: GeneratorConfig
    output: OutputConfig


@dataclass(frozen=True, slots=True)
class ParsedCLI:
    stage: str
    config: IRDatasetConfig
    sample: int
    passage_id: str | None
    query: str | None


@dataclass(frozen=True, slots=True)
class LocalRunSummary:
    prepare: PrepareSummary
    embed_reused: bool
    neighbors: NeighborSummary


def _build_dataclass(cls, values: dict[str, Any] | None):
    raw = dict(values or {})
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} config keys: {', '.join(unknown)}")
    return cls(**raw)


def load_ir_dataset_config(
    config_path: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> IRDatasetConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"IR dataset config does not exist: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if overrides:
        raw = deep_merge(raw, overrides)
    source_values = dict(raw.get("source") or {})
    output_values = dict(raw.get("output") or {})
    if source_values.get("cache_dir") is not None:
        source_values["cache_dir"] = Path(source_values["cache_dir"])
    if output_values.get("root") is not None:
        output_values["root"] = Path(output_values["root"])
    if output_values.get("generation_root") is not None:
        output_values["generation_root"] = Path(output_values["generation_root"])
    return IRDatasetConfig(
        source=_build_dataclass(SourceConfig, source_values),
        chunking=_build_dataclass(ChunkingConfig, raw.get("chunking")),
        preparation=_build_dataclass(PrepareConfig, raw.get("preparation")),
        retrieval=_build_dataclass(RetrievalConfig, raw.get("retrieval")),
        generation=_build_dataclass(GeneratorConfig, raw.get("generation")),
        output=_build_dataclass(OutputConfig, output_values),
    )


def parse_cli(argv: list[str] | None = None) -> ParsedCLI:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    stages = (
        "prepare",
        "embed",
        "neighbors",
        "inspect",
        "run",
        "select-targets",
        "prepare-generation",
        "submit-generation",
        "generation-status",
        "collect-generation",
    )
    stage_positions = [(index, token) for index, token in enumerate(raw_argv) if token in stages]
    if len(stage_positions) != 1:
        raise ValueError(f"Exactly one IR dataset stage is required: {', '.join(stages)}")
    stage_index, stage = stage_positions[0]
    del raw_argv[stage_index]

    parser = argparse.ArgumentParser(description="Prepare and inspect the local Habr IR retrieval corpus.")
    parser.add_argument("--config", default="configs/datasets/habr-ir.yaml")
    parser.add_argument("--sample", type=int, default=10)
    parser.add_argument("--passage-id")
    parser.add_argument("--query")
    args, unknown = parser.parse_known_args(raw_argv)
    config = load_ir_dataset_config(args.config, overrides=parse_unknown_overrides(unknown))
    return ParsedCLI(
        stage=stage,
        config=config,
        sample=max(1, int(args.sample)),
        passage_id=args.passage_id,
        query=args.query,
    )


def _source(config: SourceConfig) -> HabrSource:
    return HabrSource(
        repo_id=config.repo_id,
        config=config.config,
        split=config.split,
        revision=config.revision,
        cache_dir=config.cache_dir,
    )


def prepare_stage(
    config: IRDatasetConfig,
    *,
    source: HabrSource | None = None,
    chunker: MarkdownPassageChunker | None = None,
) -> PrepareSummary:
    active_source = source or _source(config.source)
    active_chunker = chunker or MarkdownPassageChunker(config.chunking)
    return prepare_passages(
        rows=active_source.iter_rows(),
        output_dir=config.output.root,
        chunker=active_chunker,
        config=config.preparation,
        source_fingerprint=active_source.fingerprint(),
    )


def _embed_fingerprint(config: IRDatasetConfig, prepare_fingerprint: str) -> str:
    payload = {
        "prepare_fingerprint": prepare_fingerprint,
        "model_name": config.retrieval.model_name,
        "model_revision": config.retrieval.model_revision,
        "batch_size": config.retrieval.batch_size,
        "bm25_index_version": BM25_INDEX_VERSION,
        "bm25_token_pattern": TECHNICAL_TOKEN_PATTERN,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        with item.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def embed_stage(
    config: IRDatasetConfig,
    prepare_summary: PrepareSummary,
    *,
    encoder: TextEncoder | None = None,
) -> bool:
    root = config.output.root
    state_path = root / "embed_state.json"
    fingerprint = _embed_fingerprint(config, prepare_summary.fingerprint)
    bm25_dir = root / "bm25"
    dense_dir = root / "dense"
    if state_path.exists() and bm25_dir.exists() and dense_dir.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            valid = (
                state.get("fingerprint") == fingerprint
                and state.get("bm25_sha256") == _sha256_tree(bm25_dir)
                and state.get("dense_sha256") == _sha256_tree(dense_dir)
            )
        except (OSError, json.JSONDecodeError):
            valid = False
        if valid:
            return True

    frame = pl.read_parquet(prepare_summary.passages_path).sort("corpus_rank")
    rows = list(zip(frame["passage_id"].to_list(), frame["serialized_passage"].to_list(), strict=True))
    BM25Index.build(rows, bm25_dir)
    active_encoder = encoder or E5TextEncoder(
        model_name=config.retrieval.model_name,
        revision=config.retrieval.model_revision,
        device=config.retrieval.device,
        max_length=config.chunking.model_max_tokens,
    )
    DenseIndex.build(
        rows,
        dense_dir,
        active_encoder,
        batch_size=config.retrieval.batch_size,
    )
    _write_json_atomic(
        state_path,
        {
            "fingerprint": fingerprint,
            "count": frame.height,
            "bm25_sha256": _sha256_tree(bm25_dir),
            "dense_sha256": _sha256_tree(dense_dir),
        },
    )
    return False


def neighbors_stage(config: IRDatasetConfig) -> NeighborSummary:
    return build_neighbor_artifact(
        passages_path=config.output.root / "passages.parquet",
        bm25_index=BM25Index.load(config.output.root / "bm25", mmap=True),
        dense_index=DenseIndex.load(config.output.root / "dense"),
        output_path=config.output.root / "neighbors.parquet",
        config=NeighborBuildConfig(
            bm25_k=config.retrieval.bm25_k,
            dense_k=config.retrieval.dense_k,
            union_k=config.retrieval.union_k,
            rrf_k=config.retrieval.rrf_k,
            query_passages=config.retrieval.query_passages,
            dense_block_size=config.retrieval.dense_block_size,
            device=config.retrieval.device,
        ),
    )


def run_local_pipeline(config: IRDatasetConfig) -> LocalRunSummary:
    prepared = prepare_stage(config)
    embed_reused = embed_stage(config, prepared)
    neighbors = neighbors_stage(config)
    return LocalRunSummary(prepare=prepared, embed_reused=embed_reused, neighbors=neighbors)


def select_targets_stage(config: IRDatasetConfig) -> pl.DataFrame:
    passages_path = config.output.root / "passages.parquet"
    if not passages_path.exists():
        raise FileNotFoundError(f"Passage artifact does not exist: {passages_path}")
    return select_target_slots(
        pl.read_parquet(passages_path),
        TargetSelectionConfig(output_dir=config.output.generation_root),
    )


def _source_corpus_manifest_fingerprint(root: Path) -> str:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Source corpus manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid source corpus manifest: {manifest_path}") from exc
    fingerprint = manifest.get("fingerprint") if isinstance(manifest, dict) else None
    if fingerprint != REQUIRED_SOURCE_CORPUS_FINGERPRINT:
        raise ValueError("source corpus manifest fingerprint does not match the required Habr corpus")
    return fingerprint


def prepare_generation_stage(config: IRDatasetConfig) -> dict[str, Any]:
    source_root = config.output.root
    generation_root = config.output.generation_root
    targets_path = generation_root / "targets.parquet"
    passages_path = source_root / "passages.parquet"
    if not targets_path.exists():
        raise FileNotFoundError(f"Target artifact does not exist: {targets_path}")
    if not passages_path.exists():
        raise FileNotFoundError(f"Passage artifact does not exist: {passages_path}")
    context_path = generation_root / "generation_context.parquet"
    if context_path.exists():
        generation_context = pl.read_parquet(context_path)
    else:
        generation_context = build_generation_context(
            pl.read_parquet(targets_path),
            pl.read_parquet(passages_path),
            BM25Index.load(source_root / "bm25", mmap=True),
            DenseIndex.load(source_root / "dense"),
            GenerationContextConfig(
                bm25_k=config.retrieval.bm25_k,
                dense_k=config.retrieval.dense_k,
                union_k=config.retrieval.union_k,
                rrf_k=config.retrieval.rrf_k,
                dense_block_size=config.retrieval.dense_block_size,
                device=config.retrieval.device,
                output_dir=generation_root,
            ),
        )
    return prepare_generation_batches(
        pl.read_parquet(targets_path),
        generation_context,
        config.generation,
        generation_root,
        source_corpus_fingerprint=_source_corpus_manifest_fingerprint(source_root),
    )


def _generation_state(config: IRDatasetConfig) -> dict[str, Any]:
    path = config.output.generation_root / "generation_state.json"
    if not path.exists():
        raise FileNotFoundError(f"Generation batch state does not exist: {path}")
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid generation batch state: {path}") from exc
    if not isinstance(state, dict):
        raise ValueError(f"Invalid generation batch state: {path}")
    return state


def _openai_client_from_env() -> Any:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI Batch stages")
    from openai import OpenAI

    options: dict[str, str] = {"api_key": api_key}
    if base_url := os.environ.get("OPENAI_BASE_URL"):
        options["base_url"] = base_url
    return OpenAI(**options)


def inspect_stage(
    config: IRDatasetConfig,
    *,
    sample: int = 10,
    passage_id: str | None = None,
    query: str | None = None,
    bm25_index: BM25Index | None = None,
    dense_index: DenseIndex | None = None,
) -> list[dict[str, Any]]:
    if query is not None:
        passages_path = config.output.root / "passages.parquet"
        if not passages_path.exists():
            raise FileNotFoundError(f"Passage artifact does not exist: {passages_path}")
        frame = pl.read_parquet(passages_path).sort("corpus_rank")
        lexical_index = bm25_index or BM25Index.load(config.output.root / "bm25", mmap=True)
        semantic_index = dense_index
        if semantic_index is None:
            encoder = E5TextEncoder(
                model_name=config.retrieval.model_name,
                revision=config.retrieval.model_revision,
                device=config.retrieval.device,
                max_length=config.chunking.model_max_tokens,
            )
            semantic_index = DenseIndex.load(config.output.root / "dense", encoder=encoder)
        lexical = lexical_index.search([query], k=config.retrieval.bm25_k)[0]
        semantic = semantic_index.search_texts(
            [query],
            k=config.retrieval.dense_k,
            batch_size=config.retrieval.batch_size,
            block_size=config.retrieval.dense_block_size,
            device=config.retrieval.device,
        )[0]
        candidates = merge_neighbors(
            "__free_query__",
            lexical,
            semantic,
            rrf_k=config.retrieval.rrf_k,
            limit=config.retrieval.union_k,
        )
        metadata = {row["passage_id"]: row for row in frame.iter_rows(named=True)}
        return [
            {
                **asdict(candidate),
                "query": query,
                "candidate_title": metadata[candidate.candidate_id]["title"],
                "candidate_section": metadata[candidate.candidate_id]["section"],
                "candidate_preview": metadata[candidate.candidate_id]["content"][:500],
            }
            for candidate in candidates
        ]

    path = config.output.root / "neighbors.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Neighbor artifact does not exist: {path}")
    frame = pl.read_parquet(path).sort(["query_id", "rrf_score"], descending=[False, True])
    if passage_id is not None:
        return frame.filter(pl.col("query_id") == str(passage_id)).to_dicts()
    query_ids = frame["query_id"].unique(maintain_order=True).head(sample).to_list()
    return frame.filter(pl.col("query_id").is_in(query_ids)).to_dicts()


def _summary_payload(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: _summary_payload(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _summary_payload(item) for key, item in value.items()}
    return value


def main(argv: list[str] | None = None) -> int:
    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path)
    promote_hf_token_env()
    parsed = parse_cli(argv)
    if parsed.stage == "prepare":
        result = prepare_stage(parsed.config)
    elif parsed.stage == "embed":
        prepared = prepare_stage(parsed.config)
        result = {"prepare": prepared, "embed_reused": embed_stage(parsed.config, prepared)}
    elif parsed.stage == "neighbors":
        result = neighbors_stage(parsed.config)
    elif parsed.stage == "inspect":
        result = inspect_stage(
            parsed.config,
            sample=parsed.sample,
            passage_id=parsed.passage_id,
            query=parsed.query,
        )
    elif parsed.stage == "run":
        result = run_local_pipeline(parsed.config)
    elif parsed.stage == "select-targets":
        result = select_targets_stage(parsed.config)
    elif parsed.stage == "prepare-generation":
        result = prepare_generation_stage(parsed.config)
    elif parsed.stage == "submit-generation":
        result = submit_pending_shards(
            _generation_state(parsed.config), _openai_client_from_env(), parsed.config.output.generation_root
        )
    elif parsed.stage == "generation-status":
        result = refresh_batch_status(
            _generation_state(parsed.config), _openai_client_from_env(), parsed.config.output.generation_root
        )
    else:
        result = collect_completed_shards(
            _generation_state(parsed.config), _openai_client_from_env(), parsed.config.output.generation_root
        )
    print(json.dumps(_summary_payload(result), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
