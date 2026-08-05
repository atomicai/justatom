from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

SOURCE_DATASET = "ir_datasets:mmarco/v2/ru"
DEFAULT_REPO_ID = "justatom/mmarco-ru-selected"
HF_TOKEN_ENV_NAMES = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN", "HF_API_KEY")
HF_COLUMNS = [
    "pair_id",
    "query_id",
    "positive_doc_id",
    "negative_doc_id",
    "query",
    "positive",
    "doc_id",
    "content",
    "source",
    "bucket",
]


@dataclass(frozen=True)
class SelectionConfig:
    seed: int = 42
    train_rows: int = 50_000
    dev_queries: int = 5_000
    eval_corpus_docs: int = 50_000
    query_min_chars: int = 3
    query_max_chars: int = 256
    positive_min_chars: int = 50
    positive_max_chars: int = 1500
    docpairs_scan_limit: int = 5_000_000
    verified_docpairs_only: bool = True


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def valid_pair(query: Any, positive: Any, cfg: SelectionConfig) -> bool:
    query_text = clean_text(query)
    positive_text = clean_text(positive)
    return (
        cfg.query_min_chars <= len(query_text) <= cfg.query_max_chars
        and cfg.positive_min_chars <= len(positive_text) <= cfg.positive_max_chars
    )


def render_dataset_card(*, repo_id: str, cfg: SelectionConfig) -> str:
    selection = asdict(cfg)
    return f"""---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
  - split: dev
    path: data/dev-*.parquet
  - split: corpus
    path: data/corpus-*.parquet
---

# mMARCO-ru-selected

Deterministic selected subset for JustAtom retrieval training and evaluation.

- Repository: `{repo_id}`
- Source: `{SOURCE_DATASET}`
- Seed: `{selection["seed"]}`
- Train rows target: `{selection["train_rows"]}`
- Dev query target: `{selection["dev_queries"]}`
- Eval corpus docs target: `{selection["eval_corpus_docs"]}`

Rows use `query` as the retrieval label and `positive` as the positive passage.
`pair_id` is unique for query-positive rows; the same passage can be relevant to
multiple queries in mMARCO, so `positive_doc_id` is intentionally not used as the
row id for train/dev splits.
The `corpus` split stores candidate documents as `doc_id, content`.
"""


def write_manifest(
    *,
    output_dir: Path,
    repo_id: str,
    cfg: SelectionConfig,
    counts: dict[str, int],
) -> Path:
    manifest = {
        "repo_id": repo_id,
        "source": SOURCE_DATASET,
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "selection": asdict(cfg),
        "counts": counts,
        "splits": {
            "train": "data/train-00000-of-00001.parquet",
            "dev": "data/dev-00000-of-00001.parquet",
            "corpus": "data/corpus-00000-of-00001.parquet",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def resolve_hf_token(env: dict[str, str] | None = None) -> str | None:
    active_env = os.environ if env is None else env
    for name in HF_TOKEN_ENV_NAMES:
        value = active_env.get(name)
        if value and str(value).strip():
            return str(value).strip()
    return None


def align_rows_for_hf(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    aligned: list[dict[str, str]] = []
    for row in rows:
        aligned.append({column: clean_text(row.get(column, "")) for column in HF_COLUMNS})
    return aligned


def _load_ir_dataset(dataset_id: str):
    try:
        import ir_datasets
    except Exception as ex:
        msg = "ir_datasets is required to materialize mMARCO-ru-selected."
        raise ImportError(msg) from ex
    return ir_datasets.load(dataset_id)


def _query_map(dataset) -> dict[str, str]:
    return {str(query.query_id): clean_text(query.text) for query in dataset.queries_iter()}


def _positive_qrels(dataset) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for qrel in dataset.qrels_iter():
        if getattr(qrel, "relevance", 0) > 0:
            rows.append((str(qrel.query_id), str(qrel.doc_id)))
    return rows


def _positive_sets(rows: list[tuple[str, str]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for query_id, doc_id in rows:
        out.setdefault(query_id, set()).add(doc_id)
    return out


def _doc_text(doc_store, doc_id: str) -> str | None:
    try:
        doc = doc_store.get(str(doc_id))
    except Exception:
        return None
    if doc is None:
        return None
    return clean_text(getattr(doc, "text", ""))


def _attach_verified_negatives(
    *,
    train_dataset,
    selected_rows: list[dict[str, Any]],
    positive_sets: dict[str, set[str]],
    scan_limit: int,
) -> None:
    by_query: dict[str, dict[str, Any]] = {}
    for row in selected_rows:
        if row.get("negative_doc_id"):
            continue
        by_query.setdefault(str(row["query_id"]), row)
    if not by_query:
        return

    found = 0
    for index, pair in enumerate(train_dataset.docpairs_iter()):
        if index >= scan_limit:
            break
        query_id = str(pair.query_id)
        row = by_query.get(query_id)
        if row is None:
            continue
        positive_id = str(pair.doc_id_a)
        if positive_id not in positive_sets.get(query_id, set()):
            continue
        negative_id = str(pair.doc_id_b)
        row["negative_doc_id"] = negative_id
        found += 1
        if found >= len(by_query):
            break


def build_train_rows(*, cfg: SelectionConfig) -> list[dict[str, Any]]:
    train = _load_ir_dataset("mmarco/v2/ru/train")
    doc_store = train.docs_store()
    queries = _query_map(train)
    qrels = _positive_qrels(train)
    positive_sets = _positive_sets(qrels)
    rng = random.Random(cfg.seed)
    rng.shuffle(qrels)

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for query_id, doc_id in qrels:
        if len(rows) >= cfg.train_rows:
            break
        key = (query_id, doc_id)
        if key in seen:
            continue
        query = queries.get(query_id)
        positive = _doc_text(doc_store, doc_id)
        if not valid_pair(query, positive, cfg):
            continue
        seen.add(key)
        rows.append(
            {
                "pair_id": f"{query_id}:{doc_id}",
                "query_id": query_id,
                "positive_doc_id": doc_id,
                "negative_doc_id": None,
                "query": clean_text(query),
                "positive": clean_text(positive),
                "source": "mmarco/v2/ru/train",
                "bucket": "qrels",
            }
        )

    if len(rows) < cfg.train_rows:
        raise RuntimeError(f"Only selected {len(rows)} train rows, target was {cfg.train_rows}.")

    if cfg.verified_docpairs_only and cfg.docpairs_scan_limit > 0:
        _attach_verified_negatives(
            train_dataset=train,
            selected_rows=rows,
            positive_sets=positive_sets,
            scan_limit=cfg.docpairs_scan_limit,
        )
    return rows


def build_dev_rows(*, cfg: SelectionConfig) -> list[dict[str, Any]]:
    dev = _load_ir_dataset("mmarco/v2/ru/dev")
    doc_store = dev.docs_store()
    queries = _query_map(dev)
    qrels = _positive_qrels(dev)
    rng = random.Random(cfg.seed + 1)
    rng.shuffle(qrels)

    rows: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    for query_id, doc_id in qrels:
        if len(rows) >= cfg.dev_queries:
            break
        if query_id in seen_queries:
            continue
        query = queries.get(query_id)
        positive = _doc_text(doc_store, doc_id)
        if not valid_pair(query, positive, cfg):
            continue
        seen_queries.add(query_id)
        rows.append(
            {
                "pair_id": f"{query_id}:{doc_id}",
                "query_id": query_id,
                "positive_doc_id": doc_id,
                "query": clean_text(query),
                "positive": clean_text(positive),
                "source": "mmarco/v2/ru/dev",
                "bucket": "qrels",
            }
        )

    if len(rows) < cfg.dev_queries:
        raise RuntimeError(f"Only selected {len(rows)} dev rows, target was {cfg.dev_queries}.")
    return rows


def build_corpus_rows(*, cfg: SelectionConfig, dev_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base = _load_ir_dataset("mmarco/v2/ru")
    doc_store = base.docs_store()
    rng = random.Random(cfg.seed + 2)

    rows: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()

    def add_doc(doc_id: str, source: str) -> bool:
        if doc_id in seen_doc_ids:
            return False
        text = _doc_text(doc_store, doc_id)
        if not text or not (cfg.positive_min_chars <= len(text) <= cfg.positive_max_chars):
            return False
        seen_doc_ids.add(doc_id)
        rows.append({"doc_id": doc_id, "content": text, "source": source})
        return True

    for row in dev_rows:
        add_doc(str(row["positive_doc_id"]), "dev_positive")

    doc_count = base.docs_count()
    attempts = 0
    max_attempts = max(cfg.eval_corpus_docs * 50, cfg.eval_corpus_docs + 10_000)
    while len(rows) < cfg.eval_corpus_docs and attempts < max_attempts:
        attempts += 1
        add_doc(str(rng.randrange(doc_count)), "random")

    if len(rows) < cfg.eval_corpus_docs:
        raise RuntimeError(f"Only selected {len(rows)} corpus docs, target was {cfg.eval_corpus_docs}.")
    return rows


def materialize(*, output_dir: Path, repo_id: str, cfg: SelectionConfig) -> dict[str, Path]:
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_rows = build_train_rows(cfg=cfg)
    dev_rows = build_dev_rows(cfg=cfg)
    corpus_rows = build_corpus_rows(cfg=cfg, dev_rows=dev_rows)

    train_path = data_dir / "train-00000-of-00001.parquet"
    dev_path = data_dir / "dev-00000-of-00001.parquet"
    corpus_path = data_dir / "corpus-00000-of-00001.parquet"
    pl.DataFrame(align_rows_for_hf(train_rows), schema=HF_COLUMNS).write_parquet(train_path)
    pl.DataFrame(align_rows_for_hf(dev_rows), schema=HF_COLUMNS).write_parquet(dev_path)
    pl.DataFrame(align_rows_for_hf(corpus_rows), schema=HF_COLUMNS).write_parquet(corpus_path)

    manifest_path = write_manifest(
        output_dir=output_dir,
        repo_id=repo_id,
        cfg=cfg,
        counts={"train": len(train_rows), "dev": len(dev_rows), "corpus": len(corpus_rows)},
    )
    readme_path = output_dir / "README.md"
    readme_path.write_text(render_dataset_card(repo_id=repo_id, cfg=cfg), encoding="utf-8")

    return {
        "train": train_path,
        "dev": dev_path,
        "corpus": corpus_path,
        "manifest": manifest_path,
        "readme": readme_path,
    }


def upload_folder(*, output_dir: Path, repo_id: str, private: bool) -> None:
    try:
        from dotenv import load_dotenv
        from huggingface_hub import HfApi, create_repo
    except Exception as ex:
        msg = "python-dotenv and huggingface_hub are required for upload."
        raise ImportError(msg) from ex

    load_dotenv()
    token = resolve_hf_token()
    if not token:
        supported = ", ".join(HF_TOKEN_ENV_NAMES)
        raise RuntimeError(f"HF token is missing. Set one of {supported} in .env or the environment.")

    create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)
    HfApi(token=token).upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output_dir),
        commit_message="Materialize mMARCO-ru-selected",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize and optionally upload justatom/mmarco-ru-selected.")
    parser.add_argument("--output-dir", default=".tmp_runs/datasets/mmarco-ru-selected")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--train-rows", type=int, default=SelectionConfig.train_rows)
    parser.add_argument("--dev-queries", type=int, default=SelectionConfig.dev_queries)
    parser.add_argument("--eval-corpus-docs", type=int, default=SelectionConfig.eval_corpus_docs)
    parser.add_argument("--seed", type=int, default=SelectionConfig.seed)
    parser.add_argument("--docpairs-scan-limit", type=int, default=SelectionConfig.docpairs_scan_limit)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = SelectionConfig(
        seed=args.seed,
        train_rows=args.train_rows,
        dev_queries=args.dev_queries,
        eval_corpus_docs=args.eval_corpus_docs,
        docpairs_scan_limit=args.docpairs_scan_limit,
    )
    output_dir = Path(args.output_dir)
    paths = materialize(output_dir=output_dir, repo_id=args.repo_id, cfg=cfg)
    for name, path in paths.items():
        print(f"{name}: {path}")
    if args.upload:
        upload_folder(output_dir=output_dir, repo_id=args.repo_id, private=bool(args.private))
        print(f"uploaded: {args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
