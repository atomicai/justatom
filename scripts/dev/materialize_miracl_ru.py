from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / ".data" / "retrieval" / "miracl-ru-train.jsonl"


def _compose_content(title: str | None, text: str | None) -> str:
    clean_title = (title or "").strip()
    clean_text = (text or "").strip()
    if clean_title and clean_text.startswith(clean_title):
        return clean_text
    if clean_title and clean_text:
        return f"{clean_title}\n\n{clean_text}"
    return clean_text or clean_title


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize MIRACL-ru train pairs into the local JSONL format used by justatom.",
    )
    parser.add_argument("--split", default="train", help="MIRACL split to read from miracl/miracl (default: train)")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output JSONL path (default: .data/retrieval/miracl-ru-train.jsonl)",
    )
    parser.add_argument(
        "--limit-queries",
        type=int,
        default=None,
        help="Optional cap on the number of MIRACL queries to process.",
    )
    return parser.parse_args()


def materialize(*, split: str, output_path: Path, limit_queries: int | None = None) -> None:
    rows = load_dataset("miracl/miracl", "ru", split=split, trust_remote_code=True)
    if limit_queries is not None:
        rows = rows.select(range(min(limit_queries, len(rows))))

    docs_by_id: dict[str, dict[str, object]] = {}
    processed_queries = 0

    for row in rows:
        query = str(row.get("query", "")).strip()
        if not query:
            continue
        processed_queries += 1

        for passage in row.get("positive_passages") or []:
            doc_id = str(passage.get("docid", "")).strip()
            if not doc_id:
                continue

            title = str(passage.get("title", "") or "").strip()
            text = str(passage.get("text", "") or "").strip()
            content = _compose_content(title=title, text=text)
            if not content:
                continue

            record = docs_by_id.setdefault(
                doc_id,
                {
                    "chunk_id": doc_id,
                    "content": content,
                    "queries": [],
                    "title": title,
                    "source_dataset": "miracl/miracl",
                    "source_config": "ru",
                    "source_split": split,
                    "language": "ru",
                },
            )
            queries = record["queries"]
            if query not in queries:
                queries.append(query)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in docs_by_id.values():
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        "Wrote {} MIRACL-ru documents with {} unique query associations to {}",
        len(docs_by_id),
        sum(len(record["queries"]) for record in docs_by_id.values()),
        output_path,
    )
    logger.info("Processed {} source queries from miracl/miracl[ru]/{}", processed_queries, split)


if __name__ == "__main__":
    args = _parse_args()
    materialize(
        split=args.split,
        output_path=Path(args.output),
        limit_queries=args.limit_queries,
    )
