from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from statistics import mean, pstdev

import torch
from torchmetrics.retrieval import RetrievalHitRate, RetrievalMAP, RetrievalMRR, RetrievalNormalizedDCG

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from justatom.tooling.dataset import DatasetRecordAdapter
from justatom.running.embeddings.local import LocalEmbeddingClient


def maybe_cuda_or_mps() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_documents(
    *,
    dataset_name_or_path: str,
    content_field: str,
    labels_field: str,
    chunk_id_col: str | None,
    limit: int | None,
) -> list[dict]:
    adapter = DatasetRecordAdapter.from_source(
        dataset_name_or_path,
        lazy=False,
        content_col=content_field,
        queries_col=labels_field,
        chunk_id_col=chunk_id_col,
        limit=limit,
    )
    return list(adapter.iterator(as_json=True))


def _unique_queries(documents: list[dict]) -> list[str]:
    return DatasetRecordAdapter.extract_queries(documents)


def _metric_objects(top_k: int):
    return {
        "HitRate": RetrievalHitRate(top_k=top_k),
        "mrr": RetrievalMRR(top_k=top_k),
        "map": RetrievalMAP(top_k=top_k),
        "ndcg": RetrievalNormalizedDCG(top_k=top_k),
    }


def _evaluate_scores(
    query_vectors: torch.Tensor,
    doc_vectors: torch.Tensor,
    queries: list[str],
    documents: list[dict],
    eval_top_k: list[int],
) -> dict[str, tuple[float, float]]:
    scores = query_vectors @ doc_vectors.T
    results: dict[str, list[float]] = {f"HitRate@{k}": [] for k in eval_top_k}
    results.update({f"mrr@{k}": [] for k in eval_top_k})
    results.update({f"map@{k}": [] for k in eval_top_k})
    results.update({f"ndcg@{k}": [] for k in eval_top_k})

    for query_idx, query in enumerate(queries):
        preds = scores[query_idx]
        target = torch.tensor(
            [1 if query in (doc.get("meta", {}).get("labels", []) or []) else 0 for doc in documents],
            dtype=torch.long,
        )
        indexes = torch.zeros(len(documents), dtype=torch.long)

        if int(target.sum().item()) == 0:
            continue

        for top_k in eval_top_k:
            for metric_name, metric in _metric_objects(top_k).items():
                value = metric(preds, target, indexes)
                metric.reset()
                results[f"{metric_name}@{top_k}"].append(float(value.item()))

    return {key: (mean(values), pstdev(values) if len(values) > 1 else 0.0) for key, values in results.items() if values}


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate embedding-only retrieval locally without Weaviate")
    parser.add_argument("--model", required=True, help="HF model id or local checkpoint path")
    parser.add_argument("--dataset", default="justatom")
    parser.add_argument("--content-field", default="content")
    parser.add_argument("--labels-field", default="queries")
    parser.add_argument("--chunk-id-col", default="chunk_id")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--query-prefix", default="query: ")
    parser.add_argument("--content-prefix", default="passage: ")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 5, 10])
    args = parser.parse_args()

    documents = _load_documents(
        dataset_name_or_path=args.dataset,
        content_field=args.content_field,
        labels_field=args.labels_field,
        chunk_id_col=args.chunk_id_col,
        limit=args.limit,
    )
    queries = _unique_queries(documents)

    device = maybe_cuda_or_mps()
    doc_client = LocalEmbeddingClient(
        model_name_or_path=args.model,
        device=device,
        prefix=args.content_prefix,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
    )
    query_client = LocalEmbeddingClient(
        model_name_or_path=args.model,
        device=device,
        prefix=args.query_prefix,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
    )

    doc_vectors = torch.tensor(await doc_client.embed([doc[args.content_field] for doc in documents]), dtype=torch.float32)
    query_vectors = torch.tensor(await query_client.embed(queries), dtype=torch.float32)

    metrics = _evaluate_scores(query_vectors, doc_vectors, queries, documents, sorted(set(args.top_k)))

    print(f"model={args.model}")
    print(f"dataset={args.dataset}")
    print(f"documents={len(documents)}")
    print(f"queries={len(queries)}")
    for name in sorted(metrics):
        metric_mean, metric_std = metrics[name]
        print(f"{name} mean={metric_mean:.4f} std={metric_std:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
