from collections.abc import Callable

import torch
from more_itertools import chunked
from torchmetrics.retrieval import (
    RetrievalHitRate,
    RetrievalMAP,
    RetrievalMRR,
    RetrievalNormalizedDCG,
)
from tqdm.asyncio import tqdm_asyncio

from justatom.modeling.metrics import IAdditiveMetric
from justatom.running.mask import IEvaluatorRunner, IRetrieverRunner


def _normalize_metric_name(metric_name: str) -> str:
    cleaned = metric_name.strip().lower().replace("_", "").replace("-", "")
    cleaned = cleaned.rstrip("@")
    aliases = {
        "hitrate": "HitRate",
        "hr": "HitRate",
        "mrr": "mrr",
        "map": "map",
        "ndcg": "ndcg",
    }
    if cleaned not in aliases:
        raise ValueError(
            f"Unsupported retrieval metric '{metric_name}'. Use one of: HitRate, mrr, map, ndcg"
        )
    return aliases[cleaned]


def _build_metric(metric_name: str, top_k: int):
    if metric_name == "HitRate":
        return RetrievalHitRate(top_k=top_k)
    if metric_name == "mrr":
        return RetrievalMRR(top_k=top_k)
    if metric_name == "map":
        return RetrievalMAP(top_k=top_k)
    if metric_name == "ndcg":
        return RetrievalNormalizedDCG(top_k=top_k)
    raise ValueError(f"Unknown retrieval metric '{metric_name}'")


class EvaluatorRunner(IEvaluatorRunner):
    def __init__(self, ir: IRetrieverRunner):
        super().__init__(ir=ir)
        self.metrics = dict()

    @torch.no_grad()
    async def evaluate_topk(
        self,
        queries: str | list[str],
        metrics: str | Callable,
        metrics_top_k: list[str | Callable],
        eval_top_k: int | list[int] | None = [1, 2, 5, 10, 12, 15, 20],
        top_k: int = 20,
        batch_size: int = 10,
    ):
        del metrics

        if eval_top_k is None:
            eval_top_k = [1, 2, 5, 10, 12, 15, 20]
        elif isinstance(eval_top_k, int):
            eval_top_k = [eval_top_k]
        eval_top_k = sorted(set(int(k) for k in eval_top_k))

        normalized_metric_names: list[str] = []
        metric_prefix_by_name: dict[str, str] = {}
        for metric_name in metrics_top_k:
            if not isinstance(metric_name, str):
                raise ValueError(
                    "Only string metric names are supported in metrics_top_k: HitRate, mrr, map, ndcg"
                )
            normalized_name = _normalize_metric_name(metric_name)
            if normalized_name not in normalized_metric_names:
                normalized_metric_names.append(normalized_name)
            metric_prefix_by_name[normalized_name] = f"{normalized_name}@"

        aggregated_metrics = {
            f"{metric_prefix_by_name[name]}{tk}": IAdditiveMetric()
            for name in normalized_metric_names
            for tk in eval_top_k
        }

        metric_objects = {
            (name, tk): _build_metric(name, top_k=tk)
            for name in normalized_metric_names
            for tk in eval_top_k
        }

        retrieval_top_k = max([top_k, *eval_top_k])
        async for batch_queries in tqdm_asyncio(chunked(queries, n=batch_size)):
            js_batch_queries = [qi for qi in batch_queries if qi is not None]
            res_topk = await self.ir.retrieve_topk(
                queries=js_batch_queries, batch_size=batch_size, top_k=retrieval_top_k
            )
            for question, docs_topk in zip(js_batch_queries, res_topk, strict=False):
                target = []
                preds = []
                indexes = []
                total_docs = len(docs_topk)

                for rank_idx, doc in enumerate(docs_topk):
                    doc_meta = getattr(doc, "meta", {}) or {}
                    labels = doc_meta.get("labels", [])
                    if isinstance(labels, str):
                        labels = [labels]
                    labels = [str(lb) for lb in labels if lb is not None]

                    target.append(1 if question in labels else 0)

                    score = getattr(doc, "score", None)
                    if score is None:
                        score = float(total_docs - rank_idx)
                    preds.append(float(score))
                    indexes.append(0)

                if not target:
                    continue

                target_t = torch.tensor(target, dtype=torch.long)
                preds_t = torch.tensor(preds, dtype=torch.float)
                indexes_t = torch.tensor(indexes, dtype=torch.long)

                for metric_name in normalized_metric_names:
                    for ev_top_k in eval_top_k:
                        metric_obj = metric_objects[(metric_name, ev_top_k)]
                        metric_value = metric_obj(preds_t, target_t, indexes_t)
                        metric_obj.reset()

                        metric_key = f"{metric_prefix_by_name[metric_name]}{ev_top_k}"
                        aggregated_metrics[metric_key].update(float(metric_value), 1)

        return aggregated_metrics


__all__ = ["EvaluatorRunner"]
