from typing import Callable, List, Union

import torch
from more_itertools import chunked
from tqdm.autonotebook import tqdm
from justatom.tooling import stl

from torchmetrics import RetrievalHitRate, RetrievalMRR
from justatom.modeling.metrics import IAdditiveMetric
from justatom.running.mask import IEvaluatorRunner, IRetrieverRunner


class EvaluatorRunner(IEvaluatorRunner):

    def __init__(self, ir: IRetrieverRunner):
        super().__init__(ir=ir)
        self.metrics = dict()

    @torch.no_grad()
    def evaluate_topk(
        self,
        queries: Union[str, List[str]],
        metrics: Union[Union[str, Callable]],
        metrics_top_k: List[Union[str, Callable]],
        eval_top_k: List[int] = None,
        top_k: int = 20,
        batch_size: int = 10,
    ):
        monitor = dict(HR2=IAdditiveMetric(), HR5=IAdditiveMetric(), MRR=IAdditiveMetric())
        hr2 = RetrievalHitRate(top_k=2)
        hr5 = RetrievalHitRate(top_k=5)
        mrr = RetrievalMRR()
        metrics_top_k_names = set(metrics_top_k)
        metrics_top_k = {f"{m}{tk}": IAdditiveMetric() for m in metrics_top_k for tk in eval_top_k}  # hr2, hr5, mrr
        for batch_queries in tqdm(chunked(queries, n=batch_size)):
            res_topk = self.ir.retrieve_topk(queries=batch_queries, batch_size=batch_size, top_k=top_k)
            # res_topk[i][j]  # prediction for the i-th sample @ j-th position.
            for question, docs_topk in zip(batch_queries, res_topk):
                labels = list(stl.flatten_list([c.meta["labels"] for c in docs_topk]))
                for ev_top_k in eval_top_k:
                    for _metric_topk in metrics_top_k_names:
                        if any([question == c for c in labels[:ev_top_k]]):
                            metrics_top_k[f"{_metric_topk}{str(ev_top_k)}"].update(1, 1)
                        else:
                            metrics_top_k[f"{_metric_topk}{str(ev_top_k)}"].update(0, 1)
        return metrics_top_k


__all__ = ["EvaluatorRunner"]
