from typing import Callable, List, Union

import torch
from more_itertools import chunked
from tqdm.autonotebook import tqdm

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
        metrics: List[Union[str, Callable]],
        top_k: int = 5,
        batch_size: int = 10,
    ):
        monitor = dict()
        for batch_queries in tqdm(chunked(queries, n=batch_size)):
            res_topk = self.ir.retrieve_topk()
            labels = [
                c.meta["labels"] for c in res_topk
            ]  # num_questions_doc1 + num_questions_doc2 + ... + num_questions_dock
            raise NotImplementedError()


__all__ = ["EvaluatorRunner"]
