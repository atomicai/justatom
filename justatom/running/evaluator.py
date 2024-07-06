from typing import Callable, List, Union

from justatom.running.mask import IEvaluatorRunner, IRetrieverRunner


class EvaluatorRunner(IEvaluatorRunner):

    def __init__(self, ir: IRetrieverRunner):
        super().__init__(ir=ir)

    def evaluate_topk(
        self,
        queries: Union[str, List[str]],
        dex_name: str,
        metrics: List[Union[str, Callable]],
        top_k: int = 5,
    ):
        pass
