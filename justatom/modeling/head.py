import torch

from justatom.modeling.mask import IHead
from justatom.training.loss import init_loss


class ANNHead(IHead):
    def __init__(self, loss_fn: str | torch.nn.Module | None = None, device="cpu", **props):
        super(ANNHead, self).__init__()  # noqa: UP008
        props_for_loss = props.get("loss", {})
        self.loss = init_loss(device=device, name=loss_fn, **props_for_loss)
        self.label_tensor_name = props.get("label_tensor_name", "labels")
        self.config = dict(loss_fn=loss_fn, loss=props_for_loss)

    @classmethod
    def load(cls, **props):
        return cls(**props)

    def forward(self, logits):
        return logits

    def prepare_labels(self, labels):
        # mapping from labels to ids
        return labels

    def logits_to_preds(self, logits, return_pred_ids=True):
        preds = torch.argmax(logits, dim=1)
        return preds

    def logits_to_loss(self, logits, labels):
        L = self.loss(logits, labels)
        return L

    @classmethod
    def dot_product_scores(cls, query_vectors: torch.Tensor, passage_vectors: torch.Tensor) -> torch.Tensor:
        """
        Calculates dot product similarity scores for two 2-dimensional tensors

        :param query_vectors: tensor of (?query) of dimension n1 x D,
                        where n1 is the number of queries/batch size and D is embedding size
        :param passage_vectors: tensor of (?context/passage) embeddings of dimension n2 x D,
                        where n2 is (batch_size * num_positives) + (batch_size * num_hard_negatives)
                        and D is embedding size

        :return: dot_product: similarity score of each query with each context/passage (dimension: n1xn2)
        """
        # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
        dot_product = torch.matmul(query_vectors, torch.transpose(passage_vectors, 0, 1))
        return dot_product


class ContrastiveHead(IHead):
    pass


class TripletHead(IHead):
    pass


class ATOMICHead(IHead):
    pass


__all__ = ["ANNHead", "ATOMICHead"]
