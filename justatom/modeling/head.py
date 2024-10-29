import json
from pathlib import Path

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
    def load(cls, data_dir: Path | str, **kwargs):
        _head_config = Path(data_dir) / "config.json"
        with open(_head_config) as f:
            config = json.load(f)
        props = {k: v for k, v in config.items() if k != "klass"}
        return cls(**props, **kwargs)

    def save(self, save_dir: str):
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        config = self.config
        # save heads
        config["klass"] = self.__class__.__name__
        output_config_file = Path(save_dir) / "config.json"
        with open(output_config_file, "w") as file:
            json.dump(config, file)

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


# linear layer
class LLHead(IHead):
    def __init__(self, loss_fn: str | torch.nn.Module | None = None, weights_path: str | None = None, device="cpu", **props):
        super().__init__()
        props_for_loss = props.get("loss", {})
        self.loss = init_loss(device=device, name=loss_fn, **props_for_loss)
        self.label_tensor_name = props.get("label_tensor_name", "labels")
        self.config = dict(loss_fn=loss_fn, loss=props_for_loss)
        in_features = props.get("in_features", 384)
        out_features = props.get("out_features", 50)
        self.model = torch.nn.Linear(in_features=in_features, out_features=out_features)
        if weights_path is not None:
            self.model.load_state_dict(torch.load(weights_path, weights_only=True))
        self.model.eval()

    @classmethod
    def load(cls, data_dir: Path | str, **kwargs):
        _head_config = Path(data_dir) / "config.json"
        with open(_head_config) as f:
            config = json.load(f)
        props = {k: v for k, v in config.items() if k != "klass"}
        return cls(**props, **kwargs)

    def save(self, save_dir: str):
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        config = self.config
        # save heads
        config["klass"] = self.__class__.__name__
        config["weights_path"] = str(save_dir / "torch_model.pt")
        output_config_file = Path(save_dir) / "config.json"
        with open(output_config_file, "w") as file:
            json.dump(config, file)
        # Save the model itself
        torch.save(self.model.state_dict(), config["weights_path"])

    def forward(self, logits):
        return self.model(logits)

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


__all__ = ["ANNHead", "ATOMICHead", "LLHead"]
