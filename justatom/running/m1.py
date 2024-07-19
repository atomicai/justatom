from pathlib import Path
from typing import List, Optional, Union

import simplejson as json
import torch
import torch.nn.functional as F
from loguru import logger

from justatom.modeling.div import loss_per_head_sum

from justatom.modeling.mask import IHead, ILanguageModel
from justatom.running.mask import IMODELRunner
from justatom.processing.mask import IProcessor


class M1LMRunner(IMODELRunner, torch.nn.Module):

    def __init__(
        self,
        model: ILanguageModel,
        prediction_heads: List[IHead],
        device="cpu",
        processor: Optional[IProcessor] = None,
    ):
        super(M1LMRunner, self).__init__()
        self.model = model
        self.prediction_heads = prediction_heads or []
        self.device = device
        self.dropout = torch.nn.Dropout(0.1)
        self.loss_aggregation_fn = loss_per_head_sum
        self.config = dict(
            prediction_heads=[hi.generate_config() for hi in prediction_heads or []]
        )
        self.processor = processor

    def to(self, device):
        logger.info(f"Moving to device {str(device)}")
        for mod in self.prediction_heads:
            if mod.device != device:
                logger.info("Head is not on the same device")
            if mod.loss.device != device:
                logger.info("Loss on the other device :(")
        super(M1LMRunner, self).to(device)

    @classmethod
    def load(cls, data_dir: Union[Path, str], config=None, **props):
        # model_config.json supposed to be present in directory
        _model_config = Path(data_dir) / "runner_config.json"
        assert _model_config.exists(), "The model file is not found for `M1Runner`"
        model = ILanguageModel.load(Path(data_dir))
        if config is None:
            _runner_config = Path(data_dir) / "runner_config.json"
            with open(_runner_config) as f:
                config = json.load(f)
        heads = []
        for head in config["prediction_heads"]:
            props = {k: v for k, v in head.items() if k != "klass"}
            hi = IHead.load(head["klass"], **props)
            heads.append(hi)
        return cls(model=model, prediction_heads=heads)

    def save(self, save_dir: str):
        """
        Dumps the config to the .json file

        :param save_dir: Directory where the files are to be saved
        :return: None
        """
        super().save(save_dir)
        if self.processor is not None:
            self.processor.save(save_dir)

    def logits_to_loss_per_head(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Collect losses from each prediction head.

        :param logits: Logits, can vary in shape and type, depending on task.
        :return: The per sample per prediciton head loss whose first two dimensions
                 have length n_pred_heads, batch_size.
        """
        all_losses = []
        for head, logits_for_one_head, labels_for_one_head in zip(
            self.prediction_heads, logits, labels
        ):
            # check if PredictionHead connected to Processor
            assert hasattr(head, "label_tensor_name"), (
                f"Label_tensor_names are missing inside the {head.task_name} Prediction Head. Did you connect the model"
                " with the processor through either 'model.connect_heads_with_processor(processor.tasks)'"
                " or by passing the processor to the Adaptive Model?"
            )
            all_losses.append(
                head.logits_to_loss(
                    logits=logits_for_one_head, labels=labels_for_one_head
                )
            )
        return all_losses

    def logits_to_loss(
        self, logits: torch.Tensor, global_step: Optional[int] = None, **kwargs
    ):
        """
        Get losses from all prediction heads & reduce to single loss *per sample*.

        :param logits: Logits, can vary in shape and type, depending on task.
        :param global_step: Number of current training step.
        :param kwargs: Placeholder for passing generic parameters.
                       Note: Contains the batch (as dict of tensors), when called from Trainer.train().
        :return: torch.tensor that is the per sample loss (len: batch_size)
        """
        all_losses = self.logits_to_loss_per_head(logits, **kwargs)
        # This aggregates the loss per sample across multiple prediction heads
        # Default is sum(), but you can configure any fn that takes [Tensor, Tensor ...] and returns [Tensor]
        loss = self.loss_aggregation_fn(
            all_losses, global_step=global_step, batch=kwargs
        )
        return loss

    def prepare_labels(self, labels):
        """
        Label conversion to original label space, per prediction head.

        :param label_maps: dictionary for mapping ids to label strings
        :type label_maps: dict[int:str]
        :return: labels in the right format
        """
        all_labels = []
        # for head, label_map_one_head in zip(self.prediction_heads):
        #     labels = head.prepare_labels(label_map=label_map_one_head, **kwargs)
        #     all_labels.append(labels)
        for head in self.prediction_heads:
            labels = head.prepare_labels(labels)
            all_labels.append(labels)
        return all_labels

    def maybe_norm(self, xs: torch.Tensor, norm: bool) -> torch.Tensor:
        if norm:
            return F.normalize(xs, p=2, dim=len(xs.shape) - 1)
        return xs

    def forward(self, batch, norm: bool = True, **props):
        # Run forward pass of (multiple) prediction heads using the output from above
        Q = self.model(**batch, **props)
        all_logits = []
        for Qi in Q:
            if len(self.prediction_heads) > 0:
                for head in self.prediction_heads:
                    all_logits.append(
                        self.maybe_norm(head(self.dropout(Qi)), norm=norm)
                    )
            else:  # If no head is initialized => simple forward pass of a model
                all_logits.append(self.maybe_norm(self.dropout(Qi), norm=norm))

        return all_logits
