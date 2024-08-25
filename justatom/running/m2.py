import os
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger

from justatom.modeling.div import loss_per_head_sum
from justatom.modeling.mask import ILanguageModel
from justatom.running.mask import IMODELRunner


class M2LMRunner(IMODELRunner, nn.Module):
    """
    Base Class for implementing M=2 `LanguageModel` models with frameworks like PyTorch and co.
    """

    def __init__(
        self,
        query_model: ILanguageModel,
        passage_model: ILanguageModel | None = None,
        prediction_heads=None,
        embeds_dropout_prob: float = 0.1,
        device: str = "cpu",
    ):
        self.query_model = query_model.to(device)
        self.query_model_out_dims = query_model.output_dims
        self.query_dropout = nn.Dropout(embeds_dropout_prob)

        self.passage_model = passage_model.to(device)
        self.passage_model_out_dims = passage_model.output_dims
        self.passage_dropout = nn.Dropout(embeds_dropout_prob)
        self.prediction_heads = [ph.to(device) for ph in prediction_heads]

        self.loss_aggregation_fn = loss_per_head_sum

    def save(self, save_dir: Path, lm1_name: str = "query_model", lm2_name: str = "passage_model"):
        """
        Saves the 2 language model weights and respective config_files in directories
        - query_model
        - passage_model
        within save_dir.

        :param save_dir: Path to save the M2LMRunner to.
        """
        os.makedirs(save_dir, exist_ok=True)
        if not os.path.exists(Path.joinpath(save_dir, Path(lm1_name))):
            os.makedirs(Path.joinpath(save_dir, Path(lm1_name)))
        if not os.path.exists(Path.joinpath(save_dir, Path(lm2_name))):
            os.makedirs(Path.joinpath(save_dir, Path(lm2_name)))
        self.query_model.save(Path.joinpath(save_dir, Path(lm1_name)))
        self.passage_model.save(Path.joinpath(save_dir, Path(lm2_name)))
        for i, ph in enumerate(self.prediction_heads):
            logger.info("prediction_head saving")
            ph.save(save_dir, i)
        # TODO: Save runner config in the directory specifying the `__class__.__name__` to load afterwards.

    def forward(self, batch, **kwargs):
        pass

    def forward_lm(
        self,
        query_input_ids: torch.Tensor | None = None,
        query_segment_ids: torch.Tensor | None = None,
        query_attention_mask: torch.Tensor | None = None,
        passage_input_ids: torch.Tensor | None = None,
        passage_segment_ids: torch.Tensor | None = None,
        passage_attention_mask: torch.Tensor | None = None,
    ):
        """
        Forward pass for the 2 `LanguageModel` models.

        :param kwargs: Holds all arguments that need to be passed to the language models.
        :return: 2 tensors of pooled_output from the 2 language models.
        """
        pooled_output = [None, None]

        if query_input_ids is not None and query_segment_ids is not None and query_attention_mask is not None:
            pooled_output1, _ = self.query_model(
                input_ids=query_input_ids, segment_ids=query_segment_ids, attention_mask=query_attention_mask
            )
            pooled_output[0] = pooled_output1

        if passage_input_ids is not None and passage_segment_ids is not None and passage_attention_mask is not None:
            max_seq_len = passage_input_ids.shape[-1]
            passage_input_ids = passage_input_ids.view(-1, max_seq_len)
            passage_attention_mask = passage_attention_mask.view(-1, max_seq_len)
            passage_segment_ids = passage_segment_ids.view(-1, max_seq_len)

            pooled_output2, _ = self.passage_model(
                input_ids=passage_input_ids, segment_ids=passage_segment_ids, attention_mask=passage_attention_mask
            )
            pooled_output[1] = pooled_output2

        return tuple(pooled_output)
