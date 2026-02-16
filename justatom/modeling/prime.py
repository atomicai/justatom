from pathlib import Path
from typing import Any

import numpy as np
import simplejson as json
import torch
import torch.nn as nn
from torch.functional import F  # pyright: ignore[reportPrivateImportUsage]
from transformers import AutoModel

from justatom.etc.pattern import cached_call, singleton
import threading
from justatom.modeling.div import IAttention, IEmbedding, MLAttention
from justatom.modeling.mask import ILanguageModel, IModel


class E5GeneralWrapper(ILanguageModel):
    def __init__(self):
        super().__init__()

    def average_pool(
        self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )

        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def maybe_norm_or_average(
        self, xs, attention_mask: torch.Tensor, norm: bool, average: bool
    ):
        if average:
            result = self.average_pool(xs, attention_mask=attention_mask)
            if norm:
                result = F.normalize(result, p=2, dim=len(result.shape) - 1)
        elif norm:
            result = F.normalize(xs, p=2, dim=len(xs.shape) - 1)
        else:
            result = xs
        return result


class E5Model(E5GeneralWrapper):
    """Base E5 family semantic model from hugging face"""

    def __init__(
        self,
        model_name_or_instance: str | nn.Module = "intfloat/multilingual-e5-base",
        device="cpu",
        **kwargs,
    ):
        super().__init__()
        self.model = (
            AutoModel.from_pretrained(model_name_or_instance)
            if isinstance(model_name_or_instance, str)
            else model_name_or_instance
        )
        self.name = "intfloat/multilingual-e5-base"
        self.model.to(device)

    @classmethod
    def load(cls, model_name_or_path: str, **kwargs):
        model = AutoModel.from_pretrained(model_name_or_path)
        return cls(model, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        group_ids: torch.Tensor = None,
        pos_input_ids: torch.Tensor = None,
        pos_attention_mask: torch.Tensor = None,
        norm: bool = True,
        average: bool = True,
    ):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        response = self.maybe_norm_or_average(
            outputs.last_hidden_state,
            attention_mask=attention_mask,
            norm=norm,
            average=average,
        )
        if pos_input_ids is not None and pos_attention_mask is not None:
            pos_outputs = self.model(
                input_ids=pos_input_ids, attention_mask=pos_attention_mask
            )
            pos_response = self.maybe_norm_or_average(
                pos_outputs.last_hidden_state,
                attention_mask=pos_attention_mask,
                norm=norm,
                average=average,
            )
            return response, pos_response

        return (response,)


class E5SModel(E5GeneralWrapper):
    """Small E5 family semantic model from huggingface"""

    def __init__(
        self,
        model_name_or_instance: str | nn.Module = "intfloat/multilingual-e5-small",
        device="cpu",
        **kwargs,
    ):
        super().__init__()
        self.model = (
            AutoModel.from_pretrained(model_name_or_instance)
            if isinstance(model_name_or_instance, str)
            else model_name_or_instance
        )
        self.name = "intfloat/multilingual-e5-small"
        self.model.to(device)

    @classmethod
    def load(cls, model_name_or_path: str, **kwargs):
        model = AutoModel.from_pretrained(model_name_or_path)
        return cls(model, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        group_ids: torch.Tensor = None,
        pos_input_ids: torch.Tensor = None,
        pos_attention_mask: torch.Tensor = None,
        norm: bool = True,
        average: bool = True,
    ):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        response = self.maybe_norm_or_average(
            outputs.last_hidden_state,
            attention_mask=attention_mask,
            norm=norm,
            average=average,
        )
        if pos_input_ids is not None and pos_attention_mask is not None:
            pos_outputs = self.model(
                input_ids=pos_input_ids,
                attention_mask=pos_attention_mask,
                norm=norm,
                average=average,
            )
            pos_response = self.maybe_norm_or_average(
                pos_outputs,
                attention_mask=attention_mask,
                norm=norm,
                average=average,
            )
            return response, pos_response

        return (response,)


class E5LModel(E5GeneralWrapper):
    """Large E5 family semantic model from huggingface"""

    def __init__(
        self,
        model_name_or_instance: str | nn.Module = "intfloat/multilingual-e5-large",
        device="cpu",
        **kwargs,
    ):
        super().__init__()
        self.model = (
            AutoModel.from_pretrained(model_name_or_instance)
            if isinstance(model_name_or_instance, str)
            else model_name_or_instance
        )
        self.name = "intfloat/multilingual-e5-large"
        self.model.to(device)

    @classmethod
    def load(cls, model_name_or_path: str, **kwargs):
        model = AutoModel.from_pretrained(model_name_or_path)
        return cls(model, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        group_ids: torch.Tensor = None,
        pos_input_ids: torch.Tensor = None,
        pos_attention_mask: torch.Tensor = None,
        norm: bool = True,
        average: bool = True,
    ):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        response = self.maybe_norm_or_average(
            outputs.last_hidden_state,
            attention_mask=attention_mask,
            norm=norm,
            average=average,
        )
        if pos_input_ids is not None and pos_attention_mask is not None:
            pos_outputs = self.model(
                input_ids=pos_input_ids, attention_mask=pos_attention_mask
            )
            pos_response = self.maybe_norm_or_average(
                pos_outputs.last_hidden_state,
                attention_mask=pos_attention_mask,
                norm=norm,
                average=average,
            )
            return response, pos_response

        return (response,)


class E5LInstructModel(E5GeneralWrapper):

    def __init__(
        self,
        model_name_or_instance: str | nn.Module = "intfloat/multilingual-e5-large-instruct",
        device="cpu",
        **kwargs,
    ):
        super().__init__()
        self.model = (
            AutoModel.from_pretrained(model_name_or_instance)
            if isinstance(model_name_or_instance, str)
            else model_name_or_instance
        )
        self.name = "intfloat/multilingual-e5-large-instruct"
        self.model.to(device)
    
    @classmethod
    def load(cls, model_name_or_path: str, **kwargs):
        model = AutoModel.from_pretrained(model_name_or_path)
        return cls(model, **kwargs)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        group_ids: torch.Tensor = None,
        pos_input_ids: torch.Tensor = None,
        pos_attention_mask: torch.Tensor = None,
        norm: bool = True,
        average: bool = True,
    ):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        response = self.maybe_norm_or_average(
            outputs.last_hidden_state,
            attention_mask=attention_mask,
            norm=norm,
            average=average,
        )
        if pos_input_ids is not None and pos_attention_mask is not None:
            pos_outputs = self.model(
                input_ids=pos_input_ids, attention_mask=pos_attention_mask
            )
            pos_response = self.maybe_norm_or_average(
                pos_outputs.last_hidden_state,
                attention_mask=pos_attention_mask,
                norm=norm,
                average=average,
            )
            return response, pos_response

        return (response,)


class MBERTModel(ILanguageModel):
    def __init__(
        self,
        model_name_or_instance: (
            str | nn.Module
        ) = "google-bert/bert-base-multilingual-cased",
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__()
        self.model = (
            AutoModel.from_pretrained(model_name_or_instance)
            if isinstance(model_name_or_instance, str)
            else model_name_or_instance
        )
        self.name = "google-bert/bert-base-multilingual-cased"
        self.model.to(device)

    @classmethod
    def load(cls, model_name_or_path: str, **kwargs):
        model = AutoModel.from_pretrained(model_name_or_path)
        return cls(model, **kwargs)

    def maybe_norm(self, xs, norm: bool):
        if norm:
            return F.normalize(xs, p=2, dim=len(xs.shape) - 1)
        return xs

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        group_ids: torch.Tensor = None,
        pos_input_ids: torch.Tensor = None,
        pos_attention_mask: torch.Tensor = None,
        norm: bool = True,
        average: bool = True,
    ):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).pooler_output

        outputs = self.maybe_norm(outputs, norm=norm)
        if pos_input_ids is not None and pos_attention_mask is not None:
            pos_outputs = self.model(
                input_ids=pos_input_ids, attention_mask=pos_attention_mask
            ).pooler_output

            pos_outputs = self.maybe_norm(pos_outputs, norm=norm)

            return outputs, pos_outputs

        return (outputs,)


class BGEModel(ILanguageModel):
    def __init__(
        self,
        model_name_or_instance: str | nn.Module = "deepvk/USER-bge-m3",
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__()
        self.model = (
            AutoModel.from_pretrained(model_name_or_instance)
            if isinstance(model_name_or_instance, str)
            else model_name_or_instance
        )
        self.name = "deepvk/USER-bge-m3"
        self.model.to(device)

    @classmethod
    def load(cls, model_name_or_path: str, **kwargs):
        model = AutoModel.from_pretrained(model_name_or_path)
        return cls(model, **kwargs)

    def maybe_norm(self, xs, norm: bool):
        if norm:
            return F.normalize(xs, p=2, dim=len(xs.shape) - 1)
        return xs

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        group_ids: torch.Tensor = None,
        pos_input_ids: torch.Tensor = None,
        pos_attention_mask: torch.Tensor = None,
        norm: bool = True,
        average: bool = True,
    ):
        # embedding of CLS token
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0]

        outputs = self.maybe_norm(outputs, norm=norm)
        if pos_input_ids is not None and pos_attention_mask is not None:
            pos_outputs = self.model(
                input_ids=pos_input_ids, attention_mask=pos_attention_mask
            ).last_hidden_state[:, 0]

            pos_outputs = self.maybe_norm(pos_outputs, norm=norm)

            return outputs, pos_outputs

        return (outputs,)


class PosFreeEncoderModel(IModel):
    """
    Positional Free Bidirectional Encoder Representation from Transformers
    """

    def __init__(
        self, embedding: nn.Module = None, attention: nn.Module = None, **props
    ):
        super(PosFreeEncoderModel, self).__init__()  # noqa: UP008
        if embedding is not None and attention is not None:
            self.embedding = embedding
            self.blocks = attention
        else:
            embedding_props = {k: props[k] for k in IEmbedding.props}
            self.embedding = IEmbedding(**embedding_props)

            if "num_blocks" in props:
                attention_props = {k: props[k] for k in MLAttention.props}
                self.blocks = MLAttention(**attention_props)
            else:
                attention_props = {k: props[k] for k in IAttention.props}
                self.blocks = IAttention(**attention_props)

    def generate_config(self):
        if isinstance(self.blocks, IAttention):  # noqa: SIM108
            num_blocks = -1
        else:
            num_blocks = self.blocks.config["num_blocks"]  # type: ignore
        self.config = {"num_blocks": num_blocks}
        return self

    @classmethod
    def load(cls, pretrained_model_name_or_path):
        # Only need to load two components.
        with open(Path(pretrained_model_name_or_path) / "model_config.json") as fin:
            config = json.load(fin)
        # (1). Load the attention heads module.
        if config.get("num_blocks") > 0:
            att = MLAttention.load(pretrained_model_name_or_path)
        else:
            att = IAttention.load(pretrained_model_name_or_path)
        # (2). Load the embedding module.
        emb = IEmbedding.load(pretrained_model_name_or_path)

        return cls(embedding=emb, attention=att)

    def save(self, save_dir: str | Path, state_dict: dict[Any, Any] | None = None):
        """
        Save the model `state_dict` and its configuration file so that it can be loaded again.

        :param save_dir: The directory in which the model should be saved.
        :param state_dict: A dictionary containing the whole state of the module, including names of layers. By default, the unchanged state dictionary of the module is used.
        """  # noqa: E501
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        # (1). Save the encoder with attention module and config(s)
        self.blocks.save(save_dir)  # pyright: ignore[reportCallIssue]
        # (2). Save the embedding module.
        self.embedding.save(save_dir)  # pyright: ignore[reportCallIssue]
        # (3). Generate and save the config
        self.generate_config().save_config(save_dir)

    def average_pool(
        self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )

        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        norm: bool = True,
        average: bool = False,
    ):
        emb = self.embedding(input_ids)  # batch_size x max_seq_len x embedding_dim
        # attention_mask.shape == (batch_size, max_seq_len)
        out = self.blocks(input_tensor=emb, attention_mask=attention_mask)
        if average:
            out = self.average_pool(
                out,
                attention_mask=attention_mask,  # pyright: ignore[reportArgumentType]
            )
        if norm:
            response = F.normalize(out, p=2, dim=len(out.shape) - 1)
        return response


@singleton
class ILMFinder:
    store: dict[str, ILanguageModel] = dict()
    _lock = threading.Lock()

    def find(self, model_name_or_path: str, **kwargs):
        key = cached_call(model_name_or_path, **kwargs)
        with self._lock:
            if key not in self.store:
                self.store[key] = ILanguageModel.load(model_name_or_path, **kwargs)
            return self.store[key]


HF_CLASS_MAPPING = {
    "intfloat/multilingual-e5-base": E5Model,
    "intfloat/multilingual-e5-small": E5SModel,
    "intfloat/multilingual-e5-large": E5LModel,
    "google-bert/bert-base-multilingual-cased": MBERTModel,
    "deepvk/USER-bge-m3": BGEModel,
    "justatom/pfbert": PosFreeEncoderModel,
}

COMMON_CLASS_MAPPING = {"justatom/pfbert": PosFreeEncoderModel}


__all__ = [
    "PosFreeEncoderModel",
    "E5SModel",
    "E5Model",
    "BGEModel",
    "E5LModel",
]
