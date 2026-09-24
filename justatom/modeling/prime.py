import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
import simplejson as json
import torch
import torch.nn as nn
from torch.functional import F  # pyright: ignore[reportPrivateImportUsage]
from transformers import AutoModel

from justatom.etc.pattern import cached_call, singleton
from justatom.modeling.div import IAttention, IEmbedding, MLAttention
from justatom.modeling.mask import ILanguageModel, IModel


class EmbeddingPoolingWrapper(ILanguageModel):
    def __init__(self):
        super().__init__()

    def average_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def maybe_norm_or_average(self, xs, attention_mask: torch.Tensor, norm: bool, average: bool):
        if average:
            result = self.average_pool(xs, attention_mask=attention_mask)
            if norm:
                result = F.normalize(result, p=2, dim=len(result.shape) - 1)
        elif norm:
            result = F.normalize(xs, p=2, dim=len(xs.shape) - 1)
        else:
            result = xs
        return result


class Qwen3EmbeddingModel(EmbeddingPoolingWrapper):
    """Qwen3 embedding model with last-token pooling and optional MRL truncation."""

    def __init__(
        self,
        model_name_or_instance: str | nn.Module = "Qwen/Qwen3-Embedding-0.6B",
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__()
        self.model = (
            AutoModel.from_pretrained(model_name_or_instance, **kwargs)
            if isinstance(model_name_or_instance, str)
            else model_name_or_instance
        )
        self.name = (
            model_name_or_instance
            if isinstance(model_name_or_instance, str)
            else getattr(self.model.config, "_name_or_path", "") or "Qwen/Qwen3-Embedding-0.6B"
        )
        self.name = getattr(self.model.config, "justatom_model_name", self.name)
        self.model.config.use_cache = False
        self.to(device)

    def to(self, *args, **kwargs):
        target = kwargs.get("device")
        if target is None and args and isinstance(args[0], (str, torch.device)):
            target = args[0]
        explicit_dtype = "dtype" in kwargs or (len(args) > 1 and isinstance(args[1], torch.dtype))
        if target is not None and torch.device(target).type == "mps" and not explicit_dtype:
            kwargs["dtype"] = torch.float32
        return super().to(*args, **kwargs)

    @classmethod
    def load(cls, model_name_or_path: str, **kwargs):
        return cls(str(model_name_or_path), **kwargs)

    def last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        positions = attention_mask.shape[1] - 1 - attention_mask.flip([1]).long().argmax(dim=1)
        return last_hidden_states[torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device), positions]

    def pool_embeddings(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.last_token_pool(hidden, attention_mask)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        norm: bool = True,
        layer_idx: int = -1,
        target_dim: int | None = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        use_hidden_states = layer_idx != -1
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=use_hidden_states,
        )

        if layer_idx == -1:
            hidden = outputs.last_hidden_state
        else:
            hidden = outputs.hidden_states[layer_idx]

        embeddings = self.pool_embeddings(hidden, attention_mask=attention_mask)
        embeddings = self.maybe_norm_or_average(
            embeddings,
            attention_mask=attention_mask,
            norm=norm,
            average=False,
        )

        if target_dim is not None:
            if not (32 <= target_dim <= embeddings.shape[1]):
                raise ValueError(f"target_dim must be in [32, {embeddings.shape[1]}], got {target_dim}")
            embeddings = embeddings[:, :target_dim]
            if norm:
                embeddings = F.normalize(embeddings, p=2, dim=len(embeddings.shape) - 1)

        return embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        group_ids: torch.Tensor = None,
        pos_input_ids: torch.Tensor = None,
        pos_attention_mask: torch.Tensor = None,
        norm: bool = True,
        average: bool = False,
        layer_idx: int = -1,
        target_dim: int | None = None,
    ):
        response = self.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            norm=norm,
            layer_idx=layer_idx,
            target_dim=target_dim,
        )
        if pos_input_ids is not None and pos_attention_mask is not None:
            pos_response = self.encode(
                input_ids=pos_input_ids,
                attention_mask=pos_attention_mask,
                norm=norm,
                layer_idx=layer_idx,
                target_dim=target_dim,
            )
            return response, pos_response

        return (response,)


class Qwen3VLEmbeddingModel(Qwen3EmbeddingModel):
    """Text-only Qwen3-VL embeddings; no generation head or vision adapters."""

    def __init__(
        self,
        model_name_or_instance: str | nn.Module = "Qwen/Qwen3-VL-Embedding-2B",
        device: str = "cpu",
        **kwargs,
    ):
        if isinstance(model_name_or_instance, str):
            # Lazy import keeps older encoders usable without the VL architecture.
            from transformers import Qwen3VLModel

            model_name_or_instance = Qwen3VLModel.from_pretrained(model_name_or_instance, **kwargs)
        super().__init__(model_name_or_instance, device=device)
        self.name = "Qwen/Qwen3-VL-Embedding-2B"
        self.model.visual.requires_grad_(False)
        self.model.config.use_cache = False
        self.model.config.text_config.use_cache = False

    @classmethod
    def load(cls, model_name_or_path: str, **kwargs):
        return cls(str(model_name_or_path), **kwargs)

    @property
    def output_dims(self):
        return self.model.config.text_config.hidden_size

    def last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        positions = attention_mask.shape[1] - 1 - attention_mask.flip([1]).long().argmax(dim=1)
        return last_hidden_states[torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device), positions]

    def encode(self, input_ids, attention_mask, norm=True, layer_idx=-1, target_dim=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if target_dim is not None and not 64 <= target_dim <= self.output_dims:
            raise ValueError(f"target_dim must be in [64, {self.output_dims}], got {target_dim}")
        return super().encode(input_ids, attention_mask, norm=norm, layer_idx=layer_idx, target_dim=target_dim)

    def resolve_lora_targets(self, requested: str | tuple[str, ...]) -> list[str]:
        """Resolve PEFT targets explicitly inside the language tower only."""
        targets = []
        for name, module in self.model.named_modules():
            if not name.startswith("language_model.") or not isinstance(module, nn.Linear):
                continue
            if isinstance(requested, str):
                matches = requested == "all-linear" or re.fullmatch(requested, name) is not None
            else:
                matches = any(name == item or name.endswith("." + item) for item in requested)
            if matches:
                targets.append(name)
        if not targets:
            raise ValueError("Qwen3-VL text LoRA requires target_modules matching language_model linear layers")
        return targets


class Nemotron3EmbeddingModel(Qwen3EmbeddingModel):
    """Bidirectional Ministral backbone with masked mean pooling."""

    def __init__(self, model_name_or_instance="nvidia/Nemotron-3-Embed-1B-BF16", device="cpu", **kwargs):
        super().__init__(model_name_or_instance, device=device, **kwargs)
        if getattr(self.model.config, "is_causal", True):
            raise ValueError("Nemotron embedding weights must declare is_causal=false")
        # Ministral attention modules in Transformers 5.x default to causal even
        # when the mask factory reads config.is_causal=false. SDPA/Flash may omit
        # a full bidirectional mask; keep their module-level fallback consistent.
        for module in self.model.modules():
            if hasattr(module, "is_causal"):
                module.is_causal = False

    def pool_embeddings(self, hidden, attention_mask):
        return self.average_pool(hidden, attention_mask)


class EmbeddingGemmaModel(Qwen3EmbeddingModel):
    """Gemma's bidirectional mean embedding plus both published dense layers."""

    def __init__(
        self,
        model_name_or_instance="google/embeddinggemma-300m",
        device="cpu",
        projection=None,
        projection_specs=None,
        **kwargs,
    ):
        from justatom.modeling.projection import load_gemma_projection

        if isinstance(model_name_or_instance, str):
            projection, projection_specs = load_gemma_projection(model_name_or_instance, **kwargs)
        elif projection is None or projection_specs is None:
            raise ValueError("EmbeddingGemma requires its pretrained projection and projection_specs")
        super().__init__(model_name_or_instance, device=device, **kwargs)
        if not getattr(self.model.config, "use_bidirectional_attention", False):
            raise ValueError("EmbeddingGemma requires use_bidirectional_attention=true")
        self.projection = projection
        self.model.config.justatom_projection = projection_specs
        self.to(device=device, dtype=next(self.model.parameters()).dtype)

    def to(self, *args, **kwargs):
        if kwargs.get("dtype") == torch.float16 or any(arg is torch.float16 for arg in args):
            raise ValueError("EmbeddingGemma supports float32/bfloat16, not float16")
        return super().to(*args, **kwargs)

    @property
    def output_dims(self):
        return self.model.config.justatom_projection[-1]["out_features"]

    def pool_embeddings(self, hidden, attention_mask):
        if hidden.dtype == torch.float16:
            raise ValueError("EmbeddingGemma activations require float32 or bfloat16")
        pooled = self.average_pool(hidden, attention_mask)
        return self.projection(pooled.to(next(self.projection.parameters()).dtype))

    def save(self, save_dir, state_dict=None):
        from justatom.modeling.projection import save_gemma_projection

        super().save(save_dir, state_dict=state_dict)
        save_gemma_projection(self.projection, save_dir)


class WeMMEmbeddingModel(Qwen3VLEmbeddingModel):
    """WeMM's text path; native Qwen3.5 backbone, no LM head or vision LoRA."""

    def __init__(self, model_name_or_instance="tencent/WeMM-Embedding-2B", device="cpu", **kwargs):
        name = model_name_or_instance if isinstance(model_name_or_instance, str) else "tencent/WeMM-Embedding-2B"
        if isinstance(model_name_or_instance, str):
            from transformers import Qwen3_5Model

            model_name_or_instance = Qwen3_5Model.from_pretrained(model_name_or_instance, **kwargs)
        super().__init__(model_name_or_instance, device=device)
        self.name = getattr(self.model.config, "justatom_model_name", name)

    def encode(self, *args, **kwargs):
        # Match the official embedding() path without an inference-only wrapper.
        # Reset positional state as in upstream; gradients remain enabled.
        backbone = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        backbone.rope_deltas = None
        return super().encode(*args, **kwargs)


class E5Model(EmbeddingPoolingWrapper):
    """Base E5 family semantic model from hugging face"""

    def __init__(
        self,
        model_name_or_instance: str | nn.Module = "intfloat/multilingual-e5-base",
        device="cpu",
        **kwargs,
    ):
        super().__init__()
        self.model = (
            AutoModel.from_pretrained(model_name_or_instance) if isinstance(model_name_or_instance, str) else model_name_or_instance
        )
        self.name = "intfloat/multilingual-e5-base"
        self.model.to(device)

    @classmethod
    def load(cls, model_name_or_path: str, **kwargs):
        model_kwargs = {} if kwargs.get("revision") is None else {"revision": kwargs["revision"]}
        model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
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
            pos_outputs = self.model(input_ids=pos_input_ids, attention_mask=pos_attention_mask)
            pos_response = self.maybe_norm_or_average(
                pos_outputs.last_hidden_state,
                attention_mask=pos_attention_mask,
                norm=norm,
                average=average,
            )
            return response, pos_response

        return (response,)


class E5SModel(EmbeddingPoolingWrapper):
    """Small E5 family semantic model from huggingface"""

    def __init__(
        self,
        model_name_or_instance: str | nn.Module = "intfloat/multilingual-e5-small",
        device="cpu",
        **kwargs,
    ):
        super().__init__()
        self.model = (
            AutoModel.from_pretrained(model_name_or_instance) if isinstance(model_name_or_instance, str) else model_name_or_instance
        )
        self.name = "intfloat/multilingual-e5-small"
        self.model.to(device)

    @classmethod
    def load(cls, model_name_or_path: str, **kwargs):
        model_kwargs = {} if kwargs.get("revision") is None else {"revision": kwargs["revision"]}
        model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
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
            )
            pos_response = self.maybe_norm_or_average(
                pos_outputs.last_hidden_state,
                attention_mask=pos_attention_mask,
                norm=norm,
                average=average,
            )
            return response, pos_response

        return (response,)


class E5LModel(EmbeddingPoolingWrapper):
    """Large E5 family semantic model from huggingface"""

    def __init__(
        self,
        model_name_or_instance: str | nn.Module = "intfloat/multilingual-e5-large",
        device="cpu",
        **kwargs,
    ):
        super().__init__()
        self.model = (
            AutoModel.from_pretrained(model_name_or_instance) if isinstance(model_name_or_instance, str) else model_name_or_instance
        )
        self.name = "intfloat/multilingual-e5-large"
        self.model.to(device)

    @classmethod
    def load(cls, model_name_or_path: str, **kwargs):
        model_kwargs = {} if kwargs.get("revision") is None else {"revision": kwargs["revision"]}
        model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
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
            pos_outputs = self.model(input_ids=pos_input_ids, attention_mask=pos_attention_mask)
            pos_response = self.maybe_norm_or_average(
                pos_outputs.last_hidden_state,
                attention_mask=pos_attention_mask,
                norm=norm,
                average=average,
            )
            return response, pos_response

        return (response,)


class E5LInstructModel(EmbeddingPoolingWrapper):

    def __init__(
        self,
        model_name_or_instance: str | nn.Module = "intfloat/multilingual-e5-large-instruct",
        device="cpu",
        **kwargs,
    ):
        super().__init__()
        self.model = (
            AutoModel.from_pretrained(model_name_or_instance) if isinstance(model_name_or_instance, str) else model_name_or_instance
        )
        self.name = "intfloat/multilingual-e5-large-instruct"
        self.model.to(device)

    @classmethod
    def load(cls, model_name_or_path: str, **kwargs):
        model_kwargs = {} if kwargs.get("revision") is None else {"revision": kwargs["revision"]}
        model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
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
            pos_outputs = self.model(input_ids=pos_input_ids, attention_mask=pos_attention_mask)
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
        model_name_or_instance: str | nn.Module = "google-bert/bert-base-multilingual-cased",
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__()
        self.model = (
            AutoModel.from_pretrained(model_name_or_instance) if isinstance(model_name_or_instance, str) else model_name_or_instance
        )
        self.name = "google-bert/bert-base-multilingual-cased"
        self.model.to(device)

    @classmethod
    def load(cls, model_name_or_path: str, **kwargs):
        model_kwargs = {} if kwargs.get("revision") is None else {"revision": kwargs["revision"]}
        model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
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
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask).pooler_output

        outputs = self.maybe_norm(outputs, norm=norm)
        if pos_input_ids is not None and pos_attention_mask is not None:
            pos_outputs = self.model(input_ids=pos_input_ids, attention_mask=pos_attention_mask).pooler_output

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
            AutoModel.from_pretrained(model_name_or_instance) if isinstance(model_name_or_instance, str) else model_name_or_instance
        )
        self.name = "deepvk/USER-bge-m3"
        self.model.to(device)

    @classmethod
    def load(cls, model_name_or_path: str, **kwargs):
        model_kwargs = {} if kwargs.get("revision") is None else {"revision": kwargs["revision"]}
        model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
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
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]

        outputs = self.maybe_norm(outputs, norm=norm)
        if pos_input_ids is not None and pos_attention_mask is not None:
            pos_outputs = self.model(input_ids=pos_input_ids, attention_mask=pos_attention_mask).last_hidden_state[:, 0]

            pos_outputs = self.maybe_norm(pos_outputs, norm=norm)

            return outputs, pos_outputs

        return (outputs,)


class PosFreeEncoderModel(IModel):
    """
    Positional Free Bidirectional Encoder Representation from Transformers
    """

    def __init__(self, embedding: nn.Module = None, attention: nn.Module = None, **props):
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

    def average_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

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
        response = out
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
    "Qwen/Qwen3-Embedding-0.6B": Qwen3EmbeddingModel,
    "Qwen/Qwen3-Embedding-4B": Qwen3EmbeddingModel,
    "nvidia/Nemotron-3-Embed-1B-BF16": Nemotron3EmbeddingModel,
    "google/embeddinggemma-300m": EmbeddingGemmaModel,
    "tencent/WeMM-Embedding-2B": WeMMEmbeddingModel,
    "Qwen/Qwen3-VL-Embedding-2B": Qwen3VLEmbeddingModel,
    "google-bert/bert-base-multilingual-cased": MBERTModel,
    "deepvk/USER-bge-m3": BGEModel,
    "justatom/pfbert": PosFreeEncoderModel,
}

COMMON_CLASS_MAPPING = {"justatom/pfbert": PosFreeEncoderModel}

# Backward compatibility alias for older imports.
E5GeneralWrapper = EmbeddingPoolingWrapper


__all__ = [
    "EmbeddingPoolingWrapper",
    "PosFreeEncoderModel",
    "E5SModel",
    "E5Model",
    "Qwen3EmbeddingModel",
    "Qwen3VLEmbeddingModel",
    "Nemotron3EmbeddingModel",
    "EmbeddingGemmaModel",
    "WeMMEmbeddingModel",
    "BGEModel",
    "E5LModel",
]
