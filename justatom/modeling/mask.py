import abc
import copy
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import simplejson as json
import torch
import torch.nn as nn
from loguru import logger

from justatom.etc.errors import ModelingError

#: Names of the attributes in various model configs which refer to the number of dimensions in the output vectors
OUTPUT_DIM_NAMES = ["dim", "hidden_size", "d_model"]

GRANTED_MODEL_NAMES = [
    "IPFBERT",
    "IGLOVE",
    "E5Model",
    "E5SModel",
    "E5LModel",
    "ATOMICModel",
    "ATOMICSModel",
    "ATOMICLModel",
]


class IBaseModel(nn.Module, abc.ABC):
    """
    This parent class for those implementation(s) that do not fit any specific kind of domain.
    Neither (1) NLP nor (2) CV ...
    """

    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        """This automatically keeps track of all available subclasses.
        Enables generic load() or all specific IModel implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def __init__(self):
        super().__init__()
        self._output_dims = None

    @classmethod
    def load(
        cls,
        pretrained_model_name_or_path,
        revision=None,
        n_added_tokens=0,
        language_model_class=None,
        **kwargs,
    ):
        config_file = Path(pretrained_model_name_or_path) / "model_config.json"
        assert config_file.exists(), "The config is not found, couldn't load the model"
        logger.info(f"Model found locally at {pretrained_model_name_or_path}")
        with open(config_file) as f:
            config = json.load(f)
        model = cls.subclasses[config["klass"]].load(pretrained_model_name_or_path)
        return model

    @abc.abstractmethod
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        raise NotImplementedError

    def save_config(self, save_dir: Union[Path, str]):
        """
        Save the configuration of the language model in format.
        """
        save_filename = Path(save_dir) / "model_config.json"
        config = copy.deepcopy(self.config)
        config["klass"] = self.__class__.__name__
        # string = json.sto_json_string()  # type: ignore [union-attr,operator]
        with open(str(save_filename), "w") as f:
            f.write(json.dumps(config))

    def save(
        self, save_dir: Union[str, Path], state_dict: Optional[Dict[Any, Any]] = None
    ):
        """
        Save the model `state_dict` and its configuration file so that it can be loaded again.

        :param save_dir: The directory in which the model should be saved.
        :param state_dict: A dictionary containing the whole state of the module, including names of layers. By default, the unchanged state dictionary of the module is used.
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        # Save Weights
        save_name = Path(save_dir) / "pytorch_model.bin"
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Only save the model itself

        if not state_dict:
            state_dict = model_to_save.state_dict()  # type: ignore [union-attr]
        torch.save(state_dict, save_name)
        self.save_config(save_dir)


class ILanguageModel(nn.Module, abc.ABC):
    """
    The parent class for any kind of model that can embed language into a semantic vector space.
    These models read in tokenized sentences and return vectors that capture the meaning of sentences or of tokens.
    """

    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        """This automatically keeps track of all available subclasses.
        Enables generic load() or all specific LanguageModel implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def __init__(self):
        super().__init__()
        self._output_dims = None

    @classmethod
    def load(
        cls,
        model_name_or_path,
        revision=None,
        n_added_tokens=0,
        language_model_class=None,
        **kwargs,
    ):
        config_file = Path(model_name_or_path) / "config.json"
        # assert config_file.exists(), "The config is not found, couldn't load the model"
        if config_file.exists():
            logger.info(f"Model found locally at {model_name_or_path}")
            # it's a local directory in FARM format
            with open(config_file) as f:
                config = json.load(f)
            language_model = cls.subclasses[config["klass"]].load(
                model_name_or_path, **kwargs
            )
        else:
            from justatom.modeling.prime import HF_CLASS_MAPPING

            logger.info(f'Loading from huggingface hub via "{model_name_or_path}"')
            klass = HF_CLASS_MAPPING[model_name_or_path]
            language_model = klass.load(model_name_or_path, **kwargs)
        return language_model

    @property
    def encoder(self):
        return self.model.encoder

    @abc.abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_ids: Optional[
            torch.Tensor
        ],  # DistilBERT does not use them, see DistilBERTLanguageModel
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: bool = False,
    ):
        raise NotImplementedError

    @property
    def output_hidden_states(self):
        """
        Controls whether the model outputs the hidden states or not
        """
        self.encoder.config.output_hidden_states = True

    @output_hidden_states.setter
    def output_hidden_states(self, value: bool):
        """
        Sets the model to output the hidden states or not
        """
        self.encoder.config.output_hidden_states = value

    @property
    def output_dims(self):
        """
        The output dimension of this language model
        """
        if self._output_dims:
            return self._output_dims

        for odn in OUTPUT_DIM_NAMES:
            try:
                value = getattr(self.model.config, odn, None)
                if value:
                    self._output_dims = value
                    return value
            except AttributeError:
                raise ModelingError(
                    "Can't get the output dimension before loading the model."
                )

        raise ModelingError(
            "Could not infer the output dimensions of the language model."
        )

    def save_config(self, save_dir: Union[Path, str]):
        """
        Save the configuration of the language model in format.
        """
        save_filename = Path(save_dir) / "config.json"
        config = self.model.config.to_dict()
        config["klass"] = self.__class__.__name__
        # string = json.sto_json_string()  # type: ignore [union-attr,operator]
        with open(str(save_filename), "w") as f:
            f.write(json.dumps(config))

    def save(
        self, save_dir: Union[str, Path], state_dict: Optional[Dict[Any, Any]] = None
    ):
        """
        Save the model `state_dict` and its configuration file so that it can be loaded again.

        :param save_dir: The directory in which the model should be saved.
        :param state_dict: A dictionary containing the whole state of the module, including names of layers. By default, the unchanged state dictionary of the module is used.
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        # Save Weights
        save_name = Path(save_dir) / "pytorch_model.bin"
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Only save the model itself

        if not state_dict:
            state_dict = model_to_save.state_dict()  # type: ignore [union-attr]
        torch.save(state_dict, save_name)
        self.save_config(save_dir)

    def formatted_preds(
        self,
        logits,
        samples,
        ignore_first_token: bool = True,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extracting vectors from a language model (for example, for extracting sentence embeddings).
        You can use different pooling strategies and layers by specifying them in the object attributes
        `extraction_layer` and `extraction_strategy`. You should set both these attirbutes using the Inferencer:
        Example:  Inferencer(extraction_strategy='cls_token', extraction_layer=-1)

        :param logits: Tuple of (sequence_output, pooled_output) from the language model.
                       Sequence_output: one vector per token, pooled_output: one vector for whole sequence.
        :param samples: For each item in logits, we need additional meta information to format the prediction (for example, input text).
                        This is created by the Processor and passed in here from the Inferencer.
        :param ignore_first_token: When set to `True`, includes the first token for pooling operations (for example, reduce_mean).
                                   Many models use a special token, like [CLS], that you don't want to include in your average of token embeddings.
        :param padding_mask: Mask for the padding tokens. These aren't included in the pooling operations to prevent a bias by the number of padding tokens.
        :param input_ids: IDs of the tokens in the vocabulary.
        :param kwargs: kwargs
        :return: A list of dictionaries containing predictions, for example: [{"context": "some text", "vec": [-0.01, 0.5 ...]}].
        """
        if not hasattr(self, "extraction_layer") or not hasattr(
            self, "extraction_strategy"
        ):
            raise ModelingError(
                "`extraction_layer` or `extraction_strategy` not specified for LM. "
                "Make sure to set both, e.g. via Inferencer(extraction_strategy='cls_token', extraction_layer=-1)`"
            )

        # unpack the tuple from LM forward pass
        sequence_output = logits[0][0]
        pooled_output = logits[0][1]

        # aggregate vectors
        if self.extraction_strategy == "pooled":
            if self.extraction_layer != -1:
                raise ModelingError(
                    f"Pooled output only works for the last layer, but got extraction_layer={self.extraction_layer}. "
                    "Please set `extraction_layer=-1`"
                )
            vecs = pooled_output.cpu().numpy()

        elif self.extraction_strategy == "per_token":
            vecs = sequence_output.cpu().numpy()

        elif self.extraction_strategy == "reduce_mean":
            vecs = self._pool_tokens(
                sequence_output,
                padding_mask,
                self.extraction_strategy,
                ignore_first_token=ignore_first_token,
            )
        elif self.extraction_strategy == "reduce_max":
            vecs = self._pool_tokens(
                sequence_output,
                padding_mask,
                self.extraction_strategy,
                ignore_first_token=ignore_first_token,
            )
        elif self.extraction_strategy == "cls_token":
            vecs = sequence_output[:, 0, :].cpu().numpy()
        else:
            raise NotImplementedError(
                f"This extraction strategy ({self.extraction_strategy}) is not supported by Haystack."
            )

        preds = []
        for vec, sample in zip(vecs, samples):
            pred = {}
            pred["context"] = sample.clear_text["text"]
            pred["vec"] = vec
            preds.append(pred)
        return preds

    def _pool_tokens(
        self,
        sequence_output: torch.Tensor,
        padding_mask: torch.Tensor,
        strategy: str,
        ignore_first_token: bool,
    ):
        token_vecs = sequence_output.cpu().numpy()
        # we only take the aggregated value of non-padding tokens
        padding_mask = padding_mask.cpu().numpy()
        ignore_mask_2d = padding_mask == 0
        # sometimes we want to exclude the CLS token as well from our aggregation operation
        if ignore_first_token:
            ignore_mask_2d[:, 0] = True
        ignore_mask_3d = np.zeros(token_vecs.shape, dtype=bool)
        ignore_mask_3d[:, :, :] = ignore_mask_2d[:, :, np.newaxis]
        if strategy == "reduce_max":
            pooled_vecs = (
                np.ma.array(data=token_vecs, mask=ignore_mask_3d).max(axis=1).data
            )
        if strategy == "reduce_mean":
            pooled_vecs = (
                np.ma.array(data=token_vecs, mask=ignore_mask_3d).mean(axis=1).data
            )

        return pooled_vecs


class IRemoteLargeLanguageModel(abc.ABC):
    """
    The parent class for any kind of remote LLM that can generate sequence of tokens.
    These models receive prefix string and return generated sequence via API.
    """

    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        """This automatically keeps track of all available subclasses.
        Enables generic load() or all specific LanguageModel implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def generate(self, prompt: str, history: List[str], **kwargs):
        pass


class ILargeLanguageModel(abc.ABC):
    """
    The parent class for any kind of local LLM that can generate sequence of tokens.
    These models receive tokenized string (prefix) and return generated sequence via forward pass.
    """

    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        """This automatically keeps track of all available subclasses.
        Enables generic load() or all specific LanguageModel implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def __init__(self):
        super().__init__()


class IVisionModel(nn.Module, abc.ABC):
    """
    The parent class for any kind of model that can embed image into a semantic vector space.
    These models read in patches of images and return vectors that capture the meaning of images or of patches.
    """

    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        """This automatically keeps track of all available subclasses.
        Enables generic load() or all specific LanguageModel implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def __init__(self):
        super().__init__()
        self._output_dims = None

    @classmethod
    def load(
        cls,
        pretrained_model_name_or_path,
        revision=None,
        n_added_tokens=0,
        language_model_class=None,
        **kwargs,
    ):
        config_file = Path(pretrained_model_name_or_path) / "cv_model_config.json"
        assert config_file.exists(), "The config is not found, couldn't load the model"
        logger.info(f"Model found locally at {pretrained_model_name_or_path}")
        # it's a local directory in FARM format
        with open(config_file) as f:
            config = json.load(f)
        cv_model = cls.subclasses[config["klass"]].load(pretrained_model_name_or_path)
        return cv_model


class IHead(nn.Module, abc.ABC):
    """
    Takes embeddings from one of [ILanguageModel | [IVisionModel] and
    generates logits for a given task. Can also convert logits to loss and and logits to predictions.
    """

    subclasses = {}  # type: Dict

    def __init_subclass__(cls, **kwargs):
        """This automatically keeps track of all available subclasses.
        Enables generic load() for all specific PredictionHead implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    @classmethod
    def load(cls, klass, **props):
        return cls.subclasses[klass].load(**props)

    def generate_config(self):
        config = self.config or {}
        config["klass"] = self.__class__.__name__
        return config

    @abc.abstractmethod
    def logits_to_loss(self, logits, labels):
        pass

    @abc.abstractmethod
    def logits_to_preds(self, logits):
        pass


class IMetric(abc.ABC):
    """Interface for all Metrics.

    Args:
        compute_on_call: Computes and returns metric value during metric call.
            Used for per-batch logging.
            default: ``True``
    """

    def __init__(self, compute_on_call: bool = True):
        """Interface for all Metrics."""
        self.compute_on_call = compute_on_call

    @abc.abstractmethod
    def reset(self) -> None:
        """Resets the metric to it's initial state.

        By default, this is called at the start of each loader
        (`on_loader_start` event).
        """
        pass

    @abc.abstractmethod
    def update(self, *args, **kwargs) -> Any:
        """Updates the metrics state using the passed data.

        By default, this is called at the end of each batch
        (`on_batch_end` event).

        Args:
            *args: some args :)
            **kwargs: some kwargs ;)
        """
        pass

    @abc.abstractmethod
    def compute(self) -> Any:
        """Computes the metric based on it's accumulated state.

        By default, this is called at the end of each loader
        (`on_loader_end` event).

        Returns:
            Any: computed value, # noqa: DAR202
            it's better to return key-value
        """
        pass

    def __call__(self, *args, **kwargs) -> Any:
        """Computes the metric based on it's accumulated state.

        By default, this is called at the end of each batch
        (`on_batch_end` event).
        Returns computed value if `compute_on_call=True`.

        Args:
            *args: Arguments passed to update method.
            **kwargs: Keyword-arguments passed to update method.

        Returns:
            Any: computed value, it's better to return key-value.
        """
        value = self.update(*args, **kwargs)
        return self.compute() if self.compute_on_call else value


class IDocEmbedder(abc.ABC):
    """Abstract class for document embedder."""

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def encode(self, texts: List[str], **kwargs) -> Iterator[np.ndarray]:
        pass


__all__ = [
    "IBaseModel",
    "IVisionModel",
    "ILanguageModel",
    "ILargeLanguageModel",
    "IRemoteLargeLanguageModel",
    "IHead",
    "IMetric",
    "IDocEmbedder",
    "GRANTED_MODEL_NAMES",
]
