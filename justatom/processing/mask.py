import abc
import inspect
from collections.abc import Iterable
from pathlib import Path

import simplejson as json
import torch
from loguru import logger
from torch.utils.data import TensorDataset

from justatom.etc.format import maybe_json
from justatom.processing.sample import SampleBasket

GRANTED_PROCESSOR_NAMES = [
    "PosFreeEncoderProcessor",
    "EncoderProcessor",
    "BiEncoderProcessor",
    "GammaHybridProcessor",
    "RuntimeProcessor",
]


def tensor_features(features):
    """
    Converts a list of feature dictionaries (one for each sample) into a PyTorch Dataset.

    :param features: A list of dictionaries. Each dictionary corresponds to one sample. Its keys are the
                     names of the type of feature and the keys are the features themselves.
    :Return: a Pytorch dataset and a list of tensor names.
    """
    # features can be an empty list in cases where down sampling occurs
    if len(features) == 0:
        return None, None
    tensor_names = list(features[0].keys())
    all_tensors = []
    for t_name in tensor_names:
        cur_tensor = torch.stack([torch.tensor(sample[t_name]) for sample in features])
        all_tensors.append(cur_tensor)

    dataset = TensorDataset(*all_tensors)
    return dataset, tensor_names


class IProcessor(abc.ABC):
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    @abc.abstractmethod
    def dataset_from_dicts(
        self,
        dicts: list[dict],
        indices: list[int] | None = None,
        return_baskets: bool = False,
        debug: bool = False,
    ):
        raise NotImplementedError()

    def add_task(
        self,
        name,
        metric,
        label_list,
        label_column_name=None,
        label_name=None,
        task_type=None,
        text_column_name=None,
    ):
        if type(label_list) is not list:
            raise ValueError(
                f"Argument `label_list` must be of type list. Got: f{type(label_list)}"
            )

        if label_name is None:
            label_name = f"{name}_label"
        label_tensor_name = label_name + "_ids"
        self.tasks[name] = {
            "label_list": label_list,
            "metric": metric,
            "label_tensor_name": label_tensor_name,
            "label_name": label_name,
            "label_column_name": label_column_name,
            "text_column_name": text_column_name,
            "task_type": task_type,
        }

    def do_prefix(self, x: str, pref: str):
        return pref.strip() + " " + x.strip()

    def _check_sample_features(self, basket: SampleBasket):
        """
        Check if all samples in the basket has computed its features.

        :param basket: the basket containing the samples

        :return: True if all the samples in the basket has computed its features, False otherwise
        """
        return basket.samples and not any(
            sample.features is None for sample in basket.samples
        )

    def save(self, save_dir: str):
        """
        Saves the vocabulary to file and also creates a json file containing all the
        information needed to load the same processor.

        :param save_dir: Directory where the files are to be saved
        :return: None
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        config = self.generate_config()
        # save tokenizer incl. attributes
        config["klass"] = self.__class__.__name__

        # Because the fast tokenizers expect a str and not Path
        # always convert Path to str here.
        self.tokenizer.save_pretrained(str(save_dir))

        output_config_file = Path(save_dir) / "processor_config.json"
        with open(output_config_file, "w") as file:
            json.dump(config, file)

    @classmethod
    def load(
        cls,
        where: str,  # TODO revert ignore
        **kwargs,
    ):
        """
        Loads the class of processor specified by processor name.

        :param processor_name: The class of processor to be loaded.
        :param data_dir: Directory where data files are located.
        :param tokenizer: A tokenizer object
        :param max_seq_len: Sequences longer than this will be truncated.
        :param train_filename: The name of the file containing training data.
        :param dev_filename: The name of the file containing the dev data.
                             If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :param test_filename: The name of the file containing test data.
        :param dev_split: The proportion of the train set that will be sliced.
                          Only works if dev_filename is set to None
        :param kwargs: placeholder for passing generic parameters
        :return: An instance of the specified processor.
        """
        config_file = Path(where) / "processor_config.json"
        if config_file.exists():
            logger.info(f"Processor found locally at {where}")
            with open(config_file) as f:
                config = json.load(f)
            processor = cls.subclasses[config["klass"]].load(
                where, config=config, **kwargs
            )
        else:
            logger.info("Loading default `INFERProcessor` instance")
            processor = cls.subclasses["INFERProcessor"].load(
                where, config={}, **kwargs
            )
        return processor

    def generate_config(self):
        """
        Generates config file from Class and instance attributes (only for sensible config parameters).
        """
        config = {}
        # self.__dict__ doesn't give parent class attributes
        for key, value in inspect.getmembers(self):
            if maybe_json(value) and key[0] != "_":
                if issubclass(type(value), Path):
                    value = str(value)
                config[key] = value
        return config

    def _create_dataset(self, baskets: list[SampleBasket]):
        features_flat: list = []
        basket_to_remove = []
        for basket in baskets:
            if self._check_sample_features(basket):
                if not isinstance(basket.samples, Iterable):
                    raise ValueError("basket.samples must contain a list of samples.")
                for sample in basket.samples:
                    if sample.features is None:
                        raise ValueError("sample.features must not be None.")
                    features_flat.extend(sample.features)
            else:
                # remove the entire basket
                basket_to_remove.append(basket)
        dataset, tensor_names = tensor_features(features=features_flat)
        return dataset, tensor_names


__all__ = ["IProcessor"]
