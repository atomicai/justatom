import abc
from pathlib import Path
from loguru import logger
import simplejson as json
import copy
import numpy as np
from bertopic.backend import BaseEmbedder
from justatom.modeling.prime import IDocEmbedder
from typing import Union, List, Dict, Tuple


class IRunner:
    """
    Base Class for implementing M1/M2/M3 etc... models with frameworks like PyTorch and co.
    """

    subclasses = {}  # type: Dict

    def __init_subclass__(cls, **kwargs):
        """
        This automatically keeps track of all available subclasses.
        Enables generic load() for all specific Mi implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    @classmethod
    def load(
        cls,
        data_dir: str,  # TODO revert ignore
        **kwargs,
    ):
        """
        Loads the class of processor specified by processor name.

        :param data_dir: Directory where data files are located.
        :param kwargs: placeholder for passing generic parameters
        :return: An instance of the specified processor.
        """
        config_file = Path(data_dir) / "runner_config.json"
        assert config_file.exists(), "The config is not found, couldn't load the processor"
        logger.info(f"Runner config found locally at {data_dir}")
        with open(config_file) as f:
            config = json.load(f)
        runner = cls.subclasses[config["klass"]].load(data_dir, config=config)
        return runner

    def save_config(self, save_dir: Union[Path, str]):
        save_filename = Path(save_dir) / "runner_config.json"
        config = copy.deepcopy(self.config)
        config["klass"] = self.__class__.__name__
        with open(str(save_filename), "w") as f:
            f.write(json.dumps(config))

    def save(self, save_dir: str):
        """
        Dumps the config to the .json file

        :param save_dir: Directory where the files are to be saved
        :return: None
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        config = self.config
        # save heads
        config["klass"] = self.__class__.__name__
        output_config_file = Path(save_dir) / "runner_config.json"
        with open(output_config_file, "w") as file:
            json.dump(config, file)
        # Save the model itself
        self.model.save(save_dir)

    def connect_heads_with_processor(self, tasks: Dict, require_labels: bool = True):
        """
        Populates prediction heads (flow) with information coming from tasks.

        :param tasks: A dictionary where the keys are the names of the tasks and
                      the values are the details of the task (e.g. label_list, metric,
                      tensor name).
        :param require_labels: If True, an error will be thrown when a task is
                               not supplied with labels.
        :return: None
        """
        for head in self.prediction_heads:
            head.label_tensor_name = tasks[head.task_name]["label_tensor_name"]
            label_list = tasks[head.task_name]["label_list"]
            if not label_list and require_labels:
                raise Exception(f"The task '{head.task_name}' is missing a valid set of labels")
            label_list = tasks[head.task_name]["label_list"]
            head.label_list = label_list
            head.metric = tasks[head.task_name]["metric"]


class ICLUSTERINGWrapperBackend(BaseEmbedder):

    def __init__(self, model: IDocEmbedder):
        self.model = model

    def embed(self, documents: List[str], verbose: bool = False) -> np.ndarray:
        """Embed a list of n documents/words into an n-dimensional
        matrix of embeddings

        Arguments:
            documents: A list of documents or words to be embedded
            verbose: Controls the verbosity of the process

        Returns:
            Document/words embeddings with shape (n, m) with `n` documents/words
            that each have an embeddings size of `m`
        """
        pass


class ICLUSTERINGRunner(abc.ABC):
    """
    Pipeline for clustering using any custom embedding module.
    """

    def __init__(self, model: BaseEmbedder, **kwargs):
        self.model = model

    @abc.abstractmethod
    def fit_transform(self, docs, **kwargs) -> Tuple[List[int], Union[np.ndarray, None]]:
        pass


class IRetrieverRunner(abc.ABC):

    @abc.abstractmethod
    def retrieve_topk(self, queries: Union[str, List[str]], top_k: int = 5):
        pass
