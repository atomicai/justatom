import abc

GRANTED_PROCESSOR_NAMES = ["PFBERTProcessor", "E5SMALLProcessor", "E5Processor", "E5LARGEProcessor"]


class IProcessor(abc.ABC):

    @abc.abstractmethod
    def dataset_from_dicts(self, dicts):
        pass
