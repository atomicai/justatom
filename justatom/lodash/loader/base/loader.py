from abc import abstractmethod
from abc import ABC


class Loader(ABC):

    @abstractmethod
    def fire(cls, name: str, **kwargs):
        raise NotImplementedError
