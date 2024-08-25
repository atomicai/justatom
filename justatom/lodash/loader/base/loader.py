from abc import ABC, abstractmethod


class Loader(ABC):
    @abstractmethod
    def fire(cls, name: str, **kwargs):
        raise NotImplementedError
