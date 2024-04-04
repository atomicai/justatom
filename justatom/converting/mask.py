import abc


class IConverter(abc.ABC):

    @abc.abstractmethod
    def convert(self, fp, **kwargs):
        pass


__all__ = ["IConverter"]
