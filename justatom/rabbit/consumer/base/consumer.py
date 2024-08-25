from abc import abstractmethod

from justatom.lodash.loader import MLoader


class Consumer(MLoader):
    @abstractmethod
    def consume(self):
        pass
