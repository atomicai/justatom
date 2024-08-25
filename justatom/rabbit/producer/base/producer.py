from abc import abstractmethod

from justatom.lodash.loader import MLoader


class Producer(MLoader):
    @abstractmethod
    def publish(self, routing_key, message, exchange=None):
        pass
