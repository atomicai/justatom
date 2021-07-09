from justatom.lodash.loader import MLoader
from abc import abstractmethod


class Producer(MLoader):
    
    
    @abstractmethod
    def publish(self, routing_key, message, exchange=None):
        pass
