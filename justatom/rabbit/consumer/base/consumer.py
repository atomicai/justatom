from justatom.lodash.loader import MLoader
from abc import abstractmethod


class Consumer(MLoader):
    
    @abstractmethod
    def consume(self):
        pass

