from abc import ABC, abstractmethod

class BaseCost(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def get_cost(self, x, y):
        pass