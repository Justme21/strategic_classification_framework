from abc import ABC, abstractmethod

class BaseUtility(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def get_utility(self, X, model):
        pass