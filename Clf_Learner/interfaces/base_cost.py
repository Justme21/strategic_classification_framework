from abc import ABC, abstractmethod
from torch import Tensor

class BaseCost(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, X:Tensor, y:Tensor) -> Tensor:
        pass