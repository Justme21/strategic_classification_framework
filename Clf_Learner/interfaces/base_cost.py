from abc import ABC, abstractmethod
from torch import Tensor

class BaseCost(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, X:Tensor, Z:Tensor) -> Tensor:
        pass

    @abstractmethod
    def set_standardiser(self, standardiser) -> None:
        pass

    @abstractmethod
    def get_standardiser(self):
        pass