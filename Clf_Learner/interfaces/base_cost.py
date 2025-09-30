from abc import ABC, abstractmethod
from torch import Tensor
from typing import Callable

class BaseCost(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        self.standardisation_inverter: Callable|None

    @abstractmethod
    def __call__(self, X:Tensor, Z:Tensor) -> Tensor:
        pass

    @abstractmethod
    def set_standardisation_inverter(self, invert_standaridsation:Callable) -> None:
        pass