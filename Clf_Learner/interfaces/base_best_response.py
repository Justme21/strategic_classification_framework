from abc import ABC, abstractmethod
from torch import Tensor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .base_cost import BaseCost
    from .base_model import BaseModel
    from .base_utility import BaseUtility

class BaseBestResponse(ABC):
    @abstractmethod
    def __init__(self, utility:'BaseUtility|None', cost:'BaseCost|None'):
        pass

    @abstractmethod
    def __call__(self, X:Tensor, model:'BaseModel') -> Tensor:
        pass