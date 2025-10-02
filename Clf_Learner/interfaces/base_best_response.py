from abc import ABC, abstractmethod
from torch import Tensor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .base_cost import BaseCost
    from .base_model import BaseModel
    from .base_utility import BaseUtility

class BaseBestResponse(ABC):
    @abstractmethod
    def __init__(self, utility:'BaseUtility', cost:'BaseCost', strategic_columns:list[int]|None,):
        assert cost is not None, f"Error: {type(self).__name__} requires a valid cost function be specified"
        assert utility is not None, f"Error: {type(self).__name__} requires a valid utility function be specified"
        self._cost = cost
        self._utility = utility

        self._strategic_columns = strategic_columns

    @abstractmethod
    def objective(self, Z:Tensor, X:Tensor, model:'BaseModel') -> Tensor:
        """Evaluate the best response objective"""
        # This is used to as part of ImplciitDifferentiation evaluation for the loss wrt best response
        raise NotImplementedError

    @abstractmethod
    def __call__(self, X:Tensor, model:'BaseModel') -> Tensor:
        pass