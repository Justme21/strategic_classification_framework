from abc import ABC, abstractmethod

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
    def get_best_response(self, X, model:'BaseModel'):
        pass