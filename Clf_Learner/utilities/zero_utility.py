from torch import Tensor

from ..interfaces import BaseUtility, BaseModel

class StrategicUtility(BaseUtility):
    """Uninformative Utility"""
    def __init__(self, **kwargs):
        pass

    def __call__(self, X:Tensor, model:'BaseModel') -> Tensor:
        return Tensor([0])