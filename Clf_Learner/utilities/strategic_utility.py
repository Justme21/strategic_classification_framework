from torch import Tensor

from ..interfaces import BaseUtility, BaseModel

class StrategicUtility(BaseUtility):
    """Letting f(x) be the utility"""
    def __init__(self, coef=1, **kwargs):
        self.coef = coef

    def __call__(self, X:Tensor, model:'BaseModel') -> Tensor:
        return model.forward_utility(X)