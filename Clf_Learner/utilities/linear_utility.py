from torch import Tensor

from ..interfaces import BaseUtility, BaseModel

class LinearUtility(BaseUtility):
    def __init__(self, coef=1, **kwargs):
        self.coef = coef

    def __call__(self, X:Tensor, model:'BaseModel') -> Tensor:
        weight = model.get_weights()
        return weight*X