import torch 
from torch import Tensor

from ..interfaces import BaseUtility, BaseModel

class StrategicUtility(BaseUtility):
    def __init__(self, coef=1, **kwargs):
        self.coef = coef

    def __call__(self, X:Tensor, model:'BaseModel') -> Tensor:
        pred = model.forward(X)
        return self.coef*pred