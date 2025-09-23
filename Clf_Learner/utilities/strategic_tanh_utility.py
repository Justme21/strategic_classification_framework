import torch 
from torch import Tensor
from typing import cast

from ..interfaces import BaseUtility, BaseModel

class StrategicTanhUtility(BaseUtility):
    def __init__(self, coef=1, alpha=7.5, **kwargs):
        self.coef = coef
        self.alpha = alpha

    def __call__(self, X:Tensor, model:'BaseModel') -> Tensor:
        model_out = model.forward(X)

        return self.coef*torch.tanh(self.alpha*model_out)