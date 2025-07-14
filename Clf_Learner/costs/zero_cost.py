import torch
from torch import Tensor

from ..interfaces import BaseCost

class ZeroCost(BaseCost):
    def __init__(self, **kwargs):
        pass
    def __call__(self, X:Tensor, Z:Tensor) -> Tensor:
        return torch.Tensor([0])