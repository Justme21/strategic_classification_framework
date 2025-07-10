import torch.linalg
from torch import Tensor

from ..interfaces import BaseCost

class QuadraticCost(BaseCost):
    def __init__(self, eps=1, **kwargs):
        self.eps = eps

    def __call__(self, X:Tensor, Z:Tensor) -> Tensor:
        dist = torch.sum(torch.pow(X-Z, 2))
        return 1/(2*self.eps)*dist