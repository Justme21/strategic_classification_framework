import torch
from torch import Tensor

from ..interfaces import BaseCost

class QuadraticCost(BaseCost):
    def __init__(self, eps=None, radius=2, **kwargs):
        if eps is not None:
            self.eps = eps
        else:
            self.eps = (radius**2)/4 # Formula to compute the epsilon corresponding to allowing best response to move by up to radius distance

    def __call__(self, X:Tensor, Z:Tensor) -> Tensor:
        dist = torch.sum(torch.pow(X-Z, 2),1)
        return 1/(2*self.eps)*dist