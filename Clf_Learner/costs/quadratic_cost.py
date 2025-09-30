import torch
from torch import Tensor
from typing import Callable

from ..interfaces import BaseCost

class QuadraticCost(BaseCost):
    def __init__(self, eps=None, radius=2, **kwargs):
        if eps is not None:
            self.eps = eps
        else:
            self.eps = (radius**2)/4 # Formula to compute the epsilon corresponding to allowing best response to move by up to radius distance

    def __call__(self, X:Tensor, Z:Tensor) -> Tensor:
        if self.standardisation_inverter is not None:
            X = self.standardisation_inverter(X)
            Z = self.standardisation_inverter(Z)

        dist = torch.sum(torch.pow(X-Z, 2),1)
        return 1/(2*self.eps)*dist

    def set_standardisation_inverter(self, invert_standardisation:Callable):
        self.standardisation_inverter = invert_standardisation