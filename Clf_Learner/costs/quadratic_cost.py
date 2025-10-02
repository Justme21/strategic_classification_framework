import torch
from torch import Tensor

from ..interfaces import BaseCost
from ..datasets.tools.dataset_tools import Standardiser

class QuadraticCost(BaseCost):
    """Mahalanobis Distance to compute ||X-Z||_{2}^{2}"""
    #torch.sum(torch.pow(X-Z, 2),1)

    def __init__(self, eps=None, radius=2, **kwargs):
        if eps is not None:
            self.eps = eps
        else:
            self.eps = (radius**2)/4 # Formula to compute the epsilon corresponding to allowing best response to move by up to radius distance

        self.standardiser = None

    def __call__(self, X:Tensor, Z:Tensor) -> Tensor:
        Q = torch.ones_like(X, device=X.device) # Should be a [batch x x_dim x x_dim] tensor, but we have no covariances, so this works
        if self.standardiser is not None:
            # We want to compute the regular distance on the standardised data, so we recale
            sigma_sq = torch.pow(self.standardiser.std,2)
            Q = sigma_sq*Q

        dist= (X-Z)*Q*(X-Z)
        coef = 1/(2*self.eps)
        return coef*torch.sum(dist, 1)

    def set_standardiser(self, standardiser:Standardiser):
        self.standardiser = standardiser

    def get_standardiser(self):
        return self.standardiser