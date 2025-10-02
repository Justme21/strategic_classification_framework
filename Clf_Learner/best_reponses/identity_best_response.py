import torch

from ..interfaces.base_best_response import BaseBestResponse

class IdentityResponse(BaseBestResponse):
    """Modelling when there is no Strategic Behaviour"""
    def __init__(self, utility, cost, **kwargs):
        # Don't need init here
        pass

    def objective(self, Z, X, model):
        return torch.ones_like(Z)

    def __call__(self, X, model, y=None):
        return X