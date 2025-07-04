import torch

from ..interfaces.base_best_response import BaseBestResponse

class LinearBestResponse(BaseBestResponse):
    """The best response to a Linear Model has a closed form solution that can be evaluated exactly"""
    def __init__(self, utility, cost):
        self.cost = cost
        self.utility = utility

        self.radius = 2 # The degree to which an input can be strategically perturbed

    def get_best_response(self, X, model):
        W = model.get_params()

        # Compute magnitude and direction of movement orthogonal to decision boundary
        norm = torch.norm(W, p=2)
        distances = model.forward(X)/norm
        X_moved = torch.stack([x - d*W/norm for x, d in zip(X, distances)])
        
        # Constraints applied to best responses
        cond1 = -self.radius <= distances # Must be within radius of decision boundary
        cond2 = distances < 0 # Must be beneath decision boundary
        cond = cond1*cond2

        cond = cond.repeat((X.size(1), 1)).T
        X_opt = torch.where(cond, X_moved, X)

        return X_opt
