import torch

from Clf_Learner.interfaces.base_model import BaseModel

from ..interfaces import BaseBestResponse

class LinearBestResponse(BaseBestResponse):
    """The best response to a Linear Model has a closed form solution that can be evaluated exactly"""
    def __init__(self, utility, cost, radius=2, **kwargs):

        self.utility = utility
        self.cost = cost
        self.radius = radius # The degree to which an input can be strategically perturbed

    def objective(self, Z: torch.Tensor, X: torch.Tensor, model: BaseModel) -> torch.Tensor:
        # Linear Best Response is discontinuous so doesn't have a meaningful gradient
        return self.utility(Z, model) - self.cost(X, Z)

    def __call__(self, X, model, **kwargs):
        W = model.get_weights(include_bias=False) # Omit bias term when computing distance

        # Compute magnitude and direction of movement orthogonal to decision boundary
        W_norm = torch.norm(W, p=2)
        distances = model.forward(X).detach()/W_norm
        X_moved = torch.cat([x - d*W/W_norm for x, d in zip(X, distances)], dim=0)

        # Rescale to account for standardisation
        scale = 1
        standardiser = self.cost.get_standardiser()
        if standardiser is not None:
            sigma = standardiser.std
            scaled_W_norm = torch.norm(W/sigma, p=2)
            scale = scaled_W_norm/W_norm
        
        # Constraints applied to best responses
        cond1 = -self.radius*scale <= distances # Must be within radius of decision boundary
        cond2 = distances < 0 # Must be beneath decision boundary
        cond = cond1*cond2

        cond = cond.repeat((X.size(1), 1)).T
        X_opt = torch.where(cond, X_moved, X)

        return X_opt
