import torch

from ..interfaces import BaseCost, BaseBestResponse, BaseUtility

ZERO_THRESHOLD = 1e-3

class SGDBestResponse(BaseBestResponse):
    """The best response to a Linear Model has a closed form solution that can be evaluated exactly"""
    def __init__(self, utility:BaseUtility, cost:BaseCost, radius=2, lr=1e-2, max_epochs=100, **kwargs):
        self.cost = cost
        self.utility = utility
        self.radius = radius # The degree to which an input can be strategically perturbed

        self.max_epochs = max_epochs
        self.lr=lr
        self.opt = torch.optim.Adam

    def __call__(self, X, model):
        Z = X.detach().clone().requires_grad_()
        opt = self.opt([Z], self.lr)
        for _ in range(self.max_epochs):
            opt.zero_grad()
            l = (self.utility(Z, model) - self.cost(X,Z)).sum()
            l.backward()
            opt.step()
            if abs(l.item())<ZERO_THRESHOLD:
                # Consider it converged
                break
        
        dist = torch.sqrt(torch.sum(torch.pow(X-Z, 2), dim=1))
        pred_old = model.predict(X)
        pred_new = model.predict(Z)

        cond1 = dist<self.radius # Only accept changes that are within radius
        cond2 = pred_new>0 # Only accept changes that result in positive prediction
        cond3 = pred_old<0 # Only make changes when not already positive
        cond=cond1*cond2*cond3
        cond = cond.repeat(X.size(1), 1).T

        X_opt = torch.where(cond, Z, X)

        return X_opt