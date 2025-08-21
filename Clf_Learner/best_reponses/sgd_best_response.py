import torch

from ..interfaces import BaseCost, BaseBestResponse, BaseModel, BaseUtility

ZERO_THRESHOLD = 1e-3

class SGDBestResponse(BaseBestResponse):
    """The best response to a Linear Model has a closed form solution that can be evaluated exactly"""
    def __init__(self, utility:BaseUtility, cost:BaseCost, radius=2, lr=1e-2, max_epochs=100, **kwargs):
        self._cost = cost
        self._utility = utility

        self.max_epochs = max_epochs
        self.lr=lr
        self.opt = torch.optim.Adam

    def _get_utility(self, Z:torch.Tensor, X:torch.Tensor, model:BaseModel) -> torch.Tensor:
        return self._utility(Z, model) - self._cost(X,Z)

    def __call__(self, X:torch.Tensor, model:BaseModel) ->torch.Tensor:
        Z = X.detach().clone().requires_grad_()
        opt = self.opt([Z], self.lr)

        pred_old = model.predict(X)
        cond1 = pred_old<0
        for _ in range(self.max_epochs):
            opt.zero_grad()
            # To maximise the objective, we do gradient descent on the negative of the loss
            util = self._get_utility(Z, X, model)

            l = -(cond1*util).sum()
            l.backward(inputs=[Z])
            opt.step()

            max_grad = torch.max(torch.abs(Z.grad)) if Z.grad is not None else 0
            if max_grad<ZERO_THRESHOLD:
                # Consider it converged
                break
        
        util_Z = self._get_utility(Z, X, model)
        util_X = self._get_utility(X, X, model)
        cond2 = util_Z>util_X

        cond=cond1*cond2
        cond = cond.repeat(X.size(1), 1).T
            
        X_opt = torch.where(cond, Z, X)
        return X_opt