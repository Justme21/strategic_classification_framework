import math
import torch

from ..interfaces import BaseCost, BaseBestResponse, BaseModel, BaseUtility

ZERO_THRESHOLD = 1e-3

class SGDBestResponse(BaseBestResponse):
    """Use Stochastic Gradient Descent on the Agent objective to determine the best response z, for a given x, and a given function"""
    def __init__(self, utility:BaseUtility, cost:BaseCost, radius=2, lr=1e-2, max_epochs=100, **kwargs):
        assert cost is not None, "Error: SGD Best Response requires a valid cost function be specified"
        assert utility is not None, "Error: SGD Best Response requires a valid utility function be specified"
        self._cost = cost
        self._utility = utility

        self.max_epochs = max_epochs
        self.lr=lr
        self.opt = torch.optim.SGD

    def _get_utility(self, Z:torch.Tensor, X:torch.Tensor, model:BaseModel) -> torch.Tensor:
        return self._utility(Z, model) - self._cost(X,Z)

    def __call__(self, X:torch.Tensor, model:BaseModel, debug=False) ->torch.Tensor:
        Z = X.detach().clone().requires_grad_()
        opt = self.opt([Z], self.lr)

        pred_old = model.predict(X)
        cond1 = pred_old<0
        for t in range(self.max_epochs):
            opt.zero_grad()
            # To maximise the objective, we do gradient descent on the negative of the loss
            util = self._get_utility(Z, X, model)

            l = -(cond1*util).sum()

            l.backward(inputs=[Z])

            # NOTE: Doesn't entirely resolve instability in results
            # Decaying the learning rate to make the later iteration rounds less noisy
            #for param_group in opt.param_groups:
            #    param_group['lr'] = self.lr/math.sqrt(t+1)
            opt.step()

            max_grad = torch.max(torch.abs(Z.grad)) if Z.grad is not None else 0
            if max_grad<ZERO_THRESHOLD:
                # Consider it converged
                break
        
        with torch.no_grad():
            util_Z = self._get_utility(Z, X, model)
            util_X = self._get_utility(X, X, model)
            cond2 = util_Z>util_X

            cond=cond1*cond2
            cond = cond.repeat(X.size(1), 1).T

            X_opt = torch.where(cond, Z, X)

        return X_opt