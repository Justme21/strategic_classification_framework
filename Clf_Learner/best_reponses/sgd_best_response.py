import math
import torch

from ..interfaces import BaseCost, BaseBestResponse, BaseModel, BaseUtility

ZERO_THRESHOLD = 1e-3

class SGDBestResponse(BaseBestResponse):
    """The best response to a Linear Model has a closed form solution that can be evaluated exactly"""
    def __init__(self, utility:BaseUtility, cost:BaseCost, radius=2, lr=1e-2, max_epochs=100, **kwargs):
        assert cost is not None, "Error: SGD Best Response requires a valid cost function be specified"
        assert utility is not None, "Error: SGD Best Response requires a valid utility function be specified"
        self._cost = cost
        self._utility = utility

        self.max_epochs = max_epochs
        self.lr=lr
        #self.opt = torch.optim.Adam
        self.opt = torch.optim.SGD

    def _get_utility(self, Z:torch.Tensor, X:torch.Tensor, model:BaseModel) -> torch.Tensor:
        return self._utility(Z, model) - self._cost(X,Z)

    def __call__(self, X:torch.Tensor, model:BaseModel, debug=False) ->torch.Tensor:
        Z = X.detach().clone().requires_grad_()
        #self.lr = 0.5
        opt = self.opt([Z], self.lr)

        pred_old = model.predict(X)
        cond1 = pred_old<0
        for t in range(self.max_epochs):
            opt.zero_grad()
            # To maximise the objective, we do gradient descent on the negative of the loss
            util = self._get_utility(Z, X, model)
            #t1 = self._utility(Z, model)
            #t1 = model(Z).squeeze()
            #t2 = self._cost(X,Z)
            #util = (t1 - t2)


            l = -(cond1*util).sum()

            #t1_grad = torch.autograd.grad(t1.sum(), Z, retain_graph=True)[0]
            #t2_grad = torch.autograd.grad(t2.sum(), Z, retain_graph=True)[0]
            #opt.zero_grad()

            #t1.retain_grad()
            #t2.retain_grad()
            #util.retain_grad()
            #l.retain_grad()

            l.backward(inputs=[Z])
            #l.backward()

            # NOTE: Doesn't entirely resolve instability in results
            # Decaying the learning rate to make the later iteration rounds less noisy
            #for param_group in opt.param_groups:
            #    param_group['lr'] = self.lr/math.sqrt(t+1)
            opt.step()
            
            if debug:
                print(f"T: {t}\t utilZ: {self._utility(Z, model)[5]}\tcostZ: {self._cost(X,Z)[5]}")
                #print(f"T: {t}: Grad: {Z.grad[5]}\t dt1/dz = {t1_grad[5]}\tdt2/dz = {t2_grad[5]}")
                import pdb
                pdb.set_trace()

            if torch.max(torch.abs(Z.grad))<ZERO_THRESHOLD:
                # Consider it converged
                break
        
        util_Z = self._get_utility(Z, X, model)
        util_X = self._get_utility(X, X, model)
        cond2 = util_Z>util_X

        cond=cond1*cond2
        cond = cond.repeat(X.size(1), 1).T

        X_opt = torch.where(cond, Z, X)

        if debug:
            t1_X = self._utility(X, model)
            t1_Z = self._utility(Z, model)

            t2_X = self._cost(X,X)
            t2_Z = self._cost(Z,X)

            import pdb
            pdb.set_trace()

        return X_opt