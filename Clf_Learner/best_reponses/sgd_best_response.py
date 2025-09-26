import torch

from ..interfaces import BaseCost, BaseBestResponse, BaseModel, BaseUtility

ZERO_THRESHOLD = 1e-2
NO_IMPROVEMENT_THRESHOLD = 20

class SGDBestResponse(BaseBestResponse):
    """Use Stochastic Gradient Descent on the Agent objective to determine the best response z, for a given x, and a given function"""
    def __init__(self, utility:BaseUtility, cost:BaseCost, lr=1e-2, max_iterations=100, **kwargs):
        #BaseBestResponse.__init__(self, utility, cost)
        self._cost = cost
        self._utility = utility

        self._cost = cost
        self._utility = utility

        self.max_iterations = max_iterations
        self.lr= lr
        self.opt = torch.optim.SGD

    def objective(self, Z:torch.Tensor, X:torch.Tensor, model:BaseModel) -> torch.Tensor:
        return self._utility(Z, model) - self._cost(X, Z)

    def __call__(self, X:torch.Tensor, model:BaseModel, debug=False, animate_rate=None) ->torch.Tensor:
        Z = X.detach().clone().requires_grad_()
        opt = self.opt([Z], self.lr)

        if animate_rate is not None:
            assert isinstance(animate_rate, int)
            Z_store = X.detach().clone().unsqueeze(0)

        l_old_val = None
        no_improvement_count = 0
        for t in range(self.max_iterations):
            opt.zero_grad()
            # To maximise the objective, we do gradient descent on the negative of the loss
            obj = self.objective(Z, X, model)

            l = -obj.sum()

            l.backward(inputs=[Z])

            # NOTE: Doesn't entirely resolve instability in results
            # Decaying the learning rate to make the later iteration rounds less noisy
            #for param_group in opt.param_groups:
            #    param_group['lr'] = self.lr/math.sqrt(t+1)
            opt.step()

            # Save frames if animating
            if animate_rate is not None and t%animate_rate==0:
                Z_store = torch.cat([Z_store, Z.detach().clone().unsqueeze(0)], dim=0)

            # Check for Convergence
            l_new_val = l.item()/len(X)
            if l_old_val is not None:
                l_diff = abs(l_old_val - l_new_val)
                if l_diff>ZERO_THRESHOLD:
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            
            if no_improvement_count>NO_IMPROVEMENT_THRESHOLD:
                # Consider it converged
                break

            l_old_val = l_new_val

        # To do the where evaluation we need a X_store the same size as Z
        if animate_rate is not None:
            if t%animate_rate !=0:
                # Store the converged Z value
                Z_store = torch.cat([Z_store, Z.detach().clone().unsqueeze(0)], dim=0) 
            X_store = X.detach().clone().unsqueeze(0).repeat([Z_store.shape[0]]+[1 for _ in range(len(X.shape))])
        
        with torch.no_grad():
            #obj_Z = self.objective(Z, X, model)
            #obj_X = self.objective(X, X, model)
            #cond2 = obj_Z>obj_X
            #cond = cond2

            pred_new = model.predict(Z)
            cond2 = pred_new>0

            cost = self._cost(X,Z)
            cond3 = cost<=2
            cond = cond2*cond3

            #cond=cond1*cond2*cond3

            cond = cond.unsqueeze(1)
            if animate_rate is not None:
                cond = cond.unsqueeze(0)
                X_opt = torch.where(cond, Z_store, X_store).clone()
            else:
                X_opt = torch.where(cond, Z, X).clone()

        return X_opt