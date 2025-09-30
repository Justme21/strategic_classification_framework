import torch
import torch.nn.functional as F

from ..interfaces import BaseCost, BaseBestResponse, BaseModel, BaseUtility

ZERO_PRIMAL_THRESHOLD = 1e-5
ZERO_DUAL_THRESHOLD = 1e-3

class AugmentedLagrangianBestResponse(BaseBestResponse):
    """ Computing the Best Response by treating the objective as a constrained minimization problem for the cost"""
    def __init__(self, utility:BaseUtility, cost:BaseCost, lr=1e-2, max_epochs=5, steps_per_epoch=100, mu_init=1.0, mu_mult=1.1, **kwargs):
        BaseBestResponse.__init__(self, utility, cost)

        self.max_epochs = max_epochs
        self.steps_per_epoch = steps_per_epoch
        self.lr=lr
        self.opt = torch.optim.SGD

        self.mu_init = mu_init # Constraint weight term
        self.mu_mult = mu_mult # Multiplier for the constraint weight term

    def _benefit_constraint_func(self, Z, model):
        # >0 if utility(Z)<0
        return -self._utility(Z, model)

    def _feasibility_constraint_func(self, cost, X, model):
        # >0 if cost>1-f(x)
        f_x = model.predict(X).detach()
        feasibility_obj = (1.0-f_x).detach()
        return cost - feasibility_obj

    def objective(self, Z:torch.Tensor, X:torch.Tensor, model:BaseModel, lam1:torch.Tensor, lam2: torch.Tensor, mu:float) -> torch.Tensor:
        cost = self._cost(X,Z)
 
        benefit_constraint = self._benefit_constraint_func(Z, model)
        benefit_comp = 0.5*mu*benefit_constraint*benefit_constraint
        benefit_lagrange = lam1*benefit_constraint

        feasibility_constraint = self._feasibility_constraint_func(cost, X, model)
        feasibility_comp = 0.5*mu*feasibility_constraint*feasibility_constraint
        feasibility_lagrange = lam2*feasibility_constraint
        
        L = cost + benefit_comp + benefit_lagrange + feasibility_comp + feasibility_lagrange

        return L

    def __call__(self, X:torch.Tensor, model:BaseModel, debug=False) ->torch.Tensor:
        Z = X.detach().clone().requires_grad_()

        c1 = self._benefit_constraint_func(X, model)
        pred_old = model.predict(X).detach()
        cond1 = pred_old<0

        # Lagrangian coefficients for each constraint
        lam_init = 0.1 #NOTE: This parameter affects the magnitude of parameter changes in each step
        lam1 = lam_init * torch.ones((len(X)))
        lam2 = lam_init * torch.ones((len(X)))

        mu = self.mu_init

        for _ in range(self.max_epochs):
            Z = Z.detach().requires_grad_() # Fresh Z for each epoch here
            opt = self.opt([Z], self.lr)

            # Primal Optimisation
            for _ in range(self.steps_per_epoch):
                opt.zero_grad()
                util = self.objective(Z, X, model, lam1, lam2, mu)

                l = util.mean()
                l.backward(inputs=[Z])
                opt.step()

                print(f"L: {l.item()}")
                max_grad = torch.max(torch.abs(Z.grad)) if Z.grad is not None else 0
                if max_grad<ZERO_PRIMAL_THRESHOLD:
                    # Consider it converged
                    break

            # Dual Optimisation
            with torch.no_grad():
                cost = self._cost(X, Z)

                c1 = self._benefit_constraint_func(Z, model)
                c2 = self._feasibility_constraint_func(cost, X, model)

                max_violation = torch.max(torch.stack([c1.max(), c2.max()])).item()

                if max_violation <= ZERO_DUAL_THRESHOLD:
                    # All constraint conditions satisfied
                    break

                # Update the dual variables
                lam1 = lam1 + mu * c1
                lam2 = lam2 + mu * c2

                # Increase penalty parameter to exert more constraint pressure on next iteration
                mu *= self.mu_mult

                print(f"Lam1: {lam1.mean()}\tLam2: {lam2.mean()}\t Mu: {mu}")

        # Produce outputs
        with torch.no_grad():
            # Only realise changes when all conditions satisfied, otherwise keep old X
            cond2 = self._benefit_constraint_func(Z, model)==0

            cost = self._cost(X,Z)
            cond3 = self._feasibility_constraint_func(cost, X, model)==0

            cond = cond2*cond3
            cond = cond.repeat(X.size(1), 1).T
            #cond = cond.unsqueeze(1).expand(-1, X.size(1)) <- clearer?

            X_opt = torch.where(cond, Z, X)

        t1 = model.predict(X_opt)
        return X_opt