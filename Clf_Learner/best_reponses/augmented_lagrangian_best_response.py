import torch
import torch.nn.functional as F

from ..interfaces import BaseCost, BaseBestResponse, BaseModel, BaseUtility

ZERO_PRIMAL_THRESHOLD = 1e-5
ZERO_DUAL_THRESHOLD = 1e-3

class AugmentedLagrangianBestResponse(BaseBestResponse):
    """ Computing the Best Response by treating the objective as a constrained minimization problem for the cost"""
    def __init__(self, utility:BaseUtility, cost:BaseCost, lr=1e-2, max_epochs=5, steps_per_epoch=100, mu_init=50.0, mu_mult=1.1, **kwargs):
        assert cost is not None, "Error: Feasibility Best Response requires a valid cost function be specified"
        assert utility is not None, "Error: Feasibility Best Response requires a valid utility function be specified"
        self._cost = cost
        self._utility = utility

        self.max_epochs = max_epochs
        self.steps_per_epoch = steps_per_epoch
        self.lr=lr
        self.opt = torch.optim.SGD

        self.mu_init = mu_init # Constraint weight term
        self.mu_mult = mu_mult # Multiplier for the constraint weight term

    def _benefit_constraint_func(self, Z, model):
        # Non-zero if utility(Z)<0
        return F.relu(-self._utility(Z, model))

    def _feasibility_constraint_func(self, cost, X, model):
        # Non-zero if cost>1-f(x)
        f_x = model.predict(X).detach()
        feasibility_obj = (1.0-f_x).detach()
        return F.relu(cost - feasibility_obj)

    def _get_utility(self, Z:torch.Tensor, X:torch.Tensor, model:BaseModel, lam1:torch.Tensor, lam2: torch.Tensor, mu:float) -> torch.Tensor:
        cost = self._cost(X,Z)
 
        benefit_constraint = self._benefit_constraint_func(Z, model)
        benefit_comp = 0.5*mu*benefit_constraint*benefit_constraint
        benefit_lagrange = lam1*benefit_constraint

        feasibility_constraint = self._feasibility_constraint_func(cost, X, model)
        feasibility_comp = 0.5*mu*feasibility_constraint*feasibility_constraint
        feasibility_lagrange = lam2*feasibility_constraint
        
        L = cost + benefit_comp + benefit_lagrange + feasibility_comp + feasibility_lagrange

        #return L
        return cost, benefit_comp, benefit_lagrange, feasibility_comp, feasibility_lagrange

    def __call__(self, X:torch.Tensor, model:BaseModel, debug=False) ->torch.Tensor:
        #print("Starting Best Response Computation")
        Z = X.detach().clone().requires_grad_()

        #print(f"Model Weights here are: {model.get_weights(include_bias=True)}")
        c1 = self._benefit_constraint_func(X, model)
        #print(f"Initial Benefit Constraint Violations: {c1.mean().item()}")
        pred_old = model.predict(X).detach()
        cond1 = pred_old<0
        #print(f"Initial number of negative classifications: {cond1.sum()}")

        # Lagrangian coefficients for each constraint
        lam_init = 100 #NOTE: This parameter affects the magnitude of parameter changes in each step
        lam1 = lam_init * torch.ones((len(X)))
        lam2 = lam_init * torch.ones((len(X)))

        mu = self.mu_init

        for _ in range(self.max_epochs):
            Z = Z.detach().requires_grad_() # Fresh Z for each epoch here
            opt = self.opt([Z], self.lr)

            # Primal Optimisation
            for _ in range(self.steps_per_epoch):
                opt.zero_grad()
                #util = self._get_utility(Z, X, model, lam1, lam2, mu)
                cost, benefit_comp, benefit_lagrange, feasibility_comp, feasibility_lagrange = self._get_utility(Z, X, model, lam1, lam2, mu)
                util = cost + benefit_comp + benefit_lagrange + feasibility_comp + feasibility_lagrange

                l = (cond1*util).mean()
                #print(f"Cost: {cost.sum().item()}\t Constr: {(util-cost).sum().item()}\t Loss: {l.item()}")
                l.backward(inputs=[Z])
                opt.step()

                max_grad = torch.max(torch.abs(Z.grad)) if Z.grad is not None else 0
                if max_grad<ZERO_PRIMAL_THRESHOLD:
                    # Consider it converged
                    #print("Gradient 0; ending inner loop")
                    break

            # Dual Optimisation
            with torch.no_grad():
                cost = self._cost(X, Z)

                c1 = self._benefit_constraint_func(Z, model)
                c2 = self._feasibility_constraint_func(cost, X, model)

                max_violation = torch.max(torch.stack([c1.max(), c2.max()])).item()

                #print(f"Max Violation is: {max_violation}")
                if max_violation <= ZERO_DUAL_THRESHOLD:
                    # All constraint conditions satisfied
                    #print("No violations; ending outer loop")
                    break

                # Update the dual variables
                lam1 = lam1 + mu * c1
                lam2 = lam2 + mu * c2

                # Increase penalty parameter to exert more constraint pressure on next iteration
                mu *= self.mu_mult

                #print(f"After update: lam1: {lam1.mean().item()}\tlam2: {lam2.mean().item()}\t Mu: {mu}")
                #print(f"Cost: {cost.mean().item()} F(Z): {self._utility(Z, model).mean().item()}")
                #print(f"C1; Max: {c1.max().item()}\t Mean: {c1.mean().item()}")
                #print(f"C2; Max: {c2.max().item()}\t Mean: {c2.mean().item()}")

        # Produce outputs
        with torch.no_grad():
            # Only realise changes when all conditions satisfied, otherwise keep old X
            cond2 = self._benefit_constraint_func(Z, model)==0

            cost = self._cost(X,Z)
            cond3 = self._feasibility_constraint_func(cost, X, model)==0

            cond=cond1*cond2*cond3
            cond = cond.repeat(X.size(1), 1).T
            #cond = cond.unsqueeze(1).expand(-1, X.size(1)) <- clearer?

            X_opt = torch.where(cond, Z, X)

        t1 = model.predict(X_opt)
        #print(f"Num pred<0 before: {len(pred_old[pred_old<0])}\tNum Pred<0 After: {len(t1[t1<0])}")
        #print("Ending Best Response Computation")
        return X_opt