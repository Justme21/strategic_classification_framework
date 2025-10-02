import torch

from ..interfaces import BaseCost, BaseBestResponse, BaseModel, BaseUtility
from ..tools.device_tools import get_device

ZERO_THRESHOLD = 1e-7
NO_IMPROVEMENT_THRESHOLD = 100

class LagrangianBestResponse(BaseBestResponse):
    """ Computing the Best Response by treating the objective as a constrained minimization problem for the cost"""
    def __init__(self, utility:BaseUtility, cost:BaseCost, lr=1e-2, max_iterations=10, margin=1e-3,\
                 lagrange_mult_lr=1e-4, lagrange_mult_cost_lr=1e-3, **kwargs):
        assert cost is not None, "Error: Feasibility Best Response requires a valid cost function be specified"
        assert utility is not None, "Error: Feasibility Best Response requires a valid utility function be specified"
        self._cost = cost
        self._utility = utility

        self.max_iterations = max_iterations
        self.lr=lr
        self.opt = torch.optim.Adam

        self.lagrange_mult_init = 0.0

        device = get_device()
        self.lagrange_mult = torch.Tensor([]).to(device)
        self.lagrange_mult_cost = torch.Tensor([]).to(device)

        self._t = 0 # Value only used for hyperparamter tuning
        self._margin = margin
        self._lagrange_mult_lr = lagrange_mult_lr
        self._lagrange_mult_cost_lr = lagrange_mult_cost_lr

    def objective(self, Z: torch.Tensor, X: torch.Tensor, model: BaseModel):
        return self._objective_impl(Z, X, model, self.lagrange_mult, self.lagrange_mult_cost)

    def _objective_impl(self, Z:torch.Tensor, X:torch.Tensor, model:BaseModel, lagrange_mult:torch.Tensor, lagrange_mult_cost:torch.Tensor) -> torch.Tensor:
        cost = self._cost(X,Z)

        # Constraint 1: utility(Z,model) > 0 
        benefit_lagrange = -lagrange_mult*(self._utility(Z, model)-self._margin)
        
        # Constraint 2: cost(X,Z) < 2 → g2(Z) = cost(X,Z) - 2 ≤ 0
        penalty_cost = lagrange_mult_cost * (cost - 2)

        L = cost + benefit_lagrange + penalty_cost
        return L

    def __call__(self, X:torch.Tensor, model:BaseModel, debug=False, animate_rate=None, y=None) ->torch.Tensor:
        Z = X.detach().clone().requires_grad_()

        lagrange_mult = torch.tensor([0.0 for _ in range(len(X))], device=X.device).requires_grad_()
        lagrange_mult_cost = torch.tensor([0.0 for _ in range(len(X))], device=X.device).requires_grad_()

        if animate_rate is not None:
            assert isinstance(animate_rate, int)
            Z_store = X.detach().clone().unsqueeze(0)

        opt_z = self.opt([Z], lr=self.lr) # Adam optimiser
        opt_lagrange = torch.optim.Adam([lagrange_mult], self._lagrange_mult_lr, maximize=True)
        opt_lagrange_cost = torch.optim.Adam([lagrange_mult_cost], self._lagrange_mult_cost_lr, maximize=True)

        pred_x = model.predict(X)<0
        cost_old_val = None
        util_old_val = None

        device = get_device()
        if device=="cuda":
            # Compilation is optimised for speedup on CUDA
            obj_comp = torch.compile(self._objective_impl)
        #elif device == 'mps':
        #    # torch.compile not supported on MPS
        #    obj_comp = torch.compile(self._objective_impl, backend='aot_eager')
        else:
            obj_comp = self._objective_impl

        for t in range(self.max_iterations):
            # #for g in opt_z.param_groups:
            # #    g['lr'] = self.lr / (1 + 0.1*t)

            opt_z.zero_grad()
            opt_lagrange.zero_grad()
            opt_lagrange_cost.zero_grad()
            
            util = obj_comp(Z, X, model, lagrange_mult, lagrange_mult_cost)
            l = util.mean()
            
            l.backward(inputs=[Z, lagrange_mult, lagrange_mult_cost])

            opt_z.step()
            opt_lagrange.step()
            opt_lagrange_cost.step()

            with torch.no_grad():
                lagrange_mult.clamp_(min=0)
                lagrange_mult_cost.clamp_(min=0)

            if debug and t%50 == 0:
                print("iter", t)
                print("Negative Points")
                print(f"  util.mean() {self._utility(Z, model)[pred_x].mean().item()}")
                print(f"  cost.mean() {self._cost(X,Z)[pred_x].mean().item()}", )
                print(f"  Z_X: {Z[pred_x, 0].mean()}\tZ_Y: {Z[pred_x,1].mean()}")
                print(f"  lagrange_mult.mean {lagrange_mult[pred_x].mean().item()} \t lagrange_cost.mean {lagrange_mult_cost[pred_x].mean().item()}")

                print("Positive Points")
                print(f"  util.mean() {self._utility(Z, model)[~pred_x].mean().item()}")
                print(f"  cost.mean() {self._cost(X,Z)[~pred_x].mean().item()}", )
                print(f"  Z_X: {Z[~pred_x, 0].mean()}\tZ_Y: {Z[~pred_x,1].mean()}")
                print(f"  lagrange_mult.mean {lagrange_mult[~pred_x].mean().item()} \t lagrange_cost.mean {lagrange_mult_cost[~pred_x].mean().item()}")

            if animate_rate is not None and t%animate_rate==0:
                Z_store = torch.cat([Z_store, Z.detach().clone().unsqueeze(0)], dim=0)

            # Convergence Check
            if t>4000 and t%NO_IMPROVEMENT_THRESHOLD==0:
                cost_diff = None
                util_diff = None
                cost_new_val = self._cost(X, Z).mean().item()
                util_new_val = self._utility(Z, model).mean().item()
                if cost_old_val is not None:
                    cost_diff = abs(cost_old_val - cost_new_val)
                if util_old_val is not None:
                    util_diff = abs(util_old_val - util_new_val)

                if (cost_diff is not None and cost_diff<ZERO_THRESHOLD) and (util_diff is not None and cost_diff<ZERO_THRESHOLD):
                        # Consider it converged
                        break
                cost_old_val = cost_new_val
                util_old_val = util_new_val

        
        # To do the where evaluation we need a X_store the same size as Z
        if animate_rate is not None:
            if t%animate_rate !=0:
                # Store the converged Z value
                Z_store = torch.cat([Z_store, Z.detach().clone().unsqueeze(0)], dim=0)
            X_store = X.detach().clone().unsqueeze(0).repeat([Z_store.shape[0]]+[1 for _ in range(len(X.shape))])

        # Store Lagrange Multiplier
        self.lagrange_mult = lagrange_mult.detach().clone()
        self.lagrange_mult_cost = lagrange_mult_cost.detach().clone()

        self._t = t

        # Produce outputs
        with torch.no_grad():
            # Only realise changes when all conditions satisfied, otherwise keep old X
            cost = self._cost(X,Z)
            cond1 = cost<2

            #util_X = self._utility(X, model)
            #util_Z = self._utility(Z, model)
            #cond2 = util_Z>util_X

            pred_X = model.predict(X)
            pred_Z = model.predict(Z)
            cond2 = pred_Z>pred_X

            cond=cond1*cond2
            #cond = cond2
            cond = cond.repeat(X.size(1), 1).T
            #cond = cond.unsqueeze(1).expand(-1, X.size(1)) <- clearer?

            if animate_rate is not None:
                cond = cond.unsqueeze(0)
                X_opt = torch.where(cond, Z_store, X_store)
            else:
                X_opt = torch.where(cond, Z, X)
                #X_opt = Z

        return X_opt