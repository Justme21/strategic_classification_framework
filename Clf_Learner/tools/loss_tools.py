import torch

from torch import Tensor
from typing import cast

from ..interfaces import BaseLoss, BaseModel

EPS_HESS = 1e-6

def _get_implicit_grad(Z_star: Tensor, X: Tensor, y:Tensor, loss_fn, model: BaseModel):
    Z_star.detach().requires_grad_()
    theta = [ p for p in model.parameters() if p.requires_grad]
    B, D_z = Z_star.shape

    # Loss Gradients
    loss = loss_fn(model, X, y, Z_star)

    grad_l_z = torch.autograd.grad(loss, Z_star, create_graph=True, retain_graph=True)[0]
    grad_l_theta = torch.autograd.grad(loss, theta, create_graph=True, retain_graph=True, allow_unused=True)
    grad_l_theta_flat = torch.cat([g.reshape(-1) if g is not None else torch.zeros_like(p) for p, g in zip(theta, grad_l_theta)])

    # Utility Gradients
    objective = model.best_response.objective
    
    # Computing the Correction Term
    correction_per_param = [torch.zeros_like(p) for p in theta]

    U = objective(Z_star, X, model)
    grad_u_z = torch.autograd.grad(U.sum(), Z_star, create_graph=True)[0]

    #for i in range(B):
    def _compute_correction(Z_row, X_row, grad_u_z_row, grad_l_z_row):
        #Z_row = Z_star[i].unsqueeze(0) # Want a 1 x x_dim tensor. Could also unsqueeze()
        #X_row = X[i].unsqueeze(0)
        #grad_u_z_row = grad_u_z[i].unsqueeze(0) # 1 x D_z
        #grad_l_z_row = grad_l_z[i].unsqueeze(-1)

        # Hessian
        # We can exploit the fact that the z's are independent of each other, so the Hessian should be block diagonal
        H_row = torch.autograd.functional.hessian(lambda z_arg: objective(z_arg, X_row, model), Z_row)
        H_row = H_row.reshape(D_z, D_z) # 1 x x_dim x 1 x x_dim
        H_row = H_row + EPS_HESS*torch.eye(D_z, dtype=H_row.dtype)

        # Hv = par_L_par_z => v = (H^{-1})par_l_par_z
        v = torch.linalg.solve(H_row, grad_l_z_row) # D_z x 1

        s = torch.dot(v.squeeze(), grad_u_z_row.squeeze()).sum() # scalar
        grad_s_theta = torch.autograd.grad(s, theta, retain_graph=True, allow_unused=True)

        return grad_s_theta
        #for j, g in enumerate(grad_s_theta):
        #    if g is not None:
        #        correction_per_param[j] += g.detach()

    batch_grad = torch.vmap(_compute_correction)(Z_star, X, grad_u_z, grad_l_z)

    import pdb
    pdb.set_trace()
    correction_flat = torch.cat([c.reshape(-1) for c in correction_per_param])

    # Final Gradient: grad_l_theta - correction
    grad_flat = grad_l_theta_flat - correction_flat
        
    return grad_flat

class ImplicitDifferentiationLossWrapper(BaseLoss):

    def __init__(self, loss:BaseLoss):
        self.loss = loss

    def __call__(self, model:BaseModel, X:Tensor, y:Tensor) -> Tensor:
        # First we compute the loss
        Z_star = model.best_response(X, model)
        Z_star = Z_star.clone().detach().requires_grad_()

        loss = self.loss(model, X, y, Z_star)

        # Then we compute the Implicit Gradient
        dl_dtheta_flat = _get_implicit_grad(Z_star, X, y, self.loss, model) # 1D tensor

        # Params can be 2+ dimensional. We just want a vector of all the parameters
        params = [p for p in model.parameters() if p.requires_grad]
        theta_flat = torch.cat([p.reshape(-1) for p in params])

        # Then we detach both from the computation graph and recombine them depending on model params
        # This trick allows backward to pass the correct gradients
        return (loss.detach() + theta_flat*(dl_dtheta_flat.detach())).sum()