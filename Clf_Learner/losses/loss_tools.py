import torch

from torch import Tensor
from typing import cast

from ..interfaces import BaseLoss, BaseModel

EPS_HESS = 1e-6

def _get_implicit_grad_vanilla(Z_star: Tensor, X: Tensor, y:Tensor, loss_fn, model: BaseModel):
    """Compute the implicit gradient by explicitly working out the Hessian for each Z"""
    # This is SLOW
    Z_star.detach().requires_grad_()
    theta = [ p for p in model.parameters() if p.requires_grad]
    B, D_z = Z_star.shape

    # Loss Gradients
    loss = loss_fn(model, X, y, Z_star)

    grad_l_z = torch.autograd.grad(loss, Z_star, create_graph=True, retain_graph=True)[0]
    grad_l_theta = torch.autograd.grad(loss, theta, create_graph=True, retain_graph=True, allow_unused=True)
    grad_l_theta_flat = torch.cat([g.reshape(-1) if g is not None else torch.zeros_like(p) for p, g in zip(theta, grad_l_theta)])

    # Computing the Correction Term
    correction_per_param = [torch.zeros_like(p) for p in theta]

    # Utility Gradients
    objective = model.best_response.objective
    U = objective(Z_star, X, model)
    grad_u_z = torch.autograd.grad(U.sum(), Z_star, create_graph=True)[0]

    for i in range(B):
        Z_row = Z_star[i].unsqueeze(0) # Want a 1 x x_dim tensor. Could also unsqueeze()
        X_row = X[i].unsqueeze(0)
        grad_u_z_row = grad_u_z[i].unsqueeze(0) # 1 x D_z
        grad_l_z_row = grad_l_z[i].unsqueeze(-1)

        # Hessian
        # We can exploit the fact that the z's are independent of each other, so the Hessian should be block diagonal
        H_row = torch.autograd.functional.hessian(lambda z_arg: objective(z_arg, X_row, model), Z_row)
        H_row = H_row.reshape(D_z, D_z) # 1 x x_dim x 1 x x_dim
        H_row = H_row + EPS_HESS*torch.eye(D_z, dtype=H_row.dtype)

        # Hv = par_L_par_z => v = (H^{-1})par_l_par_z
        v = torch.linalg.solve(H_row, grad_l_z_row) # D_z x 1

        s = torch.dot(v.squeeze(), grad_u_z_row.squeeze()).sum() # scalar
        grad_s_theta = torch.autograd.grad(s, theta, retain_graph=True, allow_unused=True)

        for j, g in enumerate(grad_s_theta):
            if g is not None:
                correction_per_param[j] += g.detach()

    correction_flat = torch.cat([c.reshape(-1) for c in correction_per_param])

    # Final Gradient: grad_l_theta - correction
    grad_flat = grad_l_theta_flat - correction_flat
        
    return grad_flat

################################################################
################################################################

def _do_hvp(Z_row, objective_fn, X_row, model, v):
    def hvp_fn(z_input):
        return objective_fn(z_input, X_row, model).sum()
    
    v_reshaped = v.reshape(Z_row.shape) # (1, D_z)
    _, hvp_out = torch.autograd.functional.hvp(hvp_fn, Z_row, v_reshaped)
    return hvp_out.reshape(-1) # (D_z, )

def _solve_Hv_eq_b(Z_row, objective_fn, X_row, model, b, tol=1e-5, max_iter=10):
    """
    Solve H v = b using Conjugate Gradient where H is the Hessian of objective_fn at Z_row.
    Z_row: (1, Dz)
    b:     (1, Dz) or (Dz,)  -> we will flatten to (Dz,)
    Returns v: (Dz,)  (1-D)
    """

    v = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    r_dot_r_old = torch.dot(r.view(-1), r.view(-1))

    if torch.sqrt(r_dot_r_old)<tol:
        # b is functionally 0. Hv=0 => b=0
        return torch.zeros_like(b)

    for _ in range(max_iter):
        Hp = _do_hvp(Z_row, objective_fn, X_row, model, p)
        Hp = Hp + EPS_HESS*p # Damping for stability
        denom = torch.dot(p.view(-1), Hp.view(-1))
        if torch.abs(denom)<tol:
            # Avoid divide by zero
            break
        alpha = r_dot_r_old/denom
        v = v + alpha*p
        r = r - alpha*Hp
        r_dot_r_new = torch.dot(r.view(-1), r.view(-1))
        if torch.sqrt(r_dot_r_new)<tol:
            break
        p = r + (r_dot_r_new/r_dot_r_old)*p
        r_dot_r_old = r_dot_r_new

    return v

def _get_implicit_grad_hvp(Z_star: Tensor, X: Tensor, y:Tensor, loss_fn, model: BaseModel):
    """Compute the implicit gradient using Hessian Vector Product and Conjugate Gradient"""
    
    Z_star.detach().requires_grad_()
    theta = [ p for p in model.parameters() if p.requires_grad]
    B, D_z = Z_star.shape

    # Loss Gradients
    loss = loss_fn(model, X, y, Z_star)

    grad_l_z = torch.autograd.grad(loss, Z_star, create_graph=True, retain_graph=True)[0]
    grad_l_theta = torch.autograd.grad(loss, theta, create_graph=True, retain_graph=True, allow_unused=True)
    grad_l_theta_flat = torch.cat([g.reshape(-1) if g is not None else torch.zeros_like(p) for p, g in zip(theta, grad_l_theta)])

    # Computing the Correction Term
    correction_per_param = [torch.zeros_like(p) for p in theta]

    # Utility Gradients
    objective = model.best_response.objective
    U = objective(Z_star, X, model)
    grad_u_z = torch.autograd.grad(U.mean(), Z_star, create_graph=True)[0]

    for i in range(B):
        Z_row = Z_star[i].unsqueeze(0) # Want a 1 x x_dim tensor. Could also unsqueeze()
        X_row = X[i].unsqueeze(0)
        grad_u_z_row = grad_u_z[i].reshape(-1) # (D_z, )
        grad_l_z_row = grad_l_z[i].reshape(-1) # (D_z, )
    
        # Hessian
        # Use Hessian Vector Product (hvp) here so as not to have to compute the hessian explicitly
        # v = grad_l_z_row*H^{-1}
        v = _solve_Hv_eq_b(Z_row, objective, X_row, model, b=grad_l_z_row) # D_z x 1
        
        # Compute (\partial-l/\partial-z_row)*H^{-1}*(\partial^{2}U/\partial-z\partial-theta)
        v_detached = v.detach() # Detaching v so gradient isn't computed through when computing grad_s_theta
        v_times_grad_u_z_row_theta = torch.autograd.grad(grad_u_z_row, theta, grad_outputs=v_detached, retain_graph=True, allow_unused=True)
        
        for j, g in enumerate(v_times_grad_u_z_row_theta):
            if g is not None:
                correction_per_param[j] += g.detach()

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
        #dl_dtheta_flat = _get_implicit_grad_vanilla(Z_star, X, y, self.loss, model) # 1D tensor
        dl_dtheta_flat = _get_implicit_grad_hvp(Z_star, X, y, self.loss, model) # 1D tensor


        # Params can be 2+ dimensional. We just want a vector of all the parameters
        params = [p for p in model.parameters() if p.requires_grad] # TODO: Replace parameters with get_weights() here
        theta_flat = torch.cat([p.reshape(-1) for p in params])

        # Then we detach both from the computation graph and recombine them depending on model params
        # This trick allows backward to pass the correct gradients
        return (loss.detach() + theta_flat*(dl_dtheta_flat.detach())).sum()