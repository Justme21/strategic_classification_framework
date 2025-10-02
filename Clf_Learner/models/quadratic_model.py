import torch
import torch.nn as nn
from ..interfaces import BaseBestResponse, BaseDataset, BaseLoss, BaseModel
from .tools.model_training_tools import vanilla_training_loop

class QuadraticModel(BaseModel, nn.Module):
    """
    General d-dimensional quadratic classifier:
        f(x) = x^T Q x + w^T x + b
    Predicts: sign(f(x))

    Supports any input dimension d>=1.
    get_boundary_vals is implemented only for d=2 (plotting).
    """
    def __init__(self, best_response:BaseBestResponse, loss:BaseLoss,
                 address:str, x_dim:int, is_primary:bool=True, **kwargs):
        BaseModel.__init__(self, best_response, loss, address, x_dim, is_primary)
        nn.Module.__init__(self)
        self.x_dim = x_dim

        if self.is_primary():
            # Symmetric quadratic matrix Q
            Q = torch.randn(x_dim, x_dim)
            self.Q = nn.Parameter((Q + Q.T)/2)       # make symmetric
            self.w = nn.Parameter(torch.randn(x_dim))
            self.b = nn.Parameter(torch.zeros(1))
        else:
            Q = torch.randn(x_dim, x_dim)
            self.Q = (Q + Q.T)/2
            self.w = torch.randn(x_dim)
            self.b = torch.zeros(1)

        self.best_response = best_response
        self.loss = loss

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        """
        Evaluate f(x) for batch of points X with shape (batch_size, x_dim).
        """
        # quadratic term: x^T Q x = sum_i sum_j x_i Q_ij x_j
        quad = torch.einsum('bi,ij,bj->b', X, self.Q, X)
        lin = X @ self.w
        return quad + lin + self.b

    def predict(self, X:torch.Tensor) -> torch.Tensor:
        y_hat = self.forward(X)
        y_hat[torch.abs(y_hat) <= 1e-10] = 0
        return torch.sign(y_hat)

    def get_boundary_vals(self, X:torch.Tensor) -> torch.Tensor:
        """
        Return 2D decision boundary points (x,y) for plotting when x_dim==2.
        Solves: a*y^2 + b*y + c = 0 for each x in X.
        """
        assert self.x_dim == 2, "Boundary visualization is only supported for 2D."
        X = X.view(-1)

        # Extract coefficients
        Q = self.Q
        w = self.w
        b = self.b

        a = Q[1,1]
        b_coef = 2*Q[0,1]*X + w[1]
        c_term = Q[0,0]*(X**2) + w[0]*X + b

        discriminant = b_coef**2 - 4*a*c_term
        mask = discriminant >= 0
        if torch.sum(mask)==0:
            return torch.empty((0,2))

        sqrt_discr = torch.sqrt(discriminant[mask])
        X_re = X[mask]
        y_plus = (-b_coef[mask] + sqrt_discr)/(2*a)
        y_minus = (-b_coef[mask] - sqrt_discr)/(2*a)

        X_plus = torch.stack([X_re,y_plus], dim=1)
        X_minus = torch.stack([X_re,y_minus], dim=1)
        X_plus = torch.flip(X_plus, dims=[0])
        return torch.cat([X_plus, X_minus], dim=0)

    def get_weights(self, include_bias=True) -> torch.Tensor:
        """
        Flatten parameters: [vec(Q), w, b(optional)].
        vec(Q) is row-major flattened Q.
        """
        flatQ = self.Q.flatten()
        if include_bias:
            return torch.cat([flatQ, self.w.flatten(), self.b.view(-1)], dim=0)
        else:
            return torch.cat([flatQ, self.w.flatten()], dim=0)

    def set_weights(self, weight_tensor:torch.Tensor) -> None:
        """
        Set parameters for non-primary models.
        Expected layout: [vec(Q), w, b].
        """
        assert not self.is_primary(), "Error: Can only set model weights for non-primary models"
        d = self.x_dim
        num_Q = d*d
        flatQ = weight_tensor[:, :num_Q]
        self.Q = flatQ.view(-1,d,d)
        self.w = weight_tensor[:, num_Q:num_Q+d]
        self.b = weight_tensor[:, num_Q+d]

    def fit(self, train_dset:BaseDataset, opt, lr:float, batch_size:int, epochs:int, validate:bool, verbose:bool=False) -> dict:
        train_losses_dict = vanilla_training_loop(self, train_dset, opt, lr, batch_size, epochs, validate, verbose)
        return train_losses_dict