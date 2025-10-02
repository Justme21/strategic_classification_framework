import torch
import torch.nn as nn

from ..interfaces import BaseModel, BaseBestResponse, BaseDataset, BaseLoss
from .tools.model_training_tools import vanilla_training_loop

class MLPModel(BaseModel, nn.Module):
    """Multi-Layer Perceptron"""
    def __init__(self, best_response:BaseBestResponse, loss:BaseLoss, address:str, x_dim:int, hidden_layers:int=1,\
                  hidden_dim:int=2, activation:str='relu', dropout:float=0.0, is_primary:bool=True, **kwargs):
        BaseModel.__init__(self, best_response, loss, address, x_dim, is_primary)
        nn.Module.__init__(self)

        self.x_dim = x_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Choose activation
        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }
        assert activation in activations, f"Unknown activation '{activation}'"
        self.activation_fn = activations[activation]()

        # Build sequential network
        layers = []
        in_dim = x_dim

        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(self.activation_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        # Final output layer (scalar score for classification boundary)
        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)

        self.best_response = best_response
        self.loss = loss

    def get_boundary_vals(self, range_t: torch.Tensor, res:int=200):
        """
        Compute decision boundary (x,z) points where f(x,z)=0.
        Returns: list of tensors, each of shape (M,2) for one contour segment.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # make a meshgrid over the input space
        x_low, x_high = range_t.min().item(), range_t.max().item()
        z_low, z_high = x_low, x_high   # assume roughly square domain
        xx, yy = np.meshgrid(
            np.linspace(x_low, x_high, res),
            np.linspace(z_low, z_high, res)
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        grid_t = torch.tensor(grid, dtype=torch.float32, device=next(self.parameters()).device)

        # evaluate ICNN on grid
        with torch.no_grad():
            zz = self(grid_t).cpu().numpy().reshape(xx.shape)

        # find 0-level contour
        cs = plt.contour(xx, yy, zz, levels=[0], colors='none')
        bcs = []
        allsegs = getattr(cs, "allsegs", None)
        if allsegs and len(allsegs) > 0 and len(allsegs[0]) > 0:
            for seg in allsegs[0]:
                if seg is not None and seg.size > 0:
                    bcs.append(torch.tensor(seg, dtype=torch.float32))
        else:
            cols = getattr(cs, "collections", None)
            if cols is not None:
                for col in cols:
                    for path in col.get_paths():
                        v = path.vertices
                        if v.shape[0] > 0:
                            bcs.append(torch.tensor(v, dtype=torch.float32))
        return bcs

    def get_weights(self, **kwargs):
        """Flatten all parameters into a single vector."""
        return torch.cat([p.flatten() for p in self.parameters()])

    def set_weights(self, weight_tensor: torch.Tensor) -> None:
        """Set weights for non-primary models."""
        assert not self.is_primary(), "Error: Can only set weights for non-primary models"
        # NOTE: you'd need to reshape and load weights into the state_dict if you want this fully working.
        raise NotImplementedError("Setting weights not yet implemented for MLPModel.")
    
    def forward(self, X):
        """Forward pass (returns raw scores)."""
        return self.network(X).squeeze(-1)

    def predict(self, X):
        y_hat = self.forward(X)
        y_hat[torch.abs(y_hat) <= 1e-10] = 0
        return torch.sign(y_hat)

    def fit(self, train_dset:BaseDataset, opt, lr:float, batch_size:int, epochs:int, validate:bool, verbose:bool=False):
        train_losses_dict = vanilla_training_loop(self, train_dset, opt, lr, batch_size, epochs, validate, verbose)
        return train_losses_dict