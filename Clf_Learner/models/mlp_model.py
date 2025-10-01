import torch
import torch.nn as nn

from ..interfaces import BaseModel, BaseBestResponse, BaseDataset, BaseLoss
from .model_training_tools import vanilla_training_loop

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