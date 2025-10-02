import torch
import torch.nn as nn

from ..interfaces import BaseBestResponse, BaseDataset, BaseLoss, BaseModel
from .tools.model_training_tools import vanilla_training_loop

class LinearModel(BaseModel, nn.Module):
    """Standard linear model"""
    def __init__(self, best_response:BaseBestResponse, loss:BaseLoss, address:str, x_dim:int, is_primary:bool=True, **kwargs):
        BaseModel.__init__(self, best_response, loss, address, x_dim, is_primary)
        nn.Module.__init__(self)
        self.x_dim = x_dim
        if self.is_primary():
            self.weight = nn.Parameter(torch.randn(1, x_dim))
            self.bias = nn.Parameter(torch.randn(1))
        else:
            self.weight = torch.randn(1, x_dim)
            self.bias = torch.randn(1)

        self.best_response = best_response
        self.loss = loss

    def get_boundary_vals(self, X):
        """(Optional) For the input 1-D X values, returns the y values that would lie
            on the model decision boundary. This is only used for data visualisation (not included in repo)"""
        if not self.is_primary():
            # Weights still include batch term
            W = self.weight.squeeze(0)[0]
            b = self.bias.squeeze(0)
        else:
            W = self.weight[0]
            b = self.bias

        y = (-W[0]*X-b)*(1.0/W[1]) 
        boundary_coords = torch.stack([X,y], dim=1)
        return boundary_coords

    def get_weights(self, include_bias=True) -> torch.Tensor:
        weights = self.weight
        if include_bias:
            bias = self.bias.unsqueeze(0)
            weights = torch.cat((weights, bias), dim=1)
        
        return weights
    
    def set_weights(self, weight_tensor:torch.Tensor) -> None:
        assert not self.is_primary(), "Error: Can only set model weights for non-primary models"
        # NOTE: This function should only be called if model is not primary.
        # Weight setting here means individual model cannot be called, and should only be called through primary model
        # When weights are set here the resulting weight tensors will have a specific batch dimension that isn't in standard def

        self.weight = weight_tensor[:,:,:self.x_dim] # <batch_size> x 1 x x_dim
        self.bias = weight_tensor[:,:,self.x_dim:].squeeze(1) # squeeze here to match conventional 1-D tensor definition <batch_size> x 1

    def forward(self, X):
        # Flatten to make output uni-dimensional to match y
        # unsqueeze to ensure model output has the same dimensionality as non-deterministic model

        if self.is_primary():
            out = torch.einsum("bi, oi->bo", X, self.weight) + self.bias
        else:
            out = torch.einsum("bi, boi->bo", X, self.weight) + self.bias

        return out.squeeze()

    def predict(self, X):
        y_hat = self.forward(X)
        y_hat[torch.abs(y_hat) <= 1e-10] = 0 # This is a dangerous stopgap, we later map negatives to 0.
        return torch.sign(y_hat)

    def fit(self, train_dset:BaseDataset, opt, lr:float, batch_size:int=128, epochs:int=100, validate:bool=False, verbose:bool=False) -> dict:
        train_losses_dict = vanilla_training_loop(self, train_dset, opt, lr, batch_size, epochs, validate, verbose)
        return train_losses_dict