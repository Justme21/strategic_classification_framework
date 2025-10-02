import math
import torch
import torch.nn as nn

from ..interfaces import BaseBestResponse, BaseDataset, BaseLoss, BaseModel
from .tools.model_training_tools import vanilla_training_loop

class ParabolicModel(BaseModel, nn.Module):
    """Parabolic Classifier of the form: f(x) = x[1] - a(x[0]-b)^2 + c"""
    def __init__(self, best_response:BaseBestResponse, loss:BaseLoss, address:str, x_dim:int, is_primary:bool=True, **kwargs):
        BaseModel.__init__(self, best_response, loss, address, x_dim, is_primary)
        nn.Module.__init__(self)
        assert x_dim == 2, "Error: At the moment Parabolic classifier can only be applied to 2D-datasets"
        self.x_dim = x_dim

        if self.is_primary():
            # Model is the primary model being trained, so model weights will be optimised by optimiser
            self.angle = nn.Parameter(2*torch.pi*torch.rand(1)) # Angle the classifier is rotated by
            self.coef = nn.Parameter(torch.tensor([1.0])) # Scales the with of the classifier
            self.offset = nn.Parameter(torch.rand([1,2])) # Moves the turning point of the classifier along x- and y- axes
        else:
            # Model is secondary to some other model (e.g. it is a component of a randomised model) so these should be treated as scalars
            self.angle = 2*math.pi*torch.rand(1) # Angle the classifier is rotated by
            self.coef = torch.tensor([1.0]) # Scales the with of the classifier
            self.offset = torch.rand([1,2]) # Moves the turning point of the classifier along x- and y- axes

        self.best_response = best_response
        self.loss = loss

    def get_boundary_vals(self, X):
        """(Optional) For the input 1-D X values, returns the y values that would lie
            on the model decision boundary. This is only used for data visualisation (not included in repo)"""
        s = torch.sin(self.angle)
        c = torch.cos(self.angle)

        A = torch.abs(self.coef)*(s**2)
        B = -2*torch.abs(self.coef)*(X*c - self.offset[:,0])*s - c
        C = -self.offset[:,1] + torch.abs(self.coef)*(X*c - self.offset[:,0])**2 - X*s

        if A > 1e-6:
            discriminant = B**2 - 4*A*C
            re_discr = discriminant[discriminant>=0] # Omit imaginary solutions
            sqrt_discr = torch.sqrt(re_discr)

            X_re = X[discriminant>=0]
            y_plus = (-B[discriminant>=0] + sqrt_discr)/(2*A)
            y_neg = (-B[discriminant>=0] - sqrt_discr)/(2*A)

            X_plus = torch.stack([X_re, y_plus],dim=1)
            X_neg = torch.stack([X_re, y_neg],dim=1)

            X_plus = torch.flip(X_plus, dims=[0]) # Flip the positives to make x dim continuous for plotting
            boundary_coords = torch.cat([X_plus, X_neg], dim=0)

        else:
            # Angle is either 0 or pi, either way y=x^2
            assert self.coef!=0, "Error: Can't produce visualisation; model scaling coefficient is 0"
            y = (1/c)*(torch.abs(self.coef)*((X*c - self.offset[:,0])**2 )- self.offset[:,1])
            boundary_coords = torch.stack([X, y], dim=1)

        return boundary_coords

    def get_weights(self, include_bias=True):
        # TODO: Figure out if offset values should be returned here
        return torch.cat((self.angle, self.coef, self.offset.squeeze()), dim=0)

    def set_weights(self, weight_tensor: torch.Tensor) -> None:
        assert not self.is_primary(), "Error: Can only set model weights for non-primary models"
        # NOTE: This function should only be called if model is not primary.
        # Weight setting here means individual model cannot be called, and should only be called through primary model
        # When weights are set here the resulting weight tensors will have a specific batch dimension that isn't in standard def
        # TODO: Probably more flexible to define "self.weights" list in init and then using that to prescribe the order of definition in both get_ and set_weights
        self.angle = weight_tensor[:, 0]
        self.coef = weight_tensor[:, 1]
        self.offset = weight_tensor[:, 2:]

    def forward(self, X):
        # Step 1: Rotate coordinate space
        s = torch.sin(self.angle)
        c = torch.cos(self.angle)

        # The minus dimensionality here makes this notation compatible with optional batch dimension
        # -1 is the "second" dimension (=column), -2 is the "first" dimension (=row)
        rot_mat = torch.stack([torch.stack([c, -s], dim=-1),
                                torch.stack([s, c], dim=-1)], dim=-2).squeeze()

        # Multiply X by the rotation matrix. We use einsum for this to handle the ambiguous dimensionablity
        X = torch.einsum("...i, ...ij -> ...j", X, torch.transpose(rot_mat, -2, -1) )

        # Step 2: Apply Classifier
        # Doing absolute value of coef because it being negative is the same as the angle being pi greater. So don't want overlap
        res = X[:,1] - torch.abs(self.coef)*(torch.pow(X[:,0]-self.offset[:,0],2)) + self.offset[:,1]
        return torch.flatten(res)

    def predict(self, X):
        y_hat = self.forward(X)
        y_hat[torch.abs(y_hat) <= 1e-10] = 0
        return torch.sign(y_hat)

    def fit(self, train_dset:BaseDataset, opt, lr:float, batch_size:int, epochs:int, validate:bool, verbose:bool=False):
        train_losses_dict = vanilla_training_loop(self, train_dset, opt, lr, batch_size, epochs, validate, verbose)
        return train_losses_dict