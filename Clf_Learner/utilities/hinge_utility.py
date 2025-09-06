import torch
import torch.nn.functional as F
from torch import Tensor

from ..interfaces import BaseUtility, BaseModel

def _hinge_loss(model:BaseModel, X:Tensor, y:Tensor, margin:float):
    #Standard hinge loss
    acc_term = y*model.forward_utility(X)
    return (1/margin)*F.relu(margin-acc_term)

class HingeUtility(BaseUtility):
    # Using the negative Hinge Loss between the prediction and 1 as a utility
    # u = -l(f(x),1)
    def __init__(self, coef=1, margin=1.0, **kwargs):
        assert margin >0, "Error: HingeUtility requires positive margin value"
        self.coef = coef
        self.margin = margin

    def __call__(self, X:Tensor, model:BaseModel):
        # Hinge Utility = - Hinge Loss
        y = torch.ones(X.shape[0])
        return -self.coef*_hinge_loss(model, X, y, margin=self.margin)