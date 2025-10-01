import torch
import torch.nn.functional as F
from torch import Tensor

from ..interfaces import BaseLoss, BaseModel

def _regularization_loss(model:BaseModel):
    W = model.get_weights()
    norm = (torch.norm(W, p=2)**2)
    return norm

def _hinge_loss(model:BaseModel, Z:Tensor, y:Tensor,):
    #Smart Strategic SVM version of the hinge loss
    acc_term = y*model.forward_loss(Z)
    return torch.mean(F.relu(1-acc_term))

class NaiveStrategicSVMHingeLoss(BaseLoss):
    # As defined in Generalised Strategic Classification
    # https://github.com/SagiLevanon1/GSC/blob/main/generalization.ipynb

    def __init__(self, gamma=0, **kwargs):
        self.reg_weight = gamma

    def __call__(self, model:BaseModel, X:Tensor, y:Tensor, Z:Tensor|None=None):
        if Z is None:
            Z = model.best_response(X, model)
        loss = _hinge_loss(model, Z, y)

        reg = self.reg_weight*_regularization_loss(model)
        return reg + loss