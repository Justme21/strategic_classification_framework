import torch
import torch.nn.functional as F
from torch import Tensor

from ..interfaces import BaseLoss, BaseModel

def _regularization_loss(model:BaseModel):
    W = model.get_weights()
    norm = (torch.norm(W, p=2)**2)
    return norm

def _hinge_loss(model:BaseModel, X:Tensor, y:Tensor):
    #Smart Strategic SVM version of the hinge loss
    acc_term = y*model.forward_loss(model.best_response(X, model))
    return torch.mean(F.relu(1-acc_term))

class NaiveStrategicSVMHingeLoss(BaseLoss):
    # As defined in Generalised Strategic Classification
    # https://github.com/SagiLevanon1/GSC/blob/main/generalization.ipynb

    def __init__(self, gamma=0, **kwargs):
        self.reg_weight = gamma

    def __call__(self, model:BaseModel, X:Tensor, y:Tensor):
        reg = self.reg_weight*_regularization_loss(model)
        loss = _hinge_loss(model, X, y)
        return reg + loss