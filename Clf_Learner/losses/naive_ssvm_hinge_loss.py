import torch
from torch import Tensor

from ..interfaces import BaseLoss, BaseModel
from ..tools.utils import RELU

def _regularization_loss(model:BaseModel):
    W = model.get_weights()
    norm = (torch.norm(W, p=2)**2)
    return norm

def _hinge_loss(model:BaseModel, X:Tensor, y:Tensor):
    #Smart Strategic SVM version of the hinge loss
    acc_term = y*model.forward(model.best_response(X, model))
    return torch.mean(RELU(1-acc_term))

class NaiveStrategicSVMHingeLoss(BaseLoss):
    # As defined in Generalised Strategic Classification
    # https://github.com/SagiLevanon1/GSC/blob/main/generalization.ipynb

    def __init__(self, reg_weight=0, **kwargs):
        self.reg_weight = reg_weight

    def __call__(self, model:BaseModel, X:Tensor, y:Tensor):
        reg = self.reg_weight*_regularization_loss(model)
        loss = _hinge_loss(model, X, y)
        return reg + loss