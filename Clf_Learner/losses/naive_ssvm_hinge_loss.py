import torch

from ..interfaces import BaseLoss, BaseModel
from ..tools.utils import RELU

def _regularization_loss(model):
    W = model.get_params()
    norm = (torch.norm(W, p=2)**2)
    return norm

def _hinge_loss(model:BaseModel, X, y):
    #Smart Strategic SVM version of the hinge loss
    W = model.get_params()

    acc_term = y*model.forward(model.best_response.get_best_response(X, model))
    return torch.mean(RELU(1-acc_term))

class NaiveStrategicSVMHingeLoss(BaseLoss):

    def __init__(self, reg_weight=0, **kwargs):
        self.reg_weight = reg_weight

    def __call__(self, model:BaseModel, X, y):
        reg = self.reg_weight*_regularization_loss(model)
        loss = _hinge_loss(model, X, y)
        return reg + loss