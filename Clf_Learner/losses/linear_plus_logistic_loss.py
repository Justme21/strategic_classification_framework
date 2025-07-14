import torch

from ..interfaces import BaseLoss, BaseModel
from ..tools.utils import RELU

def _regularization_loss(model):
    W = model.get_weights()
    norm = (torch.norm(W, p=2)**2)
    return norm

def _neg_linear_loss(model:BaseModel, X_br, y):
    return -y*model.forward(X_br) 

def _log_loss(model:BaseModel, X_br):
    exp = model.forward(X_br)
    return torch.log(1 + torch.exp(exp))

class LinearPlusLogisticLoss(BaseLoss):
    # As specified in Performative Prediction paper
    # https://github.com/jcperdomo/performative-prediction/blob/main/experiments/neurips2020/logistic_regression.py

    def __init__(self, gamma=0, **kwargs):
        self.reg_weight = gamma

    def __call__(self, model:BaseModel, X, y):
        reg = (self.reg_weight/2)*_regularization_loss(model)

        X_br = model.best_response(X, model)
        lin_loss = _neg_linear_loss(model,X_br,y)
        log_loss = _log_loss(model, X_br)
        loss = torch.mean(lin_loss + log_loss)
        return reg + loss