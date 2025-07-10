import torch

from ..interfaces import BaseLoss, BaseModel
from ..tools.utils import RELU

def _regularization_loss(model):
    W = model.get_params()
    norm = (torch.norm(W, p=2)**2)
    return norm

def _neg_linear_loss(model:BaseModel, X, y):
    return ((y.T)*model.forward(model.best_response(X, model))).squeeze() 

def _log_loss(model:BaseModel, X, y):
    exp = model.forward(model.best_response(X, model))
    return torch.log(1 + torch.exp(exp))

class LinearPlusLogisticLoss(BaseLoss):
    # As specified in Performative Prediction paper
    # https://github.com/jcperdomo/performative-prediction/blob/main/experiments/neurips2020/logistic_regression.py

    def __init__(self, gamma=0, **kwargs):
        self.reg_weight = gamma

    def __call__(self, model:BaseModel, X, y):
        reg = (self.reg_weight/2)*_regularization_loss(model)
        lin_loss = _neg_linear_loss(model,X,y)
        log_loss = _log_loss(model, X, y)
        loss = torch.mean(lin_loss + log_loss)
        return reg + loss