import torch
from torch import Tensor

from ..interfaces import BaseLoss, BaseModel

def _kl_divergence(mu:Tensor, log_sigma:Tensor):
    sigma = log_sigma.exp()
    return 0.5 * torch.sum(sigma**2 + mu**2 - 1 - 2 * log_sigma)

def _regularization_loss(model:BaseModel):
    mean = model.get_mean()
    log_std = model.get_log_std()
    return _kl_divergence(mean, log_std)

def _neg_linear_loss(model:BaseModel, X:Tensor, y:Tensor):
    return -y*model.forward(X) 

class LinearPlusKLLoss(BaseLoss):
    # As specified in Performative Prediction paper
    # https://github.com/jcperdomo/performative-prediction/blob/main/experiments/neurips2020/logistic_regression.py

    def __init__(self, gamma=0, **kwargs):
        self.reg_weight = gamma

    def __call__(self, model:BaseModel, X:Tensor, y:Tensor):
        assert not model.is_deterministic(), "Error: This loss uses KL divergence regularisation and so is only for non-deterministic models"
        reg = (self.reg_weight/2)*_regularization_loss(model)

        X_br = model.best_response(X, model)
        lin_loss = _neg_linear_loss(model,X_br,y)
        loss = torch.mean(lin_loss)
        return reg + loss