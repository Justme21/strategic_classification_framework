import torch
import torch.nn.functional as F

from torch import Tensor

from ..interfaces import BaseLoss, BaseModel

class StrategicCrossEntropyLoss(BaseLoss):

    def __init__(self, **kwargs):
        pass

    def __call__(self, model:BaseModel, X:Tensor, y:Tensor, Z:Tensor|None=None):
        if Z is None:
            Z = model.best_response(X, model)
        
        y = torch.where(y<0,0, 1).float()

        model_out = model.forward_loss(Z)
        loss = F.binary_cross_entropy_with_logits(model_out, y)

        return loss.mean()