import torch 
from torch import Tensor
from typing import cast

from ..interfaces import BaseUtility, BaseModel

SAFETY_THRESHOLD = 5e-1

class AltStrategicUtility(torch.autograd.Function):
    @staticmethod
    def forward(ctx, model_out) -> Tensor:
        ctx.save_for_backward(model_out)

        pred = torch.where(model_out>0, torch.ones_like(model_out), -torch.ones_like(model_out))
        return pred

    @staticmethod
    def backward(ctx, grad_out):
        (model_out, ) = ctx.saved_tensors

        mask = (model_out<=SAFETY_THRESHOLD).float()

        grad_input = grad_out * mask
        return grad_input

class StrategicUtility(BaseUtility):
    def __init__(self, coef=1, **kwargs):
        self.coef = coef

    def __call__(self, X:Tensor, model:'BaseModel') -> Tensor:
        model_out = model.forward_utility(X)
        pred = cast(torch.Tensor, AltStrategicUtility.apply(model_out)) # Casting just to keep with the type-hinting

        return self.coef*pred



