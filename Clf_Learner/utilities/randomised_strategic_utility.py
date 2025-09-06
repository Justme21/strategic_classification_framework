import torch 
from torch import Tensor
from typing import cast

from ..interfaces import BaseUtility, BaseModel

SAFETY_THRESHOLD = 5e-3

class AltStrategicUtility(torch.autograd.Function):
    @staticmethod
    def forward(ctx, model_out, mix_weight) -> Tensor:
        ctx.save_for_backward(model_out, mix_weight)

        pred = torch.where(model_out>0, torch.ones_like(model_out), -torch.ones_like(model_out))
        pred = torch.einsum("wo, w -> o", pred, mix_weight)
        return pred

    @staticmethod
    def backward(ctx, grad_out):
        (model_out, mix_weight) = ctx.saved_tensors

        mask = (model_out<=SAFETY_THRESHOLD).float()
        weighted_mask = mask*mix_weight.unsqueeze(1)

        grad_input = grad_out * weighted_mask
        return grad_input, None

class RandomisedStrategicUtility(BaseUtility):

    def __init__(self, coef=1, **kwargs):
        self.coef = coef

    def __call__(self, X:Tensor, model:'BaseModel') -> Tensor:
        if model.get_num_components() == 1:
            model_out = model.forward_utility(X).unsqueeze(0)        
        else:
            model_out = model.forward_utility(X, 0).unsqueeze(0)
            for i in range(1, model.get_num_components()):
                model_out = torch.cat([model_out, model.forward_utility(X, i).unsqueeze(0)], dim=0)
            
        mix_weight = model.get_mixture_probs().detach()     
        pred = cast(torch.Tensor, AltStrategicUtility.apply(model_out, mix_weight)) # Casting just to keep with the type-hinting

        return self.coef*pred