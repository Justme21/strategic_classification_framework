import torch
import torch.nn.functional as F
from torch import Tensor

from ..interfaces import BaseUtility, BaseModel

def _hinge_loss(model:BaseModel, X:Tensor, y:Tensor, margin:float):
    #Smart Strategic SVM version of the hinge loss
    if model.get_num_components() == 1:
        model_out = model.forward_utility(X).unsqueeze(0)        
    else:
        model_out = model.forward_utility(X, 0).unsqueeze(0)
        for i in range(1, model.get_num_components()):
            model_out = torch.cat([model_out, model.forward_utility(X, i).unsqueeze(0)], dim=0)
    acc_term = y*model_out
    loss_term = (1/margin)*F.relu(margin-acc_term)

    mix_weight = model.get_mixture_probs().detach()
    expected_loss = torch.matmul(mix_weight.unsqueeze(0), loss_term).squeeze(0)

    return expected_loss

class ExpectedHingeUtility(BaseUtility):
    # Using the negative Expected Hinge Loss between the prediction and 1 as a utility
    # u = - E_{f}[l(f(x),1)]
    def __init__(self, coef=1.0, margin=1.0, **kwargs):
        assert margin >0, "Error: ExpectedHingeUtility requires positive margin value"
        self.coef = coef
        self.margin = margin

    def __call__(self, X:Tensor, model:BaseModel):
        # Hinge Utility = - Hinge Loss
        y = torch.ones(X.shape[0])
        return -self.coef*_hinge_loss(model, X, y, margin=self.margin)