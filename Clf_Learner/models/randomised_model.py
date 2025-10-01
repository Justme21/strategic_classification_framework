import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import DataLoader

from ..interfaces import BaseBestResponse, BaseDataset, BaseLoss, BaseModel
from ..tools.model_evaluation_tools import validate_model

def _get_model(model_name):
    from ..models import MODEL_DICT
    return MODEL_DICT[model_name]

class RandomisedModel(BaseModel, nn.Module):
    """Generic Randomised Model Classifier. Can take any other deterministic classifier as a base model and can use it to
       learn a randomised variant of that classifier (where a randomised classifier samples model parameters from a learnt parameter distribution)"""
    def __init__(self, best_response:BaseBestResponse, loss:BaseLoss, address:str, x_dim:int, num_comps:int=3, model_name="linear", no_var=True, **kwargs):
        BaseModel.__init__(self, best_response, loss, address, x_dim)
        nn.Module.__init__(self)
        self.deterministic = False
        self.x_dim = x_dim

        self._model = _get_model(model_name)(best_response, loss, address, x_dim, is_primary=False, **kwargs)
        
        self.num_comps = num_comps
        self.tau = 1.0
        self.gumbel_hard = True

        self.mix_logits = nn.Parameter(torch.zeros(num_comps)) # Mixing distribution

        weights = self._model.get_weights() # weights are out_dim (1) x num_weights vector
        self.weights = nn.Parameter(weights.repeat([num_comps] + [1 for _ in range(len(weights.shape))])) # Arguments are how many times each dimension should be repeated

        self._no_var = no_var
        if not no_var:
            self.log_sigma = nn.Parameter(torch.full(self.means.size(), -1.0))

        self.best_response = best_response
        self.loss = loss

    def get_num_components(self):
        return self.num_comps

    def get_weights(self, include_bias=True):
        return self.weights

    def set_weights(self):
        # Randomised model will never not be primary, so set_weights will never be called
        assert False, "Error: 'set_weights' should never be called for the Randomised Classifier"

    def get_mixture_probs(self) -> Tensor:
        return torch.softmax(self.mix_logits, dim=0)

    def get_boundary_vals(self, X):
        """(Optional) For the input 1-D X values, returns the y values that would lie
            on the model decision boundary. This is only used for data visualisation (not included in repo)"""
        boundary_coords = []
        for i in range(self.num_comps):
            weights = self.weights[i].unsqueeze(0)
            self._model.set_weights(weights)

            model_boundary = self._model.get_boundary_vals(X)
            model_boundary = model_boundary

            boundary_coords.append(model_boundary)

        return boundary_coords

    def forward(self, X):
        batch_size = X.size(0)
        
        # Sample Mixture Weights per batch element
        mix_indices = F.gumbel_softmax(self.mix_logits.expand(batch_size, -1), tau=self.tau, hard=self.gumbel_hard)

        # Multiply the sampled indices (one-hot) by the means and stds to get the sampled values
        weights = torch.einsum("bk,k...->b...", mix_indices, self.weights) # ellipsis work with einsum. Flexibility in weight tensor shape

        self._model.set_weights(weights)
        out = self._model.forward(X)
        return out

    def forward_utility(self, X:Tensor, comp_id=None) -> Tensor:
        if comp_id is None:
            out = self.forward(X)
        else:
            batch_size = X.size(0)
            weight = self.weights[comp_id]
            weights = weight.repeat([batch_size] + [1 for _ in range(len(weight.shape))])

            self._model.set_weights(weights)
            out = self._model.forward(X)
        return out

    def predict(self, X):
        # TODO: Probably want "num_samples" as an argument here so that you can specify how many weight samples you want to use to
        # evaluate the expected value over the model
        y_hat = self.forward(X)
        y_hat[torch.abs(y_hat) <= 1e-10] = 0
        y_hat = torch.sign(y_hat) # Predictions

        return torch.sign(y_hat) # Translate average prediction back into prediction
    
    def fit(self, train_dset:BaseDataset, opt, lr:float, batch_size:int, epochs:int, validate:bool, verbose=False):
        # Put Data into a DataLoader
        train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=False)
        
        # Initialise Optimiser
        opt = opt(self.parameters(), lr=lr)

        # Training Loop
        total_time = time.time()
        train_losses = []
        if validate:
            valid_clean_accuracies = []
            valid_strat_accuracies = []

        for epoch in range(epochs):
            t1 = time.time()
            train_losses.append([])
            for X, y in train_loader:                    
                opt.zero_grad()
                # TODO: Specifiy number of repeats here. So loop over each batch a few times to get a better gradient
                l = self.loss(self, X, y)
                l.backward(inputs=[self.weights, self.mix_logits])
                opt.step()
                train_losses[-1].append(l.item())

            #TODO: Validation evaluation should go here
            if validate:
                with torch.no_grad():
                    clean_accuracy, strat_accuracy = validate_model(self, train_dset)
                    valid_clean_accuracies.append(clean_accuracy)
                    valid_strat_accuracies.append(strat_accuracy)

            t2 = time.time()
            self.save_params()
            if verbose:
                probs = self.get_mixture_probs()
                print(f"Mix Weights: {probs}\n Weights: {self.weights}")
                print(f"------------- epoch {epoch+1} / {epochs} | time: {t2-t1} sec | loss: {np.mean(train_losses[-1])}")
                if validate:
                    print(f"------------- validation: clean acc: {clean_accuracy} | strat acc: {strat_accuracy}")
        
        if verbose:
            print(f"Total training time: {time.time()-total_time} seconds")

        out = {'train': {'train-loss': train_losses}}
        if validate:
            out['validation'] = {'valid-clean-acc': valid_clean_accuracies, 'valid-strat-acc': valid_strat_accuracies}

        return out