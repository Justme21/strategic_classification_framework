import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import DataLoader

from ..interfaces import BaseBestResponse, BaseDataset, BaseLoss, BaseModel

def _get_model(model_name):
    from . import MODEL_DICT
    return MODEL_DICT[model_name]

class IteratedRandomisedModel(BaseModel, nn.Module):
    def __init__(self, best_response:BaseBestResponse, loss:BaseLoss, address:str, x_dim:int, num_comps:int=3, model_name="linear", **kwargs):
        BaseModel.__init__(self, best_response, loss, address, x_dim)
        nn.Module.__init__(self)
        self.deterministic = False
        self.x_dim = x_dim

        self._model = _get_model(model_name)(best_response, loss, address, x_dim, is_primary=False, **kwargs)
        
        self.num_comps = num_comps
        self.tau = 1.0
        self.gumbel_hard = True

        self._comp_index = None
        self.mix_logits = torch.zeros(num_comps) # Mixing distribution

        weights = self._model.get_weights() # weights are out_dim (1) x num_weights vector
        self.weights = nn.Parameter(weights.repeat([num_comps] + [1 for _ in range(len(weights.shape))])) # Arguments are how many times each dimension should be repeated

        self.best_response = best_response
        self.loss = loss

    def get_num_components(self):

        if self._comp_index is None:
            return self.num_comps
        else:
            # comp_index is the index of the component currently being trained.
            # At this point in training there are only that many components
            return self._comp_index + 1
    
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
        mix_indices = F.gumbel_softmax(self.mix_logits[:self.num_comps].expand(batch_size, -1), tau=self.tau, hard=self.gumbel_hard)
        mix_indices = torch.cat([mix_indices, torch.zeros(batch_size, self.num_comps-mix_indices.shape[1])], dim=1) # Pad the weights with 0's for the iterated case

        # Multiply the sampled indices (one-hot) by the means and stds to get the sampled values
        weights = torch.einsum("bk,k...->b...", mix_indices, self.weights) # ellipsis work with einsum. Flexibility in weight tensor shape

        self._model.set_weights(weights)
        out = self._model.forward(X)
        return out
    
    def forward_loss(self, X: Tensor) -> Tensor:
        batch_size = X.size(0)
        weights = self.weights[self._comp_index].expand(batch_size, -1)

        self._model.set_weights(weights)
        out = self._model.forward(X)
        return out 

    def forward_utility(self, X:Tensor, comp_id=None) -> Tensor:
        if comp_id is None:
            out = self.forward(X)
        else:
            batch_size = X.size(0)
            weights = self.weights[comp_id].expand(batch_size, -1)

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
    
    def fit(self, train_dset:BaseDataset, opt, lr, batch_size=128, epochs=100, verbose=False):
        # Put Data into a DataLoader
        train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=False)
        
        # Initialise Optimiser
        opt = opt(self.parameters(), lr=lr)

        # Training Loop
        total_time = time.time()
        train_losses = []

        for i in range(self.num_comps):
            if verbose:
                print(f"Starting training comp: {i}")

            self._comp_index = i # This will update the comp_index parameter that forward_loss uses
            self.mix_logits = torch.zeros([i+1]) # Update the distribution so it is uniform over the current number of components

            for epoch in range(epochs):
                t1 = time.time()
                train_losses.append([])
                for X, y in train_loader:                    
                    opt.zero_grad()
                    l = self.loss(self, X, y)
                    l.backward(inputs=[self.weights])
                    opt.step()
                    train_losses[-1].append(l.item())

                #TODO: Validation evaluation should go here
            
                t2 = time.time()
                if verbose:
                    probs = torch.softmax(self.get_mixture_probs(), dim=0)
                    print(f"Mix Weights: {probs}\n Weights: {self.weights}")
                    print(f"------------- epoch {epoch+1} / {epochs} | time: {t2-t1} sec | loss: {np.mean(train_losses[-1])}")
            
            if verbose:
                print(f"Comp: {i} training complete\n")

        if verbose:
            print(f"Total training time: {time.time()-total_time} seconds")

        return {'train_losses': train_losses}