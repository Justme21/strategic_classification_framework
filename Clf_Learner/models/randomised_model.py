import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import DataLoader

from ..interfaces import BaseBestResponse, BaseDataset, BaseLoss, BaseModel

def _get_model(model_name):
    from ..models import MODEL_DICT
    return MODEL_DICT[model_name]

class RandomisedModel(BaseModel, nn.Module):
    def __init__(self, best_response:BaseBestResponse, loss:BaseLoss, x_dim:int, num_comps:int=3, model_name="linear", no_var=True, **kwargs):
        BaseModel.__init__(self, best_response, loss, x_dim)
        nn.Module.__init__(self)
        self.deterministic = False
        self.x_dim = x_dim

        self._model = _get_model(model_name)(best_response, loss, x_dim, is_primary=False, **kwargs)
        
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

    #def get_mean(self) -> Tensor:
    #    return self.mean
    
    #def get_log_std(self) -> Tensor:
    #    return torch.zeros([self.num_comps, 1, self.x_dim])

    def get_weights(self, include_bias=True):
        #weights = torch.cat((self.get_mean(), self.get_log_std()), dim=-1) # TODO: Decide what do to with stds here
        #return weights
        return self.weights

    def set_weights(self):
        # Randomised model will never not be primary, so set_weights will never be called
        pass

    def get_mixture_probs(self) -> Tensor:
        return torch.softmax(self.mix_logits, dim=0)

    def forward(self, X):
        batch_size = X.size(0)
        
        # Sample Mixture Weights per batch element
        mix_indices = F.gumbel_softmax(self.mix_logits.expand(batch_size, -1), tau=self.tau, hard=self.gumbel_hard)

        # Multiply the sampled indices (one-hot) by the means and stds to get the sampled values
        weights = torch.einsum("bk,k...->b...", mix_indices, self.weights) # ellipsis work with einsum. Flexibility in weight tensor shape

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
        for epoch in range(epochs):
            t1 = time.time()
            train_losses.append([])
            for X, y in train_loader:                    
                opt.zero_grad()
                l = self.loss(self, X, y)
                l.backward()
                opt.step()
                train_losses[-1].append(l.item())

            #TODO: Validation evaluation should go here
            
            t2 = time.time()
            if verbose:
                probs = torch.softmax(self.get_mixture_probs(), dim=0)
                print(f"Mix Weights: {probs}\n Weights: {self.weights}")
                print(f"------------- epoch {epoch+1} / {epochs} | time: {t2-t1} sec | loss: {np.mean(train_losses[-1])}")
        
        if verbose:
            print(f"Total training time: {time.time()-total_time} seconds")

        return {'train_losses': train_losses}
    

    def save_params(self, address):
        torch.save(self.state_dict(), f"{address}/model_params")
 
    def load_params(self, address):
        self.load_state_dict(torch.load(f"{address}/model_params", weights_only=True))