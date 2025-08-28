import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import DataLoader

from ..interfaces import BaseBestResponse, BaseDataset, BaseLoss, BaseModel

class RandomisedLinearDeltaModel(BaseModel, nn.Module):
    def __init__(self, best_response:BaseBestResponse, loss:BaseLoss, x_dim:int, num_comps:int=3, **kwargs):
        BaseModel.__init__(self, best_response, loss, x_dim)
        nn.Module.__init__(self)
        self.deterministic = False
        self.x_dim = x_dim
        
        self.num_comps = num_comps
        self.tau = 1.0
        self.gumbel_hard = True

        self.mix_logits = nn.Parameter(torch.zeros(num_comps)) # Mixing distribution
        self.mean = nn.Parameter(torch.randn(num_comps, 1, x_dim))
        self.bias = nn.Parameter(torch.zeros(num_comps, 1))

        self.best_response = best_response
        self.loss = loss

    def get_mean(self) -> Tensor:
        return self.mean
    
    def get_log_std(self) -> Tensor:
        return torch.zeros([self.num_comps, 1, self.x_dim])

    def get_weights(self, include_bias=True):
        weights = torch.cat((self.get_mean(), self.get_log_std()), dim=-1) # TODO: Decide what do to with stds here
        return weights

    def get_mixture_probs(self) -> Tensor:
        return torch.softmax(self.mix_logits, dim=0)

    def forward(self, X):
        batch_size = X.size(0)
        
        # Sample Mixture Weights per batch element
        mix_indices = F.gumbel_softmax(self.mix_logits.expand(batch_size, -1), tau=self.tau, hard=self.gumbel_hard)

        # Multiply the sampled indices (one-hot) by the means and stds to get the sampled values
        means = torch.einsum("bk,koi->boi", mix_indices, self.mean)
        biases = torch.einsum("bk,ko->bo", mix_indices, self.bias)

        # Reparametrisation Trick
        weights = means

        X = X.unsqueeze(1)
        out = torch.einsum("boi,boi->bo", X, weights) + biases
        out = out.squeeze()
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
                print(f"Mix Weights: {probs}\n Means: {self.mean}\nBiases: {self.bias}")
                print(f"------------- epoch {epoch+1} / {epochs} | time: {t2-t1} sec | loss: {np.mean(train_losses[-1])}")
        
        if verbose:
            print(f"Total training time: {time.time()-total_time} seconds")

        return {'train_losses': train_losses}
    

    def save_params(self, address):
        torch.save(self.state_dict(), f"{address}/model_params")
 
    def load_params(self, address):
        self.load_state_dict(torch.load(f"{address}/model_params", weights_only=True))