import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import DataLoader

from ..interfaces import BaseBestResponse, BaseDataset, BaseLoss, BaseModel

class BayesianLinearModel(nn.Module):
    def __init__(self, in_dim:int, out_dim:int, num_comps:int): #bias=True)
        super().__init__()
        self.x_dim = in_dim
        self.out_dim = out_dim
        self.num_comps = num_comps

        self.mix_weights = nn.Parameter(torch.tensor([1/num_comps for _ in range(num_comps)])) # Mixing distribution initially uniform
        self.mean = nn.Parameter(torch.randn(num_comps, out_dim, in_dim))
        self.log_sigma = nn.Parameter(torch.full((num_comps, out_dim, in_dim), -1.0))

        #TODO: Bias term? To match nn.Linear functionality which allows for optional bias term

    def get_mean(self) -> Tensor:
        return self.mean
    
    def get_log_stds(self) -> Tensor:
        return self.log_sigma

    def get_weights(self, include_bias=True):
        weights = torch.cat((self.get_mean(), self.get_log_stds()), dim=-1) # TODO: Decide what do to with stds here
        #TODO: If including bias will need to include this later

        return weights

    def forward(self, X:Tensor):
        batch_size = X.size(0)
        
        # Sample Mixture Weights per batch element
        mix_indices = F.gumbel_softmax(self.mix_weights.expand(batch_size, -1), tau=1.0, hard=True)
        mix_indices = mix_indices.view(batch_size, self.num_comps, 1, 1)

        # Multiply the sampled indices (one-hot) by the means and stds to get the sampled values
        means = torch.sum(mix_indices*self.mean.unsqueeze(0), dim=1)
        stds = torch.sum(mix_indices*self.log_sigma.exp().unsqueeze(0), dim=1)

        # Reparametrisation Trick
        eps = torch.randn(batch_size, self.out_dim, self.x_dim)
        weights = means + stds*eps

        X = X.unsqueeze(1)
        out = torch.bmm(X, weights.transpose(1,2)).squeeze()
        return out


class RandomisedLinearModel(BaseModel, nn.Module):
    def __init__(self, best_response:BaseBestResponse, loss:BaseLoss, x_dim:int, num_comps:int=3, **kwargs):
        BaseModel.__init__(self, best_response, loss, x_dim)
        nn.Module.__init__(self)
        self.x_dim = x_dim
        self.model = BayesianLinearModel(in_dim=x_dim, out_dim=1, num_comps=num_comps)

        self.best_response = best_response
        self.loss = loss

    def get_mean(self) -> Tensor:
        return self.model.get_mean()
    
    def get_log_std(self) -> Tensor:
        return self.model.get_log_stds()

    def get_weights(self, include_bias=True):
        weights = self.model.get_weights(include_bias)
        return weights

    def forward(self, X):
        model_out = self.model(X)
        return model_out

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
                probs = torch.softmax(self.model.mix_weights, dim=0)
                print(f"Mix Weights: {probs}\n Means: {self.model.mean}\n Log Sigma: {self.model.log_sigma}")
                print(f"------------- epoch {epoch+1} / {epochs} | time: {t2-t1} sec | loss: {np.mean(train_losses[-1])}")
        
        if verbose:
            print(f"Total training time: {time.time()-total_time} seconds")

        return {'train_losses': train_losses}
    

    def save_params(self, address):
        torch.save(self.state_dict(), f"{address}/model_params")
 
    def load_params(self, address):
        self.load_state_dict(torch.load(f"{address}/model_params", weights_only=True))