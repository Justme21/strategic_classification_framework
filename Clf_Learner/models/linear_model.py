import numpy as np
import time
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from ..interfaces import BaseBestResponse, BaseDataset, BaseLoss, BaseModel

class LinearModel(BaseModel, nn.Module):
    def __init__(self, best_response:BaseBestResponse, loss:BaseLoss, x_dim:int, **kwargs):
        BaseModel.__init__(self, best_response, loss, x_dim)
        nn.Module.__init__(self)
        self.x_dim = x_dim
        self.fc = nn.Linear(x_dim, 1, bias=True)

        self.best_response = best_response
        self.loss = loss

    def get_params(self):
        return self.fc.weight[0]

    def forward(self, X):
        return torch.flatten(self.fc(X))

    def predict(self, X):
        y_hat = self.forward(X)
        y_hat[torch.abs(y_hat) <= 1e-10] = 0
        return torch.sign(y_hat)

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
            batch = 1
            train_losses.append([])
            for X, y in train_loader:
                opt.zero_grad()
                l = self.loss(self, X, y)
                l.backward()
                opt.step()
                train_losses[-1].append(l.item())
                #if verbose:
                #    print(f"batch {batch} / {len(train_loader)} | loss: {l.item()}")
                batch += 1

            #TODO: Validation evaluation should go here
            
            t2 = time.time()
            if verbose:
                print(f"------------- epoch {epoch+1} / {epochs} | time: {t2-t1} sec | loss: {np.mean(train_losses[-1])}")
        
        if verbose:
            print(f"Total training time: {time.time()-total_time} seconds")

        return {'train_losses': train_losses}

    def save_params(self, address):
        torch.save(self.state_dict(), f"{address}/model_params")
 
    def load_params(self, address):
        self.load_state_dict(torch.load(f"{address}/model_params", weights_only=True))