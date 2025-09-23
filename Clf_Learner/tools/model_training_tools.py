import numpy as np
import time 

from torch.nn import Module
from torch.utils.data import DataLoader
from ..interfaces import BaseModel, BaseDataset

def vanilla_training_loop(model:BaseModel, train_dset:BaseDataset, opt, lr, batch_size, epochs, verbose) -> dict[str, list]:
    """The base training loop that is common to most models"""
    # Put Data into a DataLoader
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=False)
    
    # Initialise Optimiser
    assert isinstance(model, Module), "Error: `vanilla_training_loop` can only be used for torch-based models"
    opt = opt(model.parameters(), lr=lr)

    # Training Loop
    total_time = time.time()
    train_losses = []
    for epoch in range(epochs):
        t1 = time.time()
        batch = 1
        train_losses.append([])
        for X, y in train_loader:
            opt.zero_grad()
            l = model.loss(model, X, y)
            l.backward()
            opt.step()
            train_losses[-1].append(l.item())
            batch += 1

        print(f"End of Epoch: {epoch}: {model.get_weights()}")
        #TODO: Validation evaluation should go here
        
        t2 = time.time()
        model.save_params() # Store intermediate parameter values
        if verbose:
            print(f"------------- epoch {epoch+1} / {epochs} | time: {t2-t1} sec | loss: {np.mean(train_losses[-1])}")
    
    if verbose:
        print(f"Total training time: {time.time()-total_time} seconds")

    return {'train_losses': train_losses}
