import numpy as np
import time 
import torch

from torch.nn import Module
from torch.utils.data import DataLoader
from ..interfaces import BaseModel, BaseDataset
from ..tools.model_evaluation_tools import validate_model
from ..tools.device_tools import get_device

def vanilla_training_loop(model:BaseModel, train_dset:BaseDataset, opt, lr:float, batch_size:int, epochs:int, validate:bool, verbose:bool) -> dict[str, dict[str, list]]:
    """The base training loop that is common to most models"""
    # Put Data into a DataLoader
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    
    # Put Data Standardiser onto the same device as everyone else
    DEVICE = get_device()

    train_dset.set_standardiser_device(DEVICE)

    # Initialise Optimiser
    assert isinstance(model, Module), "Error: `vanilla_training_loop` can only be used for torch-based models"
    opt = opt(model.parameters(), lr=lr)

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
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            l = model.loss(model, X, y)
            l.backward()
            opt.step()
            train_losses[-1].append(l.item())

        #TODO: Validation evaluation should go here
        if validate:
            with torch.no_grad():
                clean_accuracy, strat_accuracy = validate_model(model, train_dset)
                valid_clean_accuracies.append(clean_accuracy)
                valid_strat_accuracies.append(strat_accuracy)

        t2 = time.time()
        model.save_params() # Store intermediate parameter values
        if verbose:
            print(f"End of Epoch: {epoch+1}: {model.get_weights()}")
            print(f"------------- epoch {epoch+1} / {epochs} | time: {t2-t1} sec | loss: {np.mean(train_losses[-1])}")
            if validate:
                print(f"------------- validation: clean acc: {clean_accuracy} | strat acc: {strat_accuracy}")
    
    if verbose:
        print(f"Total training time: {time.time()-total_time} seconds")

    out = {'train': {'train-loss': train_losses}}
    if validate:
        out['validation'] = {'valid-clean-acc': valid_clean_accuracies, 'valid-strat-acc': valid_strat_accuracies}

    return out