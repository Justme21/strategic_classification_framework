from ..datasets import CSVDataset, CSV_DATASET_DICT
from ..interfaces import BaseDataset

import torch

def lookup_filename(filename:str, verbose:bool=False):
    """Allow aliasing for known datasets"""
    if filename in CSV_DATASET_DICT:
        if verbose:
            print(f"{filename} found in lookup: {CSV_DATASET_DICT[filename]}")
        return CSV_DATASET_DICT[filename]
    else:
        return filename

def get_dataset(filename:str, target_label:int|str, verbose:bool=False, dataset_args={}) -> BaseDataset|None:
    try:
        if filename.endswith('.csv'):
            dataset = CSVDataset(filename, target_label, verbose=verbose, **dataset_args)
        else:
            print(f"Error: {filename} datatype not supported")
            dataset = None
    except FileNotFoundError:
        print(f"Error: could not find dataset at {filename}")
        dataset = None

    return dataset

class Standardiser:
    def __init__(self, X:torch.Tensor):
        self.mean, self.std = self._fit(X)
    
    def _fit(self, X: torch.Tensor):
        """Compute mean and std from dataset X."""
        mean = X.mean(dim=0, keepdim=True)
        std = X.std(dim=0, keepdim=True)
        # avoid div-by-zero for constant columns
        std[std == 0] = 1.0
        return mean, std
    
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Apply standardization: (X - mean) / std"""
        return (X - self.mean) / self.std
    
    def inverse_transform(self, X_scaled: torch.Tensor) -> torch.Tensor:
        """Revert standardization back to original space."""
        return X_scaled * self.std + self.mean

def get_standardiser(X: torch.Tensor) -> Standardiser:
    standardiser = Standardiser(X)
    return standardiser

