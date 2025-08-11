from torch import Tensor
from torch.utils.data import Dataset

from ..interfaces import BaseDataset

class TensorDataset(BaseDataset, Dataset):
    # TODO: Ape the pytorch TensorDataset
    # train_dset = TensorDataset(X, r, y)

    def __init__(self, X:Tensor, y:Tensor):
        assert len(y.shape) == 1, f"Error: downstream models expect target tensor to have a single dimension, current target tensor has shape {y.shape}"
        
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def size(self):
        return self.X.size(), self.y.size()
    
    def get_all_vals(self) -> tuple[Tensor, Tensor]:
        return self.X, self.y