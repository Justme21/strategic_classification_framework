from torch import Tensor
from torch.utils.data import Dataset

from .tools.dataset_tools import get_standardiser

from ..interfaces import BaseDataset

class TensorDataset(BaseDataset, Dataset):
    # TODO: Ape the pytorch TensorDataset
    # train_dset = TensorDataset(X, r, y)

    def __init__(self, X:Tensor, y:Tensor, filename="", standardise=True):
        super().__init__(X, y, filename)
        assert len(y.shape) == 1, f"Error: downstream models expect target tensor to have a single dimension, current target tensor has shape {y.shape}"
        
        self._standardiser = None
        if standardise:
            print("Standardising Dataset")
            self._standardiser = get_standardiser(X)
            X = self._standardiser.transform(X)

        self.X = X
        self.y = y

        self.filename = filename

    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def size(self):
        return self.X.size(), self.y.size()
    
    def get_all_vals(self) -> tuple[Tensor, Tensor]:
        return self.X, self.y
    
    def set_standardiser_device(self, device):
        if self._standardiser:
            self._standardiser.to(device)

    def invert_standardisation(self, X:Tensor) -> Tensor:
        if self._standardiser:
            return self._standardiser.inverse_transform(X)
        else:
            return X

    def get_standardiser(self):
        return self._standardiser