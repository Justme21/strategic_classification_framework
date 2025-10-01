import torch 

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

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

def get_standardiser(X: torch.Tensor) -> Standardiser:
    standardiser = Standardiser(X)
    return standardiser