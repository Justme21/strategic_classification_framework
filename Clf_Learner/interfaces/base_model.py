from abc import ABC, abstractmethod
from torch import Tensor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .base_best_response import BaseBestResponse
    from .base_dataset import BaseDataset
    from .base_loss import BaseLoss

class BaseModel(ABC):
    @abstractmethod
    def __init__(self, best_response:'BaseBestResponse', loss:'BaseLoss', x_dim=None):
        # These are defined here so that the type-hinting is consistent
        self.best_response: BaseBestResponse
        self.loss: BaseLoss
        self.x_dim: int

    @abstractmethod
    def get_params(self) -> Tensor:
        """Return the model parameters
        : return: model weights
        """
        pass

    @abstractmethod
    def fit(self, dataset:'BaseDataset', opt, lr:float, batch_size:int, epochs:int, verbose:bool) -> dict:
        """ Learn to predict the true y values associated with the given Xs
        : X: Data to learn from
        : y: True values
        : return: dict of containing training metrics
        """
        pass

    @abstractmethod
    def forward(self, X) -> Tensor:
        """ Evaluate the output associated with input X
        : X: Data to be evaluated
        : return: Model Prediction
        """
        pass

    @abstractmethod
    def predict(self, X) -> Tensor:
        """ Predict the y for the given X
        : X: Data to predict
        : return: Model Prediction
        """
        # This will often just be a call to forward, but formatted to be output friendly
        pass

    @abstractmethod
    def save_params(self, address: str) -> None:
        """ Save model parameters to a file
        : address (str): address to save model parameters to
        : return: None
        """
        pass

    @abstractmethod
    def load_params(self, address: str) -> None:
        """ Load model parameters from a file
        : address (str) address of the file with the model parameters
        : return: None
        """
        pass