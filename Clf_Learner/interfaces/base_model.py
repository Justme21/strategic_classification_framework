from abc import ABC, abstractmethod
from torch import Tensor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .base_best_response import BaseBestResponse
    from .base_dataset import BaseDataset
    from .base_loss import BaseLoss

class BaseModel(ABC):
    @abstractmethod
    def __init__(self, best_response:'BaseBestResponse', loss:'BaseLoss', x_dim:int=None, is_primary:bool=True):
        # These are defined here so that the type-hinting is consistent
        self.best_response: BaseBestResponse
        self.loss: BaseLoss
        self.x_dim: int
        self.deterministic: bool=True # whether it's a deterministic model or a randomised model
        self._is_primary: bool=is_primary

    # Adding 'is_deterministic' because randomised classifiers will have functions deterministic ones won't
    def is_deterministic(self) -> bool:
        """Return whether or not the model is deterministic. Affects primarily behaviour in the loss function"""
        return self.deterministic
    
    def is_primary(self) -> bool:
        """Return whether or not the model is the primary model being run for the experiment.
           Model is primary if it has trainable parameters. Models are assumed to be primary unless specified otherwise"""
        return self._is_primary

    # Adding forward variants to handle the case where the forward function called in the best response (or the loss) might not be the standard forward
    def forward_utility(self, X:Tensor) -> Tensor:
        return self.forward(X)
    
    def forward_loss(self, X:Tensor) -> Tensor:
        return self.forward(X)

    @abstractmethod
    def get_weights(self, include_bias:bool=True) -> Tensor:
        """Return the model weights
        : return: model weights
        """
        pass

    @abstractmethod
    def set_weights(self, weights) -> None:
        """Set the model weights
        : return: None"""
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
    def forward(self, X:Tensor) -> Tensor:
        """ Evaluate the output associated with input X
        : X: Data to be evaluated
        : return: Model Prediction
        """
        pass

    @abstractmethod
    def predict(self, X:Tensor) -> Tensor:
        """ Predict the y for the given X
        : X: Data to predict
        : return: Model Prediction
        """
        # This will often just be a call to forward, but formatted to be output friendly
        pass

    @abstractmethod
    def save_params(self, address:str) -> None:
        """ Save model parameters to a file
        : address (str): address to save model parameters to
        : return: None
        """
        pass

    @abstractmethod
    def load_params(self, address:str) -> None:
        """ Load model parameters from a file
        : address (str) address of the file with the model parameters
        : return: None
        """
        pass