from abc import ABC, abstractmethod
from torch import Tensor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .base_dataset import BaseDataset
    from .base_model import BaseModel

class BaseLoss(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, model:'BaseModel', X:Tensor, y:Tensor, Z:Tensor|None=None) -> Tensor:
        """
        Return the losses incurred from evaluating model on batched inputs X compared against batched labels y.
        Z is the pre-computed best response to model at X.
        Inputs:
            model: Basemodel
            X: B x x_dim
            y: B x 1
            Z: B x x_dim [Optional]
        """
        pass