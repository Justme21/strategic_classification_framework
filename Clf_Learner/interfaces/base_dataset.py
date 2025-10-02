from abc import abstractmethod
from torch import Tensor
from torch.utils.data import Dataset

class BaseDataset(Dataset):

    @abstractmethod
    def __init__(self, X:Tensor, y:Tensor, source_file:str):
        self.filename: str
        self.strategic_columns: list[int]

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index:int) -> Tensor:
        pass

    @abstractmethod
    def size(self) -> tuple[tuple[int,int], tuple[int,int]]:
        # Returns the (num_rows, num_columns) for the X tensor and the Y tensor
        pass

    @abstractmethod
    def get_all_vals(self) -> tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def set_standardiser_device(self, device:str)->None:
        pass

    @abstractmethod
    def get_standardiser(self):
        pass

    @abstractmethod
    def invert_standardisation(self, X:Tensor) -> Tensor:
        pass

    @abstractmethod
    def get_strategic_columns(self) -> list[int]|None:
        pass
        