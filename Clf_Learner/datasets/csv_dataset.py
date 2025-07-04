import pandas as pd
import torch

from .tensor_dataset import TensorDataset
from ..tools.utils import DATA_DIR

class CSVDataset(TensorDataset):
    # TODO: Most likely each dataset will require it's own loader, as opposed to
    # each filetype. Come back to it later.
    def __init__(self, csv_file, target_col):
        assert csv_file.endswith('.csv')
        if csv_file.startswith('/'):
            #Catching the case where you pass a non-local file
            data_address = csv_file
        else:
            data_address = f"{DATA_DIR}/{csv_file}"
        data_df = pd.read_csv(data_address)
        data_tensor = torch.tensor(data_df.values, dtype=torch.float32)

        if target_col < 0:
            target_col = data_tensor.shape[1] + target_col

        y = data_tensor[:,target_col]
        inds = [x for x in range(data_tensor.shape[1]) if x!=target_col]
        X = data_tensor[:,inds]

        super().__init__(X=X, y=y)

    def __len__(self) -> int:
        return super().__len__()
    
    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def size(self):
        return super().size()

    def get_all_vals(self):
        return super().get_all_vals()