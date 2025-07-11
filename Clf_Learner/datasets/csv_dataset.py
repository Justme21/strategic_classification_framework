import pandas as pd
import torch

from .tensor_dataset import TensorDataset
from ..tools.utils import DATA_DIR

def _get_columns(df, columns):
    if isinstance(columns[0], int):
        return df.iloc[:, columns]
    else:
        return df[columns]
    
def _check_cols(df, target_col, strat_cols):
    if isinstance(target_col, int):
        num_cols = df.shape[1]
        if target_col < 0:
            target_col = df.shape[1] + target_col
        assert target_col<df.shape[1], f"Error: Column value {target_col} invalid for dataset with {num_cols} columns"
        if strat_cols is not None:
            assert all([isinstance(x, int) and x<num_cols for x in strat_cols])
    else:
        cols = df.columns
        assert target_col in cols, f"Error: Column value {target_col} not found in dataset with columns {cols}"
        assert strat_cols is None or all([x in cols for x in strat_cols]), f"Error: If target column is a string ({target_col}) then data columns must also be strings and must be from dataset columns({cols})"
    assert strat_cols is None or target_col not in strat_cols, f"Error: target column ({target_col}) can't also be in data columns ({strat_cols})"

class CSVDataset(TensorDataset):
    # TODO: Most likely each dataset will require it's own loader, as opposed to
    # each filetype. Come back to it later.
    def __init__(self, csv_file:str, target_col:int|str, strat_cols:list[int|str]|None=None):
        assert csv_file.endswith('.csv')
        if csv_file.startswith('/'):
            #Catching the case where you pass a non-local file
            data_address = csv_file
        else:
            data_address = f"{DATA_DIR}/{csv_file}"
        data_df = pd.read_csv(data_address)

        _check_cols(data_df, target_col, strat_cols)

        if strat_cols is None:
            if isinstance(target_col, str):
                strat_cols = [x for x in data_df.columns if x!=target_col]
            else:
                strat_cols = [x for x in range(data_df.shape[1]) if x!=target_col]
        
        X_df = _get_columns(data_df, strat_cols)
        y_df = _get_columns(data_df, [target_col])
        
        X = torch.tensor(X_df.values, dtype=torch.float32)
        y = torch.tensor(y_df.values, dtype=torch.float32).squeeze()

        super().__init__(X=X, y=y)

    def __len__(self) -> int:
        return super().__len__()
    
    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def size(self):
        return super().size()

    def get_all_vals(self):
        return super().get_all_vals()