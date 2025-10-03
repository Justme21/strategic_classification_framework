import pandas as pd
import torch

from .tensor_dataset import TensorDataset
from ..tools.utils import DATA_DIR

def _get_columns(df, columns):
    if isinstance(columns[0], int):
        return df.iloc[:, columns]
    else:
        return df[columns]
    
def _format_target_col(target_col, num_cols):
    if isinstance(target_col, int):
        if target_col < 0:
            target_col = num_cols + target_col
    return target_col

def _check_cols(df, target_col, strat_cols):
    if isinstance(target_col, int):
        num_cols = df.shape[1]
        assert target_col>=0, f"Error: Expected target_col to have been formatted by now ({target_col})"
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
    def __init__(self, csv_file:str, target_col:int|str, verbose:bool, strat_cols:list[int|str]|None=None):
        assert csv_file.endswith('.csv')
        if csv_file.startswith('/'):
            #Catching the case where you pass a non-local file
            data_address = csv_file
        else:
            data_address = f"{DATA_DIR}/{csv_file}"
        data_df = pd.read_csv(data_address)

        target_col = _format_target_col(target_col, data_df.shape[1])

        _check_cols(data_df, target_col, strat_cols)
        
        # Numbered columns
        data_columns = data_df.columns.to_list()
        if isinstance(target_col, str):
            non_target_cols = [i for i, x in enumerate(data_columns) if x!=target_col] 
        else:
            non_target_cols = [i for i, _ in enumerate(data_columns) if i!=target_col] 
 
        X_df = _get_columns(data_df, non_target_cols)
        y_df = _get_columns(data_df, [target_col])
        
        # Don't want these tensors to be on GPU if that is default device. 
        # Put onto device in training loop instead
        X = torch.tensor(X_df.values, dtype=torch.float32, device="cpu")
        y = torch.tensor(y_df.values, dtype=torch.float32, device="cpu").squeeze()

        # TODO: Fix this hack later
        # Handling 0-1 binary data and map to -1, 1
        if y.min()==0 and y.max()==1:
            y = torch.where(y==0, -1, 1)

        self._strat_cols = None
        if strat_cols:
            if not isinstance(strat_cols[0], int):
                assert all([x in data_columns for x in strat_cols])  
                int_strat_cols = [data_columns.index(x) for x in strat_cols]
            else:
                int_strat_cols = strat_cols

            self._strat_cols = int_strat_cols

        super().__init__(X=X, y=y, filename=csv_file)

    def __len__(self) -> int:
        return super().__len__()
    
    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def size(self):
        return super().size()

    def get_all_vals(self):
        return super().get_all_vals()

    def get_strategic_columns(self):
        return self._strat_cols