from ..datasets import CSVDataset, CSV_DATASET_DICT
from ..interfaces import BaseDataset

import torch

def lookup_filename(filename:str, verbose:bool=False):
    """Allow aliasing for known datasets"""
    if filename in CSV_DATASET_DICT:
        if verbose:
            print(f"{filename} found in lookup: {CSV_DATASET_DICT[filename]}")
        return CSV_DATASET_DICT[filename]
    else:
        return filename

def get_dataset(filename:str, target_label:int|str, verbose:bool=False, dataset_args={}) -> BaseDataset|None:
    try:
        if filename.endswith('.csv'):
            dataset = CSVDataset(filename, target_label, verbose=verbose, **dataset_args)
        else:
            print(f"Error: {filename} datatype not supported")
            dataset = None
    except FileNotFoundError:
        print(f"Error: could not find dataset at {filename}")
        dataset = None

    return dataset



