from ..datasets import CSVDataset
from ..interfaces import BaseDataset

def get_dataset(filename:str, target_label:int|str, dataset_args={}) -> BaseDataset|None:
    try:
        if filename.endswith('.csv'):
            dataset = CSVDataset(filename, target_label, **dataset_args)
        else:
            print(f"Error: {filename} datatype not supported")
            dataset = None
    except FileNotFoundError:
        print(f"Error: could not find dataset at {filename}")
        dataset = None

    return dataset


