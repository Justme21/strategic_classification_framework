from ..datasets import CSVDataset
from ..interfaces import BaseDataset

def get_dataset(filename:str, target_label) -> BaseDataset|None:
    try:
        if filename.endswith('.csv'):
            dataset = CSVDataset(filename, target_label)
        else:
            print(f"Error: {filename} datatype not supported")
            dataset = None
    except FileNotFoundError:
        print(f"Error: could not find dataset at {filename}")
        dataset = None

    return dataset


