import json
from pathlib import Path

from ..interfaces import BaseModel

def _format_dataset_filename(dataset_filename:str):
    return dataset_filename.rsplit('.',1)[0] #Strip out the file-suffix

def _format_model_spec(model_spec:dict):
    keys = sorted(list(model_spec.keys()))
    return "_".join([f"{x[0]}_{model_spec[x]}" for x in keys])

def _get_results_address(result_addr:str, dataset_filename:str, model_spec:dict):
    from .utils import RESULTS_DIR
    # Importing here speficially to accomodate when you might be calling module not from main (e.g. in Jupyter notebook)
    # To allow for RESULTS_DIR value to be prescriptively set
    dataset_dirname = _format_dataset_filename(dataset_filename)
    model_spec_dirname = _format_model_spec(model_spec)

    return f"{RESULTS_DIR}/{result_addr}/{dataset_dirname}/{model_spec_dirname}"

def get_results_directory(result_addr:str, dataset_filename:str, model_spec:dict):
    dir_addr = _get_results_address(result_addr, dataset_filename, model_spec)
    Path(dir_addr).mkdir(parents=True, exist_ok=True) # Make sure the specified directory tree exists

    return dir_addr

def store_model(model:BaseModel, results_dir_addr:str, dataset_filename:str, model_spec:dict):
    dir_addr = get_results_directory(results_dir_addr, dataset_filename, model_spec)
    model.save_params(dir_addr)

def fetch_model(model:BaseModel, results_dirname:str, dataset_filename:str, model_spec:dict):
    dir_addr = get_results_directory(results_dirname, dataset_filename, model_spec)
    model.load_params(dir_addr)
    return model

def store_results(results:dict, results_dirname:str, dataset_filename:str, model_spec:dict, verbose:bool):
    dir_addr = get_results_directory(results_dirname, dataset_filename, model_spec)

    results_dir = f"{dir_addr}/results.json" 
    with open(results_dir, 'w') as f:
        json.dump(results, f, indent=4)
    
    if verbose:
        print(f"Results successsfully saved to: {results_dir}\n")