import json
import numpy as np
import torch

from .interfaces.base_dataset import BaseDataset
from .interfaces.base_model import BaseModel
from .tools.model_building_tools import get_model, get_model_spec
from .tools.dataset_building_tools import get_dataset, lookup_filename
from .tools.model_evaluation_tools import evaluate_model
from .tools.results_tools import fetch_model, store_model, store_results

# Electing to use a default optimiser for all experiments. Choice of optimiser is outside the scope of this application
EXP_OPT = torch.optim.Adam

def _set_seed(seed_val):
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

def _run_experiment(dataset:BaseDataset, model:BaseModel, lr:float, batch_size:int, epochs: int, train:bool, validate:bool, test:bool, verbose:bool):
    results = {}
    # If training new model, pass to train
    if train:
        # TODO: Want to store training losses and validation losses (if/when validation is being run) and output them here 
        results = model.fit(dataset, opt=EXP_OPT, lr=lr, batch_size=batch_size, epochs=epochs, validate=validate, verbose=verbose)

    # If eval, load model and pass to eval
    if test:
        # Test the model, get results
        with torch.no_grad():
            test_results = evaluate_model(model, dataset)
            results['test'] = test_results

        if verbose:
            print(f"Test Results: {json.dumps(results['test'], indent=4)}") 

    return results

def run_experiments(data_files:list, model_spec_names:list, best_response_name:str, cost_name:str, loss_name:str, model_name:str, utility_name:str, comp_args:dict,\
                    seed_val:int, lr:float, batch_size:int, epochs:int, exp_result_dir:str, hist_result_dir:str, implicit:bool, train:bool, validate:bool, test:bool, store:bool, verbose:bool):

    assert train or hist_result_dir, "Error: Either you must train a new model from scratch, or you must specify a historic directory to load model from"
    if model_spec_names is not None:
        model_specs = [get_model_spec(model_spec_name=x) for x in model_spec_names]
    else:
        # So the loop will run just once
        model_specs = [get_model_spec(br_name=best_response_name, cost_name=cost_name, loss_name=loss_name, model_type_name=model_name, utility_name=utility_name)]

    for filename in data_files:
        # Load dataset from spec
        filename = lookup_filename(filename, verbose=verbose)
        dataset = get_dataset(filename, -1, verbose=verbose, dataset_args=comp_args.get('datasets', {}))

        if dataset is None:
            if verbose:
                print(f"Not performing experiment for {filename} as no dataset could be produced for this file")
            continue
        
        data_dim = dataset.size()[0][-1]
        init_args = {"x_dim": data_dim}
        for model_spec in model_specs:
            if verbose:
                print(f"Running Experiment: Dataset {filename}\n Model: {model_spec}")
        
            model = get_model(model_spec=model_spec, init_args=init_args, comp_args=comp_args, result_addr=exp_result_dir, dataset=dataset, implicit=implicit)
            
            if seed_val is not None:
                _set_seed(seed_val)

            if hist_result_dir is not None:
                # Load model state from historical data
                model = fetch_model(model, hist_result_dir, filename, model_spec)

            results = _run_experiment(dataset, model, lr, batch_size, epochs, train, validate, test, verbose)
    
            # Store results and return
            if store:
                store_model(model, exp_result_dir, filename, model_spec)
                store_results(results, exp_result_dir, filename, model_spec, verbose=verbose)

    # TODO: Summarise Results