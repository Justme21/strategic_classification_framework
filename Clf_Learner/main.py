import argparse
import datetime
import json
import os
import shutil
import torch

from .experiment_setup import run_experiments
from .tools.device_tools import find_device, set_device
from .tools.utils import RESULTS_DIR

def _create_arg_parser():
    from .best_reponses import BR_DICT
    from .costs import COST_DICT
    from .datasets import CSV_DATASET_DICT
    from .losses import LOSS_DICT
    from .models import MODEL_DICT
    from .utilities import UTILITY_DICT

    parser = argparse.ArgumentParser()
    parser.add_argument("--arg_file", help="(Optional) Specify the json file that args can be read from", default=None)
    parser.add_argument("--dirname", help="The directory where the experiment results will be written to", default=f"{datetime.datetime.now():%Y-%m-%d_%H_%M}")
    parser.add_argument("--datasets", help=f"Comma separated list of datasets experiments are to be run on. Specify filename in data directory, or else select from : {','.join(CSV_DATASET_DICT.keys()) if CSV_DATASET_DICT.keys() else '<No Values Defined>'}", nargs='+', required=True)
    parser.add_argument("--specs", help="Comma separated list of model specs experiments are to be run on", nargs='*', default=None)
    parser.add_argument("--best_response", help=f"(Required if --arg_file or --specs not specified) Best Response method to use from : {','.join(BR_DICT.keys()) if BR_DICT.keys() else '<No Values Defined>'}", default=None)
    parser.add_argument("--cost",  help=f" (Required if required by specified model) Cost method to use from : {', '.join(COST_DICT.keys()) if COST_DICT.keys() else '<No Values Defined>'}", default=None)
    parser.add_argument("--loss",  help=f"(Required if --arg_file or --specs not specified) Loss method to use from : {', '.join(LOSS_DICT.keys()) if LOSS_DICT.keys() else '<No Values Defined>'}", default=None)
    parser.add_argument("--model",  help=f"(Required if --arg_file or --specs not specified) Model to use from : {', '.join(MODEL_DICT.keys()) if MODEL_DICT.keys() else '<No Values Defined>'}", default=None)
    parser.add_argument("--utility", help=f"(Required if required by specified model) Utility method to use from : {', '.join(UTILITY_DICT.keys()) if UTILITY_DICT.keys() else '<No Values Defined>'}", default=None)
    parser.add_argument("--implicit", help="Use Implicit Gradient as part of training", action='store_true')
    parser.add_argument("--lr", help="Specify the learning rate to be used during training", type=float, default=1e-2)
    parser.add_argument("--batch", help="Specify the batch size to be used during training", type=int, default=128)
    parser.add_argument("--epochs", help="Specify the number of batches to be performed during training", type=int, default=100)
    parser.add_argument("--seed", help="Specify a random seed for reproducibility", type=int, default=None)
    parser.add_argument("--hist_result_dirname", help=f"Optional: Directory in {RESULTS_DIR} containing previous saved results and model to evaluate", default=None)
    parser.add_argument("--train", help="Include to perform training", action='store_true')
    parser.add_argument("--validate", help="Include to perform validation", action='store_true')
    parser.add_argument("--test", help="Include to perform model evaluation", action='store_true')
    parser.add_argument("--store", help="Include to store results from run", action='store_true')
    parser.add_argument("--verbose", help="Verbose mode", action='store_true')
    parser.add_argument("--args", help="(Optional) Dict formatted string passing arguments to specified component objects. Key values: [best response, cost, dataset, loss, model, utility]", type=json.loads, default={})
    parser.add_argument("--gpu", help="Include to run experiments on GPU (or MPS) device (if available)", action='store_true')

    return parser.parse_args()

def _make_results_dir(dirname, to_train):
    dir_addr = f"{RESULTS_DIR}/{dirname}"
    if os.path.exists(dir_addr) and to_train:
        # If the path exists and you are planning on training, then overwrite the old results
        shutil.rmtree(dir_addr)
    os.makedirs(dir_addr)

def _save_args(args) -> None:
    with open(f'{RESULTS_DIR}/{args.dirname}/commandline_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def _load_args(dirname: str) -> argparse.Namespace:
    args = argparse.Namespace()
    with open(dirname, "r") as f:
        args.__dict__ = json.load(f)
    return args

if __name__ == "__main__":
    # Ingest and parse arguments
    args = _create_arg_parser()

    if args.store:
        # Create new directory to store results
        _make_results_dir(args.dirname, args.train)

    if args.arg_file is not None:
        dirname = args.dirname
        args = _load_args(dirname)
        args.dirname = dirname

    if args.store:
        _save_args(args)

    if args.gpu:
        # Not sure if this is the best place to have this. 
        # Also, setting default device might cause some slight performance cost compared to
        # loading everything on to device manually. 
        device = find_device()
        torch.set_default_device(device)
        set_device(device)
        if device=="cuda":
            torch.set_float32_matmul_precision('high')

    run_experiments(data_files=args.datasets, model_spec_names=args.specs, best_response_name=args.best_response, cost_name=args.cost, loss_name=args.loss,\
                     model_name=args.model, utility_name=args.utility, comp_args=args.args, seed_val=args.seed, lr=args.lr, batch_size=args.batch, epochs=args.epochs,\
                        exp_result_dir=args.dirname, hist_result_dir=args.hist_result_dirname, implicit=args.implicit, train=args.train, validate=args.validate, test=args.test, store=args.store, verbose=args.verbose)