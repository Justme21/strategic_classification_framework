import os
import torch.nn as nn

HOME = os.getcwd()
RESULTS_DIR = f"{HOME}/results"
DATA_DIR = f"{HOME}/data"

RELU = nn.ReLU()