import os
import torch.nn as nn

# Directory defaults where data gets read from and sent to
HOME = os.getcwd()
RESULTS_DIR = f"{HOME}/results"
DATA_DIR = f"{HOME}/data"

# Easier to have a single ReLU rather than randomly reinitialising one all over the place
RELU = nn.ReLU()

# The dimension that is expanded out in model outputs to accomodate repeated samples from model
SAMPLE_DIM = 0