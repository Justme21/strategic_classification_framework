import torch

_DEVICE = "cpu"

def find_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device # default to running on cpu


def get_device() -> str:
    return _DEVICE

def set_device(device):
    global _DEVICE
    _DEVICE = device