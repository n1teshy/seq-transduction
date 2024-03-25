import os
import torch.nn as nn

from pathlib import Path


def get_param_count(model):
    return sum(p.numel() for p in model.parameters())


def get_root():
    return Path(os.path.abspath(__file__ + "/../.."))
