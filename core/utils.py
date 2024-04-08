import os
import torch

from pathlib import Path
from core.config import device


def get_param_count(model):
    return sum(p.numel() for p in model.parameters())


def get_root():
    return Path(os.path.abspath(__file__ + "/../.."))


def kaiming_init(model):
    def init(m):
        if hasattr(m, "weight") and m.weight.dim() > 1:
            torch.nn.init.kaiming_uniform(m.weight.data)

    model.apply(init)
