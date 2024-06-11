import os
import torch

from datetime import datetime
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


class DualLogger:
    def __init__(self, log_file, delim=" | "):
        self.delim = delim
        self.log_file = open(log_file, "a", encoding="utf-8")

    def log(self, message):
        message = datetime.now().strftime("%T") + self.delim + message
        self.log_file.write(message + "\n")
        self.log_file.flush()
        print(message)
