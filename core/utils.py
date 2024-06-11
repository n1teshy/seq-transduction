import os
import torch
import torch.nn.init as init

from datetime import datetime
from pathlib import Path


def get_param_count(model):
    return sum(p.numel() for p in model.parameters())


def get_root():
    return Path(os.path.abspath(__file__ + "/../.."))


def kaiming_init(m):
    if hasattr(m, "weight"):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)


class DualLogger:
    def __init__(self, log_file, delim=" | "):
        self.delim = delim
        self.log_file = open(log_file, "a", encoding="utf-8")

    def log(self, message):
        message = datetime.now().strftime("%T") + self.delim + message
        self.log_file.write(message + "\n")
        self.log_file.flush()
        print(message)
