import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel, stride=1, pool=None, drop=None
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        if pool is not None:
            assert isinstance(pool, int), "pool dimension must be an integer"
            self.max_pool = nn.MaxPool2d((pool, pool))
        if drop is not None:
            assert isinstance(drop, float), "drop value must be a float"
            self.drop = nn.Dropout2d(drop)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        if hasattr(self, "drop"):
            out = self.drop(out)
        return self.max_pool(out) if hasattr(self, "pool") else out
