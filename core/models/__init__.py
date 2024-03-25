import torch
import torch.nn as nn

from core.components import ConvLayer


class ConvNet(nn.Module):
    def __init__(self, final_channels, ep=1e-7):
        super().__init__()
        self.epsilon = ep
        self.conv1 = ConvLayer(1, 16, 4)
        self.conv2 = ConvLayer(16, 32, 3)
        self.conv3 = ConvLayer(32, 64, 2)
        self.conv4 = ConvLayer(64, 128, 2, stride=2)
        self.conv5 = ConvLayer(128, 256, 2, pool=2)
        self.conv6 = ConvLayer(256, final_channels, 2, stride=2)

    def forward(self, x):
        # normalize pixel values
        mean = torch.mean(x, dim=(-1, -2), keepdim=True)
        std = torch.std(x, dim=(-1, -2), keepdim=True)
        x = (x - mean) / (std + self.epsilon)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        # mean operation will output (BATCH_SIZE, EMBEDDING_SIZE), does it make sense?
        return x.mean(dim=(-1, -2))
