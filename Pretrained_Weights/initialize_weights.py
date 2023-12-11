import torch
from torch import nn


def initialize(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.BatchNorm1d, nn.Linear)):
            nn.init.normal_(m.weight.data)
