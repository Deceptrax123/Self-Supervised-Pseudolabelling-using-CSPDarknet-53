import torch
from torch import nn


def initialize(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.BatchNorm1d, nn.Linear)):
            if m.bias.data is not None:
                m.bias.data.zero_()
            else:
                nn.init.kaiming_normal_(
                    m.weight.data, mode='fan_in', nonlinearity='leaky_relu')
