# Code obtained from https://github.com/developer0hye/PyTorch-Darknet53/blob/master/model.py

import torch
from torch import nn
from torchsummary import summary


def conv_batch(in_num, out_num, kernel_size=3, padding=1, output_padding=1, stride=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_num, out_num, kernel_size=kernel_size,
                           stride=stride, padding=padding, output_padding=output_padding, bias=True),
        nn.BatchNorm2d(out_num),
        nn.ReLU())


# Residual block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()

        increased_channels = int(in_channels//2)

        self.layer1 = conv_batch(
            in_channels, increased_channels, kernel_size=1, padding=0, output_padding=0)
        self.layer2 = conv_batch(
            increased_channels, in_channels, output_padding=0)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out


class Darknet53(nn.Module):
    def __init__(self, block, num_classes):
        super(Darknet53, self).__init__()

        self.num_classes = num_classes

        self.fc = nn.Linear(2, 1024)

        self.conv1 = conv_batch(1024, 512, stride=2)
        self.conv2 = conv_batch(512, 256, stride=2)
        self.residual_block1 = self.make_layer(
            block, in_channels=256, num_blocks=4)
        self.conv3 = conv_batch(256, 128, stride=2)
        self.residual_block2 = self.make_layer(
            block, in_channels=128, num_blocks=8)
        self.conv4 = conv_batch(128, 64, stride=2)
        self.residual_block3 = self.make_layer(
            block, in_channels=64, num_blocks=8)
        self.conv5 = conv_batch(64, 42, stride=2)
        self.residual_block4 = self.make_layer(
            block, in_channels=32, num_blocks=2)
        self.conv6 = conv_batch(32, 3, stride=1, output_padding=0)
        self.residual_block5 = self.make_layer(
            block, in_channels=3, num_blocks=1)

    def forward(self, x):

        out = self.fc(x)

        out = out.view(out.size(0), 1024, 1, 1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.residual_block1(out)
        out = self.conv3(out)
        out = self.residual_block2(out)
        out = self.conv4(out)
        out = self.residual_block3(out)
        out = self.conv5(out)
        out = self.residual_block4(out)
        out = self.conv6(out)
        out = self.residual_block5(out)

        return out

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)

# model=Darknet53(DarkResidualBlock,2)
# summary(model, input_size=(3, 1024, 1024), batch_size=8, device='cpu')
