# Code obtained from https://github.com/developer0hye/PyTorch-Darknet53/blob/master/model.py

import torch
from torch import nn
from torchsummary import summary
from Model.model_segments.darknet import Darknet53
from Model.model_segments.darknet import DarkResidualBlock


def conv_batch(in_num, out_num, kernel_size=3, padding=1, output_padding=1, stride=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_num, out_num, kernel_size=kernel_size,
                           stride=stride, padding=padding, output_padding=output_padding, bias=True),
        nn.BatchNorm2d(out_num),
        nn.ReLU())


# Residual block
class DecoderResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DecoderResidualBlock, self).__init__()

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


class Decoder(nn.Module):
    def __init__(self, block, num_classes):
        super(Decoder, self).__init__()

        self.num_classes = num_classes

        # Encoder Module
        self.encoder = Darknet53(DarkResidualBlock, 2)

        # Linear and upsample modules
        self.fc = nn.Linear(2, 1024)
        self.up = nn.Upsample(scale_factor=8)

        # Decoder modules
        self.residual_block1 = self.make_layer(
            block, in_channels=1024, num_blocks=4)
        self.conv1 = conv_batch(in_num=1024, out_num=512, stride=2)

        self.residual_block2 = self.make_layer(
            block, in_channels=512, num_blocks=8)
        self.conv2 = conv_batch(in_num=512, out_num=256, stride=2)

        self.residual_block3 = self.make_layer(
            block, in_channels=256, num_blocks=8)
        self.conv3 = conv_batch(in_num=256, out_num=128, stride=2)

        self.residual_block4 = self.make_layer(
            block, in_channels=128, num_blocks=2)
        self.conv4 = conv_batch(in_num=128, out_num=64, stride=2)

        self.residual_block5 = self.make_layer(
            block, in_channels=64, num_blocks=1)

        self.conv5 = conv_batch(in_num=64, out_num=32, stride=2)
        self.conv6 = conv_batch(in_num=32, out_num=3,
                                stride=1, output_padding=0)

    def forward(self, x):
        # encoder output with buffer outputs for skip connnections
        x_0, x1, x2, x3, x4, x5 = self.encoder.forward(x)

        # Linear with reshape and upsampling
        out = self.fc(x_0)
        out = out.view(out.size(0), 1024, 1, 1)
        out = self.up(out)

        # Decoder Convolutions and encoder-decoder skip connections
        out = self.residual_block1(out)
        outcat1 = torch.add(out, x5)
        out = self.conv1(outcat1)
        out = self.residual_block2(out)
        outcat2 = torch.add(out, x4)
        out = self.conv2(outcat2)
        out = self.residual_block3(out)
        outcat3 = torch.add(out, x3)
        out = self.conv3(outcat3)
        out = self.residual_block4(out)
        outcat4 = torch.add(out, x2)
        out = self.conv4(outcat4)
        out = self.residual_block5(out)
        outcat5 = torch.add(out, x1)
        out = self.conv5(outcat5)
        out = self.conv6(out)

        return out

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)
