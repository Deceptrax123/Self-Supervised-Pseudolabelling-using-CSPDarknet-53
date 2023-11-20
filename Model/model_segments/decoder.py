import torch
from torch import nn
from torch.nn import BatchNorm2d, ReLU, Linear, ConvTranspose2d,AdaptiveAvgPool2d
from torchsummary import summary


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # equalizing linear module
        self.linear = Linear(in_features=2, out_features=1024)
        # Decoder
        self.conv1 = ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(
            3, 3), stride=2, padding=1, output_padding=1)
        self.bn1 = BatchNorm2d(512)
        self.relu1 = ReLU()

        self.conv2 = ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(
            3, 3), stride=2, output_padding=1, padding=1)
        self.bn2 = BatchNorm2d(256)
        self.relu2 = ReLU()

        self.conv3 = ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(
            3, 3), stride=2, output_padding=1, padding=1)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = ReLU()

        self.conv4 = ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(
            3, 3), stride=2, output_padding=1, padding=1)
        self.bn4 = BatchNorm2d(64)
        self.relu4 = ReLU()

        self.conv5 = ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(
            3, 3), stride=2, output_padding=1, padding=1)
        self.bn5 = BatchNorm2d(32)
        self.relu5 = ReLU()

        self.conv6 = ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(
            3, 3), stride=2, output_padding=1, padding=1)
        self.bn6 = BatchNorm2d(16)
        self.relu6 = ReLU()

        self.conv7 = ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(
            3, 3), stride=2, output_padding=1, padding=1)
        self.bn7 = BatchNorm2d(8)
        self.relu7 = ReLU()

        self.conv8 = ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=(
            3, 3), stride=2, padding=1, output_padding=1)
        
        self.pool=AdaptiveAvgPool2d(output_size=(224,224))

        self.apply(self._init_weights)

    def _init_weights(self,module):
        if isinstance(module,(ConvTranspose2d,BatchNorm2d,Linear)):
            if module.bias.data is not None:
                module.bias.data.zero_()
            else:
                nn.init.kaiming_normal_(module.weight.data,mode='fan_in',nonlinearity='relu')
        
    def forward(self, x):
        # Linear and reshape
        x = self.linear(x)
        x = x.view(x.size(0), 1024, 1, 1)

        # Decoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)

        x = self.conv8(x)

        x=self.pool(x)

        return x


#model = Decoder()
#summary(model, input_size=(2,), batch_size=8, device='cpu')
