import torch
from torch.nn import Linear, Conv2d, BatchNorm1d, ReLU, Dropout1d
from torch.nn import Module


class Bottleneck(Module):
    def __init__(self):
        super(Bottleneck, self).__init__()

        self.linear1 = Linear(in_features=2, out_features=8)
        self.bn1 = BatchNorm1d(8)
        self.relu1 = ReLU()

        self.linear2 = Linear(in_features=8, out_features=16)
        self.bn2 = BatchNorm1d(16)
        self.relu2 = ReLU()

        self.linear3 = Linear(in_features=16, out_features=32)
        self.bn3 = BatchNorm1d(32)
        self.relu3 = ReLU()

        self.linear4 = Linear(in_features=32, out_features=16)
        self.bn4 = BatchNorm1d(16)
        self.relu4 = ReLU()

        self.linear5 = Linear(in_features=16, out_features=8)
        self.bn5 = BatchNorm1d(8)
        self.relu5 = ReLU()

        self.linear6 = Linear(in_features=8, out_features=2)
        self.bn6 = BatchNorm1d(2)
        self.relu6 = ReLU()

        self.dp1 = Dropout1d(0.2)
        self.dp2 = Dropout1d(0.2)
        self.dp3 = Dropout1d(0.2)
        self.dp4 = Dropout1d(0.2)
        self.dp5 = Dropout1d(0.2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.dp1(x)

        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.dp2(x)

        x = self.linear3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.dp3(x)

        x = self.linear4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.dp4(x)

        x = self.linear5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.dp5(x)

        x = self.linear6(x)
        x = self.bn6(x)
        x = self.relu6(x)

        return x
