from torchsummary import summary
from torch.nn import Module


class Combined_Model(Module):
    def __init__(self, backbone, decoder):
        super(Combined_Model, self).__init__()

        self.backbone = backbone
        self.decoder = decoder

    def forward(self, x):
        x, x1, x2, x3, x4, x5 = self.backbone.forward(x)
        x = self.decoder(x, x1, x2, x3, x4, x5)

        return x
