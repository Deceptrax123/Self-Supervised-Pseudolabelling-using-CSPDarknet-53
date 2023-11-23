from Model.model_segments.darknet import Darknet53
from Model.model_segments.decoder import Decoder
from Model.model_segments.darknet import DarkResidualBlock
from Model.model_segments.bottleneck import Bottleneck
from torchsummary import summary
from torch.nn import Module


class Combined_Model(Module):
    def __init__(self):
        super(Combined_Model, self).__init__()

        self.backbone = Darknet53(block=DarkResidualBlock, num_classes=2)
        self.bottleneck = Bottleneck()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.backbone(x)
        x = self.bottleneck(x)
        x = self.decoder(x)

        return x

# model = Combined_Model()
# summary(model, input_size=(3, 224, 224), batch_size=8, device='cpu')
