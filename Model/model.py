from Model.model_segments.darknet import darknet53
from Model.model_segments.decoder import Decoder
from torchsummary import summary
from torch.nn import Module


class Combined_Model(Module):
    def __init__(self):
        super(Combined_Model, self).__init__()

        self.backbone = darknet53(2)
        self.decoder = Decoder()

    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)

        return x

# model = Combined_Model()
# summary(model, input_size=(3, 1024, 1024), batch_size=8, device='cpu')