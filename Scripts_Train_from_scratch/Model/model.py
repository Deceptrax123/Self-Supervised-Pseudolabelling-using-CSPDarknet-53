from Model.model_segments.decoder import Decoder
from Model.model_segments.decoder import DecoderResidualBlock
from torchsummary import summary
from torch.nn import Module


class Combined_Model(Module):
    def __init__(self):
        super(Combined_Model, self).__init__()

        self.model = Decoder(DecoderResidualBlock, 2)

    def forward(self, x):
        x = self.model(x)

        return x

# model = Combined_Model()
# summary(model, input_size=(3, 224, 224), batch_size=8, device='cpu')
