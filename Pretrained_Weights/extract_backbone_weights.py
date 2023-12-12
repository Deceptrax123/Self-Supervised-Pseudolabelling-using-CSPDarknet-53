import torch
from Pretrained_Weights.Model.model_segments.darknet import Darknet53, DarkResidualBlock
from Pretrained_Weights.Model.model_segments.decoder import Decoder, DecoderResidualBlock
from Pretrained_Weights.Model.model import Combined_Model


# extract the backbone weights of each model

if __name__ == '__main__':
    # Model state dict
    weights = torch.load("Pretrained_Weights/weights/run_1/model50.pth")

    weights_backbone = {k: v for k,
                        v in weights.items() if 'decoder.' not in k}

    # Refine key names to suit yolo architecture
    weights_backbone_updated = dict()
    for key, val in weights_backbone.items():
        updated_key = key[9:]

        weights_backbone_updated[updated_key] = val

    # Location to save backbone weights only.
    torch.save(weights_backbone_updated,
               'backbone_weights/pretrained/pretrained_backbone.pt')
