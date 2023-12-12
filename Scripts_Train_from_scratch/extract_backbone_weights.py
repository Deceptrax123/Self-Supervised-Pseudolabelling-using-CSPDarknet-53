import torch

if __name__ == '__main__':
    weights = torch.load(
        "weights/complete/no_box_penalty/model50.pth")  # Model state dict path

    weights_backbone = {k: v for k,
                        v in weights.items() if 'model.encoder' in k}

    # Refine keys to match yolo architectures
    weights_backbone_updated = dict()
    for key, val in weights_backbone.items():
        updated_key = key[14:]

        weights_backbone_updated[updated_key] = val

    print(weights_backbone_updated.keys())

    # Location to save backbone weights only
    torch.save(weights_backbone_updated,
               'backbone_weights/he_inititialized/Without_Regularization/backbone_noreg.pt')
