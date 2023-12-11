import torch
from torch.utils.data import DataLoader
from Pretrained_Weights.dataset_regularization import WheatMaskDataset
from Pretrained_Weights.Model.model_segments.decoder import Decoder
from Pretrained_Weights.Model.model_segments.darknet import Darknet53
from Pretrained_Weights.Model.model_segments.darknet import DarkResidualBlock as EncResBlock
from Pretrained_Weights.Model.model_segments.decoder import DecoderResidualBlock as DecResBlock
from Pretrained_Weights.Model.model import Combined_Model
from Pretrained_Weights.initialize_weights import initialize
import wandb
import torch.multiprocessing
from torch import mps
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
import gc
import os
from dotenv import load_dotenv
from torchsummary import summary


def mask_penalty(mask):
    w1 = 0.95
    w2 = 0.05

    weighted_mask = (torch.where(mask == 1.0, w1**2, w2**2)).to(device=device)

    return weighted_mask


def train_epoch():
    epoch_loss = 0

    for step, (x_sample, y_sample, mask) in enumerate(train_loader):
        x_sample = x_sample.to(device=device)
        y_sample = y_sample.to(device=device)
        mask = mask.to(device=device)

        predictions = model(x_sample)

        # Compute Loss
        # L2 Bounding Box Regularization
        loss = torch.mean(
            torch.add(objective(predictions, y_sample), lamb*mask_penalty(mask)))

        # Perform backpropagation
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Memory Management
        del x_sample
        del y_sample
        del predictions
        del mask
        mps.empty_cache()
        gc.collect(generation=2)

    loss = epoch_loss/train_steps
    return loss


def test_epoch():
    epoch_loss = 0

    for step, (x_sample, y_sample, _) in enumerate(test_loader):
        x_sample = x_sample.to(device=device)
        y_sample = y_sample.to(device=device)

        predictions = model(x_sample)

        # Compute Loss
        loss = torch.mean(objective(predictions, y_sample))

        # add losses
        epoch_loss += loss.item()

        del x_sample
        del predictions
        del y_sample
        mps.empty_cache()
        gc.collect(generation=2)

    loss = epoch_loss/test_steps
    return loss


def training_loop():

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = train_epoch()

        model.eval()
        with torch.no_grad():
            test_loss = test_epoch()
            print("Epoch {epoch}".format(epoch=epoch+1))
            print("L2 Train Loss {loss}".format(loss=train_loss))
            print("L2 Test Loss {loss}".format(loss=test_loss))

            wandb.log({
                "L2 Pixel Regularization Train Loss": train_loss,
                "L2 Pixel Test Loss": test_loss
            })

            # checkpoints
            if ((epoch+1) % 5 == 0):
                complete_path = "Pretrained_Weights/weights/run_1/model{epoch}.pth".format(
                    epoch=epoch+1)

                torch.save(model.state_dict(), complete_path)


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    load_dotenv(".env")

    trainX_path = os.getenv("TRAIN_X_PATH")

    labels = sorted(os.listdir(trainX_path))

    # remove '_' in filenames
    labs = list()
    for i in labels:
        if '_' not in i:
            labs.append(i)

    params = {
        'batch_size': 8,
        'shuffle': True,
        'num_workers': 0
    }

    train, test = train_test_split(labs, test_size=0.25, shuffle=True)

    train_set = WheatMaskDataset(paths=train)
    test_set = WheatMaskDataset(paths=test)

    wandb.init(
        project="backbone-pretraining-wheats",
        config={
            "architecture": "autoencoder with darknet53 backbone",
            "dataset": "Global Wheat"
        }
    )

    train_loader = DataLoader(train_set, **params)
    test_loader = DataLoader(test_set, **params)

    # set the device
    device = torch.device("mps")

    # Hyperparameters and losses
    LR = 0.001
    NUM_EPOCHS = 10000

    objective = nn.MSELoss(reduction='none')
    # models and optimizers

    # Backbone Initializer
    backbone = Darknet53(EncResBlock, 2).to(device=device)
    weights = torch.load("pretrained/model_best.pth.tar",
                         map_location='cpu')  # Load COCO weights
    weights['state_dict'] = {k: v for k, v in weights['state_dict'].items() if k not in [
        'fc.weight', 'fc.bias']}  # Remove linear layer weights to account for size differences
    backbone.load_state_dict(weights['state_dict'], strict=False)

    # Decoder
    decoder = Decoder(DecResBlock, 2).to(device=device)
    decoder.apply(initialize)  # Apply Normal initialized weights for decoder

    # Combined Model
    model = Combined_Model(
        backbone=backbone, decoder=decoder).to(device=device)

    # optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=0.001, betas=(0.9, 0.999))

    lamb = 1
    train_steps = (len(train)+params['batch_size']-1)//params['batch_size']
    test_steps = (len(test)+params['batch_size']-1)//params['batch_size']

    mps.empty_cache()
    gc.collect(generation=2)

    training_loop()
