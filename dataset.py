import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from dotenv import load_dotenv
from PIL import Image
import os


class NormMaskedDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # Get env variables for root paths
        load_dotenv(".env")
        root_x = os.getenv("TRAIN_X_PATH")
        root_y = os.getenv("TRAIN_Y_PATH")

        sample = self.paths[index]
        sample_x, sample_y = Image.open(
            root_x+sample), Image.open(root_y+sample)

        # Convert PIL Image to Image Tensor
        image2tensor = T.Compose([T.Grayscale(num_output_channels=3), T.Resize(size=(256, 256)), T.ToTensor(
        ), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        x_tensor, y_tensor = image2tensor(sample_x), image2tensor(sample_y)

        # Return samples
        return x_tensor, y_tensor
