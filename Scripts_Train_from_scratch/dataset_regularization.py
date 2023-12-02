import torch
import torchvision.transforms as T
import torchvision.transforms as T
import numpy as np
from dotenv import load_dotenv
from PIL import Image
import os


class WheatMaskDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # Get env variables for root paths
        load_dotenv(".env")
        root_x = os.getenv("TRAIN_Y_PATH")
        root_y = os.getenv("TRAIN_Y_PATH")
        root_mask = os.getenv("MASK")

        sample = self.paths[index]
        sample_x, sample_y = Image.open(
            root_x+sample), Image.open(root_y+sample)

        # Get Corresponding Mask
        mask = Image.open(root_mask+sample)

        # Convert PIL Image to Image Tensor
        image2tensor = T.Compose([T.Resize(size=(256, 256)), T.ToTensor(
        ), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        mask2tensor = T.Compose([T.ToTensor()])

        x_tensor, y_tensor = image2tensor(sample_x), image2tensor(sample_y)
        mask_tensor = mask2tensor(mask)

        # Return samples
        return x_tensor, y_tensor, mask_tensor
