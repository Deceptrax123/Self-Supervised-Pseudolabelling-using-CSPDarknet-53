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
        root_x = os.getenv("TRAIN_X_PATH")
        root_y = os.getenv("TRAIN_Y_PATH")

        sample = self.paths[index]
        sample_x_path, sample_y_path = sample[0], sample[1]
        sample_x, sample_y = Image.open(
            root_x+sample_x_path), Image.open(root_y+sample_y_path)

        # Convert PIL Image to Image Tensor
        image2tensor = T.Compose([T.Resize(size=(256,256)),T.ToTensor()])

        x_tensor, y_tensor = image2tensor(sample_x), image2tensor(sample_y)

        # Normalize X
        mean_img = torch.mean(x_tensor, [1, 2])
        std_img = torch.std(x_tensor, [1, 2])
        norm = T.Normalize(mean=mean_img, std=std_img)

        # Normalize Y
        mean_y=torch.mean(y_tensor,[1,2])
        std_y=torch.std(y_tensor,[1,2])
        norm_y=T.Normalize(mean=mean_y,std=std_y)

        x_tensor_norm = norm(x_tensor)
        y_tensor_norm=norm_y(y_tensor)

        # Return samples
        return x_tensor_norm, y_tensor_norm
