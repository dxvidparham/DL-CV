#!/usr/bin/env python3
######################################################################
# Authors:      <s203005> Karol Bogumil Krzak
#                     <s202385> David Parham
#                     <s202468> Alejandro Martinez Senent
#                     <s202460> Christian Jannik Metz
#
# Course:        Deep Learning for Computer Vision
# Semester:    June 2022
# Institution:  Technical University of Denmark (DTU)
#
# Module: <Purpose of the module>
######################################################################

import glob
import os

import albumentations
import albumentations.pytorch
import numpy as np
import PIL.Image as Image
import torch
from omegaconf import OmegaConf

# Load config file
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
config = OmegaConf.load(f"{BASE_DIR}/config/config.yaml")

IMG_SIZE = config.IMG_SIZE

# background, train_allstyles test_style0 train_style0 train_style1 train_style2


class ISICDataset(torch.utils.data.Dataset):
    def __init__(
        self, transform=None, data_path="/dtu/datasets1/02514/isic/train_allstyles"
    ):

        self.image_paths = glob.glob(f"{data_path}/Images/*.jpg")
        self.mask_paths = glob.glob(f"{data_path}/Segmentations/*.png")
        self.transform = transform
        self.classes = ["background", "foreground"]

    def __len__(self):
        "Returns the total number of samples"
        return len(self.image_paths)

    def __getitem__(self, idx):
        "Generates one sample of data"
        mask_path = self.mask_paths[idx]

        try:
            image_path = self.image_paths[idx]
        except IndexError:
            search_string = mask_path.split("_seg")[0]
            for i, path in enumerate(self.image_paths):
                if search_string in path:
                    idx = i
                    break

            image_path = self.image_paths[idx]

        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(mask_path))
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


if __name__ == "__main__":

    print("[INFO] Load datasets from disk...")
    dataset = ISICDataset()

    print("[INFO] Prepare dataloaders...")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        num_workers=config.N_WORKERS,
        batch_size=config.BATCH_SIZE,
    )

    dataloader_iter = iter(dataloader)
    x, y = next(dataloader_iter)
    print(x)