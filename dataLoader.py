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

from cProfile import label
import glob
import os

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
    def __init__(self, data_path, transform=None):
        self.mask_paths = []
        self.image_paths = sorted(glob.glob(f"{data_path}/Images/*.jpg"))
        for path in self.image_paths:
            file_name = path.split("/")[-1][:-4]
            self.mask_paths.append(
                sorted(glob.glob(f"{data_path}/Segmentations/{file_name}*.png"))[0]
            )
        assert len(self.mask_paths) == len(self.image_paths)

        self.transform = transform

    def __len__(self):
        "Returns the total number of samples"
        return len(self.image_paths)

    def __getitem__(self, idx):
        "Generates one sample of data"
        mask_path = self.mask_paths[idx]
        image_path = self.image_paths[idx]

        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(mask_path))
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


class ClassifierDataset(torch.utils.data.Dataset):
    def __init__(
        self, transform=None, data_path="/dtu/datasets1/02514/isic"
    ):
        self.mask_paths = []
        self.back_image_paths = sorted(glob.glob(f"{data_path}/background/*.jpg"))
        self.fore_image_paths = sorted(glob.glob(f"{data_path}/train_allstyles/Images/*.jpg"))

        test1 = {path:0 for path in self.back_image_paths}
        test2 = {path:1 for path in self.fore_image_paths}

        self.labels = list(test1.values())+list(test2.values())
        self.image_paths = list(test1.keys())+list(test2.keys())

        self.transform = transform


    def __len__(self):
        "Returns the total number of samples"
        return len(self.image_paths)

    def __getitem__(self, idx):
        "Generates one sample of data"
        target = self.labels[idx]

       
        image_path = self.image_paths[idx]
        
        image = np.array(Image.open(image_path))
        

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image, target


if __name__ == "__main__":

    print("[INFO] Load datasets from disk...")
    dataset = ISICDataset(data_path= "/dtu/datasets1/02514/isic/train_allstyles")

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

    d =  ClassifierDataset()
    dataloader_iter = iter(d)
    x, y = next(dataloader_iter)
    print(x)