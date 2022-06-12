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
BASE_DIR = os.getcwd()
config = OmegaConf.load(f"{BASE_DIR}/config/config.yaml")

IMG_SIZE = config.IMG_SIZE


class Hotdog_NotHotdog(torch.utils.data.Dataset):
    def __init__(
        self, train, augment, data_path="/dtu/datasets1/02514/hotdog_nothotdog"
    ):
        "Initialization"
        self.augment = augment
        data_path = os.path.join(data_path, "train" if train else "test")
        image_classes = [
            os.path.split(d)[1] for d in glob.glob(f"{data_path}/*") if os.path.isdir(d)
        ]

        image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(image_classes)}
        self.image_paths = glob.glob(f"{data_path}/*/*.jpg")

        if self.augment:
            self.transform = albumentations.Compose(
                [
                    albumentations.Resize(IMG_SIZE, IMG_SIZE),
                    albumentations.OneOf(
                        [
                            albumentations.HorizontalFlip(p=1),
                            albumentations.RandomRotate90(p=1),
                            albumentations.VerticalFlip(p=1),
                        ],
                        p=0.25,
                    ),
                    albumentations.OneOf(
                        [
                            albumentations.RandomBrightness(p=1),
                            albumentations.RandomBrightnessContrast(p=1),
                        ],
                        p=0.25,
                    ),
                    albumentations.pytorch.transforms.ToTensorV2(),
                ]
            )
        else:
            self.transform = albumentations.Compose(
                [
                    albumentations.Resize(IMG_SIZE, IMG_SIZE),
                    albumentations.pytorch.transforms.ToTensorV2(),
                ]
            )

    def __len__(self):
        "Returns the total number of samples"
        return len(self.image_paths)

    def __getitem__(self, idx):
        "Generates one sample of data"
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = np.array(image)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        X = self.transform(image=image)["image"]

        return X, y


if __name__ == "__main__":

    print("[INFO] Load datasets from disk...")
    dataset = Hotdog_NotHotdog(train=True, augment=True)

    print("[INFO] Prepare dataloaders...")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        num_workers=config.N_WORKERS,
        batch_size=config.BATCH_SIZE,
    )

    dataloader_iter = iter(dataloader)
    x, y = next(dataloader_iter)
