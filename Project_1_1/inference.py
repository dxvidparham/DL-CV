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

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.autograd import Variable
from torch import nn

from classifier.VGGNet import VGG_net
from dataLoader import Hotdog_NotHotdog as Dataset_fetcher


def inference():

    # set flags / seeds
    torch.backends.cudnn.benchmark = True
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # Load config file
    BASE_DIR = os.getcwd()
    config = OmegaConf.load(f"{BASE_DIR}/config/config.yaml")

    # Optimizer Hyperparameter
    BATCH_SIZE = 1

    # Other const variables
    N_WORKERS = config.N_WORKERS
    VERSION = config.VERSION

    # Model and checkpoint path
    BEST_MODEL_PATH = config.BEST_MODEL_PATH

    # Set device to CUDA if a CUDA device is available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    validation_set = Dataset_fetcher(train=False, augment=False)

    print("[INFO] Prepare dataloaders...")
    val_dataloader = torch.utils.data.DataLoader(
        validation_set, num_workers=N_WORKERS, batch_size=BATCH_SIZE
    )

    criterion = nn.CrossEntropyLoss()

    # Declare Siamese Network
    print(f"[INFO] Loading {VERSION}...")
    checkpoint = torch.load(BEST_MODEL_PATH)
    model = VGG_net(VERSION)
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    losses = []
    correct = 0
    total = 0

    def saliency(img, model):  # sourcery skip: avoid-builtin-shadow
        # we don't need gradients w.r.t. weights for a trained model
        for param in model.parameters():
            param.requires_grad = False

        # set model in eval mode
        model.eval()

        # we want to calculate gradient of higest score w.r.t. input
        # so set requires_grad to True for input
        img.requires_grad = True
        # forward pass to calculate predictions
        preds = model(img)
        score, indices = torch.max(preds, 1)
        # backward pass to get gradients of score predicted class w.r.t. input image
        score.backward()
        # get max along channel axis
        slc, _ = torch.max(torch.abs(img.grad[0]), dim=0)
        # normalize to [0..1]
        slc = (slc - slc.min()) / (slc.max() - slc.min())

        fig = plt.figure(f"Image belongs to class: {preds}", figsize=(8, 2))
        # plot image and its saleincy map
        # show first image
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(img[0].detach().cpu())
        plt.axis("off")

        # show the second image
        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(slc.cpu(), cmap=plt.cm.hot)
        plt.axis("off")

        plt.savefig(os.path.join(BEST_MODEL_PATH, "test.png"))

    for _, (image, label) in enumerate(val_dataloader):
        # wrap them in Variable
        plt.imshow(image.detach().cpu())
        plt.savefig(os.path.join(BEST_MODEL_PATH, "test.png"))
        # saliency(image.to(torch.float32).cuda(), model)

        break


if __name__ == "__main__":
    inference()
