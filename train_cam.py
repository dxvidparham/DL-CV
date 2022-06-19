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
# Module: main script for running the training loop for the segmentation
######################################################################

import os
from pyexpat import model
import time

import albumentations as A
import click
import numpy as np
import torch
import wandb
from albumentations.pytorch import ToTensorV2
from omegaconf import OmegaConf
from torch import nn, optim
from tqdm import tqdm

from dataLoader import ISICDataset, ClassifierDataset
from metrics import SegmentationMetric
from utils import EarlyStopping, get_model, print_statistics, save_model, visualize_results_classification

# set flags / seeds to speed up the training process
np.random.seed(1)
torch.manual_seed(1)
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

# Load config file
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
config = OmegaConf.load(f"{BASE_DIR}/config/config.yaml")

# Hyperparameter
EPOCHS = config.EPOCHS
BATCH_SIZE = config.BATCH_SIZE
LEARNING_RATE = config.LEARNING_RATE

# Other const variables
N_WORKERS = config.N_WORKERS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARCHITECTURE = config.ARCHITECTURE
IMG_SIZE = config.IMG_SIZE
PIN_MEMORY = config.PIN_MEMORY

PRETRAINED = True

training_set = ClassifierDataset()
testing_set = ClassifierDataset()

LEN_TRAINSET = len(training_set)
LEN_TESTSET = len(testing_set)


def train(train_loader, test_loader, model) -> None:

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    num_epochs = EPOCHS

    for epoch in tqdm(range(num_epochs), unit='epoch'):
        #For each epoch
        train_correct = 0
        model.train()
        for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, target = data.to(DEVICE), target.to(DEVICE)

            data.requires_grad_()

            #Zero the gradients computed for each weight
            optimizer.zero_grad()
            #Forward pass your image through the network
            output = model(data)
            #Compute the loss
            loss_train = criterion(output, target)
            #Backward pass through the network
            loss_train.backward()
            #Update the weights
            optimizer.step()
            
            #Compute how many were correctly classified
            predicted = output.argmax(1)
            train_correct += (target==predicted).sum().cpu().item()
        #Comput the test accuracy
        test_correct = 0
        

        model.eval()
        for data, target in test_loader:
            
            data = data.to(DEVICE)
            target = target.to(DEVICE)

            data.requires_grad_()
            
            output = model(data)
            loss_test = criterion(output, target)

            predicted = output.argmax(1)
            
            test_correct += (target==predicted).sum().item()
        train_acc = train_correct/LEN_TRAINSET
        test_acc = test_correct/LEN_TESTSET
            
        print("\n \n Epoch: {epoch}/{total} \n Accuracy train: {train:.1f}%\t test: {test:.1f}% \n Loss: \t train: {loss_train:.3f} \t test: {loss_test:.3f}\n".format(
                                                            epoch=epoch, total=EPOCHS,
                                                            test=100*test_acc, train=100*train_acc,
                                                            loss_train=loss_train, loss_test=loss_test))

#=====================================================================================================


@click.command()
@click.option(
    "--wandb", is_flag=True, default=False, help="Use this flag to enable wandb"
)
def main(wandb):

    train_transform = A.Compose(
        [
            A.Resize(*IMG_SIZE),
            # A.Rotate(limit=35, p=1.0),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    test_transforms = A.Compose(
        [
            A.Resize(*IMG_SIZE),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    config.pop("IMG_SIZE")

    print("[INFO] Load datasets from disk...")
    training_set = ClassifierDataset(train_transform)
    testing_set = ClassifierDataset(test_transforms)

    LEN_TRAINSET = len(training_set)
    LEN_TESTSET = len(testing_set)

    print("[INFO] Prepare dataloaders...")
    trainloader = torch.utils.data.DataLoader(
        training_set,
        shuffle=True,
        num_workers=N_WORKERS,
        batch_size=BATCH_SIZE,
        pin_memory=PIN_MEMORY,
    )
    testloader = torch.utils.data.DataLoader(
        testing_set,
        shuffle=False,
        num_workers=N_WORKERS,
        batch_size=BATCH_SIZE,
        pin_memory=PIN_MEMORY,
    )

    model = get_model(ARCHITECTURE)
    # Push model to GPU if available
    if torch.cuda.is_available():
        print(f"[INFO] Training model on GPU ({torch.cuda.get_device_name(0)})...")
        model.to(DEVICE)
    else:
        print("[INFO] Training model on CPU...")

    
    train(trainloader, testloader,model)


if __name__ == "__main__":

    main()
