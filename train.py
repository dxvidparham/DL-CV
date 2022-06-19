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

from dataLoader import ISICDataset
from metrics import SegmentationMetric
from utils import EarlyStopping, get_model, print_statistics, save_model

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


def train(trainloader, testloader, disable_wandb, scaler) -> None:

    # Control wandb initialization
    if disable_wandb:
        wandb.init(mode="disabled")
    else:
        wandb.init(project="Segmentation", config=dict(config), entity="dlincvg1")

    print(f"[INFO] Initializing model architecture -> {ARCHITECTURE}...")
    # Choose between FCN, UNet, UNet++, resnet101
    model = get_model(ARCHITECTURE)  # changed function to return model not class

    # Push model to GPU if available
    if torch.cuda.is_available():
        print(f"[INFO] Training model on GPU ({torch.cuda.get_device_name(0)})...")
        model.to(DEVICE)
    else:
        print("[INFO] Training model on CPU...")

    # Log gradients and parameters every N batches -> log_freq
    wandb.watch(model, log_freq=100)

    # Choose cross_entropy for multi_class classification
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=LEARNING_RATE / EPOCHS
    )

    wandb.log({"optimizer": optimizer.__class__.__name__})
    wandb.log({"loss_fn": loss_fn.__class__.__name__})
    wandb.log({"device": DEVICE})

    print("[INFO] Start training loop...\n")
    start_t = time.time()
    best_test_loss = 100000000
    early_stopping = EarlyStopping(tolerance=5, min_delta=10)

    train_metrics = SegmentationMetric()
    test_metrics = SegmentationMetric()

    for epoch in tqdm(range(EPOCHS), unit="epoch"):
        ######################
        #### Train the model ###
        ######################
        model.train()

        losses = []
        train_metrics.reset()

        for images, labels in tqdm(trainloader, total=len(trainloader)):

            images = images.float().to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            with torch.cuda.amp.autocast():
                output = model(images)
                if ARCHITECTURE == "resnet101":
                    output = output["out"]
                loss = loss_fn(output, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())
            train_metrics.update(output, labels)

        train_loss = sum(losses) / len(losses)
        train_mIoU, train_pixAcc = [
            100 * np.mean(metric) for metric in train_metrics.get()
        ]

        # Log train loss and acc
        wandb.log({"train_loss": train_loss})
        wandb.log({"train_pixAcc": train_pixAcc})
        wandb.log({"train_mIoU": train_mIoU})

        print_statistics(train_loss, train_pixAcc, train_mIoU, epoch, EPOCHS)

        ######################
        #### test the model ####
        ######################
        model.eval()
        with torch.no_grad():

            losses = []
            test_metrics.reset()

            for images, labels in testloader:

                images = images.float().to(DEVICE)
                labels = labels.float().unsqueeze(1).to(DEVICE)

                with torch.cuda.amp.autocast():
                    output = model(images)
                    if ARCHITECTURE == "resnet101":
                        output = output["out"]
                    loss = loss_fn(output, labels)

                losses.append(loss.item())
                test_metrics.update(output, labels)

        test_loss = sum(losses) / len(losses)
        test_mIoU, test_pixAcc = [
            100 * np.mean(metric) for metric in test_metrics.get()
        ]

        # Log train loss and acc
        wandb.log({"test_loss": test_loss})
        wandb.log({"test_pixAcc": test_pixAcc})
        wandb.log({"test_mIoU": test_mIoU})

        print_statistics(test_loss, test_pixAcc, test_mIoU, Training=False)

        # Evaluation loop end

        # Save best model if test_loss in current epoch is lower than the best validation loss
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            save_model(
                epoch, model, optimizer, config.BEST_MODEL_PATH, new_best_model=True
            )

        # Save model based on the frequency defined by "args.save_after"
        if (epoch + 1) % 5 == 0:
            save_model(epoch, model, optimizer, config.CHECKPOINT_PATH)

        # early stopping
        early_stopping(train_loss, test_loss)
        if early_stopping.early_stop:
            print(f"[INFO] Initializing Early stoppage at epoch: {epoch+1}...")
            break

    end_t = time.time()
    run_time = end_t - start_t

    # if checkpoint folder is meant to be saved for each experiment
    wandb.save(config.CHECKPOINT_PATH)
    print(
        f"[INFO] Successfully completed training session. Running time: {run_time/60:.2f} min"
    )


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
    training_set = ISICDataset(train_transform)
    testing_set = ISICDataset(test_transforms)

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

    scaler = torch.cuda.amp.GradScaler()
    train(trainloader, testloader, disable_wandb=not wandb, scaler=scaler)


if __name__ == "__main__":

    main()
