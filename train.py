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
from collections import ChainMap

import click
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

import wandb
from dataLoader import ISICDataset
from metrics import SegmentationMetric
from utils import (
    EarlyStopping,
    ImageTransformations,
    loss_fns,
    models,
    optimizers,
    print_statistics,
    save_model,
)

# set flags / seeds to speed up the training process
np.random.seed(1)
torch.manual_seed(1)
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)


def train(config, path_config, trainloader, testloader, disable_wandb, scaler) -> None:

    # Hyperparameter
    EPOCHS = config.EPOCHS
    LEARNING_RATE = config.LEARNING_RATE

    # Other const variables
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ARCHITECTURE = config.ARCHITECTURE
    OPTIMIZER = config.OPTIMIZER
    LOSS_FN = config.LOSS_FN

    # Control wandb initialization
    if disable_wandb:
        wandb.init(mode="disabled")
    else:
        wandb.init(
            project="Segmentation1",
            config=dict(config),
            entity="dlincvg1",
            # tags=[""],
        )

    print(f"[INFO] Initializing model architecture -> {ARCHITECTURE}...")
    # Choose between UNet, UNet++, resnet101
    model = models(ARCHITECTURE)()

    # Push model to GPU if available
    if torch.cuda.is_available():
        print(f"[INFO] Training model on GPU ({torch.cuda.get_device_name(0)})...")
        model.to(DEVICE)
    else:
        print("[INFO] Training model on CPU...")

    # Log gradients and parameters every N batches -> log_freq
    wandb.watch(model, log_freq=100)

    # Choose cross_entropy for multi_class classification
    loss_fn = loss_fns(LOSS_FN)()
    optimizer = optimizers(OPTIMIZER)(
        model.parameters(), lr=LEARNING_RATE, weight_decay=LEARNING_RATE / EPOCHS
    )

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
                epoch,
                model,
                optimizer,
                path_config.BEST_MODEL_PATH,
                new_best_model=True,
            )

        # Save model based on the frequency defined by "args.save_after"
        if (epoch + 1) % 5 == 0:
            save_model(epoch, model, optimizer, path_config.CHECKPOINT_PATH)

        # early stopping
        early_stopping(train_loss, test_loss)
        if early_stopping.early_stop:
            print(f"[INFO] Initializing Early stoppage at epoch: {epoch+1}...")
            break

    end_t = time.time()
    run_time = end_t - start_t

    # if checkpoint folder is meant to be saved for each experiment
    wandb.save(path_config.CHECKPOINT_PATH)
    print(
        f"[INFO] Successfully completed training session. Running time: {run_time/60:.2f} min"
    )


def update_config(config, sweep_config):
    for key, value in sweep_config.items():
        config[key.upper()] = value
    return config


@click.command()
@click.option(
    "--wandb", is_flag=True, default=False, help="Use this flag to enable wandb"
)
@click.option(
    "--ARCHITECTURE",
    type=click.Choice(["unet", "unet++", "restnet101"], case_sensitive=False),
    help="Choose between UNet, UNet++, resnet101",
)
@click.option("--BATCH_SIZE", type=int)
@click.option("--EPOCHS", type=int, help="Use this flag to enable wandb")
@click.option("--LEARNING_RATE", type=float, help="Use this flag to enable wandb")
@click.option(
    "--LOSS_FN",
    type=click.Choice(["BCEWithLogitsLoss", "BinaryDiceLoss"], case_sensitive=False),
    help="Use this flag to enable wandb",
)
@click.option(
    "--OPTIMIZER",
    type=click.Choice(["Adam", "SGD"], case_sensitive=False),
    help="Use this flag to enable wandb",
)
def main(wandb, **args):

    # Load config files
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    config = OmegaConf.load(f"{BASE_DIR}/config/config.yaml")
    hp_config = OmegaConf.load(f"{BASE_DIR}/config/hp_config.yaml")

    # Updates the config values if hyperparamter optimization is on
    if all(args.values()):
        wandb = True
        hp_config = update_config(hp_config, args)

    # Hyperparameter
    BATCH_SIZE = hp_config.BATCH_SIZE

    # Other const variables
    N_WORKERS = config.N_WORKERS
    IMG_SIZE = config.IMG_SIZE
    PIN_MEMORY = config.PIN_MEMORY

    # Load training paths into const variable
    TRAIN_PATHS = ChainMap(*config.TRAIN_STYLE_PATHS[::-1])
    TEST_PATH = config.TEST_STYLE_PATH

    # Get transformations for respective training split
    train_transform = ImageTransformations(is_train=True, img_size=IMG_SIZE)
    test_transform = ImageTransformations(is_train=False, img_size=IMG_SIZE)

    # Append names of transformations to config, to track data augmentation strategy
    hp_config["TRAIN_TRANSFORMATIONS"] = train_transform.__names__()

    print("[INFO] Load datasets from disk...")
    training_set = ISICDataset(
        TRAIN_PATHS.get("train_allstyles"), train_transform.augmentations
    )
    testing_set = ISICDataset(TEST_PATH, test_transform.augmentations)

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
    train(
        hp_config,
        config,
        trainloader,
        testloader,
        disable_wandb=not wandb,
        scaler=scaler,
    )


if __name__ == "__main__":

    main()
