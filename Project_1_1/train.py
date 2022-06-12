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
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn, optim
from torch.autograd import Variable
from tqdm import tqdm

import wandb
from classifier.VGGNet import VGG_net
from dataLoader import Hotdog_NotHotdog as Dataset_fetcher
from utils import EarlyStopping

# set flags / seeds
np.random.seed(1)
torch.manual_seed(1)

# Load config file
BASE_DIR = os.getcwd()
config = OmegaConf.load(f"{BASE_DIR}/config/config.yaml")

# Optimizer Hyperparameter
EPOCHS = config.EPOCHS
BATCH_SIZE = config.BATCH_SIZE
LEARNING_RATE = config.LEARNING_RATE
DROPOUT_PROBABILITY = config.DROPOUT_PROBABILITY

# Other const variables
N_WORKERS = config.N_WORKERS
VERSION = config.VERSION

# Model and checkpoint path
BEST_MODEL_PATH = config.BEST_MODEL_PATH
CHECKPOINT_PATH = config.CHECKPOINT_PATH

print("[INFO] Load datasets from disk...")
training_set = Dataset_fetcher(train=True, augment=True)
testing_set = Dataset_fetcher(train=False, augment=False)

print("[INFO] Prepare dataloaders...")
trainloader = torch.utils.data.DataLoader(
    training_set, shuffle=True, num_workers=N_WORKERS, batch_size=BATCH_SIZE
)
testloader = torch.utils.data.DataLoader(
    testing_set, shuffle=False, num_workers=N_WORKERS, batch_size=BATCH_SIZE
)

# pop config variables which are not needed to track
config.pop("BEST_MODEL_PATH")
config.pop("CHECKPOINT_PATH")

# Initialize logging with wandb and track conf settings
wandb.init(project="VGG", config=dict(config), entity="dlincvg1")


def train() -> None:

    print("[INFO] Building network...")
    # VGG11, VGG13, VGG16, VGG19
    model = VGG_net(VERSION)

    device = "cpu"
    if torch.cuda.is_available():
        print("[INFO] Pushing network to GPU...")
        device = "cuda"
        model.to(device)

    wandb.watch(model, log_freq=100)

    # specify loss function (categorical cross-entropy)
    criterion = nn.CrossEntropyLoss()

    # specify optimizer (stochastic gradient descent) and learning rate = 0.01
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=LEARNING_RATE / EPOCHS,
    )

    wandb.config.update({"optimizer": optimizer, "loss_func": criterion})

    print("[INFO] Started training the model...\n")
    start_t = time.time()
    best_val = 100000000
    early_stopping = EarlyStopping(tolerance=5, min_delta=10)

    for epoch in tqdm(range(EPOCHS), unit="epoch"):
        ######################
        # Train the model #
        ######################
        model.train()

        losses = []
        correct = 0
        total = 0

        for _, (images, labels) in tqdm(enumerate(trainloader), total=len(trainloader)):

            # wrap them in Variable
            images, labels = Variable(images.cuda()), Variable(labels.cuda())

            optimizer.zero_grad(set_to_none=True)

            output = model(images.float())
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            total += len(labels)

            _, predicted = torch.max(output.data, 1)
            correct += (predicted == labels).sum().item()

        train_loss = sum(losses) / max(1, len(losses))
        train_acc = 100 * correct // total

        # Log train loss and acc
        wandb.log({"train_loss": train_loss})
        wandb.log({"train_acc": train_acc})

        print(
            f"Epoch {epoch+1}/{EPOCHS} \n \tTraining:  "
            f" Loss={train_loss:.2f}\t Accuracy={train_acc}%\t"
        )

        ######################
        # validate the model #
        ######################
        with torch.no_grad():
            # Evaluation Loop Start
            model.eval()

            losses = []
            correct = 0
            total = 0

            for images, labels in testloader:

                # wrap them in Variable
                images, labels = Variable(images.cuda()), Variable(labels.cuda())

                output = model(images.float())
                loss = criterion(output, labels)

                losses.append(loss.item())
                total += len(labels)

                _, predicted = torch.max(output.data, 1)
                correct += (predicted == labels).sum().item()

        val_loss = sum(losses) / max(1, len(losses))
        val_acc = 100 * correct // total

        # Log train loss and acc
        wandb.log({"val_loss": val_loss})
        wandb.log({"val_acc": val_acc})

        print(f"\tValidation: Loss={val_loss:.2f}\t Accuracy={val_acc}%\t")

        # Evaluation loop end

        # Save best model if val_loss in current epoch is lower than the best validation loss
        if val_loss < best_val:
            best_val = val_loss
            print("\n[INFO] Saving new best_model...\n")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                BEST_MODEL_PATH,
            )

        # Save model based on the frequency defined by "args.save_after"
        if (epoch + 1) % 5 == 0:
            print(f"\n[INFO] Saving model as checkpoint -> epoch_{epoch+1}.pth\n")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(CHECKPOINT_PATH, f"epoch_{epoch + 1}.pth"),
            )

            # early stopping
            early_stopping(train_loss, val_loss)
            if early_stopping.early_stop:
                print(f"[INFO] Initializing Early stoppage at epoch: {epoch+1}...")
                wandb.config.update({"EPOCHS": epoch + 1, "early_stoppage": True})
                break

    end_t = time.time()
    run_time = end_t - start_t

    # if checkpoint folder is meant to be saved for each experiment
    wandb.save(CHECKPOINT_PATH)
    print(
        f"[INFO] Successfully completed training session. Running time: {run_time/60:.2f} min"
    )


if __name__ == "__main__":

    train()
