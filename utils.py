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
# Module: This module contain utility classes and functions
######################################################################

import os

import torch

from architectures import FCN, NestedUNet, UNet


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


def get_model(model_name):
    model_name = model_name.lower()
    if model_name == "unet":
        return UNet.UNet
    elif model_name == "unet++":
        return NestedUNet.NestedUNet
    elif model_name == "fcn":
        return FCN.FCN
    else:
        print("Model with model name: {model_name} not found.")
        raise ValueError


def save_model(epoch, model, optimizer, path, new_best_model=False):
    if new_best_model:
        print("\n[INFO] Saving new best_model...\n")
    else:
        print(f"\n[INFO] Saving model as checkpoint -> epoch_{epoch+1}.pth\n")
        path = os.path.join(path, f"epoch_{epoch + 1}.pth")

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def print_statistics(loss, pixAcc, mIoU, epoch=0, epochs=0, Training=True):
    if Training:
        print(
            f"Epoch {epoch+1}/{epochs} \n \tTraining:  "
            f" Loss={loss:.2f}\t pixAcc={pixAcc}%\t mIoU={mIoU}%\t"
        )
    else:
        print(f"\tTesting: Loss={loss:.2f}\t pixAcc={pixAcc}%\t mIoU={mIoU}%\t")


if __name__ == "__main__":
    pass
