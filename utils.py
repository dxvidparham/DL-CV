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
import matplotlib.pyplot as plt
import itertools
import numpy as np
import numpy.ma as ma

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
            f" Loss={loss:.2f}\t pixAcc={pixAcc:.2f}%\t mIoU={mIoU:.3f}%\t"
        )
    else:
        print(f"\tTesting: Loss={loss:.2f}\t pixAcc={pixAcc:.2f}%\t mIoU={mIoU:.3f}%\t")


if __name__ == "__main__":
    pass


def unique_file(basename, ext):
    actualname = "%s.%s" % (basename, ext)
    c = itertools.count()
    while os.path.exists(actualname):
        actualname = "%s (%d).%s" % (basename, next(c), ext)
    return actualname


def visualize_results_(images, predicted, label):
    # print(f"Image: {images.shape} {type(images)} {len(images)}, predicted: {predicted.shape} {type(predicted)}, GT: {label.shape} {type(label)}")
    plt.figure(figsize=(20,20))
    subplots = [plt.subplot(1,len(images), k+1) for k in range(len(images))]
    
    for k, (img, pred, gt) in enumerate(zip(images, predicted, label)):
        # print(f"Image: {img.shape} {type(img)}, predicted: {pred.shape} {type(pred)}, GT: {gt.shape} {type(gt)}")
        print(f"image {k} added to plot")
        img = (img.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        pred = pred.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()
        pred_mask = ma.masked_array(pred > 0, pred)
        gt = gt.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()
        gt_mask = ma.masked_array(gt > 0, gt)
        subplots[k].imshow(img.cpu().numpy())
        subplots[k].imshow(pred_mask, "hot", alpha=0.2)
        subplots[k].imshow(gt_mask, "jet", alpha=0.2)
        subplots[k].axis('off')
    plt.savefig(unique_file("out/images/test_result","png"),bbox_inches = "tight")
    print("saved_file")


def visualize_results(images, predicted, label):
    # print(f"Image: {images.shape} {type(images)} {len(images)}, predicted: {predicted.shape} {type(predicted)}, GT: {label.shape} {type(label)}") 
    for k, (img, pred, gt) in enumerate(zip(images, predicted, label)):
        # print(f"Image: {img.shape} {type(img)}, predicted: {pred.shape} {type(pred)}, GT: {gt.shape} {type(gt)}")
        plt.figure(figsize=(10,10))
        print(f"image {k} added to plot")
        img = (img.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        pred = pred.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()
        pred_mask = ma.masked_array(pred > 0, pred)
        gt = gt.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()
        gt_mask = ma.masked_array(gt > 0, gt)
        plt.imshow(img.cpu().numpy())
        plt.imshow(gt_mask, "hot", alpha=0.3)
        plt.savefig(unique_file("out/images/test_result_pred","png"),bbox_inches = "tight")
        plt.imshow(pred_mask, "jet", alpha=0.3)
        plt.axis('off')
        plt.savefig(unique_file("out/images/test_result_all","png"),bbox_inches = "tight")
    print("saved_file")