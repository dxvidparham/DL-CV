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
import torchvision

import matplotlib.pyplot as plt

from albumentations.pytorch import ToTensorV2
from omegaconf import OmegaConf
from torch import nn, optim
from tqdm import tqdm
from collections import ChainMap


from dataLoader import ISICDataset, ClassifierDataset
from metrics import SegmentationMetric
from utils import EarlyStopping, ImageTransformations, get_model, print_statistics, save_model, visualize_results, visualize_results_saliency


# set flags / seeds to speed up the training process
np.random.seed(1)
torch.manual_seed(1)
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

# Load config files
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
config = OmegaConf.load(f"{BASE_DIR}/config/config.yaml")
hp_config = OmegaConf.load(f"{BASE_DIR}/config/hp_config.yaml")

# Hyperparameter
EPOCHS = hp_config.EPOCHS
BATCH_SIZE = hp_config.BATCH_SIZE
LEARNING_RATE = hp_config.LEARNING_RATE

# Other const variables
N_WORKERS = config.N_WORKERS
IMG_SIZE = config.IMG_SIZE
PIN_MEMORY = config.PIN_MEMORY

# Load training paths into const variable
TRAIN_PATHS = ChainMap(*config.TRAIN_STYLE_PATHS[::-1])
TEST_PATH = config.TEST_STYLE_PATH

# Other const variables
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARCHITECTURE = hp_config.ARCHITECTURE


def train(train_loader, test_loader, segmentation_loader, model) -> None:

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    test_metrics = SegmentationMetric()

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
            output, acts = model(data)

            #Compute the loss
            loss_train = criterion(output, target)
            #Backward pass through the network
            loss_train.backward()
            #Update the weights
            optimizer.step()

            acts = acts.detach().cpu()
            grads = model.get_act_grads().detach().cpu()
            pooled_grads = torch.mean(grads, dim=[0,2,3]).detach().cpu()

            for i in range(acts.shape[1]):
                acts[:,i,:,:] += pooled_grads[i]

            heatmap_j = torch.mean(acts, dim = 1).squeeze()
            heatmap_j_max = heatmap_j.max(axis = 0)[0]
            heatmap_j /= heatmap_j_max

            #Compute how many were correctly classified
            predicted = output.argmax(1)
            
            train_correct += (target==predicted).sum().cpu().item()

        train_acc = train_correct/LEN_TRAINSET
        

        #Comput the test accuracy
        test_correct = 0
        
        model.eval()
        #=========================================
        # test classification
        for data, target in test_loader:
            
            data = data.to(DEVICE)
            target = target.to(DEVICE)

            data.requires_grad_()
            
            output, _ = model(data)
            loss_test = criterion(output, target)

            predicted = output.argmax(1)
            test_correct += (target==predicted).sum().item()
            

        
        test_acc = test_correct/LEN_TESTSET
        print("\n \n Classification:")    
        print("\nEpoch: {epoch}/{total} \n \tAccuracy train: {train:.1f}%\t test: {test:.1f}% \n\t Loss: \t train: {loss_train:.3f} \t test: {loss_test:.3f}\n".format(
                                                            epoch=epoch, total=EPOCHS,
                                                            test=100*test_acc, train=100*train_acc,
                                                            loss_train=loss_train, loss_test=loss_test))

        #=========================================
        # test segmentation
        class_correct = 0
        test_metrics.reset()
        for data, target_mask in segmentation_loader:
            
            data = data.to(DEVICE)
            target_mask = target_mask.to(DEVICE)
            target_label = torch.ones((len(data))).type(torch.LongTensor)
            target_label = target_label.to(DEVICE)

            data.requires_grad_()
            
            # classification part
            output_class, _ = model(data)
            loss_seg = criterion(output_class, target_label)
            predicted_class = output_class.argmax(1)
            class_correct += (target_label==predicted_class).sum().item()
            test_acc_batch = (target_label==predicted_class).sum() / len(data)
            
            # segmentation part
            heatmap, segmentation = model.segmentation_by_saliency()
            segmentation = segmentation.to(DEVICE)
            test_metrics.update(segmentation, target_mask)
            

            if epoch >= 2:
                visualize_results_saliency(data,heatmap, predicted_class, target_label)
                print('============================')
                # segmentation = segmentation[:,None,:,:]
                visualize_results(data,segmentation,target_mask)
                print('============================')
                
        class_acc = class_correct/LEN_SEGMENTSET
        mIoU, pixAcc = [
        100 * np.mean(metric) for metric in test_metrics.get()
        ]
            
        print("\n Segmentation: \n \tAccuracy classification: {score:.1f}".format(
                                                            score=100*class_acc))
        print(f"\tpixAcc={pixAcc:.2f}%\t mIoU={mIoU:.2f}%\n")
#=====================================================================================================


@click.command()
@click.option(
    "--wandb", is_flag=True, default=False, help="Use this flag to enable wandb"
)
def main(wandb):
    
    # Load config files
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    config = OmegaConf.load(f"{BASE_DIR}/config/config.yaml")
    hp_config = OmegaConf.load(f"{BASE_DIR}/config/hp_config.yaml")

    # Get transformations for respective training split
    train_transform = ImageTransformations(is_train=True, img_size=IMG_SIZE)

    # Append names of transformations to config, to track data augmentation strategy
    config["TRAIN_TRANSFORMATIONS"] = train_transform.__names__()

    print("[INFO] Load datasets from disk...")
    classifier_set = ClassifierDataset(transform=train_transform.augmentations)

    
    len_classifier_set = len(classifier_set)

    train_count = int(len_classifier_set * 0.7)
    test_count = int(len_classifier_set * 0.3)

    class_trainset, class_testset = torch.utils.data.random_split(
                                                    classifier_set, [train_count, test_count])
    global LEN_TRAINSET
    global LEN_TESTSET 
    LEN_TRAINSET = len(class_trainset)
    LEN_TESTSET = len(class_testset)

    print("[INFO] Prepare labeldataloaders...")
    class_trainloader = torch.utils.data.DataLoader(
        class_trainset,
        shuffle=True,
        num_workers=N_WORKERS,
        batch_size=BATCH_SIZE,
        )

    class_testloader = torch.utils.data.DataLoader(
        class_testset,
        shuffle=True,
        num_workers=N_WORKERS,
        batch_size=BATCH_SIZE,
        )
    #==============
    # Load segmentation test set

    test_transform = ImageTransformations(is_train=False, img_size=IMG_SIZE)

    seg_testing_set = ISICDataset(TEST_PATH, test_transform.augmentations)

    global LEN_SEGMENTSET
    LEN_SEGMENTSET = len(seg_testing_set)

    seg_testloader = torch.utils.data.DataLoader(
        seg_testing_set,
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

    train(class_trainloader, class_testloader, seg_testloader, model)


if __name__ == "__main__":

    main()
