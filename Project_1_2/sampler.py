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

import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import datasets
from omegaconf import OmegaConf
import os
from taco_dataloader import TacoDataset

# Load config file
BASE_DIR = os.getcwd()
config = OmegaConf.load(f"{BASE_DIR}/config/config.yaml")

# Optimizer Hyperparameter
BATCH_SIZE = config.BATCH_SIZE

# config  variables
N_WORKERS = config.N_WORKERS

SPLIT_DISTRIBUTION = config.SPLIT_DISTRIBUTION
_version = 0

classes = ["Plastic bag & wrapper", "Cigarette", "Bottle", "Carton", "Cup"]

# Classes needs to be passed to TacoDataset, so to only source images wich belong to those classes

dataset = TacoDataset()
dataset_size = dataset.__len__()
train_count = int(SPLIT_DISTRIBUTION.train[_version] * dataset_size)
test_count = int(SPLIT_DISTRIBUTION.test[_version] * dataset_size)
val_count = int(train_count * config.VALIDATION_SIZE)
train_count -= val_count


train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_count, test_count, val_count]
)


y_train_indices = train_dataset.indices
print(y_train_indices)

y_train = [dataset.targets[i] for i in y_train_indices]

class_sample_count = np.array(
    [len(np.where(y_train == t)[0]) for t in np.unique(y_train)]
)

weight = 1.0 / class_sample_count
samples_weight = np.array([weight[t] for t in y_train])
samples_weight = torch.from_numpy(samples_weight)
sampler = WeightedRandomSampler(
    samples_weight.type("torch.DoubleTensor"), len(samples_weight)
)

print("[INFO] Prepare labeldataloaders...")
trainloader = torch.utils.data.DataLoader(
    train_dataset, shuffle=True, num_workers=N_WORKERS, batch_size=BATCH_SIZE
)
