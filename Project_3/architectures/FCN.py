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
# Module: Implementation of a Fully-Connected Network  architecture
######################################################################

import torch
import torch.nn as nn


class FCN(nn.Module):
    def __init__(self, pretrained_net, output_channels=1):
        super().__init__()
        self.n_class = output_channels
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(
            512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1
        )
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(
            512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1
        )
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(
            256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1
        )
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1
        )
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, output_channels=1, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output["x5"]  # size=(N, 512, x.H/32, x.W/32)

        score = self.bn1(self.relu(self.deconv1(x5)))  # size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)  # size=(N,output_channels=1, x.H/1, x.W/1)

        return score


if __name__ == "__main__":

    # appending parrent directory so that we can import ISICDataset
    import sys

    sys.path.append("../")

    from dataLoader import ISICDataset

    print("[INFO] Load datasets from disk...")
    dataset = ISICDataset(transform=None)

    print("[INFO] Prepare dataloaders...")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        num_workers=4,
        batch_size=32,
    )

    dataloader_iter = iter(dataloader)
    x, y = next(dataloader_iter)

    print("[INFO] Test forward pass for model FCN...")
    model = FCN(num_classes=3)
    out = model
