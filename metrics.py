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
# Module: This module contain a class with all the common metrics
# for semantic segmentation
######################################################################

import torch
import numpy as np
from torchmetrics.functional import accuracy, jaccard_index
import torch


class SegmentationMetric(object):
    """Computes pixAcc and mIoU metric scores"""

    def __init__(self, nclass=2):
        super(SegmentationMetric, self).__init__()
        self.nclass = nclass
        self.reset()
        self.IoU = []
        self.pixAcc = []

    def update(self, output, labels):
        output = torch.sigmoid(output)
        labels = labels.int()

        acc_score = accuracy(output, labels).cpu()
        self.pixAcc.append(acc_score)

        iou_score = jaccard_index(output, labels, num_classes=self.nclass).cpu()
        self.IoU.append(iou_score)

    def get(self):
        return np.array(self.IoU), np.array(self.pixAcc)

    def reset(self):
        self.IoU = []
        self.pixAcc = []

    def compare_size(output, target):
        if torch.is_tensor(output):
            pred = torch.sigmoid(output).data.cpu().numpy()
        if torch.is_tensor(target):
            target = target.data.cpu().numpy()
        output_ = pred > 0.5
        target_ = target > 0.5

        sum_out = output_.sum()
        sum_target = target_.sum()

        return torch.tensor(sum_out), torch.tensor(sum_target)


if __name__ == "__main__":
    pass
