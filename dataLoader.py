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
# Module: Create custom dataset for taco dataset
######################################################################

import os

import PIL.Image as Image
import torch
import torch.utils.data
from omegaconf import OmegaConf
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms

from utils import collate_wrapper, get_super_categories


class TacoDataset(torch.utils.data.Dataset):
    def __init__(self, augment=False):
        self.root = "/dtu/datasets1/02514/data_wastedetection/"
        self.annotation = f"{self.root}/annotations.json"
        self.coco = COCO(self.annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.size = 224
        self.classes, self.remap = get_super_categories(self.coco)

        self.augment = augment
        if self.augment:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((self.size, self.size)),
                    transforms.ToTensor(),
                    # transforms.Lambda(lambda x: self._print_img_size(x)),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    # transforms.Resize((self.size, self.size)),
                    transforms.ToTensor(),
                    # transforms.Lambda(lambda x: self._print_img_size(x)),
                ]
            )

    def crop_img(self, img, ymin, ymax, xmin, xmax):
        return img[:, ymin:ymax, xmin:xmax]

    def _print_img_size(self, img):
        print(img.size())
        return img

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)

        # path for input image
        path = coco.loadImgs(img_id)[0]["file_name"]
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        iscrowd = []
        areas = []
        for i, ann in enumerate(coco_annotation):
            if ann["category_id"] in self.classes.values():
                xmin = coco_annotation[i]["bbox"][0]
                ymin = coco_annotation[i]["bbox"][1]
                xmax = xmin + coco_annotation[i]["bbox"][2]
                ymax = ymin + coco_annotation[i]["bbox"][3]

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.remap.get(ann["category_id"]))
                iscrowd.append(ann["iscrowd"])
                areas.append(ann["area"])

        img_id = torch.tensor([img_id])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {
            "boxes": boxes,
            "labels": labels,
            "image_id": img_id,
            "area": areas,
            "iscrowd": iscrowd,
            "path": path
        }

        if self.transform is not None:
            img = self.transform(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)


if __name__ == "__main__":
    #%% load config
    DIR= os.path.dirname(os.path.realpath(__file__))
    config = OmegaConf.load(f"{DIR}/config/config.yaml")

    print("[INFO] Load datasets from disk...")
    dataset = TacoDataset(augment=False)

    print("[INFO] Prepare dataloaders...")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        num_workers=config.N_WORKERS,
        batch_size=config.BATCH_SIZE,
        collate_fn=collate_wrapper,
    )

    dataloader_iter = iter(dataloader)
    x, y = next(dataloader_iter)
    print(x[0], y[0])
