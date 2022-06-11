import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO
import glob
import os

import albumentations
import albumentations.pytorch
import numpy as np
import PIL.Image as Image
import torch
from omegaconf import OmegaConf
from torchvision import transforms


class TacoDataset(torch.utils.data.Dataset):
    def __init__(self, augment=False):
        self.root = "/dtu/datasets1/02514/data_wastedetection/"
        self.annotation = f"{self.root}/annotations.json"
        self.coco = COCO(self.annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.augment = augment
        if self.augment:
            self.transform = transforms.Compose(
                [
                    # transforms.Resize((size, size)),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    # transforms.Resize((size, size)),
                    transforms.ToTensor(),
                ]
            )

    def crop_img(self, img, ymin, ymax, xmin, xmax):
        return img[:, ymin:ymax, xmin:xmax]

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        print(coco_annotation)
        # path for input image
        path = coco.loadImgs(img_id)[0]["file_name"]
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]["bbox"][0]
            ymin = coco_annotation[i]["bbox"][1]
            xmax = xmin + coco_annotation[i]["bbox"][2]
            ymax = ymin + coco_annotation[i]["bbox"][3]
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)

        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = [coco_annotation[i]["area"] for i in range(num_objs)]
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {
            "boxes": boxes,
            "labels": labels,
            "image_id": img_id,
            "area": areas,
            "iscrowd": iscrowd,
        }

        if self.transform is not None:
            img = self.transform(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)


if __name__ == "__main__":
    # Load config file
    BASE_DIR = os.getcwd()
    config = OmegaConf.load(f"{BASE_DIR}/config/config.yaml")

    print("[INFO] Load datasets from disk...")
    dataset = TacoDataset()

    print("[INFO] Prepare labeldataloaders...")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        num_workers=config.N_WORKERS
        batch_size=config.BATCH_SIZE
    )

    dataloader_iter = iter(dataloader)
    x, y = next(dataloader_iter)
