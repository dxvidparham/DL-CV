import os
import numpy as np
import PIL.Image as Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.sampler import WeightedRandomSampler

import matplotlib.pyplot as plt

from pycocotools.coco import COCO

from omegaconf import OmegaConf

from IoU import intersection_over_union as IoU
from utils import  generate_targets



# ======================================================
#%% Proposal Set

class ProposalDataset(torch.utils.data.Dataset):
    def __init__(self, img=None, img_annotations=None, size=None):

        assert img is not None, "image for proposals"
        assert img_annotations is not None, "annotations of proposals"
        assert size is not None, "put image size"

        self.img_annotations = img_annotations
        self.img = img
        self.size = size

        DIR= os.path.dirname(os.path.realpath(__file__))

        #--------------------------
        # load proposals
        #--------------------------
        for jpg in ["jpg","JPG"]:
            if jpg in self.img_annotations["path"]:
                prop_name = self.img_annotations["path"].replace(jpg,"txt")
            # else:
            #     print("failed")
            #     print(self.img_annotations["path"])

     

        annotation_path = f"{DIR}/project2_ds_new/"+prop_name
        # print(annotation_path)
        
        prop_anno_list = []
        with open(annotation_path) as f:
            proposal_list = f.readlines()
            for box_str in proposal_list:
                prop_box = box_str.split()
                prop_box = [int(i) for i in prop_box]
                prop_anno_list.append(prop_box)

        # print("proposals")
        # print(prop_anno_list)

        # crop and resize
        prop_img_list = []
        for prop in prop_anno_list:
            prop_img_list.append(self.transform_img(prop))

        # for i in range(len(prop_img_list)):
        #     print(prop_img_list[i].shape)
    
        

        #------------------------------------
        # Compute proposal labels
        #------------------------------------
        
        # Get groud truth
        bboxes_gt = self.img_annotations["boxes"]
        labels_gt = self.img_annotations["labels"]

        prop_labels_list = generate_targets(prop_anno_list, labels_gt, bboxes_gt)


        # background count
        self.class_sample_count = np.array([self.listCount(prop_labels_list, 0)])
        # class counts
        cnt = 0
        for l in labels_gt:
            cnt = cnt + self.listCount(prop_labels_list, l)
            
        self.class_sample_count = np.append(self.class_sample_count, cnt)

        self.target = prop_labels_list
        self.prop_img_list = prop_img_list
        self.prop_bboxes_list = prop_anno_list
       

    
    def transform_img(self, prop: list):
        prop_crop = transforms.functional.crop(self.img, prop[0],prop[1],prop[3]-prop[1],prop[2]-prop[0]) #TODO: be sure it works: top, left, height, width [xmin, ymin, xmax, ymax]
        resized = transforms.functional.resize(prop_crop,(self.size, self.size))
        return resized

    def listCount(self, lst, x):
        count = 0
        for ele in lst:
            if (ele == x):
                count = count + 1
        return count

    def crop_img(self, img, ymin, ymax, xmin, xmax):
        return img[:, ymin:ymax, xmin:xmax]

    def __getitem__(self, index):

        # Image ID
        img_id = self.img_annotations["image_id"]
        
        # open the input image
        img = self.prop_img_list[index]
        
        boxes = self.prop_bboxes_list[index]

        # Labels (In my case, I only one class: target class or background)
        labels = self.target[index]

        # Tensorise img_id
        img_id = torch.tensor([img_id])
        
        # Annotation is in dictionary format
        my_annotation = {
            "boxes": boxes,
            "labels": labels,
            "image_id": img_id,   
        }

        return img, my_annotation

    def __len__(self):
        return len(self.prop_bboxes_list)



# =======================================
#%% test prop loader
if __name__ == "__main__":

    root = "/dtu/datasets1/02514/data_wastedetection/"
    annotation = f"{root}/annotations.json"
    coco = COCO(annotation)
    ids = list(sorted(coco.imgs.keys()))

    img_id = ids[0]

    # path for input image
    path = coco.loadImgs(img_id)[0]["file_name"]

    # open the input image
    img = Image.open(os.path.join(root, path))

    transform = transforms.Compose(
                [
                    # transforms.Resize((size, size)),
                    transforms.ToTensor(),
                ]
            )
    img = transform(img)


    ann_ids = coco.getAnnIds(imgIds=img_id)
    # Dictionary: target coco_annotation file for an image
    coco_annotation = coco.loadAnns(ann_ids)
    # print(coco_annotation)

    # path for input image
    path = coco.loadImgs(img_id)[0]["file_name"]

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
        "path": path,
    }

    # ---------------------------------

    propset = ProposalDataset(img,my_annotation,250)

    weight = 1. / propset.class_sample_count

    target = propset.target
    # target = torch.from_numpy(target).long()


    samples_weight = np.array([weight[t] for t in target])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weigth = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # target = torch.from_numpy(target).long()
    # train_dataset = torch.utils.data.TensorDataset(data, target)


    proploader = torch.utils.data.DataLoader(
        propset, shuffle=False, batch_size=16, sampler=sampler,
    )

    iterprop = iter(proploader)

    a,b = next(iterprop)

