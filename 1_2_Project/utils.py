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

from pycocotools.coco import COCO
import torch
from IoU import intersection_over_union as IoU
import torchvision.transforms as transforms


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


def get_category_image_ids(coco_obj, category):
    img_ids = []
    if cat_ids := coco_obj.getCatIds(catNms=[category]):
        # Get all images containing an instance of the chosen category
        img_ids = coco_obj.getImgIds(catIds=cat_ids)
    else:
        # Get all images containing an instance of the chosen super category
        cat_ids = coco_obj.getCatIds(supNms=[category])
        for cat_id in cat_ids:
            img_ids += coco_obj.getImgIds(catIds=cat_id)
        img_ids = list(set(img_ids))

    # print(f"[INFO]: Class -> {category}, contains {len(img_ids)} images")
    return img_ids, cat_ids


def get_super_categories(coco_obj):
    coco = coco_obj
    super_cat_ids = {}
    super_cat_last_name = ""
    nr_super_cats = 0
    remap = {}
    for _, cat_it in coco.cats.items():
        super_cat_name = cat_it["supercategory"]

        if super_cat_name != super_cat_last_name:
            super_cat_ids[super_cat_name] = cat_it["id"]
            super_cat_last_name = super_cat_name
            nr_super_cats += 1
            remap[cat_it["id"]] = nr_super_cats

    return super_cat_ids, remap


# a simple custom collate function, just to show the idea
def collate_wrapper(batch):
    data = [item[0] for item in batch]
    my_annotation = [item[1] for item in batch]
    # target = torch.LongTensor(target)

    return [data, my_annotation]

def generate_targets(prop_anno_list, labels_gt, bboxes_gt):
    background_threshold = 0.5

    prop_labels_list = []
    for prop_anno in prop_anno_list:
        prop_label = torch.tensor(0)
        prop_anno = torch.tensor(prop_anno)
        for gt_idx, bbox_gt in enumerate(bboxes_gt):
            
            if IoU(prop_anno, bbox_gt)>background_threshold:
                prop_label = labels_gt[gt_idx]
    
        
        prop_labels_list.append(prop_label)

    return prop_labels_list


if __name__ == "__main__":
    dataset_path = "/dtu/datasets1/02514/data_wastedetection/"
    anns_file_path = f"{dataset_path}/annotations.json"

    # Loads dataset as a coco object
    coco = COCO(anns_file_path)
    # get_category_image_ids(coco, "Bottle")
    print(get_super_categories(coco))
