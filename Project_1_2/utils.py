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

import sys
import json
from pycocotools.coco import COCO
import random
import torch


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


def load_categories():

    # Read annotations
    with open(anns_file_path, "r") as f:
        dataset = json.loads(f.read())

    categories = dataset["categories"]
    anns = dataset["annotations"]
    imgs = dataset["images"]
    nr_cats = len(categories)
    nr_annotations = len(anns)
    nr_images = len(imgs)

    # Load categories and super categories
    cat_names = []
    super_cat_names = []
    super_cat_ids = {}
    super_cat_last_name = ""
    nr_super_cats = 0
    for cat_it in categories:
        cat_names.append(cat_it["name"])
        super_cat_name = cat_it["supercategory"]
        # Adding new supercat
        if super_cat_name != super_cat_last_name:
            super_cat_names.append(super_cat_name)
            super_cat_ids[super_cat_name] = nr_super_cats
            super_cat_last_name = super_cat_name
            nr_super_cats += 1

    print("Number of super categories:", nr_super_cats)
    print("Number of categories:", nr_cats)
    print("Number of annotations:", nr_annotations)
    print("Number of images:", nr_images)


def prepare_dataset():
    # Loads dataset as a coco object
    coco = COCO(anns_file_path)

    category = "Bottle"
    # Get image ids
    imgIds = []
    if catIds := coco.getCatIds(catNms=[category]):
        # Get all images containing an instance of the chosen category
        imgIds = coco.getImgIds(catIds=catIds)
    else:
        # Get all images containing an instance of the chosen super category
        catIds = coco.getCatIds(supNms=[category])
        for catId in catIds:
            imgIds += coco.getImgIds(catIds=catId)
        imgIds = list(set(imgIds))

    nr_images_found = len(imgIds)
    print("Number of images found: ", nr_images_found)

    # Select N random images
    nr_img_2_display = 10
    random.shuffle(imgIds)
    imgs = coco.loadImgs(imgIds[: min(nr_img_2_display, nr_images_found)])

    for img in imgs:
        image_path = dataset_path + "/" + img["file_name"]



if __name__ == "__main__":
    dataset_path = "/dtu/datasets1/02514/data_wastedetection/"
    anns_file_path = f"{dataset_path}annotations.json"
    load_categories()
    prepare_dataset()
