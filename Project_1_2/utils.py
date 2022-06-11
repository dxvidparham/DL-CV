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


if __name__ == "__main__":
    dataset_path = "/dtu/datasets1/02514/data_wastedetection/"
    anns_file_path = f"{dataset_path}/annotations.json"

    # Loads dataset as a coco object
    coco = COCO(anns_file_path)
    get_category_image_ids(coco, "Bottle")
