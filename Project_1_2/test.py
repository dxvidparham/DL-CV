# import json
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set()
# from PIL import Image, ExifTags
from pycocotools.coco import COCO

# from matplotlib.patches import Polygon, Rectangle
# from matplotlib.collections import PatchCollection
# import colorsys
# import random
# import pylab

# dataset_path = "./data"
# anns_file_path = dataset_path + "/" + "annotations.json"

# # Read annotations
# with open(anns_file_path, "r") as f:
#     dataset = json.loads(f.read())

# categories = dataset["categories"]
# anns = dataset["annotations"]
# imgs = dataset["images"]
# nr_cats = len(categories)
# nr_annotations = len(anns)
# nr_images = len(imgs)

# # Load categories and super categories
# cat_names = []
# super_cat_names = []
# super_cat_ids = {}
# super_cat_last_name = ""
# nr_super_cats = 0
# for cat_it in categories:
#     cat_names.append(cat_it["name"])
#     super_cat_name = cat_it["supercategory"]
#     # Adding new supercat
#     if super_cat_name != super_cat_last_name:
#         super_cat_names.append(super_cat_name)
#         super_cat_ids[super_cat_name] = nr_super_cats
#         super_cat_last_name = super_cat_name
#         nr_super_cats += 1

# print("Number of super categories:", nr_super_cats)
# print("Number of categories:", nr_cats)
# print("Number of annotations:", nr_annotations)
# print("Number of images:", nr_images)

# # User settings
# nr_img_2_display = 10
# category_name = (
#     "Bottle"  #  --- Insert the name of one of the categories or super-categories above
# )
# pylab.rcParams["figure.figsize"] = (14, 14)
# ####################

# # Obtain Exif orientation tag code
# for orientation in ExifTags.TAGS.keys():
#     if ExifTags.TAGS[orientation] == "Orientation":
#         break

# # Loads dataset as a coco object
# coco = COCO(anns_file_path)

# # Get image ids
# imgIds = []
# catIds = coco.getCatIds(catNms=[category_name])
# if catIds:
#     # Get all images containing an instance of the chosen category
#     imgIds = coco.getImgIds(catIds=catIds)
# else:
#     # Get all images containing an instance of the chosen super category
#     catIds = coco.getCatIds(supNms=[category_name])
#     for catId in catIds:
#         imgIds += coco.getImgIds(catIds=catId)
#     imgIds = list(set(imgIds))

# nr_images_found = len(imgIds)
# print("Number of images found: ", nr_images_found)

# # Select N random images
# random.shuffle(imgIds)
# imgs = coco.loadImgs(imgIds[0 : min(nr_img_2_display, nr_images_found)])

# for img in imgs:
#     image_path = dataset_path + "/" + img["file_name"]
#     # Load image
#     I = Image.open(image_path)


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

def get_annotations():



if __name__ == "__main__":
    dataset_path = "/dtu/datasets1/02514/data_wastedetection/"
    anns_file_path = f"{dataset_path}/annotations.json"

    # Loads dataset as a coco object
    coco = COCO(anns_file_path)
    get_category_image_ids(coco, "Bottle")
