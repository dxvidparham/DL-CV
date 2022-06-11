from pycocotools.coco import COCO


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
