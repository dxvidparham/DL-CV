import torch
from IoU import intersection_over_union

def nms(bounding_boxes: list, iou_threshold: float, threshold: float):
    """
    NSM ahs to be done on each class separatly 
    bboxes - each elements of a list of lists [[class, probability of that clas aka confidence score, coordinates fo the box], [], []] all of the same class
        list of lists containing all bboxes with each bboxes specified as [class_pred, prob_score, x1, y1, x2, y2]
    iou_threshold - value above which boxes are removed
    threshold - some additional trsh whne we remove boxes independetly of IoU
    RETURN -  bounding_boxes after performing NMS given a specific IoU threshold (list)
    """

    assert type(bounding_boxes) == list

    bounding_boxes = [box for box in bounding_boxes if box[1] > threshold] # only keep those that are higher than some probability trsh
    bounding_boxes = sorted(bounding_boxes, key=lambda x: x[1], reverse=True) # sorting the boxes from the highest probability
    bounding_boxes_after_nms = []

    while bounding_boxes:
        chosen_box = bounding_boxes.pop(0)

        bounding_boxes = [
            box for box in bounding_boxes
            if box[0] != chosen_box[0] # if its not equal to the class of the chosen box we want to keep that box
            or intersection_over_union(
                torch.tensor(chosen_box[2:]), # we take elements of box tensor from the second element -> box coordinates
                torch.tensor(box[2:])
            )
            < iou_threshold # if its lower that trsh we want to keep it
        ]

        bounding_boxes_after_nms.append(chosen_box)

    return bounding_boxes_after_nms