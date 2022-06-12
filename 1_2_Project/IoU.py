import torch

def intersection_over_union(boxes_preds: torch.tensor, boxes_labels: torch.tensor):
    """
    Function for calculating intersection over union, takes proporsals/predictions of bounding boxes (box_pred) and the ground truth/other boxes (boxes_labels)
    to compute their IoU
    Box1 = [x1, y1, x2, y2]
    Box2 = [x1, y1, x2, y2]

    """
    box1_x1 = boxes_preds[..., 0:1]
    box1_y1 = boxes_preds[..., 1:2]
    box1_x2 = boxes_preds[..., 2:3]
    box1_y2 = boxes_preds[..., 3:4]
    # slicing tensro to maintain its shape

    box2_x1 = boxes_labels[..., 0:1]
    box2_y1 = boxes_labels[..., 1:2]
    box2_x2 = boxes_labels[..., 2:3]
    box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1) # maximum x coordinate value of two boxes
    y1 = torch.max(box1_y1, box2_y1) # maximum y coordinate value of two boxes
                                    # both gave us a left top corner of the union
    x2 = torch.min(box1_x2, box2_x2) # minimum x coordinate value of two boxes
    y2 = torch.min(box1_y2, box2_y2) # minimum x coordinate value of two boxes
                                    # both gave us the right bottom corner of the union

    # clamp(0) if they do not intersect - then intersection = 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection ) # + 1e-6 -> numerical stability