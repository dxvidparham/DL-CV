import torch
from collections import Counter

from IoU import intersection_over_union

def mean_average_precision(
    pred_boxes: list, true_boxes: list, iou_threshold=0.5, num_classes=28):
    """
    Evaluation of the detection network, in order to use it we need all the bounding_boxes predictions from all the test images 
    Funstion calculates the most common metric for evaluation - mAP
     
    Parameters:
        pred_boxes - all the predicitons boxes across all the training examples over the enitre test set,  list of lists containing all bounding_boxes with each bounding_boxes
            in following format [[train_idx, class_prediction, confidence score, x1, y1, x2, y2], [], [] ...] /// train_idx - what image does it come from

        true_boxes - same as pred_boxes just with ground truth bounding boxes

        iou_threshold - threshold where predicted bounding_boxes is correct

        num_classes -  number of classes
    Returns:
         mAP -  value across all classes given a specific IoU threshold 
    """

    """
    Precision: TP/TP+FP -> for all bboxes predictions what fraction is correct, 
    Recall: TP/TP+FN -> devide by the total number of ground truth bboxes, from all target boxes what fraction we did predict correctly

    Image idx -> confidence score (sorted in descending order) ->  TP or FP -> Precision & Recall -> plot (area under the graph
        is then our avarage precision for a certain class) -> sum of all class predicitons precisions/num of classes = mAP -> finally
        repeat all for different IoU tresholds and avarage it  ==> that is our final score

    """

    # list storing all avarage precision scores for a respective classes
    average_precisions = []

    # used for numerical stability later on ( smth that I read about)
    epsilon = 1e-6

    # we need to calculate this for each claass respectively 
    for c in range(num_classes): 
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:                        # detection[1] - that is class confidence score of a single predicted bounding box
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bounding_boxes for each training example - image
        # Counter here finds how many ground tr uth bounding_boxes we get 
        # for each training example, so let's say
        # img 0 has 3,
        # img 1 has 5 
        # then we will obtain a dictionary with:
        # amount_bounding_boxes = {0:3, 1:5} # image idx:number of ground truth boxes
        amount_bboxes = Counter([gt[0] for gt in ground_truths]) # [expression for item in iterable if condition == True]

        # We go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2 (in pred_boxes list)
        detections.sort(key=lambda x: x[2], reverse=True) 
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection so they belong to the same image
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                # we are sending only the detected boxes coordinates to the iou calculating function
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:])
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            # after this step: we have taken one particular bbox for particular class in single image and we have taken all the gt boxes for that image
            # we have checked iou between that box and all gt in that image, kept the best iou   

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1 # if the box was already covered

            # if IOU is lower then the detection is a false positive
            # means that this detection was not TP as IOU was below trsh
            else:
                FP[detection_idx] = 1

        # for calculating precision and recall; [1, 0, 1, 1, 0] -> positice, negative etc detection; cumsum = [1, 1, 2, 3, 3]
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)

        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        # torch.trapz for numerical integration - we get the area under the recall/precision points
        average_precisions.append(torch.trapz(precisions, recalls)) # precision - y, recalls - x

    return sum(average_precisions) / len(average_precisions)


"""
We need to call this function for multiple IOU treshold values, store the results in a list and then avarage them
"""