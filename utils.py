import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch
import time
import math
import sys

from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm


def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """

    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, threshold, iou_threshold, box_format="midpoint"):
    """

    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert isinstance(bboxes, list)

    bboxes = np.array([box for box in bboxes if box[1] > threshold])
    if len(bboxes) == 0:
        return []

    bboxes = bboxes[bboxes[:, 1].argsort()[::-1]]  # Sort by confidence score
    bboxes_after_nms = []

    while len(bboxes) > 0:
        chosen_box = bboxes[0]
        bboxes = bboxes[1:]

        iou_scores = intersection_over_union(torch.tensor(chosen_box[2:]), torch.tensor(bboxes)[:, 2:], box_format)
        suppress_indices = np.where(iou_scores > iou_threshold)[0]

        bboxes = np.delete(bboxes, suppress_indices, axis=0)
        bboxes_after_nms.append(chosen_box.tolist())

    # print(intersection_over_union(torch.tensor(bboxes_after_nms[0][2:]), torch.tensor(bboxes_after_nms[1:][2:]), box_format))
    # print(bboxes_after_nms[:4])

    return bboxes_after_nms


def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=1 
):
    """

    This function calculates mean average precision (mAP)

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """


    # used for numerical stability later on
    epsilon = 1e-6

    detections = []
    ground_truths = []

    # Go through all predictions and targets,
    # and only add the ones that belong to the
    # current class c
    for detection in pred_boxes:
        if detection[1] == 0:
            detections.append(detection)

    for true_box in true_boxes:
        if true_box[1] == 0:
            ground_truths.append(true_box)

    # find the amount of bboxes for each training example
    # Counter here finds how many ground truth bboxes we get
    # for each training example, so let's say img 0 has 3,
    # img 1 has 5 then we will obtain a dictionary with:
    # amount_bboxes = {0:3, 1:5}
    amount_bboxes = Counter([gt[0] for gt in ground_truths])

    # We then go through each key, val in this dictionary
    # and convert to the following (w.r.t same example):
    # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
    for key, val in amount_bboxes.items():
        amount_bboxes[key] = torch.zeros(val)

    # sort by box probabilities which is index 2
    detections.sort(key=lambda x: x[2], reverse=True)
    TP = torch.zeros((len(detections)))
    FP = torch.zeros((len(detections)))
    total_true_bboxes = len(ground_truths)

    for detection_idx, detection in enumerate(detections):
        # Only take out the ground_truths that have the same
        # training idx as detection
        ground_truth_img = [
            bbox for bbox in ground_truths if bbox[0] == detection[0]
        ]

        num_gts = len(ground_truth_img)
        best_iou = 0

        for idx, gt in enumerate(ground_truth_img):
            iou = intersection_over_union(
                torch.tensor(detection[3:]),
                torch.tensor(gt[3:]),
                box_format=box_format,
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou > iou_threshold:
            # only detect ground truth detection once
            if amount_bboxes[detection[0]][best_gt_idx] == 0:
                # true positive and add this bounding box to seen
                TP[detection_idx] = 1
                amount_bboxes[detection[0]][best_gt_idx] = 1
            else:
                FP[detection_idx] = 1

        # if IOU is lower then the detection is a false positive
        else:
            FP[detection_idx] = 1

    TP_cumsum = torch.cumsum(TP, dim=0)
    FP_cumsum = torch.cumsum(FP, dim=0)
    recalls = TP_cumsum / (total_true_bboxes + epsilon)
    precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
    precisions = torch.cat((torch.tensor([1]), precisions))
    recalls = torch.cat((torch.tensor([0]), recalls))
    
    # Append precision and recall values to lists
    # precisions.append(precisions.numpy())
    # recalls.append(recalls.numpy())

    # torch.trapz for numerical integration
    AP = torch.trapz(precisions, recalls)

    return AP, precisions, recalls


def plot_single_example(image, boxes, save_path):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Disable axis and grid
    ax.axis('off')
    ax.grid(False)

    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor='g',
            facecolor="none",
        )

        # class_labels = ['person']

        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s='person',
            color="white",
            verticalalignment="top",
            bbox={"color": 'g', "pad": 0},
        )


    # Save the figure
    plt.savefig(save_path)

    # Close the figure to release memory
    plt.close()


def get_evaluation_bboxes(
    loader,
    model,
    iou_threshold,
    anchors,
    threshold,
    box_format="midpoint",
    device="cuda",
):
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        x = x.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            S = predictions[i].shape[2]
            # anchor = torch.tensor([*anchors[i]]).to(device) * S
            anchor = anchors[i]
            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # we just want one bbox for each label, not one for each scale
        true_bboxes = cells_to_bboxes(
            labels[2], anchor, S=S, is_preds=False
        )

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                threshold=threshold,
                iou_threshold=iou_threshold,               
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    # model.train()
    # print(all_pred_boxes[:4])
    # print(all_true_boxes[:4])
    return all_pred_boxes, all_true_boxes


def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.
    INPUT:
    predictions: tensor of size (N, 3, S, S, num_classes+5)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return converted_bboxes.tolist()


def check_class_accuracy(model, loader, threshold):
    model.eval()
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.to(config.DEVICE)
        with torch.no_grad():
            out = model(x)

        for i in range(3):
            y[i] = y[i].to(config.DEVICE)
            obj = y[i][..., 0] == 1 # in paper this is Iobj_i
            noobj = y[i][..., 0] == 0  # in paper this is Iobj_i

            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

    
    # print(f'correct class: {correct_class}, correct obj: {correct_obj}, noobj: {correct_noobj}')
    # print(f'total obj: {tot_obj}, total noobj: {tot_noobj}, noobj: {correct_noobj}')

    # Print accuracy metrics
    class_accuracy = (correct_class / (tot_obj+ 1e-16)) * 100
    noobj_accuracy = (correct_noobj / (tot_noobj + 1e-16)) * 100
    obj_accuracy = (correct_obj / (tot_obj + 1e-16)) * 100
    # print(f"Class accuracy is: {class_accuracy:.2f}%")
    # print(f"No obj accuracy is: {noobj_accuracy:.2f}%")
    # print(f"Obj accuracy is: {obj_accuracy:.2f}%")
    score = (20 * math.acos(noobj_accuracy / 100) + 
            20 * math.acos(obj_accuracy / 100) + 
            class_accuracy / 10)

    model.train()
    return score, class_accuracy, noobj_accuracy, obj_accuracy


'''
def check_class_accuracy(model, loader, threshold):
    model.eval()
    correct_class = 0
    correct_noobj = 0
    correct_obj = 0
    tot_noobj = 0
    tot_obj = 0

    for idx, (x, y) in enumerate(loader):
        x = x.to(config.DEVICE)
        with torch.no_grad():
            out = model(x)

        for i in range(3):
            y[i] = y[i].to(config.DEVICE)
            # print(y[i].shape)
            obj = y[i][..., 0] == 1
            noobj = y[i][..., 0] == 0
            # print(obj.shape)

            # Compute objectness score and apply threshold
            obj_preds = torch.sigmoid(out[i][..., 0])  # Normalize objectness score
            obj_preds = obj_preds > threshold

            # Compute accuracy for "no object" class
            correct_noobj += torch.sum(obj_preds[noobj] == 0)  # Predicted as "no object"
            tot_noobj += torch.sum(noobj)

            # Compute accuracy for "object" class
            correct_obj += torch.sum(obj_preds[obj] == 1)  # Predicted as "object"
            tot_obj += torch.sum(obj)

            # Compute class accuracy (as there's only one class)
            correct_class += torch.sum(obj_preds[obj] == 1)  # Predicted as "object"

    print(f'correct class: {correct_class}, correct obj: {correct_obj}, noobj: {correct_noobj}')
    print(f'total obj: {tot_obj}, total noobj: {tot_noobj}, noobj: {correct_noobj}')

    # Print accuracy metrics
    class_accuracy = (correct_class / (tot_obj+ 1e-16)) * 100
    noobj_accuracy = (correct_noobj / (tot_noobj + 1e-16)) * 100
    obj_accuracy = (correct_obj / (tot_obj + 1e-16)) * 100
    # print(f"Class accuracy is: {class_accuracy:.2f}%")
    # print(f"No obj accuracy is: {noobj_accuracy:.2f}%")
    # print(f"Obj accuracy is: {obj_accuracy:.2f}%")
    score = (20 * math.acos(noobj_accuracy / 100) + 
            20 * math.acos(obj_accuracy / 100) + 
            class_accuracy / 10)

    # model.train()
    return score, class_accuracy, noobj_accuracy, obj_accuracy
'''

def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer"])

    # # If we don't do this then it will just have learning rate of old checkpoint
    # # and it will lead to many hours of debugging \:
    # for param_group in optimizer.param_groups:
    #     param_group["lr"] = lr


def get_test_loader(test_csv_path):
    from dataset import YOLODataset

    IMAGE_SIZE = config.IMAGE_SIZE

    test_dataset = YOLODataset(
        test_csv_path,
        transform=config.test_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        data_dir=config.TEST_DIR,
        anchors=config.ANCHORS,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    return test_loader

def get_loaders(train_csv_path, val_csv_path):
    from dataset import YOLODataset

    IMAGE_SIZE = config.IMAGE_SIZE
    train_dataset = YOLODataset(
        train_csv_path,
        transform=config.train_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        data_dir=config.TRAIN_DIR,
        anchors=config.ANCHORS,
    )
    val_dataset = YOLODataset(
        val_csv_path,
        transform=config.test_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        data_dir=config.VAL_DIR,
        anchors=config.ANCHORS,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    train_eval_dataset = YOLODataset(
        train_csv_path,
        transform=config.train2_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        data_dir=config.TRAIN_DIR,
        anchors=config.ANCHORS,
    )
    train_eval_loader = DataLoader(
        dataset=train_eval_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader, train_eval_loader

def plot_examples(model, loader, thresh, iou_thresh, anchors, path):
    model.eval()
    x, y = next(iter(loader))
    x = x.to("cuda")
    with torch.no_grad():
        out = model(x)
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = anchors[i]
            boxes_scale_i = cells_to_bboxes(
                out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        model.train()

    for i in range(batch_size):
        # start_time = time.time()
        nms_boxes = non_max_suppression(
            bboxes[i], threshold=thresh, iou_threshold=iou_thresh, box_format="midpoint",#bboxes, iou_threshold, threshold, box_format="midpoint"
        )
        # end_time = time.time()
        # nms_execution_time = end_time - start_time
        # print("NMS Execution time:", nms_execution_time)
        if not os.path.exists(path):
            os.makedirs(path)

        plot_single_example(x[i].permute(1,2,0).detach().cpu(), nms_boxes, os.path.join(path, f'image_{i}.png'))



def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False