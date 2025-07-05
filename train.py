import os
import numpy as np
import config
import torch
import torch.optim as optim
import sys
import wandb
import itertools
import csv

from model import YOLOv3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    get_test_loader,
    plot_examples
)
from loss import YoloLoss
# import warnings
# warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True


def main():#prediction
    model = YOLOv3(num_classes=1).to(config.DEVICE)
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)


    test_loader = get_test_loader(test_csv_path = config.DATASET + "/test.csv")

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )
    
    score, class_accuracy, noobj_accuracy, obj_accuracy = check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
    pred_boxes, true_boxes = get_evaluation_bboxes(
        test_loader,
        model,
        iou_threshold=config.NMS_IOU_THRESH,
        anchors=scaled_anchors,
        threshold=config.CONF_THRESHOLD,
    )

    map_val, precisions, recalls = mean_average_precision(
        pred_boxes,
        true_boxes,
        iou_threshold=config.MAP_IOU_THRESH,
        box_format="midpoint",
        num_classes=config.NUM_CLASSES,
    )

    if not os.path.exists('output/pred6'):
        # Create directory recursively
        os.makedirs('output/pred6')

    # plot_examples(model, test_loader, 0.25, 0.6, scaled_anchors)#model, loader, thresh, iou_thresh_nms, anchors
    plot_examples(model, test_loader, config.CONF_THRESHOLD, config.NMS_IOU_THRESH, scaled_anchors, f'./output/pred6/images_{1}')#model, loader, thresh, iou_thresh_nms, anchors
    print(f'map: {map_val}, obj_accuracy: {obj_accuracy}, noobj_accuracy: {noobj_accuracy}')



if __name__ == "__main__":
    main()