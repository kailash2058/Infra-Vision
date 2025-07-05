import albumentations as A
import cv2
import torch

from albumentations.pytorch import ToTensorV2
from utils import seed_everything

DATASET = 'final_thermal_dataset_yolov3'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# seed_everything()  # If you want deterministic behavior
NUM_WORKERS = 4
BATCH_SIZE = 10
IMAGE_SIZE = 416
NUM_CLASSES = 1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 2000
CONF_THRESHOLD = 0.05#0.05
MAP_IOU_THRESH = 0.5#0.5
NMS_IOU_THRESH = 0.6 #0.45, 0.6
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_FILE_PRED = "best_model2.pth.tar"
CHECKPOINT_FILE = "best_model.pth.tar"
CHECKPOINT2_FILE = "best_model2.pth.tar"
CHECKPOINT3_FILE = "best_model_2000.pth.tar"
VOC_CHECKPOINT_FILE = "yolov3_pascal_78.1map.pth.tar"
TL_CHECKPOINT_FILE = 'tl.pth.tar'
TL_CHECKPOINT2_FILE = 'tl2.pth.tar'
TRAIN_DIR = DATASET + "/train/"
TEST_DIR = DATASET + "/test/"
VAL_DIR = DATASET + "/valid"
SAVE_IMAGE_PATH = './output/images2'

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Note these have been rescaled to be between [0, 1]


scale = 1.1
train_transforms = A.Compose(
    [
        A.Resize(width=412, height=412),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)

train2_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),  # Add value parameter to specify the constant border value
            always_apply=True,  # Ensures padding is always applied
        ),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE, always_apply=True, p=1.0),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        A.Rotate(limit=[90, 270], p=1),  # Randomly rotate by 90 or 270 degrees
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)


test_transforms = A.Compose(
    [
        A.Resize(width=412, height=412),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

