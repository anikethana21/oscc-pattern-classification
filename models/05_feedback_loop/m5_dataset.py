import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
m5_dataset.py
-------------
Dataset for Model 5 (Looping Network).
Returns:
- crops_224: 16 crops for ViT Phikon [16, 3, 224, 224]
- img_256: 256x256 image for CNN branch [3, 256, 256]
- label: 4-class integer label [0: cohesive, 1: dispersive, 2: no_feature, 3: noncohesive]
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLASS_NAMES = ["cohesive", "dispersive", "no_feature", "noncohesive"]
LABEL_ENC   = {n: i for i, n in enumerate(CLASS_NAMES)}


def get_heavy_aug():
    return A.Compose([
        A.RandomRotate90(p=1.0),
        A.Rotate(limit=20, p=0.6),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomResizedCrop(size=(512, 512), scale=(0.75, 1.0),
                            ratio=(0.9, 1.1), p=0.5),
        A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=25,
                             val_shift_limit=18, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.18,
                                   contrast_limit=0.15, p=0.6),
        A.GaussNoise(p=0.25),
        A.Resize(height=512, width=512),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_light_aug():
    return A.Compose([
        A.RandomRotate90(p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15,
                             val_shift_limit=10, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.10,
                                   contrast_limit=0.08, p=0.35),
        A.Resize(height=512, width=512),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transform():
    return A.Compose([
        A.Resize(height=512, width=512),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def make_16_crops(img_tensor_512):
    crops = []
    for r in range(4):
        for c in range(4):
            crop = img_tensor_512[:, r*128:(r+1)*128, c*128:(c+1)*128]
            crop = F.interpolate(crop.unsqueeze(0), size=(224, 224),
                                 mode="bilinear", align_corners=False).squeeze(0)
            crops.append(crop)
    return torch.stack(crops)


class OSCCLoopDataset(Dataset):
    def __init__(self, df, mode="train"):
        self.mode = mode
        self.heavy_aug = get_heavy_aug()
        self.light_aug = get_light_aug()
        self.val_transform = get_val_transform()
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = LABEL_ENC[row["label"]]

        try:
            img_np = np.array(Image.open(row["path"]).convert("RGB"))
        except Exception:
            img_np = np.zeros((512, 512, 3), dtype=np.uint8)

        if self.mode == "train":
            if row["label"] in ["dispersive", "noncohesive"]:
                result = self.heavy_aug(image=img_np)
            else:
                result = self.light_aug(image=img_np)
        else:
            result = self.val_transform(image=img_np)

        img_512 = result["image"]
        crops_224 = make_16_crops(img_512)
        img_256 = F.interpolate(img_512.unsqueeze(0), size=(256, 256),
                                mode="bilinear", align_corners=False).squeeze(0)
        return crops_224, img_256, label
