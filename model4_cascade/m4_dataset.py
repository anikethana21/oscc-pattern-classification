import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
m4_dataset.py
-------------
Datasets for Model 4.
- Stage1Dataset: In training mode, filters strictly to the first two classes (cohesive, no_feature).
- Stage2Dataset: Handles the full dataset, emphasizing dispersive & noncohesive with heavy augmentations.
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
LABEL_ENC_4 = {n: i for i, n in enumerate(CLASS_NAMES)}

# Stage 1 binary mapping for the first two classes
STAGE1_ENC = {
    "cohesive": 0,
    "no_feature": 1,
}

NAMED_COLS = [
    "nuclei_count", "nuclei_density",
    "mean_nucleus_area", "std_nucleus_area",
    "mean_circularity", "mean_eccentricity",
    "mean_nn_distance", "std_nn_distance",
    "red_fraction", "pink_fraction",
]


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
    C, H, W = img_tensor_512.shape
    crops = []
    for r in range(4):
        for c in range(4):
            crop = img_tensor_512[:, r*128:(r+1)*128, c*128:(c+1)*128]
            crop = F.interpolate(crop.unsqueeze(0), size=(224, 224),
                                 mode="bilinear", align_corners=False).squeeze(0)
            crops.append(crop)
    return torch.stack(crops)


class Stage1Dataset(Dataset):
    """
    Trained ONLY on cohesive and no_feature.
    In eval/test mode, can process any tile.
    """
    def __init__(self, df, mode="train", named_mean=None, named_std=None):
        self.mode = mode
        self.light_aug = get_light_aug()
        self.val_transform = get_val_transform()
        self.named_mean = named_mean
        self.named_std = named_std
        self.avail_named = [c for c in NAMED_COLS if c in df.columns]

        if mode == "train":
            mask = df["label"].isin(["cohesive", "no_feature"])
            self.df = df[mask].reset_index(drop=True)
            print(f"  Stage1Dataset (Train - First 2 Classes Only): {len(self.df)} tiles")
        else:
            self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label_str = row["label"]
        label = STAGE1_ENC.get(label_str, 0)

        try:
            img_np = np.array(Image.open(row["path"]).convert("RGB"))
        except Exception:
            img_np = np.zeros((512, 512, 3), dtype=np.uint8)

        if self.mode == "train":
            result = self.light_aug(image=img_np)
        else:
            result = self.val_transform(image=img_np)

        img_512 = result["image"]
        crops_224 = make_16_crops(img_512)

        if self.avail_named:
            named = np.array([row.get(c, 0.0) for c in NAMED_COLS], dtype=np.float32)
            if self.named_mean is not None and self.named_std is not None:
                named = (named - self.named_mean) / (self.named_std + 1e-8)
        else:
            named = np.zeros(len(NAMED_COLS), dtype=np.float32)

        return crops_224, torch.tensor(named, dtype=torch.float32), label


class Stage2Dataset(Dataset):
    """
    Dataset for Stage 2 (4-class HybridPath specialist).
    Includes all 4 classes, applying heavy augmentations to dispersive & noncohesive.
    """
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
        label = LABEL_ENC_4[row["label"]]

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
