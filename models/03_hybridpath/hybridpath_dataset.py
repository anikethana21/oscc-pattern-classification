import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
"""
hybridpath_dataset.py
---------------------
PyTorch Dataset for HybridPath.
- Two-tier augmentation: HEAVY for dispersive/noncohesive, LIGHT for others
- Returns (crops_224, img_256, label) per tile
  crops_224 : [16, 3, 224, 224]  -- for both Phikon branches
  img_256   : [3, 256, 256]      -- for CNN branch
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# -- Augmentation pipelines --------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_heavy_aug():
    """For dispersive and noncohesive (minority classes)."""
    return A.Compose([
        A.RandomRotate90(p=1.0),
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomResizedCrop(
            size=(512, 512),
            scale=(0.75, 1.0),
            ratio=(0.9, 1.1),
            p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=15,
            p=0.7,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.10,
            p=0.5,
        ),
        A.Resize(height=512, width=512),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def get_light_aug():
    """For cohesive and no_feature (majority classes)."""
    return A.Compose([
        A.RandomRotate90(p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Resize(height=512, width=512),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def get_val_transform():
    """No augmentation for val/test."""
    return A.Compose([
        A.Resize(height=512, width=512),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# -- Crop helper -------------------------------------------------------------

def make_16_crops(img_tensor_512):
    """
    img_tensor_512: [3, 512, 512]
    Returns: [16, 3, 224, 224]  (4x4 grid, each crop resized to 224)
    """
    C, H, W = img_tensor_512.shape
    gh, gw  = 4, 4
    ch, cw  = H // gh, W // gw
    crops   = []
    for r in range(gh):
        for c in range(gw):
            crop = img_tensor_512[:, r*ch:(r+1)*ch, c*cw:(c+1)*cw]
            crop = F.interpolate(
                crop.unsqueeze(0), size=(224, 224),
                mode="bilinear", align_corners=False
            ).squeeze(0)
            crops.append(crop)
    return torch.stack(crops)   # [16, 3, 224, 224]


# -- Dataset -----------------------------------------------------------------

LABEL_ENC = {
    "cohesive":    0,
    "dispersive":  1,
    "no_feature":  2,
    "noncohesive": 3,
}

AUGMENT_MULTIPLIER = {
    "dispersive":  5,
    "noncohesive": 3,
    "cohesive":    1,
    "no_feature":  1,
}

MINORITY_CLASSES = {"dispersive", "noncohesive"}


class OSCCDataset(Dataset):
    def __init__(self, df, mode="train",
                 augment_multiplier=None,
                 minority_classes=None):
        """
        df    : DataFrame with columns [path, label]
        mode  : "train" | "val" | "test"
        """
        self.mode             = mode
        self.heavy_aug        = get_heavy_aug()
        self.light_aug        = get_light_aug()
        self.val_transform    = get_val_transform()
        self.minority_classes = minority_classes or MINORITY_CLASSES
        self.multiplier       = augment_multiplier or AUGMENT_MULTIPLIER

        if mode == "train":
            rows = []
            for _, row in df.iterrows():
                n = self.multiplier.get(row["label"], 1)
                for _ in range(n):
                    rows.append(row)
            self.df = pd.DataFrame(rows).reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = LABEL_ENC[row["label"]]

        img_np = np.array(Image.open(row["path"]).convert("RGB"))

        # Apply augmentation
        if self.mode == "train":
            if row["label"] in self.minority_classes:
                result = self.heavy_aug(image=img_np)
            else:
                result = self.light_aug(image=img_np)
        else:
            result = self.val_transform(image=img_np)

        img_512 = result["image"]               # [3, 512, 512] tensor

        # For Phikon branches: 4x4 grid of 224x224 crops
        crops_224 = make_16_crops(img_512)      # [16, 3, 224, 224]

        # For CNN branch: resize 512 -> 256
        img_256 = F.interpolate(
            img_512.unsqueeze(0), size=(256, 256),
            mode="bilinear", align_corners=False
        ).squeeze(0)                             # [3, 256, 256]

        return crops_224, img_256, label
