"""
m4_2_dataset.py
---------------
Dataset loader and multi-crop extractor for Model 4.2.
Extracts:
  - 16 overlapping crops of 224x224 from 512x512 tile
  - Downsampled 256x256 view for TileCNN
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def make_16_crops(img_512_tensor: torch.Tensor) -> torch.Tensor:
    coords = [0, 96, 192, 288]
    crops = []
    for y in coords:
        for x in coords:
            crop = img_512_tensor[:, y:y+224, x:x+224]
            crops.append(crop)
    return torch.stack(crops, dim=0)  # [16, 3, 224, 224]


def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transform():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class Model4_2_Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, class_to_idx: dict, mode: str = "train"):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.mode = mode
        self.transform = get_train_transform() if mode == "train" else get_val_transform()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["path"]

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((512, 512), Image.Resampling.BILINEAR)
            img_np = np.array(img)
        except Exception:
            img_np = np.zeros((512, 512, 3), dtype=np.uint8)

        transformed = self.transform(image=img_np)
        img_512 = transformed["image"]
        crops_224 = make_16_crops(img_512)
        img_256 = F.interpolate(img_512.unsqueeze(0), size=(256, 256),
                                mode="bilinear", align_corners=False).squeeze(0)

        label_str = row.get("label", "")
        label_idx = self.class_to_idx.get(label_str, 0)

        return crops_224, img_256, torch.tensor(label_idx, dtype=torch.long)

