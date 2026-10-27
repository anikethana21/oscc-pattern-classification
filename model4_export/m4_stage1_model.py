import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
m4_stage1_model.py
------------------
Stage 1 of Model 4:
Trained STRICTLY on the first two classes: cohesive (0) and no_feature (1 in this binary setting).
During training, only tiles belonging to these two classes are passed.
Outputs binary logits: cohesive vs no_feature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel

NAMED_COLS = [
    "nuclei_count", "nuclei_density",
    "mean_nucleus_area", "std_nucleus_area",
    "mean_circularity", "mean_eccentricity",
    "mean_nn_distance", "std_nn_distance",
    "red_fraction", "pink_fraction",
]
N_NAMED = len(NAMED_COLS)


def load_phikon(device="cpu"):
    model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


class Stage1Classifier(nn.Module):
    """
    Binary classifier head trained solely on cohesive vs no_feature.
    """
    def __init__(self, in_dim=1536 + N_NAMED, hidden_dim=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),  # 0: cohesive, 1: no_feature
        )

    def forward(self, phikon_emb, named_feats):
        x = torch.cat([phikon_emb, named_feats], dim=1)
        return self.head(x)


class Stage1WithBackbone(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.phikon = load_phikon(device)
        self.head   = Stage1Classifier()

    def extract_phikon(self, crops_224):
        B, N, C, H, W = crops_224.shape
        flat = crops_224.view(B * N, C, H, W)
        with torch.no_grad():
            out = self.phikon(pixel_values=flat)
            cls = out.last_hidden_state[:, 0, :]
        cls    = cls.view(B, N, 768)
        mean_p = cls.mean(dim=1)
        max_p  = cls.max(dim=1).values
        return torch.cat([mean_p, max_p], dim=1)

    def forward(self, crops_224, named_feats):
        emb = self.extract_phikon(crops_224)
        return self.head(emb, named_feats)
