"""
m4_2_model.py
-------------
Model 4.2: Sub-classifier architecture clone of Model 4 Specialist.
Three branches:
  - Branch A: 100% frozen Phikon ViT-Base-16 (foundation anchor, dual-pooled to 1536 -> proj to 512)
  - Branch B: Partially unfrozen Phikon ViT-Base-16 (top 3 blocks unfrozen for fine-tuning -> proj to 512)
  - Branch C: TileCNN for 256x256 global tile context (proj to 512)
  - Dynamic Gating Network: Softmax-based attention fusion across all three branches
  - 4-Class Classification Head: Linear(512->256) -> GELU -> Dropout -> Linear(256->128) -> GELU -> Dropout -> Linear(128->num_classes)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel


def load_phikon(device="cpu"):
    return ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False).to(device)


def freeze_all(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_last_n_blocks(model, n=3):
    freeze_all(model)
    try:
        layers = model.encoder.layer
    except AttributeError:
        layers = model.layers
    total = len(layers)
    for i in range(total - n, total):
        for param in layers[i].parameters():
            param.requires_grad = True
    if hasattr(model, "layernorm"):
        for param in model.layernorm.parameters():
            param.requires_grad = True


class TileCNN(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, padding=2), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, out_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        return self.proj(self.pool(self.features(x)))


class GatingNetwork(nn.Module):
    def __init__(self, in_dim=3584, num_branches=3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_branches),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.gate(x)


class Model4_2_Classifier(nn.Module):
    def __init__(self, num_classes=4, unfreeze_last_n=3, device="cpu"):
        super().__init__()
        print(f"[Model 4.2] Loading Branch A: Phikon (100% frozen anchor)...")
        self.phikon_a = load_phikon(device)
        freeze_all(self.phikon_a)

        print(f"[Model 4.2] Loading Branch B: Phikon (top {unfreeze_last_n} blocks unfrozen)...")
        self.phikon_b = load_phikon(device)
        unfreeze_last_n_blocks(self.phikon_b, n=unfreeze_last_n)

        print(f"[Model 4.2] Initializing Branch C: TileCNN for 256x256 context...")
        self.cnn = TileCNN(out_dim=512)

        self.proj_a = nn.Sequential(nn.Linear(1536, 512), nn.GELU())
        self.proj_b = nn.Sequential(nn.Linear(1536, 512), nn.GELU())
        self.gate   = GatingNetwork(in_dim=1536 + 1536 + 512, num_branches=3)

        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def _extract_phikon(self, model, crops_224):
        B, N, C, H, W = crops_224.shape
        flat = crops_224.view(B * N, C, H, W)
        enable_grad = any(p.requires_grad for p in model.parameters())
        with torch.set_grad_enabled(enable_grad):
            out = model(pixel_values=flat)
            cls = out.last_hidden_state[:, 0, :]
        cls    = cls.view(B, N, 768)
        mean_p = cls.mean(dim=1)
        max_p  = cls.max(dim=1).values
        return torch.cat([mean_p, max_p], dim=1)

    def forward(self, crops_224, img_256):
        F_a = self._extract_phikon(self.phikon_a, crops_224)
        F_b = self._extract_phikon(self.phikon_b, crops_224)
        F_c = self.cnn(img_256)

        concat  = torch.cat([F_a, F_b, F_c], dim=1)
        weights = self.gate(concat)
        w_a, w_b, w_c = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]

        P_a = self.proj_a(F_a)
        P_b = self.proj_b(F_b)
        P_c = F_c

        fused = w_a * P_a + w_b * P_b + w_c * P_c
        return self.head(fused)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce   = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt   = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        return loss.mean()

