"""
model104_4_architecture.py
--------------------------
Model 104.4: 7-Class Tri-Branch Gated Deep Feature Fusion Network
Successor to Champion Model 4.3.2 for the 20x Magnification Dataset.

Classes (7):
  0: cohesive
  1: dysplasia
  2: noncohesive
  3: dispersive
  4: normal
  5: no_feature
  6: blank

Architecture:
  - Branch A: Frozen Phikon ViT (Global Foundation Histopathology, 1536-dim via mean+max pooling of 16 crops)
  - Branch B: Fine-Tuned Phikon ViT (Oral Cavity Adaptation, Top 4 Transformer Blocks unfreezed + LayerNorm, 1536-dim)
  - Branch C: Custom TileCNN (Local Cell Border, Chromatin Texture & Edge Geometry, 512-dim)
  - Dynamic Gating Network: Cross-branch adaptive softmax gating
  - Classification Head: Multi-layer perceptron (512 -> 256 -> 128 -> 7 logits)
  - Loss: Multi-Class Focal Loss (gamma = 2.0) with Class-Frequency Weighting
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

CLASSES_7 = [
    "cohesive",
    "dysplasia",
    "noncohesive",
    "dispersive",
    "normal",
    "no_feature",
    "blank"
]


def load_phikon_frozen(device="cpu"):
    """Branch A: Frozen histological foundation ViT."""
    model = AutoModel.from_pretrained("owkin/phikon")
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


def load_phikon_finetuned(device="cpu", unfreeze_blocks=4):
    """Branch B: Phikon ViT with top-N transformer blocks unfrozen."""
    model = AutoModel.from_pretrained("owkin/phikon")
    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model, "layers"):
        for block in model.layers[-unfreeze_blocks:]:
            for p in block.parameters():
                p.requires_grad = True

    if hasattr(model, "layernorm"):
        for p in model.layernorm.parameters():
            p.requires_grad = True

    return model.to(device)


class TileCNN(nn.Module):
    """Branch C: High-resolution localized edge and cytoplasmic texture extractor."""
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
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.25),
        )

    def forward(self, x):
        return self.proj(self.pool(self.features(x)))


class DynamicGatingNetwork(nn.Module):
    """Adaptive softmax weighting across Branch A, B, and C."""
    def __init__(self, in_dim=1536 + 1536 + 512, num_branches=3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_branches),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.gate(x)


class Model104_4_ClassifierNet(nn.Module):
    """
    Complete Model 104.4 Architecture:
    Tri-Branch Base + Dynamic Gating + 7-Class Classification MLP.
    """
    def __init__(self, num_classes=7, unfreeze_blocks=4, device="cpu"):
        super().__init__()
        self.num_classes = num_classes
        self.device = device

        self.phikon_a = load_phikon_frozen(device)
        self.phikon_b = load_phikon_finetuned(device, unfreeze_blocks=unfreeze_blocks)
        self.cnn = TileCNN(out_dim=512)

        self.proj_a = nn.Sequential(nn.Linear(1536, 512), nn.LayerNorm(512), nn.GELU())
        self.proj_b = nn.Sequential(nn.Linear(1536, 512), nn.LayerNorm(512), nn.GELU())
        self.gate = DynamicGatingNetwork(in_dim=1536 + 1536 + 512, num_branches=3)

        self.latent_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, num_classes),
        )

    def _extract_phikon(self, model, crops_16):
        B, N, C, H, W = crops_16.shape
        flat = crops_16.view(B * N, C, H, W)
        enable_grad = any(p.requires_grad for p in model.parameters())
        with torch.set_grad_enabled(enable_grad):
            out = model(pixel_values=flat)
            cls = out.last_hidden_state[:, 0, :]
        cls = cls.view(B, N, 768)
        mean_p = cls.mean(dim=1)
        max_p = cls.max(dim=1).values
        return torch.cat([mean_p, max_p], dim=1)  # [B, 1536]

    def forward(self, crops_16, img_512):
        F_a = self._extract_phikon(self.phikon_a, crops_16)
        F_b = self._extract_phikon(self.phikon_b, crops_16)
        F_c = self.cnn(img_512)

        concat = torch.cat([F_a, F_b, F_c], dim=1)
        weights = self.gate(concat)
        w_a, w_b, w_c = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]

        P_a = self.proj_a(F_a)
        P_b = self.proj_b(F_b)
        P_c = F_c

        fused_512 = w_a * P_a + w_b * P_b + w_c * P_c
        latent_256 = self.latent_proj(fused_512)
        logits = self.classifier(latent_256)
        return logits, latent_256, weights


class FocalLoss(nn.Module):
    """
    Multi-Class Focal Loss to focus on hard, subtle boundary classes.
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Tensor of shape [C]
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        p = torch.exp(-ce_loss)
        focal_loss = ((1.0 - p) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

