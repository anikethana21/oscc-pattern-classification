"""
model8_net.py
-------------
Model 8: Deep Unfrozen Phikon ViT Network for 3-Class OSCC Pattern Classification.
- True multi-layer unfreezing: Blocks 8, 9, 10, 11 + LayerNorm are actively trained.
- Multi-Scale 16-Crop Aggregation: Mean + Max pooling (1,536-dim).
- High-capacity classification head:
    1536 -> Linear(512) -> LayerNorm -> GELU -> Dropout(0.25)
         -> Linear(256) -> LayerNorm -> GELU -> Dropout(0.15) [Latent Embedding]
         -> Linear(3)   -> Class Logits
- No artificial outlier pruning: All tiles receive full gradient backpropagation.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

CLASSES_3 = ["cohesive", "dispersive", "noncohesive"]


def load_phikon_deep(device="cpu", num_unfrozen_blocks=4):
    """
    Loads owkin/phikon ViT-Base and unfreezes the top N transformer blocks + layernorm.
    """
    model = AutoModel.from_pretrained("owkin/phikon")
    
    # Freeze all parameters first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze the last N transformer layers
    if hasattr(model, "layers"):
        unfrozen_layers = model.layers[-num_unfrozen_blocks:]
        for block in unfrozen_layers:
            for p in block.parameters():
                p.requires_grad = True
        print(f"[Model 8] Unfrozen top {len(unfrozen_layers)} transformer blocks (indices {12-num_unfrozen_blocks} to 11).")

    # Also unfreeze the final layernorm
    if hasattr(model, "layernorm"):
        for p in model.layernorm.parameters():
            p.requires_grad = True
        print("[Model 8] Unfrozen ViT final LayerNorm.")

    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"[Model 8] Phikon Parameters: {trainable_p:,} trainable / {total_p:,} total ({trainable_p/total_p*100:.1f}% active).")
    
    return model


class Model8DeepPhikon(nn.Module):
    def __init__(self, num_classes=3, num_unfrozen_blocks=4, device="cpu"):
        super().__init__()
        self.device = device
        self.phikon = load_phikon_deep(device=device, num_unfrozen_blocks=num_unfrozen_blocks)

        self.feature_proj = nn.Sequential(
            nn.Linear(1536, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.25),
        )

        self.latent_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
        )

        self.classifier = nn.Linear(256, num_classes)

    def extract_features(self, crops_16):
        """
        Input: [B, 16, 3, 224, 224]
        Output: [B, 1536] (Mean + Max pooled)
        """
        B, N, C, H, W = crops_16.shape
        flat = crops_16.view(B * N, C, H, W)

        out = self.phikon(pixel_values=flat)
        cls_token = out.last_hidden_state[:, 0, :]  # [B*16, 768]
        cls_token = cls_token.view(B, N, 768)

        mean_p = cls_token.mean(dim=1)  # [B, 768]
        max_p = cls_token.max(dim=1).values  # [B, 768]
        return torch.cat([mean_p, max_p], dim=1)  # [B, 1536]

    def forward(self, crops_16):
        feat_1536 = self.extract_features(crops_16)
        h = self.feature_proj(feat_1536)
        latent_256 = self.latent_proj(h)
        logits = self.classifier(latent_256)
        return logits, latent_256

