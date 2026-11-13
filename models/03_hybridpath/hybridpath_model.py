import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
"""
hybridpath_model.py
--------------------
HybridPath architecture:
  Branch A : Phikon FROZEN         -> 1536-dim
  Branch B : Phikon PARTIALLY UNFROZEN (last 3 blocks) -> 1536-dim
  Branch C : CNN from scratch (256x256 input)          -> 512-dim

  All three projected to 512-dim -> Gating Network (dynamic weights)
  -> Weighted fusion -> MLP head -> 4 classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, AutoImageProcessor


# -- Helpers to load / freeze Phikon ----------------------------------------

def load_phikon(device="cpu"):
    model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
    model = model.to(device)
    return model


def freeze_all(model):
    for param in model.parameters():
        param.requires_grad = False


def freeze_phikon_partial(model, unfreeze_last_n=3):
    """Freeze all blocks, then unfreeze the last unfreeze_last_n blocks + layernorm."""
    freeze_all(model)
    layers = getattr(model, "layers", None)
    if layers is None and hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        layers = model.encoder.layer
    if layers is None:
        raise AttributeError("Could not find transformer layers on ViTModel")
    total_blocks = len(layers)
    unfreeze_from = total_blocks - unfreeze_last_n
    for i in range(unfreeze_from, total_blocks):
        for param in layers[i].parameters():
            param.requires_grad = True
    if hasattr(model, "layernorm"):
        for param in model.layernorm.parameters():
            param.requires_grad = True


# -- Branch C: CNN -----------------------------------------------------------

class TileCNN(nn.Module):
    """Lightweight CNN that processes a 256x256 tile image from scratch."""

    def __init__(self, out_dim=512):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 256 -> 128
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),

            # Block 2: 128 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),

            # Block 3: 64 -> 32
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),

            # Block 4: 32 -> 16
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # -> 256 x 4 x 4
        self.proj = nn.Sequential(
            nn.Flatten(),                          # 4096
            nn.Linear(4096, out_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.proj(x)                        # [B, 512]


# -- Gating Network ----------------------------------------------------------

class GatingNetwork(nn.Module):
    """
    Learns per-image soft weights for each branch.
    Input : concatenated raw branch outputs [B, 1536+1536+512]
    Output: [B, 3] weights summing to 1.0
    """

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
        return self.gate(x)   # [B, 3]


# -- Classification Head -----------------------------------------------------

class ClassificationHead(nn.Module):
    def __init__(self, in_dim=512, num_classes=4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(x)   # [B, num_classes]


# -- Full HybridPath Model ---------------------------------------------------

class HybridPath(nn.Module):
    """
    Three-branch hybrid model:
      A : frozen Phikon  (all 12 blocks frozen)
      B : adaptive Phikon (blocks 0-8 frozen, 9-11 unfrozen)
      C : CNN from scratch (256x256)

    Dynamic gating network learns per-image branch weights.
    """

    def __init__(self, num_classes=4, unfreeze_last_n=3, device="cpu"):
        super().__init__()

        # Branch A: fully frozen
        print("Loading Phikon Branch A (frozen)...")
        self.phikon_a = load_phikon(device)
        freeze_all(self.phikon_a)

        # Branch B: partially unfrozen
        print("Loading Phikon Branch B (partial unfreeze)...")
        self.phikon_b = load_phikon(device)
        freeze_phikon_partial(self.phikon_b, unfreeze_last_n=unfreeze_last_n)

        # Branch C: CNN
        self.cnn = TileCNN(out_dim=512)

        # Project each branch to common 512-dim space
        self.proj_a = nn.Sequential(nn.Linear(1536, 512), nn.GELU())
        self.proj_b = nn.Sequential(nn.Linear(1536, 512), nn.GELU())
        # proj_c: CNN already outputs 512

        # Gating: takes raw concat of all branch outputs
        self.gate = GatingNetwork(in_dim=1536 + 1536 + 512, num_branches=3)

        # Classification head
        self.head = ClassificationHead(in_dim=512, num_classes=num_classes)

    def _extract_phikon(self, model, crops_224):
        """
        crops_224 : [B, 16, 3, 224, 224]
        Returns   : [B, 1536]
        """
        B, N, C, H, W = crops_224.shape
        flat = crops_224.view(B * N, C, H, W)

        # Only compute gradients if this model has unfrozen params
        enable_grad = any(p.requires_grad for p in model.parameters())
        with torch.set_grad_enabled(enable_grad):
            out = model(pixel_values=flat)
            cls = out.last_hidden_state[:, 0, :]   # [B*16, 768]

        cls = cls.view(B, N, 768)
        mean_pool = cls.mean(dim=1)                 # [B, 768]
        max_pool  = cls.max(dim=1).values           # [B, 768]
        return torch.cat([mean_pool, max_pool], dim=1)  # [B, 1536]

    def forward(self, crops_224, img_256):
        """
        crops_224 : [B, 16, 3, 224, 224]  -- for Phikon branches
        img_256   : [B, 3, 256, 256]      -- for CNN branch
        """
        F_a = self._extract_phikon(self.phikon_a, crops_224)  # [B, 1536]
        F_b = self._extract_phikon(self.phikon_b, crops_224)  # [B, 1536]
        F_c = self.cnn(img_256)                               # [B, 512]

        # Gating
        concat  = torch.cat([F_a, F_b, F_c], dim=1)   # [B, 3584]
        weights = self.gate(concat)                     # [B, 3]
        w_a, w_b, w_c = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]

        # Project to common dim
        P_a = self.proj_a(F_a)   # [B, 512]
        P_b = self.proj_b(F_b)   # [B, 512]
        P_c = F_c                 # [B, 512]

        # Weighted fusion
        fused = w_a * P_a + w_b * P_b + w_c * P_c   # [B, 512]

        return self.head(fused)   # [B, 4]


# -- Focal Loss --------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss: focuses training on hard, misclassified examples.
    gamma=2.0 is standard. Combined with class weights for imbalance.
    """

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight   # [num_classes] tensor

    def forward(self, logits, targets):
        ce   = F.cross_entropy(logits, targets,
                               weight=self.weight, reduction="none")
        pt   = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        return loss.mean()


# -- Quick sanity check ------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = HybridPath(num_classes=4, unfreeze_last_n=3, device=device).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\nTotal params    : {total:,}")
    print(f"Trainable params: {trainable:,}  ({100*trainable/total:.2f}%)")

    # Dummy forward pass
    crops = torch.randn(2, 16, 3, 224, 224).to(device)
    imgs  = torch.randn(2, 3, 256, 256).to(device)
    out   = model(crops, imgs)
    print(f"\nForward pass output shape: {out.shape}")  # [2, 4]
    print("Model OK.")
