import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
m5_model.py
-----------
Model 5: Recurrent Feedback / Looping Dual-Stage Network.

Architecture flow:
1. Module 1 (Screening Network):
   - Phikon features -> Initial screening logits (first two classes: cohesive, no_feature)
   - Outputs latent state h1 [B, 512]
2. Module 2 (Infiltrative Specialist Network):
   - Multi-branch (Phikon + TileCNN + Gating)
   - Takes h1 + image inputs -> Specialist latent state h2 [B, 512]
3. Looping Feedback Connection:
   - h2 is passed BACK to Module 1 via a feedback recurrent gate:
     h_loop = Gate(h1, FeedbackLinear(h2))
4. Final Forcing Classifier:
   - Takes h_loop and forces classification into one of the 4 classes:
     [cohesive, dispersive, no_feature, noncohesive]
"""

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
    def __init__(self, in_dim=3584 + 512, num_branches=3):
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


class Model5LoopNetwork(nn.Module):
    """
    Model 5: Looping Dual-Stage Network with Feedback Connection.
    """
    def __init__(self, unfreeze_last_n=3, device="cpu"):
        super().__init__()

        # --- Shared / Dedicated Backbones ---
        print("Model 5: Loading Phikon Frozen Backbone...")
        self.phikon_frozen = load_phikon(device)
        freeze_all(self.phikon_frozen)

        print("Model 5: Loading Phikon Adaptive Backbone...")
        self.phikon_adapt = load_phikon(device)
        unfreeze_last_n_blocks(self.phikon_adapt, n=unfreeze_last_n)

        # CNN Branch
        self.cnn = TileCNN(out_dim=512)

        # Module 1 (First two classes screener projection)
        self.proj_mod1 = nn.Sequential(
            nn.Linear(1536, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        # Early auxiliary logits for first two classes (cohesive vs no_feature)
        self.early_screener_head = nn.Linear(512, 2)

        # Module 2 (Specialist multi-branch projections)
        self.proj_a = nn.Sequential(nn.Linear(1536, 512), nn.GELU())
        self.proj_b = nn.Sequential(nn.Linear(1536, 512), nn.GELU())
        self.gating = GatingNetwork(in_dim=1536 + 1536 + 512 + 512, num_branches=3)

        # Recurrent Loop Connection: Mod 2 -> Mod 1 Feedback
        self.feedback_transform = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
        )
        self.recurrent_gate = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Sigmoid(),
        )

        # Final 4-Class Forcing Classifier Head
        self.final_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4),  # Forces remaining into either of the 4 classes
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
        # 1. Module 1: Pass to first network
        F_frozen = self._extract_phikon(self.phikon_frozen, crops_224)  # [B, 1536]
        h1 = self.proj_mod1(F_frozen)                                   # [B, 512]
        early_logits = self.early_screener_head(h1)                    # [B, 2]

        # 2. Module 2: Pass remaining features to second specialist network
        F_adapt = self._extract_phikon(self.phikon_adapt, crops_224)    # [B, 1536]
        F_cnn   = self.cnn(img_256)                                    # [B, 512]

        # Specialist fusion conditioned on Module 1 latent state h1
        gate_input = torch.cat([F_frozen, F_adapt, F_cnn, h1], dim=1)  # [B, 4096]
        weights    = self.gating(gate_input)                           # [B, 3]
        w_a, w_b, w_c = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]

        P_a = self.proj_a(F_frozen)
        P_b = self.proj_b(F_adapt)
        P_c = F_cnn
        h2  = w_a * P_a + w_b * P_b + w_c * P_c                        # [B, 512]

        # 3. Looping Connection: Pass h2 back to first network representation
        fb = self.feedback_transform(h2)                               # [B, 512]
        gate_val = self.recurrent_gate(torch.cat([h1, fb], dim=1))     # [B, 512]
        h_loop = gate_val * h1 + (1.0 - gate_val) * fb                # Looped fused latent [B, 512]

        # 4. Final Classifier: Forces output into either of the 4 classes
        final_logits = self.final_classifier(h_loop)                  # [B, 4]

        return final_logits, early_logits


class MultiTaskLoopLoss(nn.Module):
    """
    Joint loss for Model 5:
    - Primary Focal Loss on final 4-class forcing classifier (heavy weights on dispersive & noncohesive)
    - Auxiliary loss on early screening logits (for cohesive & no_feature)
    """
    def __init__(self, class_weights=None, gamma=2.0, aux_weight=0.3):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights
        self.aux_weight = aux_weight

    def forward(self, final_logits, early_logits, targets):
        # 1. 4-class Focal Loss
        ce = F.cross_entropy(final_logits, targets, weight=self.class_weights, reduction="none")
        pt = torch.exp(-ce)
        loss_final = ((1.0 - pt) ** self.gamma * ce).mean()

        # 2. Auxiliary Loss on early screening (only active for samples with targets 0 or 2)
        # target 0 -> cohesive (label 0), target 2 -> no_feature (label 1)
        mask = (targets == 0) | (targets == 2)
        if mask.sum() > 0:
            aux_targets = torch.where(targets[mask] == 0, 0, 1)
            loss_aux = F.cross_entropy(early_logits[mask], aux_targets)
        else:
            loss_aux = 0.0

        return loss_final + self.aux_weight * loss_aux
