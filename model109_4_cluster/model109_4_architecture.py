"""
model109_4_architecture.py
--------------------------
Model 109.4: 5-Class Multi-Prototype Deep Metric Clustering Network
with Automated Outlier Radius Gate.
Successor to Champion Model 9.3 for the 20x Magnification Dataset.

Classes (5 Biological Classes):
  0: cohesive
  1: dysplasia
  2: noncohesive
  3: dispersive
  4: normal

Architecture:
  - Tri-Branch Backbone:
      * Branch A: Frozen Phikon ViT (1536-dim)
      * Branch B: Fine-Tuned Phikon ViT (Blocks 8-11 unfrozen, 1536-dim)
      * Branch C: Custom TileCNN (512-dim)
      * Dynamic Gating: Softmax-balanced fusion -> 512-dim fused representation
  - Latent Projection: 512-dim -> 256-dim L2-normalized embedding space
  - 10 Learnable Sub-Centroids (2 prototypes per biological class):
      Captures distinct morphological subtypes (e.g. cohesive sheets vs trabeculae,
      mild vs severe dysplasia, solitary infiltrative cells vs clusters).
  - Outlier Radius Gate (Quality Control):
      Computes statistical hypersphere envelope R_c = mu_c + 2.5 * sigma_c per class.
      Automatically rejects non-biological tissue (no_feature, blank glass, necrotic artifacts)
      without distorting biological classification margins.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel

CLASSES_5 = [
    "cohesive",
    "dysplasia",
    "noncohesive",
    "dispersive",
    "normal"
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


class Model109_4_MultiProtoNet(nn.Module):
    """
    Complete Model 109.4 Architecture:
    Tri-Branch Base + Dynamic Gating + 10 Sub-Centroids Multi-Prototype Metric Head.
    """
    def __init__(self, num_classes=5, prototypes_per_class=2, unfreeze_blocks=4, device="cpu", init_temp=0.07):
        super().__init__()
        self.num_classes = num_classes
        self.prototypes_per_class = prototypes_per_class
        self.total_prototypes = num_classes * prototypes_per_class
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

        # 10 Learnable sub-centroids in 256-D metric space
        self.prototypes = nn.Parameter(torch.randn(self.total_prototypes, 256))
        nn.init.orthogonal_(self.prototypes)

        # Learnable temperature parameter
        self.log_temp = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(init_temp)))))

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
        emb = self.latent_proj(fused_512)
        z = F.normalize(emb, p=2, dim=-1)  # [B, 256]

        c = F.normalize(self.prototypes, p=2, dim=-1)  # [10, 256]
        temp = torch.clamp(torch.exp(self.log_temp), min=0.03, max=0.20)
        sim_all = torch.matmul(z, c.t()) / temp  # [B, 10]

        # Max-pool across sub-prototypes for each of the 5 biological classes
        sim_grouped = sim_all.view(-1, self.num_classes, self.prototypes_per_class)
        class_logits = sim_grouped.max(dim=-1).values  # [B, 5]

        return class_logits, z, c, weights


class OutlierRadiusGate:
    """
    Computes statistical cluster radius per class and tags outliers/margins:
      Radius_k = mean(dist) + 2.5 * std(dist)
    """
    def __init__(self, n_sigma=2.5):
        self.n_sigma = n_sigma
        self.centroids = {}
        self.radii = {}

    def fit(self, embeddings_dict):
        """Fit cluster radii using validated embeddings per biological class."""
        for cls_name, embs in embeddings_dict.items():
            c = np.mean(embs, axis=0)
            dists = np.linalg.norm(embs - c, axis=1)
            mu, sigma = np.mean(dists), np.std(dists)
            self.centroids[cls_name] = c
            self.radii[cls_name] = float(mu + self.n_sigma * sigma)

    def evaluate(self, z):
        """
        Input: z (single 256-D embedding)
        Returns: { 'is_outlier': bool, 'closest_class': str, 'min_dist': float, 'cluster_radius': float }
        """
        closest_cls = None
        min_d = float("inf")
        for cls_name, c in self.centroids.items():
            d = float(np.linalg.norm(z - c))
            if d < min_d:
                min_d = d
                closest_cls = cls_name

        is_outlier = (min_d > self.radii[closest_cls])
        return {
            "is_outlier": is_outlier,
            "closest_class": closest_cls,
            "min_dist": min_d,
            "cluster_radius": self.radii[closest_cls]
        }

