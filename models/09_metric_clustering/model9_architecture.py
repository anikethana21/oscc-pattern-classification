"""
model9_architecture.py
----------------------
Model 9 Series: Tri-Branch Deep Clustering & Metric Learning Architecture
Derived directly from champion Model 4.3.2 / 4.3.2.1:
  - Branch A: Frozen Phikon ViT (Global Pan-Cancer Representation, 1536-dim)
  - Branch B: Fine-Tuned Phikon ViT (Oral-Specialized Blocks 8-11 + LayerNorm, 1536-dim)
  - Branch C: TileCNN (Local Nuclear-Cytoplasmic Texture & Edge Geometry, 512-dim)
  - Dynamic Gating Network: Adaptive softmax balancing between Branches A, B, and C.

Includes:
  - Model9_2_CenterLossNet: Tri-Branch + Center Loss (collapses PCA stragglers into tight balls)
  - Model9_1_ProtoNet: Tri-Branch + Prototypical Hypersphere Cosine Metric
  - Model9_3_MultiProtoNet: Tri-Branch + Multi-Prototype Sub-Clustering (2 centroids per class)
  - OutlierRadiusGate: Automated cluster radius estimation & artifact rejection
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

CLASSES_3 = ["cohesive", "dispersive", "noncohesive"]


# -------------------------------------------------------------
# 1. Backbone Loaders
# -------------------------------------------------------------
def load_phikon_frozen(device="cpu"):
    """Branch A: Pure frozen histological feature extractor."""
    model = AutoModel.from_pretrained("owkin/phikon")
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


def load_phikon_finetuned(device="cpu", unfreeze_blocks=4):
    """Branch B: Fine-tuned Phikon top blocks for oral cancer adaptation."""
    model = AutoModel.from_pretrained("owkin/phikon")
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze top N transformer blocks
    if hasattr(model, "layers"):
        for block in model.layers[-unfreeze_blocks:]:
            for p in block.parameters():
                p.requires_grad = True

    # Unfreeze final LayerNorm
    if hasattr(model, "layernorm"):
        for p in model.layernorm.parameters():
            p.requires_grad = True

    return model.to(device)


# -------------------------------------------------------------
# 2. Branch C: TileCNN (Local Texture & Edge Geometry)
# -------------------------------------------------------------
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
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.25),
        )

    def forward(self, x):
        return self.proj(self.pool(self.features(x)))


# -------------------------------------------------------------
# 3. Dynamic Adaptive Gating Network
# -------------------------------------------------------------
class DynamicGatingNetwork(nn.Module):
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


# -------------------------------------------------------------
# 4. Tri-Branch Base Backbone (Model 4 Heritage)
# -------------------------------------------------------------
class Model9TriBranchBase(nn.Module):
    def __init__(self, unfreeze_blocks=4, device="cpu"):
        super().__init__()
        self.device = device
        self.phikon_a = load_phikon_frozen(device)
        self.phikon_b = load_phikon_finetuned(device, unfreeze_blocks=unfreeze_blocks)
        self.cnn = TileCNN(out_dim=512)

        self.proj_a = nn.Sequential(nn.Linear(1536, 512), nn.LayerNorm(512), nn.GELU())
        self.proj_b = nn.Sequential(nn.Linear(1536, 512), nn.LayerNorm(512), nn.GELU())
        self.gate = DynamicGatingNetwork(in_dim=1536 + 1536 + 512, num_branches=3)

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

    def extract_fused_features(self, crops_16, img_512):
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
        return fused_512, weights


# -------------------------------------------------------------
# 5. Center Loss Module (Gravitational Compaction)
# -------------------------------------------------------------
class CenterLoss(nn.Module):
    """
    Center Loss mathematically pulls each tile towards its class centroid:
      L_center = 0.5 * sum || z_i - c_{y_i} ||^2
    """
    def __init__(self, num_classes=3, feat_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.orthogonal_(self.centers)

    def forward(self, features, labels):
        batch_size = features.size(0)
        centers_batch = self.centers[labels]
        loss = F.mse_loss(features, centers_batch, reduction="sum") / (2.0 * batch_size)
        return loss


# -------------------------------------------------------------
# 6. MODEL 9.2: Tri-Branch + Center Loss Compact Net (Flagship)
# -------------------------------------------------------------
class Model9_2_CenterLossNet(nn.Module):
    def __init__(self, num_classes=3, unfreeze_blocks=4, device="cpu"):
        super().__init__()
        self.backbone = Model9TriBranchBase(unfreeze_blocks=unfreeze_blocks, device=device)

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

    def forward(self, crops_16, img_512):
        fused_512, gate_weights = self.backbone.extract_fused_features(crops_16, img_512)
        latent_256 = self.latent_proj(fused_512)
        logits = self.classifier(latent_256)
        return logits, latent_256, gate_weights


# -------------------------------------------------------------
# 7. MODEL 9.1: Prototypical Tri-Branch Net
# -------------------------------------------------------------
class Model9_1_ProtoNet(nn.Module):
    def __init__(self, num_classes=3, unfreeze_blocks=4, device="cpu", init_temp=0.07):
        super().__init__()
        self.backbone = Model9TriBranchBase(unfreeze_blocks=unfreeze_blocks, device=device)

        self.proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.prototypes = nn.Parameter(torch.randn(num_classes, 256))
        nn.init.orthogonal_(self.prototypes)
        self.log_temp = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(init_temp)))))

    def forward(self, crops_16, img_512):
        fused_512, gate_weights = self.backbone.extract_fused_features(crops_16, img_512)
        emb = self.proj(fused_512)
        z = F.normalize(emb, p=2, dim=-1)

        c = F.normalize(self.prototypes, p=2, dim=-1)
        temp = torch.clamp(torch.exp(self.log_temp), min=0.03, max=0.20)
        logits = torch.matmul(z, c.t()) / temp
        return logits, z, c, gate_weights


# -------------------------------------------------------------
# 8. MODEL 9.3: Multi-Prototype Sub-Clustering Net
# -------------------------------------------------------------
class Model9_3_MultiProtoNet(nn.Module):
    def __init__(self, num_classes=3, prototypes_per_class=2, unfreeze_blocks=4, device="cpu", init_temp=0.07):
        super().__init__()
        self.num_classes = num_classes
        self.prototypes_per_class = prototypes_per_class
        self.total_prototypes = num_classes * prototypes_per_class

        self.backbone = Model9TriBranchBase(unfreeze_blocks=unfreeze_blocks, device=device)
        self.proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.prototypes = nn.Parameter(torch.randn(self.total_prototypes, 256))
        nn.init.orthogonal_(self.prototypes)
        self.log_temp = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(init_temp)))))

    def forward(self, crops_16, img_512):
        fused_512, gate_weights = self.backbone.extract_fused_features(crops_16, img_512)
        emb = self.proj(fused_512)
        z = F.normalize(emb, p=2, dim=-1)  # [B, 256]

        c = F.normalize(self.prototypes, p=2, dim=-1)  # [6, 256]
        temp = torch.clamp(torch.exp(self.log_temp), min=0.03, max=0.20)
        sim_all = torch.matmul(z, c.t()) / temp  # [B, 6]

        # Max-pool across sub-prototypes for each of the 3 classes
        sim_grouped = sim_all.view(-1, self.num_classes, self.prototypes_per_class)
        class_logits = sim_grouped.max(dim=-1).values  # [B, 3]
        return class_logits, z, c, gate_weights


# -------------------------------------------------------------
# 9. Outlier Radius Gate (Model 9.4 Automated Quality Control)
# -------------------------------------------------------------
class OutlierRadiusGate:
    """
    Computes mathematical cluster radius per class and tags outliers/margins:
      Radius_k = mean(dist) + 2.5 * std(dist)
    """
    def __init__(self, n_sigma=2.5):
        self.n_sigma = n_sigma
        self.centroids = {}
        self.radii = {}

    def fit(self, embeddings_dict):
        """Fit cluster radii using validated embeddings per class."""
        for cls_name, embs in embeddings_dict.items():
            c = np.mean(embs, axis=0)
            dists = np.linalg.norm(embs - c, axis=1)
            mu, sigma = np.mean(dists), np.std(dists)
            self.centroids[cls_name] = c
            self.radii[cls_name] = float(mu + self.n_sigma * sigma)

    def evaluate(self, z):
        """
        Input: z (single 256-D embedding)
        Returns: { 'is_outlier': bool, 'closest_class': str, 'min_dist': float, 'radius': float }
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
