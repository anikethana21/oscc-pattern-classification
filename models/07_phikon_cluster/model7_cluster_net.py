"""
model7_cluster_net.py
---------------------
Model 7: Prototypical Phikon Cluster Network with Geometric Outlier Pruning.
- Backbone: Phikon ViT-Base (frozen base + top 2 unfrozen transformer blocks for oral cancer domain adaptation).
- Aggregator: Multi-Crop Mean + Max pooling (1536-dim).
- Projection Head: MLP mapping 1536 -> 512 -> 256 with L2-normalized unit sphere embedding.
- Prototypical Clustering: Learnable cluster centroids on the hypersphere for Cohesive, Dispersive, Noncohesive.
- Outlier Pruning Gate: Flags and attenuates backpropagation of anomalous / noisy tiles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

CLASSES_3 = ["cohesive", "dispersive", "noncohesive"]

def load_phikon_backbone(device="cpu"):
    model = AutoModel.from_pretrained("owkin/phikon")
    # Freeze all layers first
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze only the last 2 transformer blocks for domain adaptation
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        for layer in model.encoder.layer[-2:]:
            for p in layer.parameters():
                p.requires_grad = True
    return model


class PrototypicalClusterHead(nn.Module):
    def __init__(self, in_dim=1536, emb_dim=256, num_classes=3, init_temp=0.07):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, emb_dim),
        )
        # Learnable class prototype centroids on the unit sphere
        self.prototypes = nn.Parameter(torch.randn(num_classes, emb_dim))
        nn.init.orthogonal_(self.prototypes)
        
        # Learnable temperature parameter (bounded)
        self.log_temp = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(init_temp)))))
        
    def get_normalized_prototypes(self):
        return F.normalize(self.prototypes, p=2, dim=-1)

    def forward(self, feat_1536):
        # 1. Project and L2 normalize embeddings onto unit hypersphere
        emb = self.proj(feat_1536)
        z = F.normalize(emb, p=2, dim=-1) # [B, 256]
        
        # 2. Normalized prototypes
        c = self.get_normalized_prototypes() # [3, 256]
        
        # 3. Cosine similarity between embeddings and prototypes
        temp = torch.clamp(torch.exp(self.log_temp), min=0.03, max=0.20)
        sim = torch.matmul(z, c.t()) # [B, 3] in range [-1, 1]
        logits = sim / temp
        
        return z, logits, c


class Model7PhikonCluster(nn.Module):
    def __init__(self, num_classes=3, device="cpu"):
        super().__init__()
        self.device = device
        print("Model 7: Initializing Phikon ViT Backbone (Top 2 blocks trainable)...")
        self.phikon = load_phikon_backbone(device).to(device)
        self.cluster_head = PrototypicalClusterHead(in_dim=1536, emb_dim=256, num_classes=num_classes)
        
    def extract_features(self, crops_16):
        """
        Input: [B, 16, 3, 224, 224]
        Output: [B, 1536] (Mean + Max pooled)
        """
        B, N, C, H, W = crops_16.shape
        flat = crops_16.view(B * N, C, H, W)
        
        # Forward through Phikon
        out = self.phikon(pixel_values=flat)
        cls_token = out.last_hidden_state[:, 0, :] # [B*16, 768]
        cls_token = cls_token.view(B, N, 768)
        
        mean_p = cls_token.mean(dim=1) # [B, 768]
        max_p  = cls_token.max(dim=1).values # [B, 768]
        return torch.cat([mean_p, max_p], dim=1) # [B, 1536]

    def forward(self, crops_16):
        feat = self.extract_features(crops_16)
        z, logits, prototypes = self.cluster_head(feat)
        return z, logits, prototypes


class OutlierPrunedPrototypicalLoss(nn.Module):
    """
    Computes Prototypical Cross-Entropy Loss with Geometric Outlier Pruning.
    Tiles whose cosine similarity to their assigned prototype falls below 
    the dynamic cluster boundary are attenuated (treated as label noise).
    """
    def __init__(self, outlier_threshold=0.20):
        super().__init__()
        self.outlier_threshold = outlier_threshold

    def forward(self, logits, labels, z, prototypes):
        # Cosine similarity of each tile to its true assigned prototype
        prototypes_norm = F.normalize(prototypes, p=2, dim=-1)
        true_prototypes = prototypes_norm[labels] # [B, 256]
        true_sim = torch.sum(z * true_prototypes, dim=-1) # [B]
        
        # Identify outliers: tiles that are far from their declared class cluster
        # Threshold: if cosine similarity < outlier_threshold (e.g. 0.20, orthogonal or opposite)
        is_outlier = (true_sim < self.outlier_threshold)
        
        # Sample weights: regular samples get weight 1.0, outliers are suppressed to 0.05
        weights = torch.ones_like(true_sim)
        weights[is_outlier] = 0.05
        
        # Weighted Cross Entropy
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = F.nll_loss(log_probs, labels, reduction="none")
        weighted_loss = torch.mean(weights * nll_loss)
        
        num_outliers = int(torch.sum(is_outlier).item())
        return weighted_loss, num_outliers

