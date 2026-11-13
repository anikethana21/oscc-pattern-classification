import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
model6.py
---------
Model 6: Step-by-Step Waterfall Early-Exit Cascade Network

Architecture:
  3 branches (Phikon A, Phikon B, TileCNN) -> dynamic gate -> fused [B, 512]

  Stepwise Cascade of Subnetworks:
  - fused [B, 512] -> Subnetwork 1 [B, 256] -> Exit 1 (no_feature) logit [B]
  - Subnetwork 1 [B, 256] -> Subnetwork 2 [B, 256] -> Exit 2 (dispersive) logit [B]
  - Subnetwork 2 [B, 256] -> Subnetwork 3 [B, 256] -> Exit 3 (cohesive) logit [B]
  - Subnetwork 3 [B, 256] -> Subnetwork 4 [B, 256] -> Exit 4 (noncohesive) logit [B]
  - Subnetwork 4 [B, 256] -> Concluder Subnetwork [B, 128] -> Concluder Head [B, 4]

All backbone feature extractors are preloaded from Model 4 Stage 2 (stage2_best.pt).
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel

sys.path.insert(0, "D:/Project/model4_cascade")
from m4_stage2_model import TileCNN, GatingNetwork, load_phikon, freeze_all, unfreeze_last_n_blocks


class EarlyExitModel(nn.Module):
    """
    Model 6 - Stepwise Early-Exit Waterfall Cascade Classifier
    """
    CLASS_ORDER = ["cohesive", "dispersive", "no_feature", "noncohesive"]

    def __init__(self, cfg: dict, device="cpu"):
        super().__init__()
        m_cfg = cfg["model"]
        hidden_dim = m_cfg.get("shared_hidden", 256)
        dropout_rate = m_cfg.get("dropout_fused", 0.2)
        unfreeze_n = m_cfg.get("unfreeze_last_n_phikon", 3)

        # 1. Branch A: frozen Phikon
        self.phikon_a = load_phikon(device)
        freeze_all(self.phikon_a)

        # 2. Branch B: Phikon with last N blocks unfrozen
        self.phikon_b = load_phikon(device)
        unfreeze_last_n_blocks(self.phikon_b, n=unfreeze_n)

        # 3. Branch C: TileCNN
        self.cnn = TileCNN(out_dim=512)

        # 4. Projections & Gate
        self.proj_a = nn.Sequential(nn.Linear(1536, 512), nn.GELU())
        self.proj_b = nn.Sequential(nn.Linear(1536, 512), nn.GELU())
        self.gate = GatingNetwork(in_dim=3584, num_branches=3)

        # 5. Stepwise Subnetworks
        # Subnetwork 1 (no_feature)
        self.net_stage1 = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.head_no_feature = nn.Linear(hidden_dim, 1)

        # Subnetwork 2 (dispersive)
        self.net_stage2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.head_dispersive = nn.Linear(hidden_dim, 1)

        # Subnetwork 3 (cohesive)
        self.net_stage3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.head_cohesive = nn.Linear(hidden_dim, 1)

        # Subnetwork 4 (noncohesive)
        self.net_stage4 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.head_noncohesive = nn.Linear(hidden_dim, 1)

        # Concluder Subnetwork (Fallback 4-class arbiter)
        self.net_concluder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.concluder = nn.Linear(128, 4)

    def load_m4_backbone(self, checkpoint_path: str):
        """
        Load pretrained weights for backbone branches from Model 4 Stage 2.
        """
        if not os.path.exists(checkpoint_path):
            print(f"[Warning] M4 weights path not found: {checkpoint_path}")
            return False
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model_dict = self.state_dict()
        matched = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(matched)
        self.load_state_dict(model_dict)
        print(f"[Model 6] Successfully loaded {len(matched)} backbone parameter tensors from {checkpoint_path}!")
        return True

    def _extract_phikon(self, model, crops_224: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = crops_224.shape
        flat = crops_224.view(B * N, C, H, W)
        enable_grad = any(p.requires_grad for p in model.parameters())
        with torch.set_grad_enabled(enable_grad):
            out = model(pixel_values=flat)
            cls = out.last_hidden_state[:, 0, :]
        cls = cls.view(B, N, 768)
        mean_p = cls.mean(dim=1)
        max_p = cls.max(dim=1).values
        return torch.cat([mean_p, max_p], dim=1)

    def forward(self, crops_224: torch.Tensor, img_256: torch.Tensor) -> dict:
        # Branch encodings
        F_a = self._extract_phikon(self.phikon_a, crops_224)
        F_b = self._extract_phikon(self.phikon_b, crops_224)
        F_c = self.cnn(img_256)

        # Dynamic gating
        concat = torch.cat([F_a, F_b, F_c], dim=1)
        weights = self.gate(concat)
        w_a, w_b, w_c = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]

        P_a = self.proj_a(F_a)
        P_b = self.proj_b(F_b)
        fused = w_a * P_a + w_b * P_b + w_c * F_c  # [B, 512]

        # Stepwise Subnetworks Cascade
        h1 = self.net_stage1(fused)
        logit_nf = self.head_no_feature(h1).squeeze(1)

        h2 = self.net_stage2(h1)
        logit_dis = self.head_dispersive(h2).squeeze(1)

        h3 = self.net_stage3(h2)
        logit_coh = self.head_cohesive(h3).squeeze(1)

        h4 = self.net_stage4(h3)
        logit_nc = self.head_noncohesive(h4).squeeze(1)

        h5 = self.net_concluder(h4)
        concluder_logits = self.concluder(h5)

        return {
            "head_no_feature": logit_nf,
            "head_dispersive": logit_dis,
            "head_cohesive": logit_coh,
            "head_noncohesive": logit_nc,
            "concluder": concluder_logits,
        }
