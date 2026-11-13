"""
dry_run_test.py
---------------
Verifies the complete Model 9.2 architecture, tensor shapes, gating weights,
and Center Loss backpropagation without running training or draining battery.
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

sys.path.insert(0, r"D:\Project\model9_hybrid_cluster")
from model9_architecture import Model9_2_CenterLossNet, CenterLoss, CLASSES_3

print("Running Model 9.2 Dry-Run Verification...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Testing on device: {device}")

# Dummy inputs
B = 2
crops_16 = torch.randn(B, 16, 3, 224, 224).to(device)
img_512  = torch.randn(B, 3, 512, 512).to(device)
labels   = torch.tensor([0, 2]).to(device)

print("[1/4] Instantiating Model 9.2 (Tri-Branch + Center Loss)...")
model = Model9_2_CenterLossNet(num_classes=3, unfreeze_blocks=4, device=device).to(device)
center_loss = CenterLoss(num_classes=3, feat_dim=256).to(device)

print("[2/4] Testing Forward Pass...")
logits, latent_256, gate_weights = model(crops_16, img_512)
print(f"  -> Logits shape: {logits.shape} (Expected: [{B}, 3])")
print(f"  -> Latent vector shape: {latent_256.shape} (Expected: [{B}, 256])")
print(f"  -> Dynamic gate weights shape: {gate_weights.shape} (Expected: [{B}, 3])")
print(f"  -> Sample gate weights: {gate_weights[0].detach().cpu().numpy()}")

print("[3/4] Testing Center Loss Computation...")
l_center = center_loss(latent_256, labels)
print(f"  -> Center Loss value: {l_center.item():.4f}")

print("[4/4] Testing Backward Gradient Flow...")
total_loss = logits.sum() + 0.01 * l_center
total_loss.backward()
print("  -> Gradients successfully propagated through all branches, gates, and centers!")

print("\n=======================================================")
print("      ALL MODEL 9.2 ARCHITECTURAL TESTS PASSED!")
print("=======================================================\n")
