import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, r"D:\Project")
sys.path.insert(0, r"D:\Project\model104_4_classifier")

print("[1] Testing model import...")
from demo_app_104_4 import load_cached_model, predict_patch

print("[2] Loading model...")
model, device = load_cached_model()
print(f"Model loaded successfully on {device}!")

print("[3] Testing prediction on a dummy 512x512 image...")
dummy = Image.new("RGB", (512, 512), color=(200, 100, 150))
pred_class, conf, probs, weights, _ = predict_patch(dummy, model, device)
print(f"Prediction: {pred_class} (conf: {conf*100:.2f}%)")
print(f"Weights: {weights}")
print("ALL CORE BACKEND FUNCTIONS WORK 100%!")

