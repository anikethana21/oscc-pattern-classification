"""
Model 4 Specialist 4-Class Predictor
------------------------------------
Uses the Champion Model 4 Specialist Network (Stage 2 Gated Network)
with dual Phikon branches (frozen + fine-tuned) and TileCNN.
Properly extracts 16 multi-scale 128->224 grid crops as trained.
Predicts all 4 OSCC patterns: cohesive, dispersive, no_feature, noncohesive.
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from m4_stage2_model import Stage2Specialist4Class

CLASS_NAMES = ["cohesive", "dispersive", "no_feature", "noncohesive"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def make_16_crops_correct(img_tensor_512: torch.Tensor) -> torch.Tensor:
    """
    Extracts 16 grid crops (128x128 resized to 224x224 via bilinear interpolation)
    matching the exact training pipeline of Model 4 Stage 2.
    Input: [3, 512, 512] -> Output: [16, 3, 224, 224]
    """
    crops = []
    for r in range(4):
        for c in range(4):
            crop = img_tensor_512[:, r*128:(r+1)*128, c*128:(c+1)*128]
            crop = F.interpolate(
                crop.unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)
            crops.append(crop)
    return torch.stack(crops, dim=0)


class Model4Predictor:
    def __init__(
        self,
        s1_weights: str = "D:/Project/model4_cascade/weights_s1/stage1_best.pt",
        s2_weights: str = "D:/Project/model4_3_2_retrained/weights/model4_3_2_best.pt",
        cascade_threshold: float = 0.75,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        print(f"[Model4Predictor] Initializing Model 4.3.2 (Champion 4-Class Specialist) on {self.device}...")

        # If absolute path doesn't exist, try local relative weights
        if not os.path.exists(s2_weights):
            local_fallback = os.path.join(current_dir, "weights", "model4_3_2_best.pt")
            if os.path.exists(local_fallback):
                s2_weights = local_fallback

        # Load Stage 2 Specialist 4-Class model
        self.s2_model = Stage2Specialist4Class(num_classes=4, device=str(self.device)).to(self.device)
        s2_ckpt = torch.load(s2_weights, map_location=self.device, weights_only=False)
        self.s2_model.load_state_dict(s2_ckpt["model_state"])
        self.s2_model.eval()

        self.use_amp = (self.device.type == "cuda")
        print("[Model4Predictor] Loaded Model 4 Specialist 4-Class model successfully!")

    def preprocess_image(self, img_input):
        """Accepts path (str), PIL.Image, or numpy array. Returns (crops_224, img_256)."""
        if isinstance(img_input, (str, Path)):
            img = Image.open(str(img_input)).convert("RGB")
        elif isinstance(img_input, np.ndarray):
            img = Image.fromarray(img_input).convert("RGB")
        elif isinstance(img_input, Image.Image):
            img = img_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(img_input)}")

        # Resize to 512x512
        img = img.resize((512, 512), Image.Resampling.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0

        # Standard ImageNet normalization
        img_norm = (img_np - IMAGENET_MEAN) / IMAGENET_STD
        img_tensor = torch.tensor(img_norm, dtype=torch.float32).permute(2, 0, 1)  # [3, 512, 512]

        crops_224 = make_16_crops_correct(img_tensor).unsqueeze(0).to(self.device)  # [1, 16, 3, 224, 224]
        img_256 = F.interpolate(img_tensor.unsqueeze(0), size=(256, 256),
                                mode="bilinear", align_corners=False).to(self.device)  # [1, 3, 256, 256]

        return crops_224, img_256

    @torch.no_grad()
    def predict(self, img_input) -> dict:
        """
        Runs Model 4 4-class specialist inference on a single tile.
        Returns prediction dictionary across all 4 classes:
          cohesive, dispersive, no_feature, noncohesive
        """
        crops_224, img_256 = self.preprocess_image(img_input)

        with torch.amp.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
            logits = self.s2_model(crops_224, img_256)

        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

        effective_probs = {
            "cohesive": round(float(probs[0]), 4),
            "dispersive": round(float(probs[1]), 4),
            "no_feature": round(float(probs[2]), 4),
            "noncohesive": round(float(probs[3]), 4),
        }

        return {
            "pred_class": pred_class,
            "confidence": round(confidence, 4),
            "decided_by": "Model 4 (Specialist)",
            "probabilities": effective_probs,
        }
