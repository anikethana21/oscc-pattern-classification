"""
Model 4 Standalone Predictor
Provides easy-to-use API for tile prediction:
    from model4_export import Model4Predictor
    predictor = Model4Predictor()
    result = predictor.predict("path/to/tile.png")
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

from m4_stage1_model import Stage1WithBackbone
from m4_stage2_model import Stage2Specialist4Class

CLASS_NAMES = ["cohesive", "dispersive", "no_feature", "noncohesive"]
NAMED_COLS = [
    "area", "perimeter", "solidity", "eccentricity", "extent",
    "major_axis_length", "minor_axis_length", "aspect_ratio",
    "n_objects", "coverage_fraction"
]


def make_16_crops(img_512_tensor: torch.Tensor) -> torch.Tensor:
    """Extracts 16 overlapping 224x224 crops from a 512x512 tensor [3, 512, 512]."""
    coords = [0, 96, 192, 288]
    crops = []
    for y in coords:
        for x in coords:
            crop = img_512_tensor[:, y:y+224, x:x+224]
            crops.append(crop)
    return torch.stack(crops, dim=0)  # [16, 3, 224, 224]


class Model4Predictor:
    def __init__(
        self,
        s1_weights: str = "D:/Project/model4_cascade/weights_s1/stage1_best.pt",
        s2_weights: str = "D:/Project/model4_cascade/weights_s2/stage2_best.pt",
        cascade_threshold: float = 0.75,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        self.threshold = cascade_threshold
        print(f"[Model4Predictor] Initializing on {self.device} (threshold={self.threshold})...")

        # Load normalization stats
        s1_dir = Path(s1_weights).parent
        nm_file = s1_dir / "named_mean.npy"
        ns_file = s1_dir / "named_std.npy"
        self.named_mean = np.load(nm_file) if nm_file.exists() else None
        self.named_std  = np.load(ns_file) if ns_file.exists() else None

        # Load Stage 1
        self.s1_model = Stage1WithBackbone(device=str(self.device)).to(self.device)
        s1_ckpt = torch.load(s1_weights, map_location=self.device, weights_only=False)
        self.s1_model.load_state_dict(s1_ckpt["model_state"])
        self.s1_model.eval()

        # Load Stage 2
        self.s2_model = Stage2Specialist4Class(num_classes=4, device=str(self.device)).to(self.device)
        s2_ckpt = torch.load(s2_weights, map_location=self.device, weights_only=False)
        self.s2_model.load_state_dict(s2_ckpt["model_state"])
        self.s2_model.eval()

        self.use_amp = (self.device.type == "cuda")
        print("[Model4Predictor] Loaded Stage 1 & Stage 2 models successfully!")

    def preprocess_image(self, img_input):
        """Accepts path (str), PIL.Image, or numpy array. Returns (crops_224, img_256, named_tensor)."""
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
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_np - mean) / std
        img_tensor = torch.tensor(img_norm, dtype=torch.float32).permute(2, 0, 1)  # [3, 512, 512]

        crops_224 = make_16_crops(img_tensor).unsqueeze(0).to(self.device)  # [1, 16, 3, 224, 224]
        img_256 = F.interpolate(img_tensor.unsqueeze(0), size=(256, 256),
                                mode="bilinear", align_corners=False).to(self.device)  # [1, 3, 256, 256]

        named_zero = torch.zeros((1, len(NAMED_COLS)), dtype=torch.float32, device=self.device)
        return crops_224, img_256, named_zero

    @torch.no_grad()
    def predict(self, img_input) -> dict:
        """
        Runs full Model 4 cascade inference on a single tile.
        Returns detailed prediction dictionary.
        """
        crops_224, img_256, named_feats = self.preprocess_image(img_input)

        with torch.amp.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
            s1_logits = self.s1_model(crops_224, named_feats)
            s2_logits = self.s2_model(crops_224, img_256)

        p1 = torch.softmax(s1_logits, dim=1).cpu().numpy()[0]
        p2 = torch.softmax(s2_logits, dim=1).cpu().numpy()[0]

        s1_coh = float(p1[0])
        s1_nf  = float(p1[1])
        s1_max = max(s1_coh, s1_nf)
        s1_choice = "cohesive" if s1_coh >= s1_nf else "no_feature"

        s2_coh = float(p2[0])
        s2_dis = float(p2[1])
        s2_nf  = float(p2[2])
        s2_nc  = float(p2[3])
        s2_idx = int(np.argmax(p2))
        s2_pred_cls = CLASS_NAMES[s2_idx]
        s2_conf     = float(p2[s2_idx])

        # Cascade Decision
        if s1_max >= self.threshold:
            pred_class = s1_choice
            confidence = s1_max
            decided_by = f"Stage1 ({s1_choice})"
            effective_probs = {
                "cohesive": round(s1_coh, 4),
                "dispersive": round(s2_dis, 4),
                "no_feature": round(s1_nf, 4),
                "noncohesive": round(s2_nc, 4),
            }
        else:
            pred_class = s2_pred_cls
            confidence = s2_conf
            decided_by = "Stage2 (Specialist)"
            effective_probs = {
                "cohesive": round(s2_coh, 4),
                "dispersive": round(s2_dis, 4),
                "no_feature": round(s2_nf, 4),
                "noncohesive": round(s2_nc, 4),
            }

        return {
            "pred_class": pred_class,
            "confidence": round(confidence, 4),
            "decided_by": decided_by,
            "probabilities": effective_probs,
            "stage1_probs": {"cohesive": round(s1_coh, 4), "no_feature": round(s1_nf, 4)},
            "stage2_probs": {
                "cohesive": round(s2_coh, 4),
                "dispersive": round(s2_dis, 4),
                "no_feature": round(s2_nf, 4),
                "noncohesive": round(s2_nc, 4),
            }
        }
