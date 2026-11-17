import os
import sys
from pathlib import Path
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# Set environment variable to avoid OpenMP duplicate runtime warning
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -----------------------------------------------------------------------------
# Base Repository Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path("D:/Project")
if not PROJECT_ROOT.exists():
    # Fallback to current directory or parent
    PROJECT_ROOT = Path(__file__).resolve().parent
    if not (PROJECT_ROOT / "models").exists() and (PROJECT_ROOT.parent / "models").exists():
        PROJECT_ROOT = PROJECT_ROOT.parent

# Add models path to sys.path
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "104.4_tri_branch_wsi_7class"))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "109.4_metric_clustering_5class"))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "04_cascade_series" / "4.3.2_champion_40x"))

# -----------------------------------------------------------------------------
# Tri-Branch Model 104.4 Architecture Definition (Self-Contained & Production-Ready)
# -----------------------------------------------------------------------------
from transformers import AutoModel

def load_phikon_frozen(device="cpu"):
    """Branch A: Frozen histological foundation ViT (owkin/phikon ViT-B)."""
    model = AutoModel.from_pretrained("owkin/phikon")
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


def load_phikon_finetuned(device="cpu", unfreeze_blocks=4):
    """Branch B: Phikon ViT with top-N transformer blocks unfrozen for oral adaptation."""
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
    """Branch C: Localized cell border and chromatin texture extractor (512-dim)."""
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


class TriBranchWSIClassifier(nn.Module):
    """
    Exact Model 104.4 Architecture:
    Tri-Branch Gated Feature Fusion Network (Frozen ViT + Fine-Tuned ViT + TileCNN)
    """
    def __init__(self, num_classes=7, unfreeze_blocks=4, device="cpu"):
        super().__init__()
        self.num_classes = num_classes
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

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, num_classes),
        )

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
        latent_256 = self.latent_proj(fused_512)
        logits = self.classifier(latent_256)
        return logits, latent_256, weights


# -----------------------------------------------------------------------------
# Configuration & Class Dictionaries
# -----------------------------------------------------------------------------
MODELS_CATALOG = {
    "Model 104.4 · Tri-Branch WSI (7-Class Champion)": {
        "id": "104.4",
        "checkpoint": PROJECT_ROOT / "models" / "104.4_tri_branch_wsi_7class" / "weights" / "model104_4_best.pt",
        "classes": ["cohesive", "dysplasia", "noncohesive", "dispersive", "normal", "no_feature", "blank"],
        "display_names": {
            "cohesive": "Cohesive Invasive Pattern",
            "dysplasia": "Epithelial Dysplasia (Pre-Malignant)",
            "noncohesive": "Non-Cohesive Invasive Pattern",
            "dispersive": "Dispersive Invasive Cells (High Risk)",
            "normal": "Normal Squamous Epithelium",
            "no_feature": "Stroma / Fibrovascular / No Feature",
            "blank": "Glass Slide / Blank Background"
        },
        "badge_colors": {
            "cohesive": "#1f77b4",
            "dysplasia": "#ff7f0e",
            "noncohesive": "#2ca02c",
            "dispersive": "#d62728",
            "normal": "#17becf",
            "no_feature": "#7f7f7f",
            "blank": "#bcbd22"
        },
        "acc": "97.19%",
        "f1": "0.9747",
        "zoom": "20x WSI"
    },
    "Model 109.4 · Metric Sub-Clustering (5-Class)": {
        "id": "109.4",
        "checkpoint": PROJECT_ROOT / "models" / "109.4_metric_clustering_5class" / "weights" / "model109_4_best.pt",
        "classes": ["cohesive", "dysplasia", "noncohesive", "dispersive", "normal"],
        "display_names": {
            "cohesive": "Cohesive Invasive Pattern",
            "dysplasia": "Epithelial Dysplasia",
            "noncohesive": "Non-Cohesive Invasive Pattern",
            "dispersive": "Dispersive Invasive Cells",
            "normal": "Normal Squamous Epithelium"
        },
        "badge_colors": {
            "cohesive": "#1f77b4",
            "dysplasia": "#ff7f0e",
            "noncohesive": "#2ca02c",
            "dispersive": "#d62728",
            "normal": "#17becf"
        },
        "acc": "97.33%",
        "f1": "0.9750",
        "zoom": "20x WSI"
    },
    "Model 4.3.2 · Cascade Champion (40x 3-Class)": {
        "id": "4.3.2",
        "checkpoint": PROJECT_ROOT / "models" / "04_cascade_series" / "4.3.2_champion_40x" / "weights" / "model4_3_2_best.pt",
        "classes": ["cohesive", "dispersive", "noncohesive"],
        "display_names": {
            "cohesive": "Cohesive Neoplastic Architecture",
            "dispersive": "Dispersive Invasive Strands",
            "noncohesive": "Non-Cohesive Cellular Islands"
        },
        "badge_colors": {
            "cohesive": "#1f77b4",
            "dispersive": "#d62728",
            "noncohesive": "#2ca02c"
        },
        "acc": "88.09%",
        "f1": "0.8809",
        "zoom": "40x High-Power"
    }
}

# ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# -----------------------------------------------------------------------------
# Preprocessing & Crop Generator
# -----------------------------------------------------------------------------
def make_16_crops(img_tensor_512: torch.Tensor) -> torch.Tensor:
    """Splits 512x512 tensor into 16 localized 128x128 crops, resized to 224x224."""
    crops = []
    for r in range(4):
        for c in range(4):
            crop = img_tensor_512[:, r * 128:(r + 1) * 128, c * 128:(c + 1) * 128]
            crop = F.interpolate(crop.unsqueeze(0), size=(224, 224),
                                 mode="bilinear", align_corners=False).squeeze(0)
            crops.append(crop)
    return torch.stack(crops, dim=0)


def load_image_safely(file_or_path):
    """Safely opens image or whole slide crop, preventing TIFF memory errors."""
    try:
        if hasattr(file_or_path, "name") and file_or_path.name.lower().endswith((".tif", ".tiff")):
            import tifffile
            with tifffile.TiffFile(file_or_path) as tif:
                page = tif.series[0].levels[-1] if len(tif.series[0].levels) > 1 else tif.pages[0]
                arr = page.asarray()
                if arr.ndim == 2:
                    arr = np.stack([arr]*3, axis=-1)
                elif arr.shape[-1] > 3:
                    arr = arr[..., :3]
                return Image.fromarray(arr).convert("RGB")
        return Image.open(file_or_path).convert("RGB")
    except Exception:
        return Image.open(file_or_path).convert("RGB")


# -----------------------------------------------------------------------------
# Streamlit App Layout & Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="OSCC Pattern Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar: Model Selection & Hardware Diagnostics
st.sidebar.title("🔬 Diagnostics Control")
st.sidebar.markdown("---")

selected_model_name = st.sidebar.selectbox(
    "Active Deep Learning Model",
    options=list(MODELS_CATALOG.keys()),
    index=0
)
model_meta = MODELS_CATALOG[selected_model_name]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Mode"

st.sidebar.subheader("Hardware & Model Specs")
st.sidebar.info(f"**Runtime Device:** `{device.type.upper()}` ({gpu_name})")
st.sidebar.success(f"**Optical Zoom:** `{model_meta['zoom']}`")
st.sidebar.metric("Validation Accuracy", model_meta['acc'])
st.sidebar.metric("Validation Macro-F1", model_meta['f1'])
st.sidebar.caption(f"Checkpoint: `{model_meta['checkpoint'].name}`")

st.sidebar.markdown("---")
input_mode = st.sidebar.radio("Select Image Source", ["Upload Custom Patch", "Pick Sample from Dataset"])


# -----------------------------------------------------------------------------
# Model Caching & Loader
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model(model_name: str):
    meta = MODELS_CATALOG[model_name]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if meta["id"] == "104.4":
        model = TriBranchWSIClassifier(num_classes=len(meta["classes"]), unfreeze_blocks=4, device=dev).to(dev)
    elif meta["id"] == "109.4":
        from model109_4_architecture import Model109_4_ClusterNet
        model = Model109_4_ClusterNet(num_classes=5, prototypes_per_class=2, unfreeze_blocks=4, device=dev).to(dev)
    else:
        from m4_stage2_model import TriBranchClassifier
        model = TriBranchClassifier(num_classes=3).to(dev)

    ckpt_path = meta["checkpoint"]
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=dev)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        clean_dict = {k.replace("module.", ""): v for k, v in ckpt.items()}
        model.load_state_dict(clean_dict, strict=False)
    
    model.eval()
    return model, dev


# -----------------------------------------------------------------------------
# Main Application Header
# -----------------------------------------------------------------------------
st.title("🔬 Oral Squamous Cell Carcinoma (OSCC) Classifier")
st.markdown(
    f"Real-time diagnostic triage powered by **{selected_model_name}** | "
    f"Target Resolution: **{model_meta['zoom']}** | Target Classes: **{len(model_meta['classes'])}**"
)

# -----------------------------------------------------------------------------
# Input Handling
# -----------------------------------------------------------------------------
image_to_process = None
sample_title = ""

if input_mode == "Upload Custom Patch":
    uploaded = st.file_uploader(
        "Upload Histopathology Image Tile or Patch...",
        type=["jpg", "jpeg", "png", "tif", "tiff"]
    )
    if uploaded is not None:
        image_to_process = load_image_safely(uploaded)
        sample_title = uploaded.name
else:
    # Discover available sample tiles
    sample_files = list((PROJECT_ROOT / "40x-20261024T155519Z-1-001" / "40x").glob("*/*.png"))
    if not sample_files:
        sample_files = list(PROJECT_ROOT.glob("**/*.png"))[:15]
    
    if sample_files:
        sample_choices = {f"{f.parent.name} / {f.name}": f for f in sample_files[:12]}
        selected_choice = st.selectbox("Choose a pre-indexed histological tile:", list(sample_choices.keys()))
        image_to_process = load_image_safely(sample_choices[selected_choice])
        sample_title = selected_choice
    else:
        st.warning("No local sample tiles discovered. Please use 'Upload Custom Patch'.")


# -----------------------------------------------------------------------------
# Inference Execution & Results Display
# -----------------------------------------------------------------------------
if image_to_process is not None:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("🖼️ Histology Input View")
        st.image(image_to_process, caption=f"Sample: {sample_title}", use_container_width=True)
        st.caption(f"Original Resolution: `{image_to_process.width} × {image_to_process.height}` px | Mode: `{image_to_process.mode}`")

    with col2:
        st.subheader("📊 Diagnostic Prediction")
        with st.spinner("Executing deep neural inference..."):
            model, dev = get_model(selected_model_name)

            # Preprocessing
            img_512 = image_to_process.resize((512, 512), Image.Resampling.BILINEAR)
            tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            t_512 = tf(img_512)
            crops_16 = make_16_crops(t_512).unsqueeze(0).to(dev)
            t_512 = t_512.unsqueeze(0).to(dev)

            # Forward pass
            with torch.no_grad():
                with torch.amp.autocast("cuda" if dev.type == "cuda" else "cpu"):
                    if model_meta["id"] == "104.4":
                        logits, latent, gate_weights = model(crops_16, t_512)
                        weights_arr = gate_weights.squeeze(0).cpu().numpy()
                    elif model_meta["id"] == "109.4":
                        logits, _, _, _ = model(crops_16, t_512)
                        weights_arr = None
                    else:
                        logits = model(crops_16, t_512) if hasattr(model, "crops") else model(t_512)
                        weights_arr = None

                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            pred_idx = int(np.argmax(probs))
            pred_class = model_meta["classes"][pred_idx]
            pred_label = model_meta["display_names"][pred_class]
            confidence = float(probs[pred_idx])

        # Prediction display card
        st.success(f"### **Prediction:** {pred_label.upper()}")
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Top Confidence Score", f"{confidence * 100:.2f}%")
        with m_col2:
            st.metric("Pattern Category", pred_class.capitalize())

        # Gating contribution breakdown (Model 104.4 specific)
        if weights_arr is not None:
            st.markdown("##### 🧬 Tri-Branch Attention Gating Contribution")
            g_col1, g_col2, g_col3 = st.columns(3)
            with g_col1:
                st.metric("Frozen ViT (Phikon)", f"{weights_arr[0]*100:.1f}%")
            with g_col2:
                st.metric("Tuned ViT (Oral)", f"{weights_arr[1]*100:.1f}%")
            with g_col3:
                st.metric("TileCNN Texture", f"{weights_arr[2]*100:.1f}%")

        st.markdown("---")
        
        # Probabilities Expander
        with st.expander("🔍 Complete Class Probability Distribution", expanded=True):
            for i, cls_name in enumerate(model_meta["classes"]):
                p_val = float(probs[i])
                display_str = model_meta["display_names"][cls_name]
                st.write(f"**{display_str}**")
                st.progress(min(p_val, 1.0))
                st.caption(f"Probability: `{p_val * 100:.2f}%`")
else:
    st.info("👈 Please upload a histopathology image or select a pre-indexed sample tile from the sidebar to begin classification.")
