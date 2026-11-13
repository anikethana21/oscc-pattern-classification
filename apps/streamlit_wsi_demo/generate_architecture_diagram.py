"""
generate_architecture_diagram.py
---------------------------------
Generates the publication-grade dark-themed neural architecture diagram for Model 104.4,
matching the exact design language of the user's reference diagram:
  - Deep dark background (#111115)
  - Rounded purple input branch blocks (Phikon frozen, Phikon fine-tuned, TileCNN)
  - Dynamic Softmax Attention Gating block (weighted sum)
  - Multi-layer dense synaptic network (512 fused -> 256 latent -> 128 head)
  - 7 output nodes with distinct labels and Focal Loss formulation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

OUTPUT_PATH = r"D:\Project\model104_4_architecture_diagram.png"
DESKTOP_PATH = r"C:\Users\anike\Desktop\model104_4_architecture_diagram.png"
ARTIFACT_PATH = r"C:\Users\anike\.gemini\antigravity\brain\40d4b308-7194-4358-b8e7-0f6a076836ae\model104_4_architecture_diagram.png"

def create_diagram():
    fig = plt.figure(figsize=(15, 7.5), facecolor="#101116")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#101116")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Colors
    PURPLE_BG = "#343268"
    PURPLE_BORDER = "#5C58B0"
    GATE_BG = "#22232A"
    GATE_BORDER = "#4E5160"
    NODE_GREEN = "#1ABC9C"
    NODE_GREEN_OUTLINE = "#16A085"
    LINE_COLOR = "#2C3E50"
    TEXT_WHITE = "#FFFFFF"
    TEXT_MUTED = "#A0A5BA"
    GOLD_ACCENT = "#F39C12"

    # --- 1. Input Branches (Left) ---
    branches = [
        ("Phikon frozen\n(Branch A · 1536→512)", 5.6),
        ("Phikon adaptive\n(Branch B · Blocks 8-11)", 3.8),
        ("TileCNN\n(Branch C · 512-dim)", 2.0)
    ]
    box_w, box_h = 2.4, 1.2
    box_x = 0.8

    for name, y in branches:
        rect = patches.FancyBboxPatch(
            (box_x, y - box_h/2), box_w, box_h,
            boxstyle="round,pad=0.1,rounding_size=0.15",
            facecolor=PURPLE_BG, edgecolor=PURPLE_BORDER, linewidth=2
        )
        ax.add_patch(rect)
        ax.text(box_x + box_w/2, y, name, color=TEXT_WHITE,
                fontsize=11, fontweight="bold", ha="center", va="center")

    # --- 2. Dynamic Gating Block (Middle Left) ---
    gate_x = 4.2
    gate_y = 3.8
    gate_w = 1.9
    gate_h = 3.6
    gate_rect = patches.FancyBboxPatch(
        (gate_x, gate_y - gate_h/2), gate_w, gate_h,
        boxstyle="round,pad=0.1,rounding_size=0.2",
        facecolor=GATE_BG, edgecolor=GATE_BORDER, linewidth=2
    )
    ax.add_patch(gate_rect)
    ax.text(gate_x + gate_w/2, gate_y + 0.5, "Gate", color=TEXT_WHITE,
            fontsize=15, fontweight="bold", ha="center", va="center")
    ax.text(gate_x + gate_w/2, gate_y - 0.2, "dynamic\nweighted sum\n(softmax)", color=TEXT_MUTED,
            fontsize=10.5, ha="center", va="center")

    # Arrows from branches to gate
    for _, y in branches:
        ax.annotate("", xy=(gate_x, y), xytext=(box_x + box_w, y),
                    arrowprops=dict(arrowstyle="-|>", color="#7F8C8D", lw=1.8, mutation_scale=15))

    # Arrow from Gate to First Dense Layer
    ax.annotate("", xy=(7.0, 3.8), xytext=(gate_x + gate_w, 3.8),
                arrowprops=dict(arrowstyle="-|>", color="#BDC3C7", lw=2.2, mutation_scale=18))

    # --- 3. Neural Dense Layers (Middle) ---
    # Layer 1: 512 Fused (5 nodes)
    l1_x = 7.3
    l1_nodes = np.linspace(2.2, 5.4, 5)
    for y in l1_nodes:
        circle = plt.Circle((l1_x, y), 0.16, color=NODE_GREEN, ec=NODE_GREEN_OUTLINE, lw=1.5, zorder=4)
        ax.add_patch(circle)
    ax.text(l1_x, 1.6, "512\nfused", color=TEXT_WHITE, fontsize=11, fontweight="bold", ha="center")

    # Layer 2: 256 Shared Latent (4 nodes)
    l2_x = 8.8
    l2_nodes = np.linspace(2.6, 5.0, 4)
    for y in l2_nodes:
        circle = plt.Circle((l2_x, y), 0.16, color=NODE_GREEN, ec=NODE_GREEN_OUTLINE, lw=1.5, zorder=4)
        ax.add_patch(circle)
    ax.text(l2_x, 1.6, "256\nshared", color=TEXT_WHITE, fontsize=11, fontweight="bold", ha="center")

    # Connect Layer 1 to Layer 2
    for y1 in l1_nodes:
        for y2 in l2_nodes:
            ax.plot([l1_x, l2_x], [y1, y2], color="#203336", lw=1.0, zorder=2)

    # --- 4. Output Layer: 7 Classes (Right) ---
    l3_x = 10.7
    classes_7 = [
        ("cohesive", "Invasive Sheets (Grade I/II)"),
        ("dysplasia", "Pre-Malignant Oral Lesion"),
        ("noncohesive", "Infiltrative Micro-Nests"),
        ("dispersive", "Single-Cell Pattern (High Risk)"),
        ("normal", "Benign Oral Mucosa"),
        ("no_feature", "Stroma / Extravasated RBCs"),
        ("blank", "Optical Glass Background")
    ]
    l3_nodes = np.linspace(1.2, 6.4, 7)[::-1]  # Top to bottom

    for idx, ((cname, subtext), y) in enumerate(zip(classes_7, l3_nodes)):
        # Node circle
        circle = plt.Circle((l3_x, y), 0.16, color=NODE_GREEN, ec=NODE_GREEN_OUTLINE, lw=1.5, zorder=4)
        ax.add_patch(circle)
        # Class Label
        ax.text(l3_x + 0.35, y, cname, color=TEXT_WHITE,
                fontsize=11.5, fontweight="bold", ha="left", va="center")
        ax.text(l3_x + 2.05, y, f"· {subtext}", color=TEXT_MUTED,
                fontsize=9.5, ha="left", va="center")

        # Connect Layer 2 to Output Layer
        for y2 in l2_nodes:
            ax.plot([l2_x, l3_x], [y2, y], color="#1E2A38", lw=0.8, zorder=2)

    # --- 5. Annotations & Header ---
    # Right-side legend
    ax.text(12.8, 7.1, "7-Class Unified Head", color=TEXT_WHITE, fontsize=13, fontweight="bold", ha="center")
    ax.text(12.8, 6.7, "Multi-Class Focal Loss (γ = 2.0)", color=GOLD_ACCENT, fontsize=10.5, fontweight="bold", ha="center")

    # Model Title at Top
    ax.text(7.5, 7.5, "MODEL 104.4 NEURAL ARCHITECTURE: 7-CLASS GATED FEATURE FUSION",
            color=TEXT_WHITE, fontsize=14, fontweight="bold", ha="center", va="center")

    # Footer note at bottom
    ax.text(7.5, 0.45, "training: dynamic softmax gating + class-frequency focal balancing on every 20x tile\n"
                       "Validation Accuracy: 97.19%  |  Validation Macro-F1: 0.9747  |  Dysplasia & Normal Recall: 100.0%",
            color=TEXT_MUTED, fontsize=10, ha="center", va="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.savefig(DESKTOP_PATH, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.savefig(ARTIFACT_PATH, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print("Architecture diagram created successfully at:")
    print(f"  - {OUTPUT_PATH}")
    print(f"  - {DESKTOP_PATH}")

if __name__ == "__main__":
    create_diagram()

