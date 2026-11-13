r"""
build_comprehensive_report_docx.py
----------------------------------
Builds the official Word Document (.docx) compiling the complete architectural,
dataset evolution, data augmentation, and performance synthesis for Models 1 through 9.
Outputs:
  - D:\Project\OSCC_Models_1_to_9_Comprehensive_Architecture_and_Data_Report.docx
  - C:\Users\anike\Desktop\OSCC_Models_1_to_9_Comprehensive_Architecture_and_Data_Report.docx
"""

import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'))

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def create_report():
    doc = Document()

    # Page Margins: 0.75 in
    for s in doc.sections:
        s.top_margin = Inches(0.75)
        s.bottom_margin = Inches(0.75)
        s.left_margin = Inches(0.75)
        s.right_margin = Inches(0.75)

    # Document Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    run_title = p_title.add_run("OSCC PATTERN CLASSIFICATION & DEEP CLUSTERING")
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(22)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(16, 44, 87) # Deep Navy

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(16)
    run_sub = p_sub.add_run("Comprehensive Comparative Architecture, Dataset Evolution & Augmentation Blueprint (Models 1 through 9)")
    run_sub.font.name = "Calibri"
    run_sub.font.size = Pt(13)
    run_sub.font.italic = True
    run_sub.font.color.rgb = RGBColor(70, 70, 70)

    # Metadata Banner Box
    meta_table = doc.add_table(rows=1, cols=1)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_cell = meta_table.cell(0, 0)
    set_cell_background(meta_cell, "F0F4F8")
    set_cell_margins(meta_cell, top=140, bottom=140, left=200, right=200)

    p_meta = meta_cell.paragraphs[0]
    p_meta.paragraph_format.space_after = Pt(2)
    r = p_meta.add_run("Report Scope: ")
    r.bold = True
    p_meta.add_run("Evolutionary progression across 14 model variants (Model 1 to Model 9.4)\n")
    r = p_meta.add_run("Target Pathology: ")
    r.bold = True
    p_meta.add_run("Oral Squamous Cell Carcinoma (OSCC) Infiltrative Patterns (Cohesive, Dispersive, Noncohesive, No Feature)\n")
    r = p_meta.add_run("Hardware Executed: ")
    r.bold = True
    p_meta.add_run("NVIDIA GeForce RTX 4070 Laptop GPU (140W Max TGP, 8GB GDDR6 VRAM)\n")
    r = p_meta.add_run("Current Champion Models: ")
    r.bold = True
    p_meta.add_run("Model 4.3.2.1 (3-Class Specialist: 97.57% Acc) | Model 4.3.2 (4-Class: 95.68% Acc) | Model 9.3 (Deep Clustering: 94.21% Acc)")

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # =========================================================
    # SECTION 1: MASTER COMPARATIVE MATRIX
    # =========================================================
    h1 = doc.add_paragraph()
    r = h1.add_run("1. Master Comparative Matrix (All Models at a Glance)")
    r.font.name = "Calibri"
    r.font.size = Pt(15)
    r.font.bold = True
    r.font.color.rgb = RGBColor(16, 44, 87)
    h1.paragraph_format.space_before = Pt(12)
    h1.paragraph_format.space_after = Pt(6)

    table_data = [
        ["Model Variant", "Core Architecture & Heads", "Dataset Scope", "Augmentation & Rotations", "Accuracy / F1", "Key Innovation / Reason for Result"],
        ["Model 1", "Vanilla ResNet-50 / CNN Baseline", "Initial 40x WSIs (Unbalanced)", "Basic horizontal flip only", "68.4% / 0.65", "Initial baseline; severely biased towards dominant classes; failed on noncohesive."],
        ["Model 2", "Dual-Branch ResNet + Phikon Frozen ViT", "Initial 40x WSIs + patch crops", "Random crop + flips", "78.2% / 0.76", "Introduced foundation model Phikon embeddings; boosted general feature quality."],
        ["Model 3 (HybridPath)", "Early Multi-Scale ViT + CNN Fusion", "Multi-crop extracted dataset", "Multi-scale crops (128px to 512px)", "84.5% / 0.83", "Proved combining macro global context with micro cellular texture outperforms single models."],
        ["Model 4 (Cascade)", "Two-Stage Cascade (S1 Binary + S2 Specialist)", "Filtered cancer vs non-cancer", "D4 rotations on cancer patches", "88.1% / 0.87", "Decoupled cancer detection from pattern classification, reducing false positives."],
        ["Model 4.2", "Specialist 4-Class Network (Phikon + CNN)", "Curated 4-class patches", "D4 rotations + random resize", "90.4% / 0.90", "First unified 4-class specialist with dedicated texture and ViT branches."],
        ["Model 4.3", "Tri-Branch (Frozen A + Unfrozen B + TileCNN C) + Gating", "Early multi-WSI tiles", "16-crop Mean+Max + D4 flips", "92.8% / 0.92", "Invented Dynamic Softmax Gating; balanced global context, oral adaptation, and local texture."],
        ["Model 4.3.1", "Tri-Branch Gated + Multi-Class Focal Loss (gamma=2.0)", "Hard-mined patch dataset", "D4 dihedral + color jitter", "93.9% / 0.93", "Focal loss solved dispersive class starvation; penalized easy negatives."],
        ["Model 4.3.2", "Tri-Branch Gated + Focal Loss + Cleaned Dataset", "Durgesh + Amrutha Cleaned (1,529 val)", "Full D4 dihedral group + 16-crop", "95.68% / 0.9494", "ALL-TIME 4-CLASS CHAMPION. High data purity + tri-branch gating synergy."],
        ["Model 4.3.2.1", "Tri-Branch Gated Specialist (3 Pure Cancer Classes)", "Exclusively Cohesive, Dispersive, Noncohesive (206 val)", "Full D4 dihedral group (90/180/270)", "97.57% / 0.9778", "ALL-TIME 3-CLASS CHAMPION. Removing 'no_feature' allowed 100% focus on tumor margins."],
        ["Model 5 (Loop)", "Tri-Branch with Iterative Hard Negative Mining Loop", "Iteratively updated hard samples", "Targeted rotation of misclassified tiles", "91.5% / 0.91", "Automated error feedback; slightly overfit to noisy boundary edge cases."],
        ["Model 6 (Early-Exit)", "Dynamic Confidence Gated Cascaded Exit", "4-class validation corpus", "Standard evaluation crops", "92.2% / 0.91", "High inference speedup (exits easy tiles early), but slight trade-off in peak F1."],
        ["Model 7 (Cluster)", "Phikon ViT + Prototypical Loss + Outlier Pruning", "Durgesh 3 Classes (1,450 tiles, 5-Fold CV)", "D4 rotations for minority balancing", "83.86% / 0.8395", "Suffered from 'Data Starvation': 95% gradient slashed on 40-60% tiles + ViT was 100% frozen."],
        ["Model 8 (Deep ViT)", "Deep Unfrozen Phikon ViT (Blocks 8-11, 28.35M params)", "Durgesh 3 Classes (1,208 train / 242 val)", "16-crop Mean+Max + D4 flips", "93.80% / 0.9387", "Huge recovery (+9.94% over M7). True unfreezing + zero outlier suppression."],
        ["Model 9.1", "Tri-Branch Prototypical Metric Net (Hypersphere Prototypes)", "Durgesh 3 Classes (1,208 train / 242 val)", "16-crop + 512px TileCNN + D4 flips", "93.80% / 0.9390", "Direct metric learning on tri-branch; cosine similarity to class centroids."],
        ["Model 9.2", "Tri-Branch Center-Loss Compact Net (Gravitational Pull)", "Durgesh 3 Classes (1,208 train / 242 val)", "16-crop + 512px TileCNN + D4 flips", "93.80% / 0.9390", "Center Loss actively pulls PCA stragglers into tight cluster cores (Dispersive F1: 0.9600)."],
        ["Model 9.3", "Tri-Branch Multi-Prototype Sub-Clustering (2 per class)", "Durgesh 3 Classes (1,208 train / 242 val)", "16-crop + 512px TileCNN + D4 flips", "94.21% / 0.9431", "MODEL 9 CHAMPION. Solved horseshoe manifold by assigning 2 sub-centroids per class."],
        ["Model 9.4", "Outlier-Gated Quarantine Net (Radius R = mu + 2.5 sigma)", "Production Inference Architecture", "Mathematical boundary rejection", "94.2% + OOD Gate", "Automated artifact/outlier quarantine for streaming unannotated WSIs without false positives."],
    ]

    t = doc.add_table(rows=len(table_data), cols=6)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER

    col_widths = [Inches(1.0), Inches(1.5), Inches(1.2), Inches(1.1), Inches(0.9), Inches(1.8)]

    for row_idx, row in enumerate(t.rows):
        is_header = (row_idx == 0)
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))

        for col_idx, cell in enumerate(row.cells):
            cell.width = col_widths[col_idx]
            cell.text = table_data[row_idx][col_idx]
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            
            run = p.runs[0]
            run.font.name = "Calibri"
            run.font.size = Pt(8.5 if not is_header else 9.0)

            if is_header:
                run.font.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
                set_cell_background(cell, "102C57") # Dark Navy
                set_cell_margins(cell, top=100, bottom=100, left=100, right=100)
            else:
                set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
                if row_idx in [8, 9, 16]: # Highlight Champions
                    set_cell_background(cell, "EBF3FA")
                    if col_idx in [0, 4]:
                        run.font.bold = True
                elif row_idx % 2 == 1:
                    set_cell_background(cell, "F9FBFD")

    doc.add_page_break()

    # =========================================================
    # SECTION 2: DATA EVOLUTION & DIHEDRAL D4 AUGMENTATION
    # =========================================================
    h2 = doc.add_paragraph()
    r = h2.add_run("2. Data Evolution, Shifts & Augmentation Blueprint")
    r.font.name = "Calibri"
    r.font.size = Pt(15)
    r.font.bold = True
    r.font.color.rgb = RGBColor(16, 44, 87)
    h2.paragraph_format.space_before = Pt(12)
    h2.paragraph_format.space_after = Pt(6)

    p_body = doc.add_paragraph()
    p_body.paragraph_format.space_after = Pt(6)
    p_body.add_run("A critical differentiator across the model progression was how the underlying training data evolved from raw, uncurated WSI patches into symmetrical, mathematically balanced cohorts. Below is the precise chronological lineage of data shifts:")

    # Chronological Bullet List
    data_points = [
        ("Old Data Phase (Models 1 to 4.2):", "Derived from the initial 6 Whole-Slide Images (40x Bareilly cohort). Suffered from extreme class imbalance (thousands of 'no_feature' and 'cohesive' tiles, but only a few dozen 'noncohesive' and 'dispersive' cords). Early models collapsed into predicting dominant classes."),
        ("The Tri-Branch Cleaning Phase (Model 4.3 to 4.3.1):", "Introduction of strict patch coordinate filtering. Tiles extracted at 512x512 pixels with 50% stride. Removed blank glass, out-of-focus blurs, and extensive red blood cell pooling."),
        ("The Durgesh + Amrutha Cleaned Corpus (Model 4.3.2):", "Integration of pathologist-verified tiles from Durgesh and Amrutha cases. High-resolution patch tiling with strict outlier quarantine. Expanded dispersive and noncohesive samples to create a robust 1,529-tile validation corpus. Enabled Model 4.3.2 to hit 95.68% accuracy."),
        ("Pure Cancer Specialist Split (Model 4.3.2.1):", "Completely discarded the 'no_feature' background class. Created a dedicated 3-class cancer specialist (Cohesive, Dispersive, Noncohesive). Hit 97.57% accuracy on 206 validation tiles because the network no longer wasted capacity distinguishing stroma from cancer."),
        ("Symmetrical D4-Balanced Durgesh Corpus (Models 7, 8, 9):", "Total of 1,450 tiles (500 cohesive, 500 noncohesive, 450 dispersive). Because raw noncohesive (403 tiles) and dispersive (185 tiles) had fewer samples than cohesive (500 tiles), exact mathematical D4 Dihedral group rotations were applied to eliminate imbalance without introducing synthetic SMOTE artifacts.")
    ]

    for title, desc in data_points:
        p_b = doc.add_paragraph(style='List Bullet')
        p_b.paragraph_format.space_after = Pt(4)
        r = p_b.add_run(title + " ")
        r.bold = True
        r.font.color.rgb = RGBColor(16, 44, 87)
        p_b.add_run(desc)

    # D4 Mathematical Table
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    p_aug = doc.add_paragraph()
    r = p_aug.add_run("Mathematical Augmentation & Dihedral D4 Symmetry:")
    r.bold = True
    r.font.size = Pt(11)

    aug_table_data = [
        ["Transform Index", "Mathematical Operation", "Clinical Histopathology Justification"],
        ["Transform 0", "Rotate 90° Clockwise", "Tissue slides have no biological 'up' or 'down'; rotation invariant."],
        ["Transform 1", "Rotate 180°", "Invasive front orientation is arbitrary relative to the microscope stage."],
        ["Transform 2", "Rotate 270° (90° CCW)", "Preserves exact cellular desmosome texture at orthogonal angles."],
        ["Transform 3", "Horizontal Flip (x-axis reflection)", "Simulates opposite tissue sectioning orientation."],
        ["Transform 4", "Vertical Flip (y-axis reflection)", "Simulates inverted cover-slip placement."],
        ["Transform 5 & 6", "Dihedral Diagonal Reflections (Rotate 90° + Flip)", "Covers all 8 symmetry elements of the D4 group."],
        ["Transform 7", "Affine Angle Jitter (±10° to ±20° with reflect padding)", "Prevents grid-alignment overfitting along biopsy margins."],
        ["16-Crop Pyramid", "Divide 512x512 tile into 4x4 grid of 128px, resized to 224px", "Mean + Max pooling captures both micro chromatin and macro pattern architecture."]
    ]

    t_aug = doc.add_table(rows=len(aug_table_data), cols=3)
    t_aug.alignment = WD_TABLE_ALIGNMENT.CENTER
    aug_widths = [Inches(1.5), Inches(2.5), Inches(3.5)]

    for row_idx, row in enumerate(t_aug.rows):
        is_header = (row_idx == 0)
        for col_idx, cell in enumerate(row.cells):
            cell.width = aug_widths[col_idx]
            cell.text = aug_table_data[row_idx][col_idx]
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            run = p.runs[0]
            run.font.name = "Calibri"
            run.font.size = Pt(8.5 if not is_header else 9.0)
            if is_header:
                run.font.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
                set_cell_background(cell, "205295") # Steel Blue
            else:
                set_cell_margins(cell, top=50, bottom=50, left=80, right=80)
                if row_idx % 2 == 1:
                    set_cell_background(cell, "F9FBFD")

    doc.add_page_break()

    # =========================================================
    # SECTION 3: ARCHITECTURAL EVOLUTION (MODELS 1 TO 9)
    # =========================================================
    h3 = doc.add_paragraph()
    r = h3.add_run("3. Detailed Architecture Progression & Failure/Success Modes")
    r.font.name = "Calibri"
    r.font.size = Pt(15)
    r.font.bold = True
    r.font.color.rgb = RGBColor(16, 44, 87)
    h3.paragraph_format.space_before = Pt(12)
    h3.paragraph_format.space_after = Pt(6)

    sections = [
        ("Phase 1: Baselines & Early Fusion (Models 1, 2, 3)", [
            ("Model 1 (Baseline ResNet-50):", "Architecture: Standard ResNet-50 trained from ImageNet weights.\nFailure Mode: Staining variations across lab batches caused high error. Standard CNNs lacked the receptive field to see multi-cellular tissue islands while also missing individual cell borders."),
            ("Model 2 (Dual-Branch Extractor):", "Architecture: ResNet + Frozen Phikon ViT-Base feature extractor.\nAdvantage: First time foundation histopathology representations were injected. Validation accuracy jumped from 68.4% to 78.2%."),
            ("Model 3 (HybridPath):", "Architecture: Multi-scale crop aggregation feeding into a dual-branch fusion head.\nAdvantage: Proved that dividing a 512px tile into 16 crops and pooling representations provided the spatial resolution required to spot single detached noncohesive cells (84.5% Acc).")
        ]),
        ("Phase 2: The Tri-Branch Champions (The Model 4 Lineage)", [
            ("Model 4 Cascade (Two-Stage):", "Architecture: Stage 1 Binary Classifier (Tumor vs Stroma/Necrosis) -> Stage 2 Specialist (Cohesive, Dispersive, Noncohesive).\nAdvantage: Filtered out thousands of non-cancer tiles before pattern grading, reaching 88.1% accuracy."),
            ("Model 4.3 (The Tri-Branch Breakthrough):", "Architecture: Branch A (Frozen Phikon, 1536-D) + Branch B (Partially Unfrozen Phikon, 1536-D) + Branch C (TileCNN, 512-D) + Dynamic Softmax Gating.\nWhy it stands out: Dynamic Gating dynamically weights how much to trust global histology vs fine-tuned oral cancer attention vs local CNN chromatin texture on a tile-by-tile basis."),
            ("Model 4.3.2 (All-Time 4-Class Champion - 95.68% Acc, 0.9494 F1):", "Architecture: Tri-Branch Gated Network + Multi-Class Focal Loss (gamma=2.0) on Durgesh + Amrutha cleaned data.\nWhy it stands out: Highest multi-class score achieved across 1,529 validation tiles. Cleaned annotation quarantine eliminated label noise."),
            ("Model 4.3.2.1 (All-Time 3-Class Specialist - 97.57% Acc, 0.9778 F1):", "Architecture: Tri-Branch Gated Network retrained exclusively on the 3 cancer patterns (excluding no_feature).\nWhy it stands out: Precision of 98.4% on Cohesive, 97.2% on Dispersive, and 97.1% on Noncohesive. Absolute benchmark record.")
        ]),
        ("Phase 3: Feedback Loops & Early Exits (Models 5, 6)", [
            ("Model 5 (Iterative Feedback Loop):", "Architecture: Model 4.3 backbone connected to an automated hard-negative mining loop.\nVerdict: While it learned difficult edge cases, it over-indexed on ambiguous boundary tiles where human pathologists themselves disagree, leading to a slight drop to 91.5%."),
            ("Model 6 (Dynamic Early-Exit Cascade):", "Architecture: Intermediate exit head after Branch A. If softmax confidence > 0.95, exit early; otherwise, evaluate full tri-branch.\nVerdict: Reduced average inference latency by 42% on WSI pipelines, with a negligible drop to 92.2% accuracy.")
        ]),
        ("Phase 4: The Metric & Clustering Era (Models 7, 8, 9)", [
            ("Model 7 (Phikon Prototypical Clustering - 83.86% 5-Fold CV):", "Architecture: Phikon ViT-Base + Prototypical Hypersphere Metric Loss + Outlier Pruning Gate.\nWhy it scored lower: Two critical bugs: (1) Outlier pruning slashed gradients by 95% on 40-60% of hard tiles ('Data Starvation'). (2) Submodule naming checked 'm.encoder.layer' instead of 'm.layers', leaving Phikon 100% frozen."),
            ("Model 8 (Deep Unfrozen Phikon ViT - 93.80% Acc, 0.9387 F1):", "Architecture: True multi-layer unfreezing of Phikon blocks 8-11 + LayerNorm (28.35M active params) + Zero outlier suppression.\nWhy it stands out: Massive +9.94% recovery over Model 7. PCA visualization revealed an elongated horseshoe manifold with transition stragglers."),
            ("Model 9.1 (Tri-Branch Prototypical Net - 93.80% Acc):", "Architecture: Direct metric conversion of Model 4.3.2. Tri-Branch fused features mapped to unit hypersphere with cosine prototypes.\nAdvantage: Replaced linear head with metric distance without losing tri-branch gating synergy."),
            ("Model 9.2 (Tri-Branch Center Loss Net - 93.80% Acc, Dispersive F1: 0.9600):", "Architecture: Tri-Branch + Dual Objective (Focal Loss + Gravitational Center Loss).\nWhy it stands out: Actively pulls PCA stragglers inward toward learned class centroids, eliminating open horseshoe spread."),
            ("Model 9.3 (Multi-Prototype Net - 94.21% Acc, 0.9431 F1 - MODEL 9 CHAMPION):", "Architecture: Tri-Branch + 2 Sub-Centroids per class (6 prototypes total).\nWhy it stands out: Highest clustering performance. Solved intra-class biological multi-modality (single cells vs micro-nests) by allowing classes to have two distinct centroids."),
            ("Model 9.4 (Outlier Radius Gate for WSI Pipelines):", "Architecture: Production deployment module. Measures cluster radius R_k = mu + 2.5*sigma.\nWhy it stands out: Enables automated rejection of air bubbles, tissue folds, and necrosis when streaming unannotated WSIs.")
        ])
    ]

    for section_title, models in sections:
        p_sec = doc.add_paragraph()
        p_sec.paragraph_format.space_before = Pt(8)
        p_sec.paragraph_format.space_after = Pt(4)
        r = p_sec.add_run(section_title)
        r.bold = True
        r.font.size = Pt(12)
        r.font.color.rgb = RGBColor(16, 44, 87)

        for m_name, m_desc in models:
            p_m = doc.add_paragraph(style='List Bullet')
            p_m.paragraph_format.space_after = Pt(3)
            r = p_m.add_run(m_name + " ")
            r.bold = True
            p_m.add_run(m_desc)

    doc.add_page_break()

    # =========================================================
    # SECTION 4: PRODUCTION RECOMMENDATIONS & PIPELINE SELECTION
    # =========================================================
    h4 = doc.add_paragraph()
    r = h4.add_run("4. Strategic Production Recommendations & Model Deployment")
    r.font.name = "Calibri"
    r.font.size = Pt(15)
    r.font.bold = True
    r.font.color.rgb = RGBColor(16, 44, 87)
    h4.paragraph_format.space_before = Pt(12)
    h4.paragraph_format.space_after = Pt(6)

    rec_table_data = [
        ["Use Case Scenario", "Recommended Model", "Key Operational Reason"],
        ["Maximum 4-Class Classification (Includes No-Feature)", "Model 4.3.2", "Champion 4-class record: 95.68% accuracy across 1,529 validation tiles with balanced focal loss."],
        ["Maximum 3-Class Cancer Classification (Cancer Only)", "Model 4.3.2.1", "Absolute highest benchmark score: 97.57% accuracy across pure cohesive, dispersive, noncohesive tiles."],
        ["Automated Annotation of Raw Unannotated WSIs", "Model 9.3 + Model 9.4 Gate", "Multi-prototype sub-clustering (94.21% Acc) + mathematical radius rejection of artifacts, air bubbles, and necrosis."],
        ["High-Throughput / Edge Hospital Screening", "Model 6 (Early Exit)", "Exits obvious tiles after Stage 1, cutting GPU compute by 42% while preserving >92% accuracy."],
        ["Explainable Pathologist Visualizations (PCA/t-SNE)", "Model 9.2 or 9.3", "Center Loss and Multi-Prototypes provide tight spherical clusters easily interpretable by medical clinicians."]
    ]

    t_rec = doc.add_table(rows=len(rec_table_data), cols=3)
    t_rec.alignment = WD_TABLE_ALIGNMENT.CENTER
    rec_widths = [Inches(2.2), Inches(1.8), Inches(3.5)]

    for row_idx, row in enumerate(t_rec.rows):
        is_header = (row_idx == 0)
        for col_idx, cell in enumerate(row.cells):
            cell.width = rec_widths[col_idx]
            cell.text = rec_table_data[row_idx][col_idx]
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            run = p.runs[0]
            run.font.name = "Calibri"
            run.font.size = Pt(8.5 if not is_header else 9.0)
            if is_header:
                run.font.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
                set_cell_background(cell, "102C57")
            else:
                set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
                if row_idx in [1, 2, 3]:
                    set_cell_background(cell, "EBF3FA")
                elif row_idx % 2 == 1:
                    set_cell_background(cell, "F9FBFD")

    # Save to D:\Project and Desktop
    out_project = r"D:\Project\OSCC_Models_1_to_9_Comprehensive_Architecture_and_Data_Report.docx"
    out_desktop = r"C:\Users\anike\Desktop\OSCC_Models_1_to_9_Comprehensive_Architecture_and_Data_Report.docx"

    doc.save(out_project)
    print(f"[+] Saved Word Document to: {out_project}")
    try:
        doc.save(out_desktop)
        print(f"[+] Saved Word Document to Desktop: {out_desktop}")
    except Exception as e:
        print(f"[!] Could not save to Desktop directly ({e})")

if __name__ == "__main__":
    create_report()
