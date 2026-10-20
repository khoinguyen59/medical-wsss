"""
Publication-Grade Academic Monochrome Figures Generator for SE374.R11.
Renders all 3 formal academic architectural figures in strict Academic Grayscale / Monochrome:
- Figure 1: Standard WSI Processing Pipeline (Two-Stage Decoupled Framework)
- Figure 2: Three-Tier System Architecture (SE374 Pathology Workstation)
- Figure 3: End-to-End Pipeline Data Flow (Inference & Clinical Biomarker Workflow)

Complies with Springer LNCS, IEEE Transactions, and UIT formal thesis standards:
- 100% Academic Grayscale: Pure white background (#FFFFFF), crisp black outlines (#000000),
  subtle grayscale fills (#F8FAFC, #F1F5F9, #E2E8F0), and high-contrast typography.
- Zero colorful/pastel/neon marketing palettes.
- High resolution (300 DPI) raster PNG and vector SVG.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


def draw_arrow(ax, x1, y1, x2, y2, label=None, label_offset=(0, 0.25), color="#000000", lw=1.5):
    """Draws a clean, proportional monochrome arrow with guaranteed non-overlapping spacing."""
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=10,
            shrinkA=4,
            shrinkB=4
        )
    )
    if label:
        lx = (x1 + x2) / 2 + label_offset[0]
        ly = (y1 + y2) / 2 + label_offset[1]
        ax.text(lx, ly, label, fontsize=8.2, fontweight="semibold", color="#000000", ha="center", va="center")


# ==============================================================================
# FIGURE 1: STANDARD WSI PROCESSING PIPELINE (ACADEMIC GRAYSCALE)
# ==============================================================================
def generate_figure_1(output_png: str, output_svg: str = None):
    """Generates Figure 1 in high-contrast academic monochrome."""
    fig, ax = plt.subplots(figsize=(16, 6.2), dpi=300)
    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # 1. INPUT WSI (x: 0.5 to 2.2)
    slide_bg = FancyBboxPatch((0.5, 1.8), 1.7, 3.4, boxstyle="round,pad=0.06,rounding_size=0.12",
                              facecolor="#FFFFFF", edgecolor="#000000", linewidth=1.5)
    ax.add_patch(slide_bg)

    label_bg = FancyBboxPatch((0.6, 4.35), 1.5, 0.7, boxstyle="round,pad=0.02,rounding_size=0.05",
                              facecolor="#F1F5F9", edgecolor="#000000", linewidth=1.0)
    ax.add_patch(label_bg)
    ax.text(1.35, 4.7, "TCGA-BRCA\nSPECIMEN", fontsize=7.5, fontweight="bold", color="#000000", ha="center", va="center")

    # Tissue biopsy illustration (grayscale morphology with contour)
    t_x = np.array([0.9, 1.2, 1.7, 1.9, 1.6, 1.1, 0.8])
    t_y = np.array([2.5, 3.2, 3.5, 2.9, 2.2, 2.1, 2.3])
    ax.fill(t_x, t_y, color="#E2E8F0", alpha=0.9)
    ax.plot(t_x, t_y, color="#000000", linewidth=1.2)

    c_x = np.array([1.1, 1.4, 1.6, 1.3])
    c_y = np.array([2.7, 3.1, 2.8, 2.4])
    ax.fill(c_x, c_y, color="#CBD5E1", alpha=0.9)
    ax.plot(c_x, c_y, color="#334155", linewidth=1.0, linestyle="--")

    ax.text(1.35, 1.35, "Input WSI", fontsize=11.5, fontweight="bold", color="#000000", ha="center")
    ax.text(1.35, 0.95, "Gigapixel Slide\n(~100k x 100k px)", fontsize=8.0, color="#334155", ha="center")

    # Arrow 1: Input WSI -> Tiling
    draw_arrow(ax, 2.25, 3.5, 3.15, 3.5, label="Patch\ntiling", label_offset=(0, 0.35))

    # 2. PATCH TILING (x: 3.25 to 4.35)
    p_ys = [4.3, 3.6, 2.9, 2.2]
    for py in p_ys:
        pb = FancyBboxPatch((3.3, py), 0.55, 0.55, boxstyle="round,pad=0.02,rounding_size=0.06",
                            facecolor="#FFFFFF", edgecolor="#000000", linewidth=1.2)
        ax.add_patch(pb)
        ax.plot([3.45, 3.6, 3.7], [py + 0.15, py + 0.35, py + 0.2], color="#475569", lw=1.2)

    # Magnified single patch
    mag_box = FancyBboxPatch((4.05, 4.6), 0.85, 0.85, boxstyle="round,pad=0.02,rounding_size=0.06",
                             facecolor="#F8FAFC", edgecolor="#000000", linewidth=1.5)
    ax.add_patch(mag_box)
    ax.text(4.47, 5.08, "H&E Patch", fontsize=8.0, fontweight="bold", color="#000000", ha="center")
    ax.text(4.47, 4.75, "224x224 px", fontsize=7.5, color="#334155", ha="center")
    ax.plot([3.85, 4.05], [4.58, 4.9], color="#000000", lw=1.0, linestyle=":")

    ax.text(3.7, 1.35, "Patch Tiles", fontsize=11, fontweight="bold", color="#000000", ha="center")
    ax.text(3.7, 0.95, "OTSU Filtered\n(Tissue only)", fontsize=8.0, color="#334155", ha="center")

    # Arrow 2: Tiling -> Feature Extractor
    draw_arrow(ax, 4.4, 3.5, 5.3, 3.5)

    # 3. FEATURE EXTRACTOR (FROZEN) (x: 5.4 to 8.5)
    fe_box = FancyBboxPatch((5.4, 1.4), 3.1, 4.4, boxstyle="round,pad=0.06,rounding_size=0.15",
                            facecolor="#FFFFFF", edgecolor="#000000", linewidth=1.8)
    ax.add_patch(fe_box)

    badge_fz = FancyBboxPatch((5.65, 5.35), 2.6, 0.45, boxstyle="round,pad=0.02,rounding_size=0.08",
                              facecolor="#000000", edgecolor="none")
    ax.add_patch(badge_fz)
    ax.text(6.95, 5.57, "[FROZEN] FEATURE EXTRACTOR", fontsize=8.5, fontweight="bold", color="#FFFFFF", ha="center")

    ax.text(6.95, 4.95, "Pretrained Foundation Models", fontsize=10.0, fontweight="bold", color="#000000", ha="center")
    ax.text(6.95, 4.65, "(Large-scale pathology pretrained)", fontsize=8.0, color="#475569", ha="center")

    models = [
        ("ResNet", "ResNet-50 / ResNet-38 (ConvNet Baseline)"),
        ("CTransPath", "Swin Transformer (SRCL Pretrained)"),
        ("UNI / UNI-2", "ViT-L/16 (100M+ Patches, DINOv2)"),
        ("CONCH", "Vision-Language CoCa (1.17M Pairs)"),
        ("Prov-GigaPath", "Billion-Parameter Whole-Slide ViT")
    ]
    for i, (m_name, m_desc) in enumerate(models):
        y_pos = 4.15 - i * 0.53
        b_dot = FancyBboxPatch((5.65, y_pos - 0.12), 0.22, 0.24, boxstyle="round,pad=0.01,rounding_size=0.04",
                               facecolor="#F1F5F9", edgecolor="#000000", lw=0.8)
        ax.add_patch(b_dot)
        ax.text(5.76, y_pos, ">", fontsize=7.5, fontweight="bold", color="#000000", ha="center")
        ax.text(5.98, y_pos + 0.05, m_name, fontsize=9.2, fontweight="bold", color="#000000")
        ax.text(5.98, y_pos - 0.14, m_desc, fontsize=7.2, color="#475569")

    # Arrow 3: Feature Extractor -> Embeddings
    draw_arrow(ax, 8.55, 3.5, 9.4, 3.5)

    # 4. PATCH EMBEDDINGS (x: 9.5 to 10.3)
    for py in p_ys:
        eb = FancyBboxPatch((9.55, py + 0.08), 0.75, 0.38, boxstyle="round,pad=0.02,rounding_size=0.06",
                            facecolor="#F8FAFC", edgecolor="#000000", linewidth=1.2)
        ax.add_patch(eb)
        for vi in range(4):
            ax.axvline(9.68 + vi * 0.13, ymin=(py + 0.12)/6.5, ymax=(py + 0.42)/6.5, color="#64748B", lw=1.0)

    ax.text(9.92, 1.45, "Patch\nEmbeddings", fontsize=9.5, fontweight="bold", color="#000000", ha="center")
    ax.text(9.92, 0.95, "Fixed Vectors\n(512-1024d)", fontsize=8.0, color="#475569", ha="center")

    # Arrow 4: Embeddings -> Trainable Downstream Model
    draw_arrow(ax, 10.35, 3.5, 11.2, 3.5)

    # 5. TRAINABLE DOWNSTREAM MODEL (x: 11.3 to 14.1)
    tm_box = FancyBboxPatch((11.3, 1.4), 2.8, 4.4, boxstyle="round,pad=0.06,rounding_size=0.15",
                            facecolor="#FFFFFF", edgecolor="#000000", linewidth=1.8)
    ax.add_patch(tm_box)

    badge_tr = FancyBboxPatch((11.5, 5.35), 2.4, 0.45, boxstyle="round,pad=0.02,rounding_size=0.08",
                              facecolor="#000000", edgecolor="none")
    ax.add_patch(badge_tr)
    ax.text(12.7, 5.57, "[TRAINABLE] DOWNSTREAM", fontsize=8.5, fontweight="bold", color="#FFFFFF", ha="center")

    ax.text(12.7, 4.95, "Task-Specific Models", fontsize=10.0, fontweight="bold", color="#000000", ha="center")
    ax.text(12.7, 4.65, "(Lightweight GPU Footprint <2GB)", fontsize=8.0, color="#475569", ha="center")

    mil_models = [
        ("CLAM", "Clustering-constrained MIL"),
        ("TransMIL", "Transformer Correlated MIL"),
        ("ABMIL / DSMIL", "Attention / Dual-Stream MIL"),
        ("WSSS Decoder", "CAM + PDA + MLPS (Topic 6)")
    ]
    for i, (m_name, m_desc) in enumerate(mil_models):
        y_pos = 4.15 - i * 0.65
        b_dot2 = FancyBboxPatch((11.5, y_pos - 0.12), 0.22, 0.24, boxstyle="round,pad=0.01,rounding_size=0.04",
                                facecolor="#F1F5F9", edgecolor="#000000", lw=0.8)
        ax.add_patch(b_dot2)
        ax.text(11.61, y_pos, "*", fontsize=9, fontweight="bold", color="#000000", ha="center")
        ax.text(11.82, y_pos + 0.06, m_name, fontsize=9.2, fontweight="bold", color="#000000")
        ax.text(11.82, y_pos - 0.14, m_desc, fontsize=7.2, color="#475569")

    # Arrow 5: Downstream -> Output
    draw_arrow(ax, 14.15, 3.5, 14.9, 3.5)

    # 6. CLINICAL OUTPUT (x: 15.0 to 16.0)
    out_box = FancyBboxPatch((15.0, 2.6), 1.0, 1.8, boxstyle="round,pad=0.04,rounding_size=0.1",
                             facecolor="#F8FAFC", edgecolor="#000000", linewidth=1.5)
    ax.add_patch(out_box)
    ax.text(15.5, 3.9, "OUTPUT", fontsize=9.0, fontweight="bold", color="#000000", ha="center")
    ax.text(15.5, 3.4, "Multi-Class\nMasks", fontsize=8.0, fontweight="semibold", color="#000000", ha="center")
    ax.text(15.5, 2.9, "sTILs Score %\nSurvival Risk", fontsize=7.2, color="#475569", ha="center")

    # Overall Caption
    ax.text(8.25, 0.35,
            "Fig. 1: Illustration of the standard WSI-processing pipeline in Computational Pathology.\n"
            "Two-stage decoupled framework: Frozen foundation feature extractor (zero OOM) -> Trainable downstream task model.",
            fontsize=9.5, fontstyle="italic", color="#000000", ha="center")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="#FFFFFF", edgecolor="none")
    if output_svg:
        os.makedirs(os.path.dirname(os.path.abspath(output_svg)), exist_ok=True)
        plt.savefig(output_svg, format="svg", bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 1 (Academic Monochrome): {output_png}")


# ==============================================================================
# FIGURE 2: THREE-TIER SYSTEM ARCHITECTURE (ACADEMIC GRAYSCALE)
# ==============================================================================
def generate_figure_2(output_png: str, output_svg: str = None):
    """Generates Figure 2 in high-contrast academic monochrome."""
    fig, ax = plt.subplots(figsize=(14, 9.5), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9.5)
    ax.axis("off")

    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # ----------------------------------------------------
    # TIER 1: CLINICAL CLIENT (y: 6.8 to 8.8)
    # ----------------------------------------------------
    t1_box = FancyBboxPatch((0.8, 6.7), 12.4, 2.2, boxstyle="round,pad=0.08,rounding_size=0.15",
                            facecolor="#FFFFFF", edgecolor="#000000", linewidth=1.8)
    ax.add_patch(t1_box)

    badge1 = FancyBboxPatch((1.1, 8.45), 4.2, 0.4, boxstyle="round,pad=0.02,rounding_size=0.06",
                            facecolor="#000000", edgecolor="none")
    ax.add_patch(badge1)
    ax.text(3.2, 8.65, "TIER 1: CLINICAL CLIENT (WEB INTERFACE)", fontsize=8.5, fontweight="bold", color="#FFFFFF", ha="center")

    c1 = FancyBboxPatch((1.2, 6.95), 3.6, 1.35, boxstyle="round,pad=0.04,rounding_size=0.08",
                        facecolor="#F8FAFC", edgecolor="#000000", lw=1.0)
    ax.add_patch(c1)
    ax.text(3.0, 7.95, "Web UI Responsive", fontsize=9.5, fontweight="bold", color="#000000", ha="center")
    ax.text(3.0, 7.5, "HTML5 / Vanilla JS / Modern CSS\nClinical dark/light standard UI", fontsize=8.0, color="#475569", ha="center")

    c2 = FancyBboxPatch((5.2, 6.95), 3.6, 1.35, boxstyle="round,pad=0.04,rounding_size=0.08",
                        facecolor="#F8FAFC", edgecolor="#000000", lw=1.0)
    ax.add_patch(c2)
    ax.text(7.0, 7.95, "DeepZoom WSI Viewer", fontsize=9.5, fontweight="bold", color="#000000", ha="center")
    ax.text(7.0, 7.5, "OpenSeadragon Pyramidal Engine\nSmooth navigation 1x -> 40x", fontsize=8.0, color="#475569", ha="center")

    c3 = FancyBboxPatch((9.2, 6.95), 3.6, 1.35, boxstyle="round,pad=0.04,rounding_size=0.08",
                        facecolor="#F8FAFC", edgecolor="#000000", lw=1.0)
    ax.add_patch(c3)
    ax.text(11.0, 7.95, "Overlay & Quantification", fontsize=9.5, fontweight="bold", color="#000000", ha="center")
    ax.text(11.0, 7.5, "WebGL Shader Overlay (5 channels)\nsTILs score (%) & PDF Export", fontsize=8.0, color="#475569", ha="center")

    # Arrow Tier 1 -> Tier 2
    draw_arrow(ax, 7.0, 6.7, 7.0, 5.9, label="RESTful JSON / Binary Tile Streaming (DZI)", label_offset=(0, 0))

    # ----------------------------------------------------
    # TIER 2: BACKEND MEDICAL API SERVICE (y: 3.6 to 5.7)
    # ----------------------------------------------------
    t2_box = FancyBboxPatch((0.8, 3.6), 12.4, 2.2, boxstyle="round,pad=0.08,rounding_size=0.15",
                            facecolor="#FFFFFF", edgecolor="#000000", linewidth=1.8)
    ax.add_patch(t2_box)

    badge2 = FancyBboxPatch((1.1, 5.35), 4.5, 0.4, boxstyle="round,pad=0.02,rounding_size=0.06",
                            facecolor="#000000", edgecolor="none")
    ax.add_patch(badge2)
    ax.text(3.35, 5.55, "TIER 2: MEDICAL API BACKEND (FASTAPI & TIATOOLBOX)", fontsize=8.5, fontweight="bold", color="#FFFFFF", ha="center")

    c4 = FancyBboxPatch((1.2, 3.85), 3.6, 1.35, boxstyle="round,pad=0.04,rounding_size=0.08",
                        facecolor="#F8FAFC", edgecolor="#000000", lw=1.0)
    ax.add_patch(c4)
    ax.text(3.0, 4.85, "Pyramidal Tile Server", fontsize=9.5, fontweight="bold", color="#000000", ha="center")
    ax.text(3.0, 4.4, "Dynamic DZI Tiler\nSub-50ms tile response latency", fontsize=8.0, color="#475569", ha="center")

    c5 = FancyBboxPatch((5.2, 3.85), 3.6, 1.35, boxstyle="round,pad=0.04,rounding_size=0.08",
                        facecolor="#F8FAFC", edgecolor="#000000", lw=1.0)
    ax.add_patch(c5)
    ax.text(7.0, 4.85, "Tissue Pre-Filtering", fontsize=9.5, fontweight="bold", color="#000000", ha="center")
    ax.text(7.0, 4.4, "OTSU Tissue Masking\nFilters 80% non-tissue background", fontsize=8.0, color="#475569", ha="center")

    c6 = FancyBboxPatch((9.2, 3.85), 3.6, 1.35, boxstyle="round,pad=0.04,rounding_size=0.08",
                        facecolor="#F8FAFC", edgecolor="#000000", lw=1.0)
    ax.add_patch(c6)
    ax.text(11.0, 4.85, "Case & Report Engine", fontsize=9.5, fontweight="bold", color="#000000", ha="center")
    ax.text(11.0, 4.4, "Digital Case Management\nCAP-compliant PDF export engine", fontsize=8.0, color="#475569", ha="center")

    # Arrow Tier 2 -> Tier 3
    draw_arrow(ax, 7.0, 3.6, 7.0, 2.8, label="Tensor Batches (224x224x3) / Patch Embeddings", label_offset=(0, 0))

    # ----------------------------------------------------
    # TIER 3: MEDICAL AI CORE ENGINE (y: 0.7 to 2.6)
    # ----------------------------------------------------
    t3_box = FancyBboxPatch((0.8, 0.7), 12.4, 2.0, boxstyle="round,pad=0.08,rounding_size=0.15",
                            facecolor="#FFFFFF", edgecolor="#000000", linewidth=1.8)
    ax.add_patch(t3_box)

    badge3 = FancyBboxPatch((1.1, 2.25), 4.8, 0.4, boxstyle="round,pad=0.02,rounding_size=0.06",
                            facecolor="#000000", edgecolor="none")
    ax.add_patch(badge3)
    ax.text(3.5, 2.45, "TIER 3: CORE MEDICAL AI ENGINE (PROTO-MAMBA WSSS)", fontsize=8.5, fontweight="bold", color="#FFFFFF", ha="center")

    ai_sub = [
        ("1. Frozen Extractor", "CTransPath / CONCH / ResNet\nFixed patch representations"),
        ("2. Visual Mamba CMB", "4-Way Selective Scan O(L)\nGlobal slide-level context"),
        ("3. Attention PDA & CGM", "Progressive Dropout Attention\nClassification Gate Mechanism"),
        ("4. WSI Stitching", "50% Overlap Patch Blending\nSeamless whole-slide assembly")
    ]
    for idx, (title, desc) in enumerate(ai_sub):
        bx = 1.1 + idx * 2.95
        b_sub = FancyBboxPatch((bx, 0.9), 2.85, 1.25, boxstyle="round,pad=0.03,rounding_size=0.06",
                               facecolor="#F8FAFC", edgecolor="#000000", lw=1.0)
        ax.add_patch(b_sub)
        ax.text(bx + 1.42, 1.75, title, fontsize=9.0, fontweight="bold", color="#000000", ha="center")
        ax.text(bx + 1.42, 1.3, desc, fontsize=7.5, color="#475569", ha="center")

    # Overall Caption
    ax.text(7.0, 0.25,
            "Fig. 2: Decoupled three-tier system architecture of the SE374 Pathology Workstation.\n"
            "Clean Architecture separation between Clinical Web Client, Medical API Service, and Core AI Engine.",
            fontsize=9.5, fontstyle="italic", color="#000000", ha="center")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="#FFFFFF", edgecolor="none")
    if output_svg:
        os.makedirs(os.path.dirname(os.path.abspath(output_svg)), exist_ok=True)
        plt.savefig(output_svg, format="svg", bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 2 (Academic Monochrome): {output_png}")


# ==============================================================================
# FIGURE 3: PIPELINE DATA FLOW (ACADEMIC GRAYSCALE)
# ==============================================================================
def generate_figure_3(output_png: str, output_svg: str = None):
    """Generates Figure 3 in high-contrast academic monochrome."""
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # Row 1 (Top, Left to Right): Preprocessing
    # 1. WSI Input
    b1 = FancyBboxPatch((0.8, 6.4), 3.0, 1.4, boxstyle="round,pad=0.04,rounding_size=0.1",
                        facecolor="#FFFFFF", edgecolor="#000000", lw=1.5)
    ax.add_patch(b1)
    ax.text(2.3, 7.35, "1. Raw WSI Slide", fontsize=10.0, fontweight="bold", color="#000000", ha="center")
    ax.text(2.3, 6.85, "Formats: .svs / .tif / .ndpi\nGigapixel: ~100k x 100k px", fontsize=8.0, color="#475569", ha="center")

    draw_arrow(ax, 3.85, 7.1, 4.65, 7.1)

    # 2. OTSU Tissue Masking
    b2 = FancyBboxPatch((4.7, 6.4), 3.0, 1.4, boxstyle="round,pad=0.04,rounding_size=0.1",
                        facecolor="#FFFFFF", edgecolor="#000000", lw=1.5)
    ax.add_patch(b2)
    ax.text(6.2, 7.35, "2. TIAToolbox Masking", fontsize=10.0, fontweight="bold", color="#000000", ha="center")
    ax.text(6.2, 6.85, "OTSU Background Thresholding\nIdentifies biological tissue polygon", fontsize=8.0, color="#475569", ha="center")

    draw_arrow(ax, 7.75, 7.1, 8.55, 7.1)

    # 3. Patch Tiling
    b3 = FancyBboxPatch((8.6, 6.4), 3.0, 1.4, boxstyle="round,pad=0.04,rounding_size=0.1",
                        facecolor="#FFFFFF", edgecolor="#000000", lw=1.5)
    ax.add_patch(b3)
    ax.text(10.1, 7.35, "3. Tiling Service", fontsize=10.0, fontweight="bold", color="#000000", ha="center")
    ax.text(10.1, 6.85, "50% Overlapping Tiling\n224x224 px patches at 10x/20x", fontsize=8.0, color="#475569", ha="center")

    # Arrow Down to Row 2
    draw_arrow(ax, 10.1, 6.35, 10.1, 5.55)

    # Row 2 (Middle, Right to Left): AI Core Pipeline
    # 4. Frozen Extractor
    b4 = FancyBboxPatch((8.6, 4.1), 3.0, 1.4, boxstyle="round,pad=0.04,rounding_size=0.1",
                        facecolor="#F8FAFC", edgecolor="#000000", lw=1.5)
    ax.add_patch(b4)
    ax.text(10.1, 5.05, "4. Frozen Extractor [FROZEN]", fontsize=10.0, fontweight="bold", color="#000000", ha="center")
    ax.text(10.1, 4.55, "CTransPath / CONCH / ResNet\nFixed embeddings (Zero OOM)", fontsize=8.0, color="#475569", ha="center")

    draw_arrow(ax, 8.55, 4.8, 7.75, 4.8)

    # 5. Stage 1 CAM with PDA
    b5 = FancyBboxPatch((4.7, 4.1), 3.0, 1.4, boxstyle="round,pad=0.04,rounding_size=0.1",
                        facecolor="#FFFFFF", edgecolor="#000000", lw=1.5)
    ax.add_patch(b5)
    ax.text(6.2, 5.05, "5. Stage 1: CAM + PDA", fontsize=10.0, fontweight="bold", color="#000000", ha="center")
    ax.text(6.2, 4.55, "Patch-level weak classification\nMulti-layer pseudo-masks (MLPS)", fontsize=8.0, color="#475569", ha="center")

    draw_arrow(ax, 4.65, 4.8, 3.85, 4.8)

    # 6. Stage 2 Segmentation & CGM
    b6 = FancyBboxPatch((0.8, 4.1), 3.0, 1.4, boxstyle="round,pad=0.04,rounding_size=0.1",
                        facecolor="#FFFFFF", edgecolor="#000000", lw=1.5)
    ax.add_patch(b6)
    ax.text(2.3, 5.05, "6. Stage 2: MLPS & CGM", fontsize=10.0, fontweight="bold", color="#000000", ha="center")
    ax.text(2.3, 4.55, "Dense semantic segmentation decoder\nClassification Gate eliminates ghosting", fontsize=8.0, color="#475569", ha="center")

    # Arrow Down to Row 3
    draw_arrow(ax, 2.3, 4.05, 2.3, 3.25)

    # Row 3 (Bottom, Left to Right): Post-processing & Clinical Delivery
    # 7. Probability Stitching
    b7 = FancyBboxPatch((0.8, 1.8), 3.0, 1.4, boxstyle="round,pad=0.04,rounding_size=0.1",
                        facecolor="#F8FAFC", edgecolor="#000000", lw=1.5)
    ax.add_patch(b7)
    ax.text(2.3, 2.75, "7. WSI Stitching", fontsize=10.0, fontweight="bold", color="#000000", ha="center")
    ax.text(2.3, 2.25, "Whole-slide spatial probability assembly\nGaussian boundary blending", fontsize=8.0, color="#475569", ha="center")

    draw_arrow(ax, 3.85, 2.5, 4.65, 2.5)

    # 8. Web DeepZoom Viewer & Biomarkers
    b8 = FancyBboxPatch((4.7, 1.8), 3.0, 1.4, boxstyle="round,pad=0.04,rounding_size=0.1",
                        facecolor="#FFFFFF", edgecolor="#000000", lw=1.5)
    ax.add_patch(b8)
    ax.text(6.2, 2.75, "8. Web Viewer & TME", fontsize=10.0, fontweight="bold", color="#000000", ha="center")
    ax.text(6.2, 2.25, "OpenSeadragon Pyramidal Navigation\nAutomated sTILs (%) & TAS ratio", fontsize=8.0, color="#475569", ha="center")

    draw_arrow(ax, 7.75, 2.5, 8.55, 2.5)

    # 9. CAP Pathology Report PDF
    b9 = FancyBboxPatch((8.6, 1.8), 3.0, 1.4, boxstyle="round,pad=0.04,rounding_size=0.1",
                        facecolor="#FFFFFF", edgecolor="#000000", lw=1.5)
    ax.add_patch(b9)
    ax.text(10.1, 2.75, "9. Clinical PDF Report", fontsize=10.0, fontweight="bold", color="#000000", ha="center")
    ax.text(10.1, 2.25, "CAP-compliant pathology report\nTissue area and microenvironment metrics", fontsize=8.0, color="#475569", ha="center")

    # Overall Caption
    ax.text(7.5, 0.75,
            "Fig. 3: End-to-end WSI data flow and inference pipeline.\n"
            "From gigapixel digital slide to web visualization and clinical pathology report.",
            fontsize=9.5, fontstyle="italic", color="#000000", ha="center")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="#FFFFFF", edgecolor="none")
    if output_svg:
        os.makedirs(os.path.dirname(os.path.abspath(output_svg)), exist_ok=True)
        plt.savefig(output_svg, format="svg", bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 3 (Academic Monochrome): {output_png}")


if __name__ == "__main__":
    p_dir = os.path.join("product", "proposal")
    d_dir = os.path.join("docs", "design")

    generate_figure_1(os.path.join(p_dir, "fig1_standard_wsi_pipeline.png"), os.path.join(d_dir, "fig1_standard_wsi_pipeline.svg"))
    generate_figure_1(os.path.join("docs", "lectures", "fig1_standard_wsi_pipeline.png"))

    generate_figure_2(os.path.join(p_dir, "fig2_system_architecture.png"), os.path.join(d_dir, "fig2_system_architecture.svg"))

    generate_figure_3(os.path.join(p_dir, "fig3_pipeline_dataflow.png"), os.path.join(d_dir, "fig3_pipeline_dataflow.svg"))

    print("All 3 academic monochrome figures generated successfully.")
