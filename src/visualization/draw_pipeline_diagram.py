"""
Scientific Figure Generator for SE374.R11 Medical AI.
Generates publication-grade, high-resolution (300 DPI) vector/raster diagrams
for the Standard Whole Slide Image (WSI) Processing Pipeline.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle, FancyArrowPatch
import numpy as np


def generate_standard_wsi_pipeline_figure(output_png_path: str, output_svg_path: str = None):
    """
    Renders Fig. 1: Illustration of the standard WSI-processing pipeline.
    Recreates the instructor's slide with publication-grade clarity, crisp typography,
    and professional medical imaging color palette.
    """
    # 16:7 aspect ratio for wide academic figures
    fig, ax = plt.subplots(figsize=(15, 6.5), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    # Background canvas
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # Font styling
    font_title = {"fontsize": 13, "fontweight": "bold", "color": "#1A202C", "ha": "center", "va": "center"}
    font_sub = {"fontsize": 9.5, "color": "#4A5568", "ha": "center", "va": "center"}
    font_item = {"fontsize": 10, "fontweight": "semibold", "color": "#2D3748", "ha": "left", "va": "center"}
    font_tag = {"fontsize": 8.5, "fontweight": "bold", "color": "#FFFFFF", "ha": "center", "va": "center"}

    # ----------------------------------------------------
    # BLOCK 1: INPUT WSI (x: 0.5 to 2.3)
    # ----------------------------------------------------
    # Glass slide background
    slide_bg = FancyBboxPatch((0.5, 1.8), 1.9, 3.4, boxstyle="round,pad=0.08,rounding_size=0.15",
                              facecolor="#F7FAFC", edgecolor="#CBD5E0", linewidth=1.5)
    ax.add_patch(slide_bg)

    # Slide label area (frost area)
    slide_label = FancyBboxPatch((0.6, 4.3), 1.7, 0.75, boxstyle="round,pad=0.02,rounding_size=0.05",
                                 facecolor="#E2E8F0", edgecolor="#CBD5E0", linewidth=1.0)
    ax.add_patch(slide_label)
    ax.text(1.45, 4.67, "SLIDE ID\nTCGA-BRCA", fontsize=7.5, fontweight="bold", color="#718096", ha="center", va="center")

    # Tissue biopsy illustration (pink/purple irregular morphology)
    t_x = np.array([1.0, 1.3, 1.8, 2.0, 1.7, 1.2, 0.9])
    t_y = np.array([2.5, 3.2, 3.5, 2.9, 2.2, 2.1, 2.3])
    ax.fill(t_x, t_y, color="#D53F8C", alpha=0.35)
    ax.plot(t_x, t_y, color="#97266D", linewidth=1.2, alpha=0.7)

    # Core high-grade tumor area
    c_x = np.array([1.2, 1.5, 1.7, 1.4])
    c_y = np.array([2.7, 3.1, 2.8, 2.4])
    ax.fill(c_x, c_y, color="#805AD5", alpha=0.5)

    ax.text(1.45, 1.35, "Input WSI", font_title)
    ax.text(1.45, 0.95, "Gigapixel Slide\n(~100,000 x 100,000 px)", font_sub)

    # ----------------------------------------------------
    # ARROW 1 -> Patch Tiling
    # ----------------------------------------------------
    arr1 = FancyArrowPatch((2.55, 3.5), (3.45, 3.5), arrowstyle="-|>,head_length=6,head_width=4",
                           color="#4A5568", linewidth=1.8, mutation_scale=15)
    ax.add_patch(arr1)
    ax.text(3.0, 3.85, "Patch\ntiling", fontsize=9.5, fontweight="semibold", color="#2D3748", ha="center", va="center")

    # ----------------------------------------------------
    # BLOCK 2: TILING & PATCHES (x: 3.6 to 4.7)
    # ----------------------------------------------------
    patch_y_positions = [4.3, 3.6, 2.9, 2.2]
    for i, py in enumerate(patch_y_positions):
        p_box = FancyBboxPatch((3.7, py), 0.55, 0.55, boxstyle="round,pad=0.02,rounding_size=0.06",
                               facecolor="#EDF2F7", edgecolor="#718096", linewidth=1.2)
        ax.add_patch(p_box)
        # Small histology texture inside patch
        ax.plot([3.85, 4.0, 4.1], [py + 0.15, py + 0.35, py + 0.2], color="#97266D", lw=1.2, alpha=0.6)

    # Magnified Single Patch Callout
    mag_box = FancyBboxPatch((4.5, 4.7), 0.9, 0.9, boxstyle="round,pad=0.03,rounding_size=0.08",
                             facecolor="#FFF5F5", edgecolor="#E53E3E", linewidth=1.5)
    ax.add_patch(mag_box)
    ax.text(4.95, 5.15, "H&E", fontsize=8, fontweight="bold", color="#E53E3E", ha="center", va="center")
    ax.text(4.95, 4.45, "224x224 px", fontsize=7.5, color="#718096", ha="center", va="center")

    # Connecting line from patch list to magnified view
    ax.plot([4.25, 4.5], [4.58, 4.9], color="#E53E3E", lw=1.2, linestyle="--")

    # ----------------------------------------------------
    # ARROW 2 -> Feature Extractor
    # ----------------------------------------------------
    arr2 = FancyArrowPatch((4.9, 3.5), (5.7, 3.5), arrowstyle="-|>,head_length=6,head_width=4",
                           color="#4A5568", linewidth=1.8, mutation_scale=15)
    ax.add_patch(arr2)

    # ----------------------------------------------------
    # BLOCK 3: FEATURE EXTRACTOR (FROZEN ❄) (x: 5.8 to 8.5)
    # ----------------------------------------------------
    fe_box = FancyBboxPatch((5.8, 1.4), 2.9, 4.4, boxstyle="round,pad=0.08,rounding_size=0.2",
                            facecolor="#F0FFF4", edgecolor="#38A169", linewidth=2.0)
    ax.add_patch(fe_box)

    # Badge: Frozen ❄
    badge_frozen = FancyBboxPatch((6.3, 5.3), 1.9, 0.45, boxstyle="round,pad=0.04,rounding_size=0.15",
                                  facecolor="#2B6CB0", edgecolor="none")
    ax.add_patch(badge_frozen)
    ax.text(7.25, 5.52, "[FROZEN] FEATURE EXTRACTOR", font_tag)

    # Header inside Feature Extractor
    ax.text(7.25, 4.85, "Pretrained Foundation Models", fontsize=10.5, fontweight="bold", color="#22543D", ha="center")
    ax.text(7.25, 4.55, "(Trained on large-scale histology)", fontsize=8, color="#48BB78", ha="center")

    # Model items with bullet icons
    models = [
        ("ResNet", "ResNet-50 / ResNet-38 (ConvNet Baseline)"),
        ("CTransPath", "Swin Transformer (SRCL Pretrained)"),
        ("UNI / UNI-2", "ViT-L/16 (100M+ Patches, DINOv2)"),
        ("CONCH", "Vision-Language CoCa (1.17M Pairs)"),
        ("Prov-GigaPath", "Billion-Parameter Whole-Slide ViT")
    ]
    for i, (m_name, m_desc) in enumerate(models):
        y_pos = 4.0 - i * 0.52
        # Bullet badge
        b_dot = FancyBboxPatch((6.05, y_pos - 0.12), 0.22, 0.24, boxstyle="round,pad=0.01,rounding_size=0.05",
                               facecolor="#C6F6D5", edgecolor="#38A169", lw=0.8)
        ax.add_patch(b_dot)
        ax.text(6.16, y_pos, ">", fontsize=7.5, fontweight="bold", color="#22543D", ha="center", va="center")
        ax.text(6.38, y_pos + 0.05, m_name, fontsize=9.5, fontweight="bold", color="#1A202C")
        ax.text(6.38, y_pos - 0.14, m_desc, fontsize=7.2, color="#4A5568")

    # ----------------------------------------------------
    # ARROW 3 -> Embedding Vectors
    # ----------------------------------------------------
    arr3 = FancyArrowPatch((8.85, 3.5), (9.55, 3.5), arrowstyle="-|>,head_length=6,head_width=4",
                           color="#4A5568", linewidth=1.8, mutation_scale=15)
    ax.add_patch(arr3)

    # ----------------------------------------------------
    # BLOCK 4: FEATURE EMBEDDINGS (x: 9.6 to 10.3)
    # ----------------------------------------------------
    for py in patch_y_positions:
        emb_box = FancyBboxPatch((9.7, py + 0.08), 0.7, 0.38, boxstyle="round,pad=0.02,rounding_size=0.08",
                                 facecolor="#EBF8FF", edgecolor="#3182CE", linewidth=1.2)
        ax.add_patch(emb_box)
        # Vector segments inside
        for vi in range(4):
            ax.axvline(9.82 + vi * 0.12, ymin=(py + 0.12)/6.5, ymax=(py + 0.42)/6.5, color="#63B3ED", lw=1.0)

    ax.text(10.05, 1.7, "Patch\nEmbeddings\n(512-1024d)", fontsize=8.5, fontweight="semibold", color="#2B6CB0", ha="center")

    # ----------------------------------------------------
    # ARROW 4 -> Trainable Model
    # ----------------------------------------------------
    arr4 = FancyArrowPatch((10.55, 3.5), (11.25, 3.5), arrowstyle="-|>,head_length=6,head_width=4",
                           color="#4A5568", linewidth=1.8, mutation_scale=15)
    ax.add_patch(arr4)

    # ----------------------------------------------------
    # BLOCK 5: TRAINABLE MODEL (FIRE 🔥) (x: 11.3 to 13.6)
    # ----------------------------------------------------
    tm_box = FancyBboxPatch((11.3, 1.4), 2.5, 4.4, boxstyle="round,pad=0.08,rounding_size=0.2",
                            facecolor="#FFFAF0", edgecolor="#DD6B20", linewidth=2.0)
    ax.add_patch(tm_box)

    # Badge: Trainable 🔥
    badge_train = FancyBboxPatch((11.6, 5.3), 1.9, 0.45, boxstyle="round,pad=0.04,rounding_size=0.15",
                                 facecolor="#C53030", edgecolor="none")
    ax.add_patch(badge_train)
    ax.text(12.55, 5.52, "[TRAINABLE] DOWNSTREAM", font_tag)

    ax.text(12.55, 4.85, "Task-Specific Models", fontsize=10.5, fontweight="bold", color="#7B341E", ha="center")

    # Downstream architectures
    mil_models = [
        ("CLAM", "Clustering-constrained MIL"),
        ("TransMIL", "Transformer-based MIL"),
        ("ABMIL / DSMIL", "Attention / Dual-Stream"),
        ("WSSS Decoder", "CAM + PDA + MLPS (Topic 6)")
    ]
    for i, (m_name, m_desc) in enumerate(mil_models):
        y_pos = 4.1 - i * 0.65
        b_dot2 = FancyBboxPatch((11.5, y_pos - 0.12), 0.22, 0.24, boxstyle="round,pad=0.01,rounding_size=0.05",
                                facecolor="#FEEBC8", edgecolor="#DD6B20", lw=0.8)
        ax.add_patch(b_dot2)
        ax.text(11.61, y_pos, "*", fontsize=9, fontweight="bold", color="#C05621", ha="center", va="center")
        ax.text(11.82, y_pos + 0.06, m_name, fontsize=9.5, fontweight="bold", color="#1A202C")
        ax.text(11.82, y_pos - 0.15, m_desc, fontsize=7.2, color="#744210")

    # ----------------------------------------------------
    # ARROW 5 -> Output
    # ----------------------------------------------------
    arr5 = FancyArrowPatch((13.9, 3.5), (14.35, 3.5), arrowstyle="-|>,head_length=6,head_width=4",
                           color="#4A5568", linewidth=1.8, mutation_scale=15)
    ax.add_patch(arr5)

    # ----------------------------------------------------
    # BLOCK 6: CLINICAL OUTPUT (x: 14.4 to 14.85)
    # ----------------------------------------------------
    out_box = FancyBboxPatch((14.4, 2.8), 0.55, 1.4, boxstyle="round,pad=0.04,rounding_size=0.1",
                             facecolor="#9F7AEA", edgecolor="#6B46C1", linewidth=1.5)
    ax.add_patch(out_box)
    ax.text(14.67, 3.8, "OUT", fontsize=8.5, fontweight="bold", color="#FFFFFF", ha="center", va="center")
    ax.text(14.67, 3.4, "Mask", fontsize=7.5, color="#FFFFFF", ha="center", va="center")
    ax.text(14.67, 3.05, "sTILs", fontsize=7.5, color="#FFFFFF", ha="center", va="center")

    # Overall Caption at bottom
    ax.text(7.5, 0.35,
            "Fig. 1: Illustration of the standard WSI-processing pipeline in Computational Pathology.\n"
            "Two-Stage Decoupled Framework: Frozen Foundation Feature Extractor (Zero OOM) -> Trainable Downstream Task Model.",
            fontsize=10.5, fontstyle="italic", color="#2D3748", ha="center", va="center")

    plt.tight_layout()

    # Ensure directories exist
    os.makedirs(os.path.dirname(os.path.abspath(output_png_path)), exist_ok=True)
    plt.savefig(output_png_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"Saved PNG to: {output_png_path}")

    if output_svg_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_svg_path)), exist_ok=True)
        plt.savefig(output_svg_path, format="svg", bbox_inches="tight")
        print(f"Saved SVG to: {output_svg_path}")

    plt.close()


if __name__ == "__main__":
    png_dest = os.path.join("product", "proposal", "fig1_standard_wsi_pipeline.png")
    svg_dest = os.path.join("docs", "design", "fig1_standard_wsi_pipeline.svg")
    lecture_dest = os.path.join("docs", "lectures", "fig1_standard_wsi_pipeline_scientific.png")

    generate_standard_wsi_pipeline_figure(png_dest, svg_dest)
    generate_standard_wsi_pipeline_figure(lecture_dest)
    print("Scientific figure generation complete.")
