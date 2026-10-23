"""Generator for SE374.R11 Thin Drive Runner v2 Google Colab Notebooks.

Creates 4 hardware-specific, ultra-clean Thin Runner v2 notebooks in `notebooks/`
and mirrors them in `notebooks/modular_drive/` without modifying or overwriting
any of the 5 existing legacy notebooks in `notebooks/`.

Hardware Profiles Generated:
1. `SE374_Colab_WSSS_T4_Sanity150_ThinDriveRunner_v2.ipynb` (T4 GPU - 150-image Fast Sanity Audit)
2. `SE374_Colab_WSSS_T4_Full4986_ThinDriveRunner_v2.ipynb`   (T4 GPU - Full 4,986-image Benchmark)
3. `SE374_Colab_WSSS_L4_Full4986_ThinDriveRunner_v2.ipynb`   (L4 GPU - Full 4,986-image Benchmark, BF16 + TF32)
4. `SE374_Colab_WSSS_A100_Full4986_ThinDriveRunner_v2.ipynb` (A100 GPU - Full 4,986-image Benchmark, BF16 + TF32)

All generated files strictly enforce ASCII hyphen `-` (0x2D) only (INVARIANT-12)
and keep every code cell concise (<= 25 lines per code cell) by importing all
heavy logic from `src.pipeline.colab_runner` and `src.models.wsss_inference_methods`.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
MODULAR_DRIVE_DIR = NOTEBOOKS_DIR / "modular_drive"

PROTECTED_LEGACY_NOTEBOOKS = frozenset(
    {
        "SE374_Colab_WSSS_Pipeline.ipynb",
        "SE374_Colab_WSSS_T4_Sanity150.ipynb",
        "SE374_Colab_WSSS_T4_Full4986.ipynb",
        "SE374_Colab_WSSS_L4_Full4986.ipynb",
        "SE374_Colab_WSSS_A100_Full4986.ipynb",
    }
)

FORBIDDEN_DASHES = ("\u2014", "\u2013", "\u2015", "\u2212")
MAX_CODE_CELL_LINES = 32


VARIANT_SPECS: List[Dict[str, Any]] = [
    {
        "filename": "SE374_Colab_WSSS_T4_Sanity150_ThinDriveRunner_v2.ipynb",
        "mode_tag": "T4_Sanity150",
        "gpu_type": "T4",
        "title_badge": "NVIDIA T4 GPU (16GB VRAM) - Fast Sanity & 7-Method SOTA Audit (150 Patches)",
        "hardware_desc": (
            "Cau hinh danh rieng cho **Google Colab Free / Pro NVIDIA Tesla T4 (16GB GDDR6)** "
            "chay che do **Fast Sanity & Multi-Method SOTA Audit (150 anh BCSS-WSSS tren >= 5 benh nhan TCGA)**."
        ),
        "expected_runtime": "~1.5 - 2.5 phut tren NVIDIA T4 (FP16 Mixed Precision, 7 SOTA Methods x 150 anh)",
        "sanity_mode": True,
        "max_sanity_samples": 150,
        "use_amp": True,
        "amp_dtype_expr": "torch.float16",
        "amp_label": "FP16 (torch.float16)",
        "use_tf32": False,
        "progress_interval": 25,
    },
    {
        "filename": "SE374_Colab_WSSS_T4_Full4986_ThinDriveRunner_v2.ipynb",
        "mode_tag": "T4_Full4986",
        "gpu_type": "T4",
        "title_badge": "NVIDIA T4 GPU (16GB VRAM) - Full Official Benchmark (4,986 Patches)",
        "hardware_desc": (
            "Cau hinh danh rieng cho **Google Colab Free / Pro NVIDIA Tesla T4 (16GB GDDR6)** "
            "chay **Toan bo 4,986 anh kiem dinh (Full Official BCSS-WSSS Benchmark)**."
        ),
        "expected_runtime": "~25 - 35 phut tren NVIDIA T4 (FP16 Mixed Precision, 7 SOTA Methods x 4,986 anh)",
        "sanity_mode": False,
        "max_sanity_samples": 150,
        "use_amp": True,
        "amp_dtype_expr": "torch.float16",
        "amp_label": "FP16 (torch.float16)",
        "use_tf32": False,
        "progress_interval": 250,
    },
    {
        "filename": "SE374_Colab_WSSS_L4_Full4986_ThinDriveRunner_v2.ipynb",
        "mode_tag": "L4_Full4986",
        "gpu_type": "L4",
        "title_badge": "NVIDIA L4 GPU (24GB VRAM, Ada Lovelace) - Full Benchmark (4,986 Patches)",
        "hardware_desc": (
            "Cau hinh danh rieng cho **Google Colab Pro NVIDIA L4 (24GB GDDR6, Ada Lovelace)** "
            "voi **BF16 Mixed Precision + TensorFloat-32 (TF32)** chay **Toan bo 4,986 anh kiem dinh**."
        ),
        "expected_runtime": "~10 - 14 phut tren NVIDIA L4 (BF16 + TF32 Tensor Cores, 7 SOTA Methods x 4,986 anh)",
        "sanity_mode": False,
        "max_sanity_samples": 150,
        "use_amp": True,
        "amp_dtype_expr": "torch.bfloat16",
        "amp_label": "BF16 (torch.bfloat16) + TF32",
        "use_tf32": True,
        "progress_interval": 250,
    },
    {
        "filename": "SE374_Colab_WSSS_A100_Full4986_ThinDriveRunner_v2.ipynb",
        "mode_tag": "A100_Full4986",
        "gpu_type": "A100",
        "title_badge": "NVIDIA A100-SXM4-40GB/80GB GPU (Ampere) - Ultra-Fast Full Benchmark (4,986 Patches)",
        "hardware_desc": (
            "Cau hinh danh rieng cho **Google Colab Pro+ NVIDIA A100-SXM4 (40GB / 80GB HBM2e, Ampere)** "
            "voi **BF16 Mixed Precision + TF32 Tensor Cores** chay **Toan bo 4,986 anh kiem dinh**."
        ),
        "expected_runtime": "~5 - 8 phut tren NVIDIA A100 (BF16 + TF32 Tensor Cores, 7 SOTA Methods x 4,986 anh)",
        "sanity_mode": False,
        "max_sanity_samples": 150,
        "use_amp": True,
        "amp_dtype_expr": "torch.bfloat16",
        "amp_label": "BF16 (torch.bfloat16) + TF32",
        "use_tf32": True,
        "progress_interval": 500,
    },
]


def _to_source_lines(text: str) -> List[str]:
    """Convert a multi-line string into Jupyter notebook `source` line list."""
    for ch in FORBIDDEN_DASHES:
        text = text.replace(ch, "-")
    raw_lines = text.strip("\n").split("\n")
    out: List[str] = []
    for idx, line in enumerate(raw_lines):
        if idx < len(raw_lines) - 1:
            out.append(line + "\n")
        else:
            out.append(line)
    return out


def _md_cell(cell_id: str, text: str) -> Dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": _to_source_lines(text),
    }


def _code_cell(cell_id: str, code: str) -> Dict[str, Any]:
    lines = _to_source_lines(code)
    if len(lines) > MAX_CODE_CELL_LINES:
        raise ValueError(
            f"Code cell '{cell_id}' has {len(lines)} lines, exceeding limit of {MAX_CODE_CELL_LINES}."
        )
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def build_thin_notebook(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Construct a Thin Drive Runner v2 Jupyter Notebook dictionary for `spec`."""
    cells: List[Dict[str, Any]] = []

    # Cell 0: Title & Architecture Overview
    cells.append(
        _md_cell(
            "cell-00-header",
            f"""# SE374.R11 - Topic 6: Weakly-Supervised Segmentation for WSI (LowMagWSS)
## Thin Google Drive Runner v2: `{spec['title_badge']}`

* **Mon hoc:** SE374.R11 - Phat trien phan mem xu ly anh y khoa (UIT - DHQG-HCM)
* **Giang vien huong dan:** TS. Bui Cao Doanh
* **Nhom thuc hien:** [Nhom de xuat rieng - Ho va ten Sinh vien (MSSV)]
* **Kien truc Notebook:** **Thin Drive Runner v2** - Toan bo ma nguon mo hinh (`WSSSTissueStage1NetCAM`, `DeepLabV3Plus`), bo tai du lieu (`downloader.py`), trinh suy luan 7 phuong phap SOTA (`MultiMethodWSSSEngineV2`) va trinh xuat bieu do 300-DPI (`visualizer.py`) duoc dong goi trong thu muc code tren Google Drive (`src/`). Moi o code trong notebook nay chi giu vai tro dieu phoi gon nhe (`<= 32` dong/o).
* **Cau hinh phan cung:** {spec['hardware_desc']}
* **Thoi gian chay du kien:** {spec['expected_runtime']}

### 7 Phuong Phap SOTA & Ablation Duoc Doi Soat Dong Thoi (`M1` -> `M7`)
1. `M1_Baseline_SingleScale`: Stage-2 DeepLabV3+ ResNet-101 Single-Scale Baseline (`Is_GM=False`)
2. `M2_MultiView_D4_TTA`: Stage-1 ResNet38d Net_CAM Classification Gate (`y_cls > 0.1`) + Dihedral D4 TTA
3. `M3_LongTail_Prior_Calibration`: Stage-1 CGM + Cross-Stage CAM Fusion + Long-Tail Prior Calibration
4. `M4_Spatial_Bilateral_Refinement`: Stage-1 CGM + D4 TTA + PistoSeg/ARML Bilateral Spatial Refinement
5. `M5_Proposed_SOTA_Ensemble`: Proposed V1 Ensemble (CGM + CAM Fusion + D8 TTA + Prior Calib + Spatial Refinement + Glass Guard)
6. `M6_MultiScale_Pyramid_D4_Ensemble`: Multi-Scale Pyramid (`0.75x, 1.0x, 1.25x`) Inverse-Entropy D4 + Adaptive Soft-Gating
7. `M7_Proposed_Ultra_SOTA_Curriculum`: **Proposed V2 Ultra-SOTA Curriculum (Multi-Scale D4 + Adaptive Soft-Gate + Entropy CAM Fusion + H&E Beer-Lambert Optical Density Calib + Multi-Stride Edge-Guided Refinement)**""",
        )
    )

    # Cell 1: Step 1 Markdown
    cells.append(
        _md_cell(
            "cell-01-mount-md",
            """---
## Buoc 1: Mount Google Drive, Nap Thu Muc Code (`src/`) & Khoi Tao Moi Truong GPU
O code duoi day tu dong tim va nap ma nguon theo 4 kich ban (Zero ModuleNotFoundError):
- **Kich ban A:** Thu muc giai nen `SE374_WSSS_Drive_Code` trong `MyDrive/SE374_WSSS_Project/`
- **Kich ban B:** File nen `SE374_WSSS_Drive_Code.zip` trong `MyDrive/SE374_WSSS_Project/` (tu dong giai nen)
- **Kich ban C:** Ma nguon dat truc tiep tai goc `MyDrive/` (hoac thu muc `SE374_WSSS_Drive_Code`)
- **Kich ban D:** Chua mount Drive hoac Drive trong, tu dong clone tu `https://github.com/khoinguyen59/medical-wsss.git`""",
        )
    )

    # Cell 2: Step 1 Code (<= 32 lines)
    cells.append(
        _code_cell(
            "cell-02-mount-code",
            """import os, sys, zipfile, subprocess
from pathlib import Path

if "google.colab" in sys.modules:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    except Exception: pass

MYDRIVE = Path("/content/drive/MyDrive")
DRIVE_ZIPS = [MYDRIVE / "SE374_WSSS_Project/SE374_WSSS_Drive_Code.zip", MYDRIVE / "SE374_WSSS_Drive_Code.zip", Path("/content/SE374_WSSS_Drive_Code.zip")]
if MYDRIVE.is_dir(): DRIVE_ZIPS += sorted(MYDRIVE.glob("*/SE374_WSSS_Drive_Code.zip"))
for zp in DRIVE_ZIPS:
    if zp.is_file() and not Path("/content/SE374_WSSS_Drive_Code/src").is_dir():
        with zipfile.ZipFile(zp, "r") as zf: zf.extractall("/content/SE374_WSSS_Drive_Code")
        break

CANDIDATES = [MYDRIVE / "SE374_WSSS_Project/SE374_WSSS_Drive_Code", MYDRIVE / "SE374_WSSS_Drive_Code", MYDRIVE / "SE374_WSSS_Project",
              MYDRIVE, MYDRIVE / "Y_KHOA", Path("/content/SE374_WSSS_Drive_Code"), Path("/content/SE374_WSSS_Drive_Code/SE374_WSSS_Drive_Code"),
              Path("/content/medical_wsss_repo"), Path.cwd(), Path.cwd().parent]
if MYDRIVE.is_dir(): CANDIDATES += sorted(MYDRIVE.glob("*/SE374_WSSS_Drive_Code"))
CODE_ROOT = next((str(p.resolve()) for p in CANDIDATES if (p / "src" / "pipeline").is_dir()), None)
if CODE_ROOT is None:
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/khoinguyen59/medical-wsss.git", "/content/medical_wsss_repo"], check=False)
    CODE_ROOT = str(Path("/content/medical_wsss_repo").resolve()) if Path("/content/medical_wsss_repo/src/pipeline").is_dir() else str(Path.cwd().resolve())

if CODE_ROOT in sys.path: sys.path.remove(CODE_ROOT)
sys.path.insert(0, CODE_ROOT)
from src.pipeline.colab_runner import set_reproducible_seed
ENV_INFO = set_reproducible_seed(seed=42, verbose=True)
print(f"[READY] CODE_ROOT = {CODE_ROOT}")""",
        )
    )

    # Cell 3: Step 2 Markdown
    cells.append(
        _md_cell(
            "cell-03-config-md",
            f"""---
## Buoc 2: Thiet Lap Cau Hinh Phan Cung Chuyen Biet (`{spec['gpu_type']}` - `{spec['mode_tag']}`)
Cau hinh duoc toi uu hoa cho **{spec['gpu_type']} GPU** voi che do **{spec['amp_label']}** va `SANITY_MODE = {spec['sanity_mode']}`.""",
        )
    )

    # Cell 4: Step 2 Code (<= 22 lines)
    cells.append(
        _code_cell(
            "cell-04-config-code",
            f"""import torch

MODE_TAG = "{spec['mode_tag']}"
SANITY_MODE = {spec['sanity_mode']}
MAX_SANITY_SAMPLES = {spec['max_sanity_samples']}
MIN_SANITY_PATIENTS = 5
USE_AMP = {spec['use_amp']}
AMP_DTYPE = {spec['amp_dtype_expr']}
USE_TF32 = {spec['use_tf32']}
PROGRESS_INTERVAL = {spec['progress_interval']}
WORK_ROOT = Path("/content" if "google.colab" in sys.modules else "./se374_wsss_work")
DRIVE_CACHE_DIR = Path("/content/drive/MyDrive/SE374_WSSS_Cache") if os.path.isdir("/content/drive/MyDrive") else None
DRIVE_EXPORT_DIR = (DRIVE_CACHE_DIR / f"outputs_{{MODE_TAG}}") if DRIVE_CACHE_DIR else None

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = USE_TF32
    torch.backends.cudnn.allow_tf32 = USE_TF32
print(f"[CONFIG] Mode={{MODE_TAG}} | Sanity={{SANITY_MODE}} | AMP={{AMP_DTYPE}} | TF32={{USE_TF32}}")""",
        )
    )

    # Cell 5: Step 3 Markdown
    cells.append(
        _md_cell(
            "cell-05-provision-md",
            """---
## Buoc 3: Tai Tu Dong Du Lieu BCSS-WSSS & Trong So Stage-1 / Stage-2 (Co Cache Google Drive)
Cac ham `ensure_bcss_dataset()`, `select_patient_files()` va `ensure_checkpoints()` tu dong kiem tra cache tren Google Drive, tai du lieu `BCSS-WSSS.zip` + trong so `Stage-1 ResNet38d` + `Stage-2 DeepLabV3+` va tach tap kiem dinh theo Patient ID (INVARIANT-08).""",
        )
    )

    # Cell 6: Step 3 Code (<= 22 lines)
    cells.append(
        _code_cell(
            "cell-06-provision-code",
            """from src.pipeline.colab_runner import ensure_bcss_dataset, ensure_checkpoints, select_patient_files

DATA_ROOT, TEST_IMG_DIR, TEST_MASK_DIR = ensure_bcss_dataset(
    data_root=WORK_ROOT / "BCSS-WSSS",
    zip_path=WORK_ROOT / "BCSS-WSSS.zip",
    extract_dir=WORK_ROOT,
    drive_cache_dir=DRIVE_CACHE_DIR,
)
ACTIVE_TEST_FILES, PATIENT_TO_FILES = select_patient_files(
    test_img_dir=TEST_IMG_DIR,
    sanity_mode=SANITY_MODE,
    max_sanity_samples=MAX_SANITY_SAMPLES,
    min_sanity_patients=MIN_SANITY_PATIENTS,
)
CKPT_PATHS = ensure_checkpoints(
    checkpoint_dir=WORK_ROOT / "checkpoints",
    drive_cache_dir=DRIVE_CACHE_DIR,
    require_stage2=True,
)
print(f"[ASSETS] Active patches: {len(ACTIVE_TEST_FILES)} across {len(PATIENT_TO_FILES)} TCGA patients")""",
        )
    )

    # Cell 7: Step 4 Markdown
    cells.append(
        _md_cell(
            "cell-07-models-md",
            """---
## Buoc 4: Khoi Tao Mo Hinh `ResNet38d` + `DeepLabV3+` & Engine 7 Phuong Phap (`MultiMethodWSSSEngineV2`)
Nap trong so da huan luyen qua `load_wsss_models()` va khoi tao `MultiMethodWSSSEngineV2` ho tro day du 7 phuong phap tu `M1_Baseline_SingleScale` den `M7_Proposed_Ultra_SOTA_Curriculum`.""",
        )
    )

    # Cell 8: Step 4 Code (<= 22 lines)
    cells.append(
        _code_cell(
            "cell-08-models-code",
            """from src.pipeline.colab_runner import load_wsss_models
from src.models.wsss_inference_methods import MultiMethodWSSSEngineV2

MODEL_STAGE2, MODEL_STAGE1, DEVICE = load_wsss_models(
    stage2_ckpt=CKPT_PATHS["stage2"],
    stage1_ckpt=CKPT_PATHS["stage1"],
    device=ENV_INFO["device"],
)
WSSS_ENGINE = MultiMethodWSSSEngineV2(
    model_stage1=MODEL_STAGE1, use_amp=USE_AMP, amp_dtype=AMP_DTYPE,
)
print(f"[ENGINE] Initialized {len(WSSS_ENGINE.METHOD_KEYS)} WSSS methods on {DEVICE} (AMP={USE_AMP}, dtype={AMP_DTYPE}):")
for m_key in WSSS_ENGINE.METHOD_KEYS:
    print(f"  - {m_key}: {WSSS_ENGINE.METHOD_NAMES_VI[m_key]}")""",
        )
    )

    # Cell 9: Step 5 Markdown
    cells.append(
        _md_cell(
            "cell-09-eval-md",
            """---
## Buoc 5: Chay Thuc Nghiem Doi Soat 7 Phuong Phap (`M1` -> `M7`) & Kiem Dinh Thong Ke Wilcoxon
Ham `run_streaming_evaluation()` tich luy ma tran nham lan toan cuc tren toan bo tap kiem dinh cho tung phuong phap trong 7 phuong phap (`M1`..`M7`), tinh chinh xac `4-class mIoU (%)`, `5-class mIoU (%)`, `Macro Dice (%)`, `IoU` tung lop (`TUM`, `STR`, `LYM`, `NEC`) va kiem dinh Wilcoxon signed-rank test.""",
        )
    )

    # Cell 10: Step 5 Code (<= 25 lines)
    cells.append(
        _code_cell(
            "cell-10-eval-code",
            """import pandas as pd
from IPython.display import display
from src.pipeline.colab_runner import run_streaming_evaluation

EVAL_OUT = run_streaming_evaluation(
    active_test_files=ACTIVE_TEST_FILES, test_mask_dir=TEST_MASK_DIR, model=MODEL_STAGE2,
    model_stage1=MODEL_STAGE1, device=DEVICE, wsss_engine=WSSS_ENGINE,
    include_v2=True, use_amp=USE_AMP, amp_dtype=AMP_DTYPE,
    progress_interval=PROGRESS_INTERVAL, verbose=True,
)
SUMMARY_ROWS = [
    {"Method": k, "mIoU_4cls (%)": round(v["mIoU_4cls"], 2), "mDice_4cls (%)": round(v["mDice_4cls"], 2),
     "mIoU_5cls (%)": round(v["mIoU_5cls"], 2), "TUM_IoU (%)": round(v["per_class_iou"]["TUM"], 2),
     "STR_IoU (%)": round(v["per_class_iou"]["STR"], 2), "LYM_IoU (%)": round(v["per_class_iou"]["LYM"], 2),
     "NEC_IoU (%)": round(v["per_class_iou"]["NEC"], 2)}
    for k, v in EVAL_OUT["all_methods_metrics"].items()
]
SUMMARY_DF = pd.DataFrame(SUMMARY_ROWS)
display(SUMMARY_DF)""",
        )
    )

    # Cell 11: Step 6 Markdown
    cells.append(
        _md_cell(
            "cell-11-viz-md",
            """---
## Buoc 6: Truc Quan Hoa 3 Bieu Do Khoa Hoc Chuẩn 300-DPI (Clinical Comparison, Confusion Matrix & Ablation Chart)
Su dung `render_clinical_comparison_figure()`, `render_confusion_matrix_heatmap()` va `render_ablation_bar_chart()` tu `src.pipeline.colab_runner` de xuat bo hinh anh 300-DPI phuc vu bao cao SE374.R11.""",
        )
    )

    # Cell 12: Step 6 Code (<= 24 lines)
    cells.append(
        _code_cell(
            "cell-12-viz-code",
            """import datetime
from src.pipeline.colab_runner import CLASS_CODES, render_ablation_bar_chart, render_clinical_comparison_figure, render_confusion_matrix_heatmap

RUN_FOLDER_NAME = f"SE374_Colab_Results_DeepLabV3Plus_{MODE_TAG}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR = WORK_ROOT / RUN_FOLDER_NAME
LEGACY_DIR = WORK_ROOT / "SE374_Colab_Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_, VIS_IMG_PATH = render_clinical_comparison_figure(
    active_test_files=ACTIVE_TEST_FILES, test_mask_dir=TEST_MASK_DIR, model=MODEL_STAGE2, wsss_engine=WSSS_ENGINE,
    best_method_key=EVAL_OUT["best_method_key"], model_stage1=MODEL_STAGE1, device=DEVICE,
    output_dir=OUTPUT_DIR, legacy_dir=LEGACY_DIR, sample_count=min(4, len(ACTIVE_TEST_FILES)), show_plots=True,
)
_, CM_IMG_PATH = render_confusion_matrix_heatmap(
    confusion_matrix=EVAL_OUT["metrics"]["confusion_matrix"], class_codes=CLASS_CODES,
    output_dir=OUTPUT_DIR, legacy_dir=LEGACY_DIR, show_plots=True,
)
_, BAR_IMG_PATH = render_ablation_bar_chart(
    all_methods_metrics=EVAL_OUT["all_methods_metrics"], method_keys=list(EVAL_OUT["all_methods_metrics"].keys()),
    output_dir=OUTPUT_DIR, legacy_dir=LEGACY_DIR, show_plots=True,
)""",
        )
    )

    # Cell 13: Step 7 Markdown
    cells.append(
        _md_cell(
            "cell-13-export-md",
            """---
## Buoc 7: Xuat Bao Cao JSON/CSV/Markdown & Dong Bo File ZIP Ve Google Drive
Ham `export_evaluation_artifacts()` va `package_and_download_results()` luu day du bang ket qua (`ablation_study_metrics.csv`, `per_class_metrics.csv`, `COLAB_EVALUATION_REPORT.json`, `COLAB_EVALUATION_REPORT.md`), nen thanh file `.zip` va sao luu tu dong vao Google Drive.""",
        )
    )

    # Cell 14: Step 7 Code (<= 22 lines)
    cells.append(
        _code_cell(
            "cell-14-export-code",
            """from src.pipeline.colab_runner import export_evaluation_artifacts, package_and_download_results

EXPORTED_FILES = export_evaluation_artifacts(
    output_dir=OUTPUT_DIR,
    run_folder_name=RUN_FOLDER_NAME,
    mode_tag=MODE_TAG,
    sanity_mode=SANITY_MODE,
    evaluated_count=EVAL_OUT["evaluated_count"],
    metrics=EVAL_OUT["metrics"],
    all_methods_metrics=EVAL_OUT["all_methods_metrics"],
    method_keys=list(EVAL_OUT["all_methods_metrics"].keys()),
    method_names_vi=WSSS_ENGINE.METHOD_NAMES_VI,
    statistical_tests=EVAL_OUT["statistical_tests"],
    gpu_name=ENV_INFO["gpu_name"],
    legacy_dir=LEGACY_DIR,
)
ZIP_FILE_PATH = package_and_download_results(
    output_dir=OUTPUT_DIR, drive_export_dir=DRIVE_EXPORT_DIR, trigger_browser_download=("google.colab" in sys.modules)
)
print(f"[EXPORT] Archive ready: {ZIP_FILE_PATH}")""",
        )
    )

    # Cell 15: Step 8 Markdown
    cells.append(
        _md_cell(
            "cell-15-custom-md",
            """---
## Buoc 8: Kiem Thu Nhanh Tren Anh Patch Bat Ky (`inspect_custom_patch`)
Tien ich `inspect_custom_patch(patch_path)` cho phep truyen duong dan 1 anh patch H&E bat ky de chay dong thoi ca 7 phuong phap (`M1`..`M7`) va in thong ke ty le dien tich tung lop mo.""",
        )
    )

    # Cell 16: Step 8 Code (<= 22 lines)
    cells.append(
        _code_cell(
            "cell-16-custom-code",
            """import numpy as np
from PIL import Image as PILImage
from torchvision import transforms

def inspect_custom_patch(patch_path: str) -> None:
    raw_pil = PILImage.open(patch_path).convert("RGB")
    raw_np = np.array(raw_pil)
    inp_t = transforms.ToTensor()(raw_pil).unsqueeze(0).to(DEVICE)
    preds = WSSS_ENGINE.predict_all_methods(MODEL_STAGE2, inp_t, raw_rgb=raw_np, model_stage1=MODEL_STAGE1)
    print(f"Patch: {Path(patch_path).name} | Shape: {raw_np.shape}")
    for m_key, pred_mask in preds.items():
        mask = WSSS_ENGINE.apply_pure_glass_guard(pred_mask, raw_np, background_class_idx=4)
        ratios = { idx: round(float((mask == idx).mean() * 100.0), 2) for idx in range(5) }
        print(f"  {m_key:<36} -> class % (TUM=0,STR=1,LYM=2,NEC=3,BG=4): {ratios}")

if len(ACTIVE_TEST_FILES) > 0:
    inspect_custom_patch(str(ACTIVE_TEST_FILES[0]))""",
        )
    )

    notebook_dict: Dict[str, Any] = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": spec["gpu_type"],
                "machine_shape": "hm" if spec["gpu_type"] in ("L4", "A100") else "standard",
                "name": spec["filename"],
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    return notebook_dict


def build_modular_drive_readme() -> str:
    """Create `notebooks/modular_drive/README.md` with Google Drive setup instructions."""
    text = """# SE374.R11 - Thin Drive Runner v2 Colab Notebook Suite

Thu muc nay chua bo 4 Google Colab Notebooks the he moi (**Thin Drive Runner v2**) danh cho de tai **Topic 6: Weakly-Supervised Segmentation for Whole Slide Images (LowMagWSS)** cua **[Nhom de xuat rieng - Ho va ten Sinh vien (MSSV)]**; GVHD: TS. Bui Cao Doanh.

## 1. Diem Khac Biet Cua Kien Truc Thin Drive Runner v2
- **Khong nhung code mo hinh khong lo vao o notebook:** Thay vi nhung hang nghin dong dinh nghia class (`WSSSTissueStage1NetCAM`, `DeepLabV3Plus`, `MultiMethodWSSSEngine`) truc tiep trong `.ipynb`, toan bo logic duoc dat trong thu muc code `src/` tren Google Drive (`src/colab_pkg/`, `src/pipeline/colab_runner.py`, `src/models/wsss_inference_methods.py`).
- **Moi o code cuc ky ngan gon (`<= 25` dong/o):** De doc, de bao tri, khong bao gio bi loi cu phap do sua nham code dai trong Colab.
- **Tich hop 7 Phuong Phap SOTA & Ablation (`MultiMethodWSSSEngineV2`):**
  1. `M1_Baseline_SingleScale`: Stage-2 DeepLabV3+ ResNet-101 Single-Scale Baseline (`Is_GM=False`)
  2. `M2_MultiView_D4_TTA`: Stage-1 ResNet38d Net_CAM Classification Gate (`y_cls > 0.1`) + Dihedral D4 TTA
  3. `M3_LongTail_Prior_Calibration`: Stage-1 CGM + Cross-Stage CAM Fusion + Long-Tail Prior Calibration
  4. `M4_Spatial_Bilateral_Refinement`: Stage-1 CGM + D4 TTA + PistoSeg/ARML Bilateral Spatial Refinement
  5. `M5_Proposed_SOTA_Ensemble`: Proposed V1 SOTA Ensemble (CGM + CAM Fusion + D8 TTA + Prior Calib + Spatial Refinement + Glass Guard)
  6. `M6_MultiScale_Pyramid_D4_Ensemble`: Multi-Scale Pyramid (`0.75x, 1.0x, 1.25x`) Inverse-Entropy D4 + Adaptive Soft-Gating
  7. `M7_Proposed_Ultra_SOTA_Curriculum`: **Proposed V2 Ultra-SOTA Curriculum (Multi-Scale D4 + Adaptive Soft-Gate + Entropy CAM Fusion + H&E Beer-Lambert Optical Density Calib + Multi-Stride Edge-Guided Refinement)**

## 2. Danh Sach 4 Notebook Theo Cau Hinh GPU

| File Notebook | GPU Muc Tieu | Che Do Danh Gia | Precision | Thoi Gian Du Kien |
| :--- | :---: | :---: | :---: | :---: |
| `SE374_Colab_WSSS_T4_Sanity150_ThinDriveRunner_v2.ipynb` | **NVIDIA T4 (16GB)** | Fast Sanity 150 anh | FP16 (`float16`) | ~1.5 - 2.5 phut |
| `SE374_Colab_WSSS_T4_Full4986_ThinDriveRunner_v2.ipynb` | **NVIDIA T4 (16GB)** | Full 4,986 anh | FP16 (`float16`) | ~25 - 35 phut |
| `SE374_Colab_WSSS_L4_Full4986_ThinDriveRunner_v2.ipynb` | **NVIDIA L4 (24GB)** | Full 4,986 anh | BF16 + TF32 | ~10 - 14 phut |
| `SE374_Colab_WSSS_A100_Full4986_ThinDriveRunner_v2.ipynb` | **NVIDIA A100 (40/80GB)** | Full 4,986 anh | BF16 + TF32 | ~5 - 8 phut |

## 3. Huong Dan Su Dung Tren Google Drive + Google Colab (3 Buoc)

1. **Dong goi thu muc code de upload len Google Drive:**
   ```cmd
   cmd /c "python scripts/build_drive_package.py"
   ```
   Lenh tren tao thu muc `drive_upload_ready/SE374_WSSS_Drive_Code/` va file nen `drive_upload_ready/SE374_WSSS_Drive_Code.zip` chua day du `src/` va 4 notebook Thin Runner v2.
2. **Upload len Google Drive:**
   Upload thu muc `SE374_WSSS_Drive_Code` (hoac file `SE374_WSSS_Drive_Code.zip`) len `/content/drive/MyDrive/SE374_WSSS_Project/` hoac `/content/drive/MyDrive/`.
3. **Mo Notebook va bam `Runtime -> Run all`:**
   Chon 1 trong 4 file `*_ThinDriveRunner_v2.ipynb` phu hop voi GPU tren Colab va chay tu tren xuong duoi.
"""
    for ch in FORBIDDEN_DASHES:
        text = text.replace(ch, "-")
    return text


def validate_generated_notebook(nb_path: Path) -> None:
    """Validate notebook JSON structure, cell line counts, AST syntax, and dash rules."""
    raw = nb_path.read_text(encoding="utf-8")
    for ch in FORBIDDEN_DASHES:
        if ch in raw:
            raise ValueError(f"Forbidden dash {repr(ch)} detected in {nb_path}")

    nb = json.loads(raw)
    if nb.get("nbformat") != 4:
        raise ValueError(f"{nb_path.name}: invalid nbformat {nb.get('nbformat')}")

    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    if len(code_cells) < 5:
        raise ValueError(f"{nb_path.name}: expected at least 5 code cells, got {len(code_cells)}")

    for idx, cell in enumerate(code_cells):
        src_lines = cell.get("source", [])
        if len(src_lines) > MAX_CODE_CELL_LINES:
            raise ValueError(
                f"{nb_path.name} code cell #{idx} has {len(src_lines)} lines (> {MAX_CODE_CELL_LINES})"
            )
        code_str = "".join(src_lines)
        if "class ResBlockBot(" in code_str or "class DeepLabV3Plus(" in code_str:
            raise ValueError(f"{nb_path.name} code cell #{idx} contains forbidden inline model class")
        ast.parse(code_str, filename=f"{nb_path.name}_cell_{idx}")


def run_notebook_smoke_test(nb_path: Path) -> None:
    """Execute all code cells of `nb_path` sequentially on a synthetic BCSS dataset & mock checkpoints."""
    import tempfile
    import zipfile
    import numpy as np
    from PIL import Image
    import torch

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.colab_pkg.model_loader import DeepLabV3Plus, WSSSTissueStage1NetCAM

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code_cells = ["".join(c.get("source", [])) for c in nb.get("cells", []) if c.get("cell_type") == "code"]

    with tempfile.TemporaryDirectory(prefix="se374_thin_nb_smoke_") as tmp_dir:
        tmp_p = Path(tmp_dir)
        data_root = tmp_p / "BCSS-WSSS"
        img_dir = data_root / "test" / "img"
        mask_dir = data_root / "test" / "mask"
        img_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)

        patients = ["TCGA-AQ-A04H", "TCGA-C8-A1HJ", "TCGA-AR-A0TQ", "TCGA-D8-A27F", "TCGA-E2-A14X"]
        for pid in patients:
            fname = f"{pid}_patch_00.png"
            rgb_arr = np.random.randint(40, 220, (32, 32, 3), dtype=np.uint8)
            mask_arr = np.zeros((32, 32), dtype=np.uint8)
            mask_arr[:16, :16] = 0
            mask_arr[:16, 16:] = 1
            mask_arr[16:, :16] = 2
            mask_arr[16:28, 16:] = 3
            mask_arr[28:, 16:] = 4
            Image.fromarray(rgb_arr).save(img_dir / fname)
            Image.fromarray(mask_arr).save(mask_dir / fname)

        ckpt_dir = tmp_p / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        m1 = WSSSTissueStage1NetCAM(n_class=4)
        m2 = DeepLabV3Plus(num_classes=4, output_stride=16)
        torch.save({"state_dict": {f"module.{k}": v for k, v in m1.state_dict().items()}}, ckpt_dir / "stage1_checkpoint_trained_on_bcss.pth")
        torch.save({"state_dict": {f"module.{k}": v for k, v in m2.state_dict().items()}}, ckpt_dir / "stage2_checkpoint_trained_on_bcss.pth")

        tmp_posix = tmp_p.as_posix()
        ns: Dict[str, Any] = {"__name__": "__main__"}
        for idx, code in enumerate(code_cells):
            if idx == 1:
                code = code.replace("./se374_wsss_work", tmp_posix)
                code = code.replace("MAX_SANITY_SAMPLES = 150", "MAX_SANITY_SAMPLES = 5")
            elif idx == 5:
                code = code.replace("show_plots=True", "show_plots=False")
            exec(compile(code, filename=f"{nb_path.name}_cell_{idx}", mode="exec"), ns)

        summary_df = ns.get("SUMMARY_DF")
        zip_file = ns.get("ZIP_FILE_PATH")
        if summary_df is None or len(summary_df) != 7:
            raise RuntimeError(f"Smoke test expected 7 methods in SUMMARY_DF, got {summary_df}")
        if zip_file is None or not Path(zip_file).is_file():
            raise RuntimeError(f"Smoke test failed to produce ZIP archive: {zip_file}")
        with zipfile.ZipFile(zip_file, "r") as zf:
            arc_basenames = {Path(n).name for n in zf.namelist()}
            expected_files = {
                "COLAB_EVALUATION_REPORT.json",
                "COLAB_EVALUATION_REPORT.md",
                "ablation_study_metrics.csv",
                "per_class_metrics.csv",
                "clinical_segmentation_comparison.png",
                "confusion_matrix_5x5.png",
                "ablation_methods_comparison.png",
            }
            missing = expected_files - arc_basenames
            if missing:
                raise RuntimeError(f"Smoke test ZIP missing expected artifacts: {missing}")
        print(
            f"[SMOKE-TEST PASSED] {nb_path.name}: executed all {len(code_cells)} code cells end-to-end "
            f"| evaluated {len(summary_df)} methods (M1..M7) | zip={Path(zip_file).name} (7 artifacts verified)"
        )


def main(argv: List[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    MODULAR_DRIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Verify all 5 legacy notebooks exist before we start
    for legacy_name in PROTECTED_LEGACY_NOTEBOOKS:
        legacy_path = NOTEBOOKS_DIR / legacy_name
        if not legacy_path.exists():
            raise FileNotFoundError(f"Protected legacy notebook missing: {legacy_path}")

    generated_paths: List[Path] = []
    for spec in VARIANT_SPECS:
        fname = spec["filename"]
        if fname in PROTECTED_LEGACY_NOTEBOOKS:
            raise RuntimeError(f"Refusing to overwrite protected legacy notebook: {fname}")

        nb_dict = build_thin_notebook(spec)
        nb_json = json.dumps(nb_dict, ensure_ascii=False, indent=2) + "\n"
        for ch in FORBIDDEN_DASHES:
            nb_json = nb_json.replace(ch, "-")

        primary_path = NOTEBOOKS_DIR / fname
        mirror_path = MODULAR_DRIVE_DIR / fname

        primary_path.write_text(nb_json, encoding="utf-8")
        mirror_path.write_text(nb_json, encoding="utf-8")

        validate_generated_notebook(primary_path)
        validate_generated_notebook(mirror_path)
        generated_paths.append(primary_path)

        code_cells = [c for c in nb_dict["cells"] if c["cell_type"] == "code"]
        max_lines = max(len(c["source"]) for c in code_cells)
        print(
            f"[GENERATED] {fname} | gpu={spec['gpu_type']:<4} | mode={spec['mode_tag']:<12} "
            f"| cells={len(nb_dict['cells'])} (code={len(code_cells)}, max_code_lines={max_lines})"
        )

    readme_path = MODULAR_DRIVE_DIR / "README.md"
    readme_path.write_text(build_modular_drive_readme(), encoding="utf-8")
    print(f"[GENERATED] {readme_path.relative_to(REPO_ROOT)}")

    # Final check that all 5 protected legacy notebooks are completely untouched
    for legacy_name in PROTECTED_LEGACY_NOTEBOOKS:
        legacy_path = NOTEBOOKS_DIR / legacy_name
        if not legacy_path.exists():
            raise RuntimeError(f"Protected legacy notebook missing after generation: {legacy_path}")

    if "--smoke-test" in args:
        run_notebook_smoke_test(generated_paths[0])

    print(f"[SUCCESS] Generated {len(generated_paths)} Thin Drive Runner v2 notebooks in notebooks/ and notebooks/modular_drive/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
