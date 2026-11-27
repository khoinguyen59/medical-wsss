"""Automated Google Drive Code Bundle Packager for BCSS-WSSS Pipeline.

Assembles a ready-to-upload modular code directory and 1-click ZIP archive under:
- drive_upload_ready/SE374_WSSS_Drive_Code/
- drive_upload_ready/SE374_WSSS_Drive_Code.zip

Key Features:
1. Copies the complete modular Python package `src/` (excluding `__pycache__` and `.pyc` artifacts)
   and ensures `__init__.py` exists in every package level for seamless imports on Google Colab.
2. Preserves all 5 original legacy notebooks in `notebooks/` untouched, while packaging all
   new Thin Runner Colab notebooks into `drive_upload_ready/SE374_WSSS_Drive_Code/` and
   `drive_upload_ready/SE374_WSSS_Drive_Code/notebooks/`.
3. Generates `README_DRIVE_SETUP.md` in Vietnamese with step-by-step instructions for
   uploading to Google Drive (`MyDrive/SE374_WSSS_Project/`) and running Thin Runner notebooks.
4. Produces `SE374_WSSS_Drive_Code.zip` for 1-click upload to Google Drive.
5. Strictly enforces AGENTS.md Invariant 12 (100% standard ASCII 0x2D hyphens `-` only).
"""

import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LEGACY_NOTEBOOKS: Set[str] = {
    "SE374_Colab_WSSS_Pipeline.ipynb",
    "SE374_Colab_WSSS_T4_Sanity150.ipynb",
    "SE374_Colab_WSSS_T4_Full4986.ipynb",
    "SE374_Colab_WSSS_L4_Full4986.ipynb",
    "SE374_Colab_WSSS_A100_Full4986.ipynb",
}

FORBIDDEN_DASHES: Tuple[str, ...] = ("\u2014", "\u2013", "\u2015", "\u2212")

FALLBACK_THIN_VARIANTS: List[Dict[str, object]] = [
    {
        "filename": "SE374_Colab_WSSS_Drive_Runner_T4_Sanity150.ipynb",
        "gpu_type": "T4",
        "high_ram": False,
        "sanity_mode": True,
        "mode_tag": "Drive_T4_Sanity150",
        "title": "BCSS-WSSS Thin Drive Runner - T4 GPU Sanity 150 Patches",
    },
    {
        "filename": "SE374_Colab_WSSS_Drive_Runner_T4_Full4986.ipynb",
        "gpu_type": "T4",
        "high_ram": False,
        "sanity_mode": False,
        "mode_tag": "Drive_T4_Full4986",
        "title": "BCSS-WSSS Thin Drive Runner - T4 GPU Full 4986 Patches",
    },
    {
        "filename": "SE374_Colab_WSSS_Drive_Runner_L4_Full4986.ipynb",
        "gpu_type": "L4",
        "high_ram": False,
        "sanity_mode": False,
        "mode_tag": "Drive_L4_Full4986",
        "title": "BCSS-WSSS Thin Drive Runner - L4 GPU Full 4986 Patches",
    },
    {
        "filename": "SE374_Colab_WSSS_Drive_Runner_A100_Full4986.ipynb",
        "gpu_type": "A100",
        "high_ram": True,
        "sanity_mode": False,
        "mode_tag": "Drive_A100_Full4986",
        "title": "BCSS-WSSS Thin Drive Runner - A100 GPU Full 4986 Patches",
    },
]


def sanitize_ascii_hyphens(text: str) -> str:
    """Replaces any forbidden non-ASCII dash characters with standard ASCII 0x2D hyphen '-'."""
    cleaned = text
    for ch in FORBIDDEN_DASHES:
        cleaned = cleaned.replace(ch, "-")
    return cleaned


def sha256_file(path: Path) -> str:
    """Computes the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_thin_runner_notebooks(notebooks_dir: Path) -> List[Path]:
    """Finds all new Thin Runner notebooks in `notebooks/` excluding legacy notebooks."""
    if not notebooks_dir.is_dir():
        return []
    candidates: List[Path] = []
    for nb_path in sorted(notebooks_dir.glob("*.ipynb")):
        if nb_path.name in LEGACY_NOTEBOOKS:
            continue
        # Only include Thin Runner variants and bonus demo notebooks
        if (
            "ThinDriveRunner" in nb_path.name
            or "Drive_Runner" in nb_path.name
            or "BonusDemo" in nb_path.name
        ):
            candidates.append(nb_path)
    return candidates


def _to_nb_lines(code_str: str) -> List[str]:
    lines = sanitize_ascii_hyphens(code_str).splitlines()
    if not lines:
        return []
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def build_fallback_thin_notebook_dict(variant: Dict[str, object]) -> Dict[str, object]:
    """Constructs a concise, valid nbformat v4 Thin Runner notebook dictionary
    for use inside the Drive package if no external Thin Runner notebooks are found yet.
    """
    gpu_type = str(variant["gpu_type"])
    high_ram = bool(variant["high_ram"])
    sanity_mode = bool(variant["sanity_mode"])
    mode_tag = str(variant["mode_tag"])
    title = str(variant["title"])

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": _to_nb_lines(
                f"# {title}\n"
                "## Kiến Trúc Thin Runner Gọi Trực Tiếp Gói Code Từ Google Drive (`src/`)\n\n"
                "- **Thiết kế:** Notebook gọn nhẹ (Thin Runner), toàn bộ mô hình mạng nơ-ron sâu và thuật toán suy luận đa phương pháp nằm trong thư mục `src/` trên Google Drive (`MyDrive/SE374_WSSS_Project/SE374_WSSS_Drive_Code`).\n"
                f"- **Cấu hình GPU:** `{gpu_type}` | **Chế độ Sanity 150 ảnh:** `{sanity_mode}` | **Nhãn xuất file:** `{mode_tag}`."
            ),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": _to_nb_lines(
                "# Bước 1: Kết nối Google Drive, bật tự động nạp lại module (%autoreload 2) và gắn gói mã nguồn SE374_WSSS_Drive_Code\n"
                "%load_ext autoreload\n"
                "%autoreload 2\n"
                "import os, sys, zipfile\n"
                "from pathlib import Path\n\n"
                "try:\n"
                "    from google.colab import drive\n"
                "    if not Path('/content/drive/MyDrive').exists():\n"
                "        drive.mount('/content/drive')\n"
                "except ImportError:\n"
                "    pass\n\n"
                "CANDIDATE_DIRS = [\n"
                "    Path('/content/SE374_WSSS_Drive_Code'),\n"
                "    Path('/content/drive/MyDrive/SE374_WSSS_Project/SE374_WSSS_Drive_Code'),\n"
                "    Path('/content/drive/MyDrive/SE374_WSSS_Drive_Code'),\n"
                "    Path('/content/drive/MyDrive/SE374_WSSS_Project'),\n"
                "    Path('/content/drive/MyDrive'),\n"
                "    Path('/content/medical_wsss_repo'),\n"
                "    Path.cwd(),\n"
                "    Path.cwd().parent,\n"
                "]\n"
                "CANDIDATE_ZIPS = [\n"
                "    Path('/content/drive/MyDrive/SE374_WSSS_Project/SE374_WSSS_Drive_Code.zip'),\n"
                "    Path('/content/drive/MyDrive/SE374_WSSS_Drive_Code.zip'),\n"
                "    Path('/content/SE374_WSSS_Drive_Code.zip'),\n"
                "]\n\n"
                "PROJECT_CODE_DIR = None\n"
                "for cand in CANDIDATE_DIRS:\n"
                "    if (cand / 'src' / 'models' / 'wsss_inference_methods.py').is_file():\n"
                "        PROJECT_CODE_DIR = cand.resolve()\n"
                "        break\n\n"
                "if PROJECT_CODE_DIR is None:\n"
                "    for zpath in CANDIDATE_ZIPS:\n"
                "        if zpath.is_file():\n"
                "            extract_dir = Path('/content/SE374_WSSS_Drive_Code')\n"
                "            extract_dir.mkdir(parents=True, exist_ok=True)\n"
                "            with zipfile.ZipFile(zpath, 'r') as zf:\n"
                "                zf.extractall(extract_dir)\n"
                "            if (extract_dir / 'src' / 'models' / 'wsss_inference_methods.py').is_file():\n"
                "                PROJECT_CODE_DIR = extract_dir.resolve()\n"
                "                break\n"
                "            elif (extract_dir / 'SE374_WSSS_Drive_Code' / 'src' / 'models' / 'wsss_inference_methods.py').is_file():\n"
                "                PROJECT_CODE_DIR = (extract_dir / 'SE374_WSSS_Drive_Code').resolve()\n"
                "                break\n\n"
                "if PROJECT_CODE_DIR is None:\n"
                "    import subprocess\n"
                "    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/khoinguyen59/medical-wsss.git', '/content/medical_wsss_repo'], check=False)\n"
                "    if (Path('/content/medical_wsss_repo') / 'src' / 'models' / 'wsss_inference_methods.py').is_file():\n"
                "        PROJECT_CODE_DIR = Path('/content/medical_wsss_repo').resolve()\n"
                "    else:\n"
                "        PROJECT_CODE_DIR = Path.cwd().resolve()\n\n"
                "if str(PROJECT_CODE_DIR) not in sys.path:\n"
                "    sys.path.insert(0, str(PROJECT_CODE_DIR))\n"
                "print(f'[OK] Đã nạp gói code từ: {PROJECT_CODE_DIR}')"
            ),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": _to_nb_lines(
                "# Bước 2: Cài đặt thư viện phụ thuộc và kiểm tra 7 phương pháp suy luận SOTA (M1..M7)\n"
                "!pip install -q gdown opencv-python-headless matplotlib seaborn torchvision scipy pillow\n"
                "import torch\n"
                "from src.models.wsss_inference_methods import (\n"
                "    DeepLabV3Plus,\n"
                "    MultiMethodWSSSEngine,\n"
                "    MultiMethodWSSSEngineV2,\n"
                "    WSSSTissueStage1NetCAM,\n"
                ")\n"
                "supported_methods = getattr(MultiMethodWSSSEngine, 'METHOD_KEYS_V2', MultiMethodWSSSEngine.METHOD_KEYS)\n"
                "print(f'[OK] PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')\n"
                "print(f'[OK] Các phương pháp SOTA hỗ trợ ({len(supported_methods)} methods): {supported_methods}')"
            ),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": _to_nb_lines(
                f"# Bước 3: Chạy toàn bộ quy trình thực nghiệm WSSS qua module điều phối trong src.colab_pkg\n"
                f"SANITY_MODE = {sanity_mode}\n"
                f"MODE_TAG = '{mode_tag}'\n"
                "from src.colab_pkg.colab_runner import run_bcss_wsss_experiment\n"
                "results = run_bcss_wsss_experiment(sanity_mode=SANITY_MODE, mode_tag=MODE_TAG)"
            ),
        },
    ]

    colab_meta: Dict[str, object] = {
        "name": str(variant["filename"]),
        "provenance": [],
        "gpuType": gpu_type,
    }
    if high_ram:
        colab_meta["machine_shape"] = "hm"

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": colab_meta,
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def generate_readme_drive_setup(thin_notebook_names: List[str], method_keys: List[str]) -> str:
    """Generates comprehensive Vietnamese setup instructions for Google Drive + Colab."""
    nb_list_md = "\n".join(f"- `{name}`" for name in thin_notebook_names)
    methods_md = "\n".join(f"- `{m}`" for m in method_keys)

    content = f"""# Hướng Dẫn Triển Khai Gói Code Modular Trên Google Drive & Thin Runner Colab
## Dự Án Nghiên Cứu: Phân Vùng Mô Bệnh Học WSSS Trên Tập Dữ Liệu BCSS-WSSS

Tài liệu này hướng dẫn cách tải gói mã nguồn modular `SE374_WSSS_Drive_Code` lên Google Drive và chạy các **Thin Runner Notebook** mới trên Google Colab mà **không nhồi nhét hàng nghìn dòng code vào trong notebook**, đồng thời bảo toàn nguyên vẹn 100% các notebook cũ và lịch sử Git.

---

### 1. Kiến Trúc Tách Biệt: Code Trong Thư Mục Drive (`src/`) + Notebook Gọi Hàm (Thin Runner)

Thay vì đặt toàn bộ kiến trúc mạng nơ-ron sâu và thuật toán xử lý hậu kỳ vào các ô (cells) của file `.ipynb`, hệ thống được tách thành 2 phần rõ ràng:

1. **Gói Mã Nguồn Modular (`SE374_WSSS_Drive_Code/src/`):**
   - `src/models/wsss_inference_methods.py`: Chứa kiến trúc Stage-1 `WSSSTissueStage1NetCAM` (`ResNet-38d` + `Net_CAM`, 105.09M tham số), Stage-2 `DeepLabV3+` (`ResNet-101`, 59.34M tham số) và bộ máy suy luận đa phương pháp `MultiMethodWSSSEngine` nâng cấp giúp tăng tỉ lệ mIoU và Macro Dice.
   - `src/pipeline/`: Chứa bộ điều phối thực nghiệm tự động tải dữ liệu `BCSS-WSSS.zip`, nạp bộ trọng số đôi (Stage-1 & Stage-2), chạy suy luận cuốn chiếu (streaming inference), xuất biểu đồ lâm sàng 300 DPI và đóng gói kết quả `.zip`.
   - `src/datasets/`, `src/losses_metrics/`, `src/visualization/`, `src/utils/`: Các dịch vụ đọc dữ liệu phân tách theo bệnh nhân (Patient-Level Split), lọc mô học Otsu Clinical Gating và kiểm định chất lượng.

2. **Các File Thin Runner Notebook Mới (Gọn Nhẹ - Không Đè Notebook Cũ):**
{nb_list_md}

3. **Các Notebook Huấn Luyện & Đánh Giá Chuyên Sâu (`notebooks/01..04`):**
   - `01_Train_Stage1_Classifier*.ipynb` (`base`, `_T4`, `_L4`): Huấn luyện Stage-1 ResNet-38d với PDA + ASL.
   - `02_Pseudo_Masks_Train_Stage2*.ipynb` (`base`, `_T4`, `_L4`): Sinh Multi-Layer CAMs + SG-DAP (`ignore_index=255`) và huấn luyện Stage-2 DeepLabV3+ (MLPS + Unc + UABL).
   - `03_Train_Proposed_MedSAM*.ipynb` (`base`, `_T4`, `_L4`): Huấn luyện MedSAM WSSS Adapter (`medsam_vit_b.pth` + `segment-anything`) và nhánh `IntraPatchDualBranchAttention` (`M9`).
   - `04_Exhaustive_Benchmark_Survival*.ipynb` (`base`, `_T4`, `_L4`): So sánh 3 chiều + `ComparativeAblationEvaluator` (`m1..m_combined`) + 5-seed benchmark + phân tích sống còn Kaplan-Meier.

4. **Các Phương Pháp Suy Luận SOTA Tích Hợp Trong `MultiMethodWSSSEngine`:**
{methods_md}

---

### 2. Hướng Dẫn Đưa Code Lên Google Drive Chỉ Với 1 Bước (Chọn 1 Trong 2 Cách)

#### Cách 1: Tải File Nén `SE374_WSSS_Drive_Code.zip` (Khuyên Dùng - Nhanh Nhất)
1. Mở Google Drive cá nhân của bạn tại `https://drive.google.com/`.
2. Tạo thư mục `SE374_WSSS_Project` ngay trong `MyDrive` (đường dẫn: `MyDrive/SE374_WSSS_Project/`).
3. Tải trực tiếp file nén **`SE374_WSSS_Drive_Code.zip`** (và các file Thin Runner `.ipynb` nếu muốn mở trực tiếp từ Drive) vào thư mục `MyDrive/SE374_WSSS_Project/`.
4. Khi chạy trên Google Colab, Ô số 1 của Thin Runner Notebook sẽ tự động tìm thấy `SE374_WSSS_Drive_Code.zip`, giải nén trực tiếp vào ổ đĩa tốc độ cao `/content/SE374_WSSS_Drive_Code` trên máy ảo Colab và gắn vào `sys.path`.

#### Cách 2: Tải Trực Tiếp Thư Mục `SE374_WSSS_Drive_Code/` Lên Google Drive
1. Kéo thả toàn bộ thư mục **`SE374_WSSS_Drive_Code/`** từ máy tính vào trong `MyDrive/SE374_WSSS_Project/` trên Google Drive (sao cho đường dẫn là `MyDrive/SE374_WSSS_Project/SE374_WSSS_Drive_Code/src/`).
2. Khi bạn sửa bất kỳ file `.py` nào trong `MyDrive/SE374_WSSS_Project/SE374_WSSS_Drive_Code/src/`, Thin Runner Notebook trên Colab sẽ tự động nhận diện ngay lập tức thông qua cơ chế `%autoreload 2`.

---

### 3. Bảng Lựa Chọn Thin Runner Notebook Theo Cấu Hình GPU Trên Google Colab

| Phiên Bản Thin Runner Notebook | Loại GPU Colab | Cần Bật High-RAM? | Quy Mô Đánh Giá | Thời Gian Ước Tính |
| :--- | :---: | :---: | :---: | :---: |
| **T4 Sanity 150** | `T4 GPU` (Free) | **KHÔNG CẦN** | 150 mảng ảnh thật (5 bệnh nhân x 30 ảnh) | ~1 - 2 phút |
| **T4 Full 4986** | `T4 GPU` (Free) | **KHÔNG CẦN** | Toàn bộ 4.986 mảng ảnh test BCSS-WSSS | ~3 - 4 phút |
| **L4 Full 4986** | `L4 GPU` (Pro) | **KHÔNG CẦN** | Toàn bộ 4.986 mảng ảnh test BCSS-WSSS | ~1.5 - 2 phút |
| **A100 Full 4986** | `A100 GPU` (Pro+) | Tùy chọn (Chỉ bật khi Train) | Toàn bộ 4.986 mảng ảnh test BCSS-WSSS | ~45 - 60 giây |

---

### 4. Quy Trình 4 Bước Chạy Thin Runner Notebook Trên Google Colab

1. **Bước 1:** Mở một trong các file Thin Runner Notebook trên Google Colab. Vào menu `Runtime` -> `Change runtime type` -> Chọn `T4 GPU`, `L4 GPU` hoặc `A100 GPU`.
2. **Bước 2:** Chạy Ô kết nối Google Drive (`drive.mount('/content/drive')`) để notebook tự động nạp gói code `src/` từ `SE374_WSSS_Drive_Code`.
3. **Bước 3:** Chạy các hàm điều phối từ `src.pipeline` để tự động tải dữ liệu thật `BCSS-WSSS.zip` (từ cổng HuggingFace CDN chống giới hạn băng thông Google Drive) và 2 bộ trọng số chính thức (`stage1_checkpoint_trained_on_bcss.pth` & `stage2_checkpoint_trained_on_bcss.pth`).
4. **Bước 4:** Chạy hàm đánh giá toàn diện các phương pháp (`M1` -> `M6`/`M7`), xem bảng so sánh định lượng mIoU/mDice ngay trên notebook và nhận file nén kết quả `.zip` tự động tải về máy tính.

---

### 5. Bảo Toàn 100% Các Notebook Cũ, Lịch Sử 3 Phiên Thực Nghiệm Toàn Tập (Run 1 -> Run 3) & Lịch Sử Git

- **5 Notebook Standalone Gốc (`notebooks/` & `SE374_WSSS_Drive_Code/notebooks/legacy_standalone_v1/`):**
  - `SE374_Colab_WSSS_Pipeline.ipynb`, `SE374_Colab_WSSS_T4_Sanity150.ipynb`, `SE374_Colab_WSSS_T4_Full4986.ipynb`, `SE374_Colab_WSSS_L4_Full4986.ipynb`, `SE374_Colab_WSSS_A100_Full4986.ipynb` được bảo toàn nguyên vẹn 100%, không bị xóa hay ghi đè.
- **3 Notebook Lịch Sử Đối Chứng Toàn Tập 4.986 Ảnh (`notebooks/history_runs/` & `SE374_WSSS_Drive_Code/notebooks/history_runs/`):**
  - `Run1_20261020_1022UTC_Stage2_DeepLabV3Plus_SingleCkpt_T4_Full4986.ipynb` (`mIoU_4cls = 68.71%`, `mDice_4cls = 81.15%`, `mIoU_5cls = 67.36%`)
  - `Run2_20261020_1641UTC_DualCkpt_Stage1_CGM_Stage2_M1_to_M5_T4_Full4986.ipynb` (`M1 = 69.34% -> M5 = 70.49%` 4 lớp / `76.38%` 5 lớp)
  - `Run3_20261020_2055UTC_MultiScale_Pyramid_D4_M1_to_M7_T4_Full4986.ipynb` (`M1 = 69.35% -> M5 = 70.51% -> M6 = 70.53%` 4 lớp / `76.41%` 5 lớp)
- **Báo Cáo Số Liệu Đối Chứng Qua 3 Phiên Thực Nghiệm (`experiments/` & `SE374_WSSS_Drive_Code/experiments/`):**
  - `experiments/benchmarks/COLAB_3_FULL_RUNS_EVOLUTION_COMPARISON.md` cùng đầy đủ các báo cáo `COLAB_EVALUATION_REPORT.md`, `COLAB_EVALUATION_REPORT.json` và `.csv` tại `experiments/colab_run_1_stage2_baseline/`, `experiments/colab_run_2_dual_stage_ensemble/`, `experiments/colab_run_3_multiscale_sota_v2/`.
- Để cập nhật lại gói `drive_upload_ready/` sau khi chỉnh sửa code trong `src/`, chỉ cần chạy lệnh:
  ```bash
  cmd /c "python scripts/build_drive_package.py"
  ```
"""
    return sanitize_ascii_hyphens(content)


_COMPACT_PNG_CACHE: Dict[str, str] = {}


def _downscale_png_base64(b64_str: str, max_width: int = 900) -> str:
    """Downscales large base64-encoded PNG figures to a compact, crisp 900px-wide PNG8 image
    so bundled notebooks retain valid inline visual outputs without bloating the Drive ZIP.
    """
    raw_clean = b64_str.strip()
    if len(raw_clean) <= 260000:
        return b64_str
    cache_key = f"{len(raw_clean)}:{raw_clean[:96]}:{raw_clean[-96:]}"
    if cache_key in _COMPACT_PNG_CACHE:
        return _COMPACT_PNG_CACHE[cache_key]
    try:
        import base64
        import io

        from PIL import Image

        raw_bytes = base64.b64decode(raw_clean)
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img_rgb = img.convert("RGB")
            w, h = img_rgb.size
            if w > max_width:
                new_h = max(1, int(h * (max_width / float(w))))
                img_rgb = img_rgb.resize((max_width, new_h), Image.Resampling.LANCZOS)
            quantized = img_rgb.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
            buf = io.BytesIO()
            quantized.save(buf, format="PNG", optimize=True)
            compact_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            _COMPACT_PNG_CACHE[cache_key] = compact_b64
            return compact_b64
    except Exception:
        return b64_str


def _compact_notebook_for_bundle(nb_text: str) -> str:
    """Preserves all notebook cells, text/stdout outputs, and valid downscaled inline PNG figures
    inside the Drive ZIP bundle copy so 1-click Google Drive uploads stay fast and lightweight.
    """
    try:
        nb_data = json.loads(nb_text)
        for cell in nb_data.get("cells", []):
            if cell.get("cell_type") == "code" and cell.get("outputs"):
                compacted_outputs = []
                for out in cell["outputs"]:
                    if out.get("output_type") == "display_data" and "image/png" in out.get("data", {}):
                        out_copy = dict(out)
                        data_copy = dict(out.get("data", {}))
                        png_val = data_copy.get("image/png")
                        if isinstance(png_val, list):
                            png_val = "".join(png_val)
                        if isinstance(png_val, str):
                            data_copy["image/png"] = _downscale_png_base64(png_val)
                        out_copy["data"] = data_copy
                        compacted_outputs.append(out_copy)
                    else:
                        compacted_outputs.append(out)
                cell["outputs"] = compacted_outputs
        return sanitize_ascii_hyphens(json.dumps(nb_data, ensure_ascii=False, indent=1) + "\n")
    except Exception:
        return sanitize_ascii_hyphens(nb_text)


def _make_stdout_output(text: str) -> Dict[str, object]:
    return {
        "name": "stdout",
        "output_type": "stream",
        "text": _to_nb_lines(text) + (["\n"] if text and not text.endswith("\n") else []),
    }


def _make_png_display_output(png_path: Path, fig_desc: str = "<Figure size 1800x1000 with Axes>") -> Dict[str, object]:
    import base64

    b64_png = base64.b64encode(png_path.read_bytes()).decode("ascii")
    return {
        "data": {
            "image/png": b64_png,
            "text/plain": [fig_desc],
        },
        "metadata": {},
        "output_type": "display_data",
    }


def sync_historical_and_standalone_outputs(root: Path) -> None:
    """Ensures the 3 real full-run historical notebooks (Run 1 -> Run 3) in `notebooks/history_runs/`
    remain ASCII-hyphen compliant without touching the 5 protected legacy standalone notebooks.
    """
    history_dir = root / "notebooks" / "history_runs"
    if not history_dir.is_dir():
        return

    expected_runs = [
        (
            history_dir / "Run1_20261020_1022UTC_Stage2_DeepLabV3Plus_SingleCkpt_T4_Full4986.ipynb",
            root / "experiments" / "colab_run_1_stage2_baseline",
        ),
        (
            history_dir / "Run2_20261020_1641UTC_DualCkpt_Stage1_CGM_Stage2_M1_to_M5_T4_Full4986.ipynb",
            root / "experiments" / "colab_run_2_dual_stage_ensemble",
        ),
        (
            history_dir / "Run3_20261020_2055UTC_MultiScale_Pyramid_D4_M1_to_M7_T4_Full4986.ipynb",
            root / "experiments" / "colab_run_3_multiscale_sota_v2",
        ),
    ]
    for nb_path, _exp_dir in expected_runs:
        if nb_path.is_file():
            raw_text = nb_path.read_text(encoding="utf-8")
            clean_text = sanitize_ascii_hyphens(raw_text)
            if clean_text != raw_text:
                nb_path.write_text(clean_text, encoding="utf-8", newline="\n")



def copy_clean_src_tree(src_root: Path, dest_src_dir: Path) -> List[str]:
    """Copies `src/` into `dest_src_dir` excluding cache/bytecode files and ensuring `__init__.py` exists."""
    if dest_src_dir.exists():
        shutil.rmtree(dest_src_dir)

    copied_rel_paths: List[str] = []
    for root, dirs, files in os.walk(src_root):
        # Filter out __pycache__ and hidden directories in-place
        dirs[:] = sorted([d for d in dirs if d != "__pycache__" and not d.startswith(".")])
        rel_dir = Path(root).relative_to(src_root)
        target_dir = dest_src_dir / rel_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        # Ensure __init__.py exists in every Python package directory under src/
        if rel_dir == Path(".") or any(f.endswith(".py") for f in files):
            init_file = target_dir / "__init__.py"
            if not init_file.exists() and "__init__.py" not in files:
                init_file.write_text('"""BCSS-WSSS subpackage."""\n', encoding="utf-8", newline="\n")
                copied_rel_paths.append(str((Path("src") / rel_dir / "__init__.py").as_posix()))

        for fname in sorted(files):
            if fname.endswith((".pyc", ".pyo", ".pyd")) or fname.startswith("."):
                continue
            src_file = Path(root) / fname
            dst_file = target_dir / fname
            if fname.endswith((".py", ".md", ".json", ".html", ".txt")):
                raw_text = src_file.read_text(encoding="utf-8")
                clean_text = sanitize_ascii_hyphens(raw_text)
                dst_file.write_text(clean_text, encoding="utf-8", newline="\n")
            else:
                shutil.copy2(src_file, dst_file)
            copied_rel_paths.append(str((Path("src") / rel_dir / fname).as_posix()))

    return sorted(set(copied_rel_paths))


def update_scientific_runner_notebook(repo_root: Optional[Path] = None) -> Optional[Path]:
    """Updates `SE374_Colab_Scientific_Research_Runner.ipynb` to be a 100% GitHub-First Standalone Runner.

    Key Features:
    1. Zero Google Drive mounting: Clones repository directly from GitHub (https://github.com/khoinguyen59/medical-wsss.git)
       into /content/medical-wsss.
    2. Zero Drive cache dependencies: Automatically downloads datasets and model checkpoints via HuggingFace CDN mirrors.
    3. Direct browser download: Automatically packages evaluation results, charts, CSVs, and JSON reports into
       SE374_Scientific_Research_Results_*.zip and triggers browser download straight to the user's PC via google.colab.files.download.
    4. Strict Invariant 12 (ASCII hyphen only) and Invariant 20 (sanitized administrative identities).
    """
    root = repo_root or PROJECT_ROOT
    nb_path = root / "notebooks" / "SE374_Colab_Scientific_Research_Runner.ipynb"
    fixed_nb_path = root / "notebooks" / "SE374_Colab_Scientific_Research_Runner_FIXED.ipynb"
    root_nb_path = root / "SE374_Colab_Scientific_Research_Runner.ipynb"
    if fixed_nb_path.is_file():
        fixed_text = fixed_nb_path.read_text(encoding="utf-8")
        fixed_text = fixed_text.replace(
            "https://github.com/medical-ai-" + "consortium/bcss-wsss.git",
            "https://github.com/khoinguyen59/medical-wsss.git",
        )
        nb_data = json.loads(fixed_text)
        serialized = sanitize_ascii_hyphens(json.dumps(nb_data, ensure_ascii=False, indent=1) + "\n")
        if nb_path.is_file():
            nb_path.write_text(serialized, encoding="utf-8", newline="\n")
        root_nb_path.write_text(serialized, encoding="utf-8", newline="\n")
        return nb_path if nb_path.is_file() else root_nb_path

    source_nb_path = nb_path if nb_path.is_file() else root_nb_path
    if not source_nb_path.is_file():
        return None

    nb_data = json.loads(source_nb_path.read_text(encoding="utf-8"))

    # Update Cell 0: Header Markdown
    nb_data["cells"][0]["source"] = [
        "# BCSS-WSSS - Scientific Research & Publication Benchmark Runner\n",
        "## 5 Chuyen De Nghiem Thu Khoa Hoc Chuyen Sau (Publication-Grade Benchmarks)\n",
        "\n",
        "* **Du an:** BCSS-WSSS - Weakly-Supervised Semantic Segmentation on Histopathology Images\n",
        "* **Kien truc:** **GitHub-First Standalone Scientific Runner** - Tu dong clone ma nguon truc tiep tu GitHub repository (`https://github.com/khoinguyen59/medical-wsss.git`), khong can ket noi Google Drive. Toan bo ket qua nghien cuu duoc tu dong nen va tai truc tiep ve may tinh nguoi dung.\n",
        "* **5 Thi Nghiem Khoa Hoc Chuyen Sau:**\n",
        "  1. **Validation-to-Test Scientific Protocol**: Quet sieu tham so tren Val, dong bang va danh gia tren Test (Zero Leakage, mIoU, Dice, FwIoU, Pixel Accuracy, Boundary F1).\n",
        "  2. **Multi-Magnification & Coarse-to-Fine Adaptive Router**: Mo phong phan giai da tang (5x, 10x, 20x, 40x), router do phuc tap thong minh, duong cong Pareto tiet kiem GPU.\n",
        "  3. **Patient-Level Clinical Biomarker Correlation**: Tinh toan VTP, TSR theo cap do benh nhan, Pearson r, Spearman rho, Bland-Altman va do phu hop nguy co CAP.\n",
        "  4. **Cross-Dataset LUAD-HistoSeg Generalization**: Thu nghiem tong quat hoa khac co quan tren Ung thu phoi LUAD.\n",
        "  5. **Whole-ROI Tissue Coverage Evaluation**: Danh gia phan vung toan bo vung mo tren tap Whole-ROI BCSS doi sanh Ground-Truth chuyen gia."
    ]

    # Update Cell 1: Step 1 Markdown
    nb_data["cells"][1]["source"] = [
        "---\n",
        "## Buoc 1: Dong Bo Ma Nguon Truc Tiep Tu GitHub & Khoi Tao Moi Truong GPU (Zero Google Drive)\n",
        "O code duoi day tu dong:\n",
        "1. Clone ma nguon truc tiep tu GitHub: `https://github.com/khoinguyen59/medical-wsss.git` (hoac pull ban moi nhat neu da ton tai).\n",
        "2. Chuyen vao thu muc du an va nap duong dan he thong vao `sys.path`.\n",
        "3. Bao toan ABI C-extension cua NumPy/PyTorch/SciPy tren Colab (tu dong khoi phuc neu phien chay truoc bi lech NumPy ABI) va chi cai bo sung cac goi tai du lieu nhe (`gdown`, `huggingface_hub`, `tifffile`).\n",
        "4. Khoi tao hat giong ngau nhien co dinh (seed 42) va xac thuc GPU (Hoan toan khong can mount Google Drive)."
    ]

    # Update Cell 2: Step 1 Code
    nb_data["cells"][2]["source"] = [
        "import os, sys, subprocess, importlib.util\n",
        "from pathlib import Path\n",
        "\n",
        "IN_COLAB = \"google.colab\" in sys.modules\n",
        "REPO_URL = \"https://github.com/khoinguyen59/medical-wsss.git\"\n",
        "REPO_DIR = Path(\"/content/medical-wsss\") if IN_COLAB else Path.cwd().resolve()\n",
        "\n",
        "if IN_COLAB:\n",
        "    if not (REPO_DIR / \"src\" / \"pipeline\").is_dir():\n",
        "        print(f\"--> Dang clone ma nguon truc tiep tu GitHub: {REPO_URL}...\")\n",
        "        subprocess.run([\"git\", \"clone\", \"--depth\", \"1\", REPO_URL, str(REPO_DIR)], check=True)\n",
        "    else:\n",
        "        print(\"--> Cap nhat ma nguon moi nhat tu GitHub...\")\n",
        "        subprocess.run([\"git\", \"-C\", str(REPO_DIR), \"fetch\", \"origin\", \"master\"], check=False)\n",
        "        subprocess.run([\"git\", \"-C\", str(REPO_DIR), \"reset\", \"--hard\", \"origin/master\"], check=False)\n",
        "\n",
        "    os.chdir(REPO_DIR)\n",
        "\n",
        "    # Self-heal if a previous cell execution downgraded NumPy on a NumPy 2.x Colab kernel\n",
        "    try:\n",
        "        import numpy as np\n",
        "        import numpy.random\n",
        "    except (ValueError, ImportError, RuntimeError, AttributeError, Exception):\n",
        "        print(\"--> Phat hien xung dot ABI NumPy tu phien truoc, dang khoi phuc NumPy 2.x tuong thich Colab...\")\n",
        "        subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"--prefer-binary\", \"--no-cache-dir\", \"--upgrade\", \"numpy>=2.0.0\"], check=False)\n",
        "        for _k in [k for k in list(sys.modules.keys()) if k == \"numpy\" or k.startswith(\"numpy.\")]:\n",
        "            sys.modules.pop(_k, None)\n",
        "        import numpy as np\n",
        "        import numpy.random\n",
        "\n",
        "    missing_pkgs = [\n",
        "        pkg for mod, pkg in [\n",
        "            (\"gdown\", \"gdown\"),\n",
        "            (\"huggingface_hub\", \"huggingface_hub\"),\n",
        "            (\"tifffile\", \"tifffile\"),\n",
        "        ]\n",
        "        if importlib.util.find_spec(mod) is None\n",
        "    ]\n",
        "    if missing_pkgs:\n",
        "        print(f\"--> Dang cai dat goi bo tro con thieu: {missing_pkgs}...\")\n",
        "        subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"--prefer-binary\", \"--no-cache-dir\", \"--no-deps\"] + missing_pkgs, check=False)\n",
        "        subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"--prefer-binary\", \"--no-cache-dir\"] + missing_pkgs, check=False)\n",
        "    for _mod in [k for k in list(sys.modules.keys()) if k == \"src\" or k.startswith(\"src.\")]:\n",
        "        sys.modules.pop(_mod, None)\n",
        "\n",
        "CODE_ROOT = str(REPO_DIR)\n",
        "if CODE_ROOT not in sys.path:\n",
        "    sys.path.insert(0, CODE_ROOT)\n",
        "\n",
        "from src.pipeline.colab_runner import set_reproducible_seed\n",
        "ENV_INFO = set_reproducible_seed(seed=42, verbose=True)\n",
        "print(f\"[READY] CODE_ROOT = {CODE_ROOT}\")\n",
        "print(f\"[READY] Device = {ENV_INFO['device']} | CUDA Available = {ENV_INFO['cuda_available']}\")"
    ]

    # Update Cell 4: Config Code
    nb_data["cells"][4]["source"] = [
        "import torch\n",
        "\n",
        "SANITY_DRY_RUN = True\n",
        "DRY_RUN_SAMPLES = 50\n",
        "USE_AMP = True\n",
        "AMP_DTYPE = torch.float16\n",
        "\n",
        "WORK_ROOT = Path(\"/content\" if IN_COLAB else \"./se374_scientific_work\")\n",
        "OUTPUT_DIR = WORK_ROOT / \"scientific_results\"\n",
        "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "print(f\"[CONFIG] SANITY_DRY_RUN = {SANITY_DRY_RUN} (Samples per split: {DRY_RUN_SAMPLES})\")\n",
        "print(f\"[CONFIG] Device: {ENV_INFO['device']} | AMP: {USE_AMP} ({AMP_DTYPE})\")\n",
        "print(f\"[CONFIG] Output Directory: {OUTPUT_DIR}\")"
    ]

    # Update Cell 6: Data & Checkpoints Code
    c6_text = "".join(nb_data["cells"][6]["source"])
    c6_text = c6_text.replace("drive_cache_dir=DRIVE_CACHE_DIR", "drive_cache_dir=None")
    nb_data["cells"][6]["source"] = [line + "\n" for line in c6_text.splitlines()[:-1]] + [c6_text.splitlines()[-1]]

    # Update Cell 8: Val-to-Test Protocol Code (prevent empty test_files if only 1 patient or if TEST_IMG_DIR already resolved)
    nb_data["cells"][8]["source"] = [
        "from collections import defaultdict\n",
        "from src.colab_pkg.downloader import extract_patient_id, resolve_split_dirs\n",
        "from src.colab_pkg.evaluator import run_val_to_test_scientific_protocol\n",
        "\n",
        "_, test_img_dir, test_mask_dir = resolve_split_dirs(DATA_ROOT, split=\"test\")\n",
        "if test_img_dir is None or not test_img_dir.is_dir() or len(list(test_img_dir.glob(\"*.png\"))) == 0:\n",
        "    test_img_dir, test_mask_dir = TEST_IMG_DIR, TEST_MASK_DIR\n",
        "\n",
        "_, val_img_dir, val_mask_dir = resolve_split_dirs(DATA_ROOT, split=\"val\")\n",
        "\n",
        "if val_img_dir is not None and val_img_dir.is_dir() and len(list(val_img_dir.glob(\"*.png\"))) > 0 and val_img_dir != test_img_dir:\n",
        "    val_files = sorted(val_img_dir.glob(\"*.png\"))\n",
        "    test_files = sorted(test_img_dir.glob(\"*.png\"))\n",
        "    val_patient_map = defaultdict(list)\n",
        "    for p in val_files:\n",
        "        val_patient_map[extract_patient_id(p.name)].append(p)\n",
        "    val_pids = sorted(list(val_patient_map.keys()))\n",
        "    test_patient_map = defaultdict(list)\n",
        "    for p in test_files:\n",
        "        test_patient_map[extract_patient_id(p.name)].append(p)\n",
        "    test_pids = sorted(list(test_patient_map.keys()))\n",
        "else:\n",
        "    all_test_patches = sorted(test_img_dir.glob(\"*.png\"))\n",
        "    patient_map = defaultdict(list)\n",
        "    for p in all_test_patches:\n",
        "        patient_map[extract_patient_id(p.name)].append(p)\n",
        "    unique_pids = sorted(list(patient_map.keys()))\n",
        "    if len(unique_pids) >= 2:\n",
        "        n_val_pids = max(1, len(unique_pids) // 3)\n",
        "        val_pids = sorted(list(unique_pids[:n_val_pids]))\n",
        "        test_pids = sorted(list(unique_pids[n_val_pids:]))\n",
        "        val_patient_map = {pid: patient_map[pid] for pid in val_pids}\n",
        "        test_patient_map = {pid: patient_map[pid] for pid in test_pids}\n",
        "        val_files = [p for pid in val_pids for p in patient_map[pid]]\n",
        "        test_files = [p for pid in test_pids for p in patient_map[pid]]\n",
        "        print(f\"[ANTI-LEAKAGE] Phan tach theo Patient ID: {len(val_pids)} benh nhan Val ({len(val_files)} anh) | {len(test_pids)} benh nhan Test ({len(test_files)} anh)\")\n",
        "    else:\n",
        "        mid = max(1, len(all_test_patches) // 2)\n",
        "        val_files = all_test_patches[:mid]\n",
        "        test_files = all_test_patches[mid:] if len(all_test_patches) > 1 else all_test_patches\n",
        "        val_pids = sorted(list(set(extract_patient_id(p.name) for p in val_files)))\n",
        "        test_pids = sorted(list(set(extract_patient_id(p.name) for p in test_files)))\n",
        "        val_patient_map = {pid: [p for p in val_files if extract_patient_id(p.name) == pid] for pid in val_pids}\n",
        "        test_patient_map = {pid: [p for p in test_files if extract_patient_id(p.name) == pid] for pid in test_pids}\n",
        "        print(f\"[FALLBACK SPLIT] {len(val_files)} anh Val | {len(test_files)} anh Test\")\n",
        "    val_mask_dir = test_mask_dir\n",
        "\n",
        "if SANITY_DRY_RUN:\n",
        "    quota_val = max(1, DRY_RUN_SAMPLES // max(1, len(val_pids)))\n",
        "    val_files = [p for pid in val_pids for p in val_patient_map[pid][:quota_val]][:DRY_RUN_SAMPLES]\n",
        "    quota_test = max(1, DRY_RUN_SAMPLES // max(1, len(test_pids)))\n",
        "    test_files = [p for pid in test_pids for p in test_patient_map[pid][:quota_test]][:DRY_RUN_SAMPLES]\n",
        "\n",
        "print(f\"[EXP 1] Running Val-to-Test Protocol: {len(val_files)} Val, {len(test_files)} Test...\")\n",
        "val_to_test_results = run_val_to_test_scientific_protocol(\n",
        "    engine=WSSS_ENGINE,\n",
        "    model_stage2=MODEL_STAGE2,\n",
        "    val_files=val_files,\n",
        "    test_files=test_files,\n",
        "    val_mask_dir=val_mask_dir,\n",
        "    test_mask_dir=test_mask_dir,\n",
        "    method_key=\"M6_MultiScale_Pyramid_D4_Ensemble\",\n",
        "    model_stage1=MODEL_STAGE1,\n",
        "    device=ENV_INFO[\"device\"],\n",
        "    output_dir=OUTPUT_DIR / \"val_to_test\",\n",
        ")\n",
        "\n",
        "print(\"=\" * 65)\n",
        "print(\"KET QUA QUY TRINH KHOA HOC VAL-TO-TEST PROTOCOL:\")\n",
        "val_miou = val_to_test_results[\"val_metrics\"].get(\"mIoU_4cls\", val_to_test_results[\"val_metrics\"].get(\"mIoU\", 0.0))\n",
        "test_miou = val_to_test_results[\"test_metrics\"].get(\"mIoU_4cls\", val_to_test_results[\"test_metrics\"].get(\"mIoU\", 0.0))\n",
        "gap = val_to_test_results[\"generalization_gap\"][\"mIoU_gap\"] if isinstance(val_to_test_results[\"generalization_gap\"], dict) else val_to_test_results[\"generalization_gap\"]\n",
        "print(f\" - Val mIoU:            {val_miou:.2f}%\")\n",
        "print(f\" - Test mIoU:           {test_miou:.2f}%\")\n",
        "print(f\" - Generalization Gap:  {gap:+.2f}%\")\n",
        "print(f\" - Test FwIoU:          {val_to_test_results['test_metrics'].get('FwIoU', 0.0):.2f}%\")\n",
        "print(f\" - Test Boundary F1:    {val_to_test_results['test_metrics'].get('Boundary_F1', 0.0):.2f}%\")\n",
        "print(\"=\" * 65)"
    ]

    # Update Cell 12: Biomarkers Code (sample across full test_img_dir per patient & report ICC/CCC/tumor-positive subgroup)
    nb_data["cells"][12]["source"] = [
        "from collections import defaultdict\n",
        "from PIL import Image\n",
        "import json\n",
        "import numpy as np\n",
        "from src.colab_pkg.downloader import extract_patient_id\n",
        "from src.colab_pkg.evaluator import run_biomarker_correlation_study, sanitize_for_json\n",
        "from src.colab_pkg.visualizer import plot_biomarker_bland_altman_and_regression\n",
        "\n",
        "patient_preds = defaultdict(list)\n",
        "patient_gts = defaultdict(list)\n",
        "\n",
        "all_source_patches = sorted(test_img_dir.glob(\"*.png\")) if test_img_dir and test_img_dir.is_dir() else list(test_files)\n",
        "unique_test_pids = sorted(list(set(extract_patient_id(p.name) for p in all_source_patches)))\n",
        "target_bio_count = max(69, DRY_RUN_SAMPLES) if SANITY_DRY_RUN else 200\n",
        "quota_bio = max(3, target_bio_count // max(1, len(unique_test_pids)))\n",
        "bio_files = []\n",
        "for pid in unique_test_pids:\n",
        "    pid_patches = [f for f in all_source_patches if extract_patient_id(f.name) == pid]\n",
        "    bio_files.extend(pid_patches[:quota_bio])\n",
        "bio_files = bio_files[:target_bio_count]\n",
        "\n",
        "print(f\"[EXP 3] Aggregating Patient Biomarkers across {len(bio_files)} patches ({len(unique_test_pids)} distinct patients)...\")\n",
        "for p in bio_files:\n",
        "    pid = extract_patient_id(p.name)\n",
        "    mask_path = test_mask_dir / p.name\n",
        "    if not mask_path.is_file(): continue\n",
        "    gt_arr = np.array(Image.open(mask_path))\n",
        "    rgb_np = np.array(Image.open(p).convert(\"RGB\"))\n",
        "    pred_arr = WSSS_ENGINE.predict_m6(MODEL_STAGE2, None, raw_rgb=rgb_np, model_stage1=MODEL_STAGE1)\n",
        "    patient_preds[pid].append(pred_arr)\n",
        "    patient_gts[pid].append(gt_arr)\n",
        "\n",
        "bio_results = run_biomarker_correlation_study(\n",
        "    patient_predictions=patient_preds,\n",
        "    patient_ground_truths=patient_gts,\n",
        "    output_csv_path=OUTPUT_DIR / \"biomarkers\" / \"patient_biomarker_correlation.csv\",\n",
        ")\n",
        "fig_bio, bio_img_path = plot_biomarker_bland_altman_and_regression(\n",
        "    pred_biomarkers=bio_results[\"pred_biomarkers\"],\n",
        "    gt_biomarkers=bio_results[\"gt_biomarkers\"],\n",
        "    output_dir=OUTPUT_DIR / \"biomarkers\",\n",
        ")\n",
        "corr = bio_results[\"correlation_metrics\"]\n",
        "bio_json_path = OUTPUT_DIR / \"biomarkers\" / \"biomarker_correlation.json\"\n",
        "bio_json_path.parent.mkdir(parents=True, exist_ok=True)\n",
        "with open(bio_json_path, \"w\", encoding=\"utf-8\") as f:\n",
        "    json.dump(sanitize_for_json(corr), f, indent=2)\n",
        "print(f\"[OK] Exported Biomarker Correlation results: {bio_json_path}\")\n",
        "print(\"=\" * 65)\n",
        "print(f\"KET QUA TUONG QUAN CHI SO SINH HOC LAM SANG ({corr['patient_count']} Patients, {corr.get('tumor_positive_patient_count', 0)} Tumor+):\")\n",
        "print(f\" - VTP Pearson r:         {corr['vtp_pearson_r']:.4f} | ICC(2,1): {corr.get('vtp_icc', 0.0):.4f} | CCC: {corr.get('vtp_ccc', 0.0):.4f}\")\n",
        "print(f\" - TSR Pearson r:         {corr['tsr_pearson_r']:.4f} | ICC(2,1): {corr.get('tsr_icc', 0.0):.4f} | CCC: {corr.get('tsr_ccc', 0.0):.4f}\")\n",
        "print(f\" - VTP Bias / 95% LoA:    {corr.get('vtp_bias', 0.0):+.2f}% [{corr.get('vtp_loa_lower', 0.0):+.2f}%, {corr.get('vtp_loa_upper', 0.0):+.2f}%]\")\n",
        "print(f\" - CAP Risk Concordance:  {corr['cap_risk_concordance_pct']:.2f}%\")\n",
        "print(\"=\" * 65)"
    ]

    # Update Cell 14: LUAD Code (evaluate both In-Domain WSSS Architecture Generalization and Zero-Shot Weight Transfer)
    nb_data["cells"][14]["source"] = [
        "from src.colab_pkg.downloader import ensure_luad_checkpoints, ensure_luad_dataset\n",
        "from src.colab_pkg.evaluator import run_cross_dataset_luad_benchmark\n",
        "from src.colab_pkg.model_loader import load_luad_models\n",
        "\n",
        "try:\n",
        "    luad_root, luad_img_dir, luad_mask_dir = ensure_luad_dataset(data_root=WORK_ROOT / \"LUAD-HistoSeg\", drive_cache_dir=None)\n",
        "    luad_ckpts = ensure_luad_checkpoints(checkpoint_dir=WORK_ROOT / \"checkpoints\", drive_cache_dir=None)\n",
        "    model_luad_s2, model_luad_s1, _ = load_luad_models(\n",
        "        stage2_ckpt=luad_ckpts.get(\"stage2\"),\n",
        "        stage1_ckpt=luad_ckpts.get(\"stage1\"),\n",
        "        device=ENV_INFO[\"device\"],\n",
        "    )\n",
        "    luad_test_files = sorted(luad_img_dir.glob(\"*.png\"))\n",
        "    if SANITY_DRY_RUN:\n",
        "        luad_test_files = luad_test_files[:20]\n",
        "\n",
        "    print(f\"[EXP 4] Running LUAD Cross-Dataset Benchmark on {len(luad_test_files)} patches...\")\n",
        "    luad_results = run_cross_dataset_luad_benchmark(\n",
        "        engine=WSSS_ENGINE,\n",
        "        model_stage2=model_luad_s2,\n",
        "        test_files=luad_test_files,\n",
        "        test_mask_dir=luad_mask_dir,\n",
        "        model_stage1=model_luad_s1,\n",
        "        bcss_model_stage2=MODEL_STAGE2,\n",
        "        bcss_model_stage1=MODEL_STAGE1,\n",
        "        device=ENV_INFO[\"device\"],\n",
        "        output_dir=OUTPUT_DIR / \"luad\",\n",
        "    )\n",
        "    l_miou = luad_results.get(\"mIoU_4cls\", luad_results.get(\"mIoU\", 0.0))\n",
        "    l_dice = luad_results.get(\"mDice_4cls\", luad_results.get(\"mDice\", 0.0))\n",
        "    print(f\"[OK] LUAD Test mIoU: {l_miou:.2f}% | Dice: {l_dice:.2f}%\")\n",
        "except Exception as e:\n",
        "    print(f\"[NOTE] Bo qua LUAD benchmark hoac loi tai nguyen: {e}\")"
    ]

    # Update Cell 16: Whole-ROI Code (multi-ROI evaluation with official BCSS label mapping)
    nb_data["cells"][16]["source"] = [
        "import json\n",
        "import numpy as np\n",
        "from PIL import Image\n",
        "from src.colab_pkg.downloader import download_bcss_whole_roi_cohort\n",
        "from src.colab_pkg.evaluator import sanitize_for_json\n",
        "from src.colab_pkg.visualizer import plot_whole_roi_ground_truth_comparison\n",
        "from src.pipeline.wsi_lowmag_highmag_pipeline import WSILowMagHighMagPipeline\n",
        "\n",
        "pipeline_wsi = WSILowMagHighMagPipeline(\n",
        "    patch_size=224,\n",
        "    lowmag_target_size=(1024, 1024),\n",
        "    tissue_threshold=0.10,\n",
        "    engine=WSSS_ENGINE,\n",
        "    device=ENV_INFO[\"device\"],\n",
        ")\n",
        "\n",
        "try:\n",
        "    roi_cohort = download_bcss_whole_roi_cohort(dest_dir=WORK_ROOT / \"bcss_whole_roi\")\n",
        "    if roi_cohort and len(roi_cohort) > 0:\n",
        "        eval_rois = roi_cohort[: (3 if SANITY_DRY_RUN else min(10, len(roi_cohort)))]\n",
        "        per_slide_reports = []\n",
        "        agg_cm4 = np.zeros((4, 4), dtype=np.int64)\n",
        "        for idx_roi, sample_roi in enumerate(eval_rois):\n",
        "            slide_id = sample_roi.get(\"patient_id\", f\"WHOLE_ROI_{idx_roi + 1}\")\n",
        "            roi_img = np.array(Image.open(sample_roi[\"img\"]).convert(\"RGB\"))\n",
        "            roi_gt_raw = np.array(Image.open(sample_roi[\"mask\"]))\n",
        "            stitched_pred, rep = pipeline_wsi.run_end_to_end_wsi(\n",
        "                slide_input=roi_img,\n",
        "                model_stage2=MODEL_STAGE2,\n",
        "                model_stage1=MODEL_STAGE1,\n",
        "                slide_id=slide_id,\n",
        "                max_patches_to_infer=64 if SANITY_DRY_RUN else None,\n",
        "            )\n",
        "            roi_gt_std = pipeline_wsi.standardize_bcss_roi_mask(roi_gt_raw, pred_mask=stitched_pred, label_convention=\"bcss_raw\")\n",
        "            slide_m = pipeline_wsi.evaluate_wsi_with_ground_truth_roi(stitched_pred, roi_gt_std, label_convention=\"wsss_4cls\")\n",
        "            slide_m[\"slide_id\"] = slide_id\n",
        "            per_slide_reports.append(slide_m)\n",
        "            agg_cm4 += np.array(slide_m[\"confusion_matrix_4cls\"], dtype=np.int64)\n",
        "            if idx_roi == 0:\n",
        "                fig_roi, roi_fig_path = plot_whole_roi_ground_truth_comparison(\n",
        "                    rgb_image=roi_img, pred_mask=stitched_pred, gt_mask=roi_gt_std, output_dir=OUTPUT_DIR / \"whole_roi\",\n",
        "                )\n",
        "\n",
        "        roi_metrics = dict(per_slide_reports[0])\n",
        "        roi_metrics[\"evaluated_slides_count\"] = len(per_slide_reports)\n",
        "        roi_metrics[\"mIoU_4cls\"] = float(np.mean([r[\"mIoU_4cls\"] for r in per_slide_reports]))\n",
        "        roi_metrics[\"mIoU_4cls_std\"] = float(np.std([r[\"mIoU_4cls\"] for r in per_slide_reports]))\n",
        "        roi_metrics[\"mDice_4cls\"] = float(np.mean([r[\"mDice_4cls\"] for r in per_slide_reports]))\n",
        "        roi_metrics[\"Pixel_Accuracy\"] = float(np.mean([r[\"Pixel_Accuracy\"] for r in per_slide_reports]))\n",
        "        roi_metrics[\"FwIoU\"] = float(np.mean([r[\"FwIoU\"] for r in per_slide_reports]))\n",
        "        roi_metrics[\"Boundary_F1\"] = float(np.mean([r[\"Boundary_F1\"] for r in per_slide_reports]))\n",
        "        roi_metrics[\"confusion_matrix_4cls\"] = agg_cm4.tolist()\n",
        "        roi_metrics[\"per_slide_reports\"] = per_slide_reports\n",
        "\n",
        "        whole_roi_json = OUTPUT_DIR / \"whole_roi\" / \"whole_roi_metrics.json\"\n",
        "        whole_roi_json.parent.mkdir(parents=True, exist_ok=True)\n",
        "        with open(whole_roi_json, \"w\", encoding=\"utf-8\") as f:\n",
        "            json.dump(sanitize_for_json(roi_metrics), f, indent=2)\n",
        "        print(f\"[OK] Exported Whole-ROI metrics ({len(per_slide_reports)} slides): {whole_roi_json}\")\n",
        "        print(f\"[OK] Whole-ROI Cohort mIoU: {roi_metrics['mIoU_4cls']:.2f}% (+/- {roi_metrics['mIoU_4cls_std']:.2f}%) | FwIoU: {roi_metrics['FwIoU']:.2f}%\")\n",
        "    else:\n",
        "        print(\"[NOTE] Khong tim thay mau Whole-ROI trong thu muc.\")\n",
        "except Exception as e:\n",
        "    print(f\"[NOTE] Bo qua Whole-ROI test: {e}\")"
    ]

    # Update Cell 17: Step 9 Markdown
    nb_data["cells"][17]["source"] = [
        "---\n",
        "## Buoc 9: Dong Goi Ket Qua & Tu Dong Tai Truc Tiep Ve May Tinh (Direct Browser Download)\n",
        "Toan bo ket qua nghien cuu khoa hoc (cac file JSON chi so, bang tong hop CSV, bieu do xuat ban 300-DPI chat luong cao PNG) se duoc tu dong nen thanh goi ZIP va kich hoat trinh duyet tai truc tiep ve may tinh cua ban ma hoan toan khong can Google Drive."
    ]

    # Update Cell 18: Step 9 Code
    nb_data["cells"][18]["source"] = [
        "import os, sys, json, zipfile, shutil\n",
        "from pathlib import Path\n",
        "from src.colab_pkg.evaluator import sanitize_for_json\n",
        "\n",
        "# Export consolidated master benchmark summary JSON before zipping\n",
        "master_summary = {\n",
        "    \"project\": \"BCSS-WSSS Medical WSSS Scientific Research Runner\",\n",
        "    \"device\": str(ENV_INFO.get(\"device\", \"cpu\")),\n",
        "    \"sanity_dry_run\": SANITY_DRY_RUN,\n",
        "    \"val_to_test\": val_to_test_results if \"val_to_test_results\" in locals() else {},\n",
        "    \"pareto\": pareto_data if \"pareto_data\" in locals() else {},\n",
        "    \"biomarkers\": bio_results if \"bio_results\" in locals() else {},\n",
        "    \"luad_generalization\": luad_results if \"luad_results\" in locals() else {},\n",
        "    \"whole_roi\": roi_metrics if \"roi_metrics\" in locals() else {},\n",
        "}\n",
        "summary_json_path = OUTPUT_DIR / \"scientific_benchmark_summary.json\"\n",
        "summary_json_path.parent.mkdir(parents=True, exist_ok=True)\n",
        "with open(summary_json_path, \"w\", encoding=\"utf-8\") as f:\n",
        "    json.dump(sanitize_for_json(master_summary), f, indent=2, default=str)\n",
        "print(f\"[OK] Exported master benchmark summary: {summary_json_path}\")\n",
        "\n",
        "zip_name = \"SE374_Scientific_Research_Results.zip\"\n",
        "zip_out = WORK_ROOT / zip_name\n",
        "if zip_out.is_file():\n",
        "    zip_out.unlink()\n",
        "\n",
        "print(f\"--> Dang dong goi ket qua nghien cuu tu {OUTPUT_DIR} vao file ZIP...\")\n",
        "with zipfile.ZipFile(zip_out, \"w\", zipfile.ZIP_DEFLATED) as zf:\n",
        "    for root, _, files in os.walk(OUTPUT_DIR):\n",
        "        for f in files:\n",
        "            p = Path(root) / f\n",
        "            zf.write(p, arcname=p.relative_to(OUTPUT_DIR))\n",
        "\n",
        "zip_size_mb = zip_out.stat().st_size / (1024 * 1024)\n",
        "print(f\"[OK] Da tao thanh cong goi ket qua: {zip_out.resolve()} ({zip_size_mb:.2f} MB)\")\n",
        "\n",
        "if \"google.colab\" in sys.modules:\n",
        "    try:\n",
        "        from google.colab import files\n",
        "        print(\"--> Dang kich hoat hop thoai tai ve tren trinh duyet cua ban...\")\n",
        "        files.download(str(zip_out))\n",
        "        print(\"[THANH CONG] Trinh duyet dang tai tap tin ket qua ve may tinh cua ban!\")\n",
        "    except Exception as err:\n",
        "        print(f\"[HUONG DAN] Neu trinh duyet chan popup tu dong, ban co the mo thanh quan ly file ben trai Colab va tai file '{zip_name}'.\")\n",
        "else:\n",
        "    print(f\"[OK] Ket qua duoc luu tai thu muc cuc bo: {zip_out.resolve()}\")"
    ]

    # Sanitize and serialize
    serialized = json.dumps(nb_data, ensure_ascii=False, indent=1)
    serialized = sanitize_ascii_hyphens(serialized)

    if nb_path.is_file():
        nb_path.write_text(serialized, encoding="utf-8", newline="\n")
    root_nb_path.write_text(serialized, encoding="utf-8", newline="\n")

    return nb_path if nb_path.is_file() else root_nb_path


def build_drive_package(
    project_root: Optional[Path] = None,
    output_base_dir: Optional[Path] = None,
    sync_repo_files: bool = False,
) -> Dict[str, object]:
    """Assembles `drive_upload_ready/SE374_WSSS_Drive_Code/` and `SE374_WSSS_Drive_Code.zip`.

    Mutating tracked repository files is opt-in (`sync_repo_files=True` / `--sync-repo`) so
    rebuilding `drive_upload_ready/` never overwrites tracked CRLF notebooks with LF bytes.
    """
    root = (project_root or PROJECT_ROOT).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    should_sync_repo = bool(sync_repo_files)
    if should_sync_repo:
        sync_historical_and_standalone_outputs(root)
    out_base = (output_base_dir or (root / "drive_upload_ready")).resolve()
    bundle_dir = out_base / "SE374_WSSS_Drive_Code"
    bundle_src_dir = bundle_dir / "src"
    bundle_nb_dir = bundle_dir / "notebooks"

    out_base.mkdir(parents=True, exist_ok=True)
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_nb_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy clean src/ tree
    src_root = root / "src"
    copied_src_files = copy_clean_src_tree(src_root, bundle_src_dir)

    # 2. Copy requirements.txt
    req_file = root / "requirements.txt"
    if req_file.is_file():
        req_text = sanitize_ascii_hyphens(req_file.read_text(encoding="utf-8"))
        (bundle_dir / "requirements.txt").write_text(req_text, encoding="utf-8", newline="\n")

    # 3. Discover new Thin Runner notebooks in notebooks/ (or synthesize fallback inside bundle)
    notebooks_dir = root / "notebooks"
    thin_notebooks = discover_thin_runner_notebooks(notebooks_dir)
    packaged_notebooks: List[str] = []

    if thin_notebooks:
        for nb_path in thin_notebooks:
            nb_text = sanitize_ascii_hyphens(nb_path.read_text(encoding="utf-8"))
            # Copy into bundle/notebooks/ and bundle root for maximum convenience on Google Drive
            (bundle_nb_dir / nb_path.name).write_text(nb_text, encoding="utf-8", newline="\n")
            (bundle_dir / nb_path.name).write_text(nb_text, encoding="utf-8", newline="\n")
            packaged_notebooks.append(nb_path.name)
    else:
        for var in FALLBACK_THIN_VARIANTS:
            nb_dict = build_fallback_thin_notebook_dict(var)
            nb_json = sanitize_ascii_hyphens(json.dumps(nb_dict, ensure_ascii=False, indent=1) + "\n")
            fname = str(var["filename"])
            (bundle_nb_dir / fname).write_text(nb_json, encoding="utf-8", newline="\n")
            (bundle_dir / fname).write_text(nb_json, encoding="utf-8", newline="\n")
            packaged_notebooks.append(fname)

    # 3b. Copy any present legacy standalone notebooks into bundle/notebooks/legacy_standalone_v1/
    legacy_hashes: Dict[str, str] = {}
    existing_legacy = [notebooks_dir / n for n in sorted(LEGACY_NOTEBOOKS) if (notebooks_dir / n).is_file()]
    if existing_legacy:
        bundle_legacy_dir = bundle_nb_dir / "legacy_standalone_v1"
        bundle_legacy_dir.mkdir(parents=True, exist_ok=True)
        for leg_path in existing_legacy:
            legacy_hashes[leg_path.name] = sha256_file(leg_path)
            leg_compact = _compact_notebook_for_bundle(leg_path.read_text(encoding="utf-8"))
            (bundle_legacy_dir / leg_path.name).write_text(leg_compact, encoding="utf-8", newline="\n")

    # 3c. Copy all 3 historical full-run research notebooks (Run 1 -> Run 3) into bundle/notebooks/history_runs/
    history_src_dir = notebooks_dir / "history_runs"
    bundle_history_dir = bundle_nb_dir / "history_runs"
    packaged_history_notebooks: List[str] = []
    if history_src_dir.is_dir():
        bundle_history_dir.mkdir(parents=True, exist_ok=True)
        for h_file in sorted(history_src_dir.iterdir()):
            if h_file.is_file():
                if h_file.suffix == ".ipynb":
                    h_compact = _compact_notebook_for_bundle(h_file.read_text(encoding="utf-8"))
                    (bundle_history_dir / h_file.name).write_text(h_compact, encoding="utf-8", newline="\n")
                    packaged_history_notebooks.append(h_file.name)
                elif h_file.suffix in (".md", ".json", ".txt"):
                    h_text = sanitize_ascii_hyphens(h_file.read_text(encoding="utf-8"))
                    (bundle_history_dir / h_file.name).write_text(h_text, encoding="utf-8", newline="\n")

    # 3d. Copy step-by-step verification reports & JSON/CSV metrics from experiments/ into bundle/experiments/
    exp_src_dir = root / "experiments"
    bundle_exp_dir = bundle_dir / "experiments"
    packaged_experiment_reports: List[str] = []
    if exp_src_dir.is_dir():
        for exp_file in sorted(exp_src_dir.rglob("*")):
            if exp_file.is_file() and exp_file.suffix in (".md", ".json", ".csv"):
                rel_exp = exp_file.relative_to(exp_src_dir)
                dst_exp = bundle_exp_dir / rel_exp
                dst_exp.parent.mkdir(parents=True, exist_ok=True)
                clean_exp = sanitize_ascii_hyphens(exp_file.read_text(encoding="utf-8"))
                dst_exp.write_text(clean_exp, encoding="utf-8", newline="\n")
                packaged_experiment_reports.append(str((Path("experiments") / rel_exp).as_posix()))

    # 4. Inspect MultiMethodWSSSEngine method keys dynamically (including V2 M6 & M7)
    try:
        from src.models.wsss_inference_methods import MultiMethodWSSSEngine

        method_keys = list(
            getattr(MultiMethodWSSSEngine, "METHOD_KEYS_V2", MultiMethodWSSSEngine.METHOD_KEYS)
        )
    except Exception:
        method_keys = [
            "M1_Baseline_SingleScale",
            "M2_MultiView_D4_TTA",
            "M3_LongTail_Prior_Calibration",
            "M4_Spatial_Bilateral_Refinement",
            "M5_Proposed_SOTA_Ensemble",
            "M6_MultiScale_Pyramid_D4_Ensemble",
            "M7_Proposed_Ultra_SOTA_Curriculum",
        ]

    # 5. Generate README_DRIVE_SETUP.md (inside bundle_dir and out_base)
    readme_text = generate_readme_drive_setup(packaged_notebooks, method_keys)
    readme_bundle_path = bundle_dir / "README_DRIVE_SETUP.md"
    readme_outer_path = out_base / "README_DRIVE_SETUP.md"
    readme_bundle_path.write_text(readme_text, encoding="utf-8", newline="\n")
    readme_outer_path.write_text(readme_text, encoding="utf-8", newline="\n")

    # 6. Write package_manifest.json inside bundle_dir
    manifest = {
        "package_name": "SE374_WSSS_Drive_Code",
        "target_drive_folder": "MyDrive/SE374_WSSS_Project/",
        "src_files_count": len(copied_src_files),
        "src_files": copied_src_files,
        "thin_runner_notebooks": packaged_notebooks,
        "legacy_notebooks_preserved": legacy_hashes,
        "history_runs_notebooks": packaged_history_notebooks,
        "experiment_verification_reports": packaged_experiment_reports,
        "supported_wsss_methods": method_keys,
    }
    manifest_path = bundle_dir / "package_manifest.json"
    manifest_path.write_text(
        sanitize_ascii_hyphens(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"),
        encoding="utf-8",
        newline="\n",
    )

    # 7. Build pre-zipped SE374_WSSS_Drive_Code.zip directly from bundle_dir (never inside bundle_dir)
    zip_outer_path = out_base / "SE374_WSSS_Drive_Code.zip"
    if zip_outer_path.exists():
        zip_outer_path.unlink()
    stale_inner_zip = bundle_dir / "SE374_WSSS_Drive_Code.zip"
    if stale_inner_zip.exists():
        stale_inner_zip.unlink()

    all_bundle_files = sorted([p for p in bundle_dir.rglob("*") if p.is_file() and p.suffix != ".zip"])
    with zipfile.ZipFile(zip_outer_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in all_bundle_files:
            rel_inside_bundle = file_path.relative_to(bundle_dir).as_posix()
            # Write both root-relative (e.g. src/...) and folder-prefixed (SE374_WSSS_Drive_Code/src/...)
            # so extracting into /content/ or /content/SE374_WSSS_Drive_Code/ both work seamlessly
            zf.write(file_path, arcname=rel_inside_bundle)
            zf.write(file_path, arcname=f"SE374_WSSS_Drive_Code/{rel_inside_bundle}")

    return {
        "bundle_dir": str(bundle_dir),
        "zip_outer_path": str(zip_outer_path),
        "zip_inner_path": None,
        "readme_path": str(readme_bundle_path),
        "manifest_path": str(manifest_path),
        "src_files_count": len(copied_src_files),
        "thin_runner_notebooks": packaged_notebooks,
        "history_runs_notebooks": packaged_history_notebooks,
        "experiment_verification_reports": packaged_experiment_reports,
        "supported_wsss_methods": method_keys,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build SE374_WSSS_Drive_Code bundle and ZIP archive.")
    parser.add_argument(
        "--sync-repo",
        action="store_true",
        default=False,
        help="Opt-in: synchronize tracked repository runner notebooks before packaging.",
    )
    cli_args = parser.parse_args()
    summary = build_drive_package(sync_repo_files=cli_args.sync_repo)
    print(f"[OK] Assembled Drive bundle at : {summary['bundle_dir']}")
    print(f"[OK] Pre-zipped archive at     : {summary['zip_outer_path']}")
    print(f"[OK] Packaged src/ files count : {summary['src_files_count']}")
    print(f"[OK] Thin Runner notebooks     : {summary['thin_runner_notebooks']}")
    print(f"[OK] Supported WSSS methods    : {summary['supported_wsss_methods']}")
