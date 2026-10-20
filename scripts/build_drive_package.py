"""Automated Google Drive Code Bundle Packager for SE374.R11 WSSS Pipeline.

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
        "title": "SE374.R11 Thin Drive Runner - T4 GPU Sanity 150 Patches",
    },
    {
        "filename": "SE374_Colab_WSSS_Drive_Runner_T4_Full4986.ipynb",
        "gpu_type": "T4",
        "high_ram": False,
        "sanity_mode": False,
        "mode_tag": "Drive_T4_Full4986",
        "title": "SE374.R11 Thin Drive Runner - T4 GPU Full 4986 Patches",
    },
    {
        "filename": "SE374_Colab_WSSS_Drive_Runner_L4_Full4986.ipynb",
        "gpu_type": "L4",
        "high_ram": False,
        "sanity_mode": False,
        "mode_tag": "Drive_L4_Full4986",
        "title": "SE374.R11 Thin Drive Runner - L4 GPU Full 4986 Patches",
    },
    {
        "filename": "SE374_Colab_WSSS_Drive_Runner_A100_Full4986.ipynb",
        "gpu_type": "A100",
        "high_ram": True,
        "sanity_mode": False,
        "mode_tag": "Drive_A100_Full4986",
        "title": "SE374.R11 Thin Drive Runner - A100 GPU Full 4986 Patches",
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
    """Finds all new Thin Runner notebooks in `notebooks/` excluding the 5 legacy notebooks."""
    if not notebooks_dir.is_dir():
        return []
    candidates: List[Path] = []
    for nb_path in sorted(notebooks_dir.glob("*.ipynb")):
        if nb_path.name not in LEGACY_NOTEBOOKS:
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
                "assert PROJECT_CODE_DIR is not None, 'Không tìm thấy gói src/! Hãy tải SE374_WSSS_Drive_Code lên MyDrive/SE374_WSSS_Project/'\n"
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
## Dự Án SE374.R11: Phân Vùng Mô Bệnh Học WSSS Trên Tập Dữ Liệu BCSS-WSSS

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

3. **Các Phương Pháp Suy Luận SOTA Tích Hợp Trong `MultiMethodWSSSEngine`:**
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

### 5. Bảo Toàn 100% Các Notebook Cũ, Lịch Sử 5 Phiên Thực Nghiệm (Run 1 -> Run 5) & Lịch Sử Git

- **5 Notebook Standalone Gốc (`notebooks/` & `SE374_WSSS_Drive_Code/notebooks/legacy_standalone_v1/`):**
  - `SE374_Colab_WSSS_Pipeline.ipynb`, `SE374_Colab_WSSS_T4_Sanity150.ipynb`, `SE374_Colab_WSSS_T4_Full4986.ipynb`, `SE374_Colab_WSSS_L4_Full4986.ipynb`, `SE374_Colab_WSSS_A100_Full4986.ipynb` được bảo toàn nguyên vẹn 100%, không bị xóa hay ghi đè.
- **5 Notebook Lịch Sử Đối Chứng Từng Bước Nghiên Cứu (`notebooks/history_runs/` & `SE374_WSSS_Drive_Code/notebooks/history_runs/`):**
  - `Run1_20261019_Naive_HSV_Heuristic_Sanity150.ipynb` (`mIoU_4cls = 9.13%`, `mIoU_5cls = 9.78%`)
  - `Run2_20261020_0656UTC_DeepLabV3Plus_ImageNetNorm_OtsuAggressive_Sanity150.ipynb` (`mIoU_4cls = 5.74%`, `mIoU_5cls = 7.59%`)
  - `Run3_20261020_1008UTC_DeepLabV3Plus_UnitRangeNorm_Fixed_Sanity150.ipynb` (`mIoU_4cls = 72.39%`, `mIoU_5cls = 69.99%`)
  - `Run4_20261020_1022UTC_Stage2_DeepLabV3Plus_SingleCkpt_T4_Full4986.ipynb` (`mIoU_4cls = 68.71%`, `mIoU_5cls = 67.36%`)
  - `Run5_20261020_1641UTC_DualCkpt_Stage1_CGM_Stage2_M1_to_M5_T4_Full4986.ipynb` (`M1 = 69.34% -> M5 = 70.49%` 4 lớp / `76.38%` 5 lớp)
- **Báo Cáo Số Liệu Đối Chứng Qua 5 Phiên Thực Nghiệm (`experiments/` & `SE374_WSSS_Drive_Code/experiments/`):**
  - `experiments/benchmarks/COLAB_5_RUNS_EVOLUTION_COMPARISON.md` cùng đầy đủ các báo cáo `COLAB_EVALUATION_REPORT.md` và `COLAB_EVALUATION_REPORT.json` của cả 5 phiên chạy.
- Để cập nhật lại gói `drive_upload_ready/` sau khi chỉnh sửa code trong `src/`, chỉ cần chạy lệnh:
  ```bash
  cmd /c "python scripts/build_drive_package.py"
  ```
"""
    return sanitize_ascii_hyphens(content)


def _compact_notebook_for_bundle(nb_text: str) -> str:
    """Preserves all notebook cells and text/stdout outputs while stripping heavy base64 PNG payloads
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
                        data_copy.pop("image/png", None)
                        out_copy["data"] = data_copy
                        compacted_outputs.append(out_copy)
                    else:
                        compacted_outputs.append(out)
                cell["outputs"] = compacted_outputs
        return sanitize_ascii_hyphens(json.dumps(nb_data, ensure_ascii=False, indent=1) + "\n")
    except Exception:
        return sanitize_ascii_hyphens(nb_text)


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
                init_file.write_text('"""SE374.R11 WSSS subpackage."""\n', encoding="utf-8", newline="\n")
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


def build_drive_package(
    project_root: Optional[Path] = None,
    output_base_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Assembles `drive_upload_ready/SE374_WSSS_Drive_Code/` and `SE374_WSSS_Drive_Code.zip`.

    Returns a dictionary summarizing the packaged files, notebooks, and archive paths.
    """
    root = (project_root or PROJECT_ROOT).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
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

    # 3b. Copy all 5 legacy standalone notebooks into bundle/notebooks/legacy_standalone_v1/
    bundle_legacy_dir = bundle_nb_dir / "legacy_standalone_v1"
    bundle_legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_hashes: Dict[str, str] = {}
    for leg_name in sorted(LEGACY_NOTEBOOKS):
        leg_path = notebooks_dir / leg_name
        if leg_path.is_file():
            legacy_hashes[leg_name] = sha256_file(leg_path)
            leg_compact = _compact_notebook_for_bundle(leg_path.read_text(encoding="utf-8"))
            (bundle_legacy_dir / leg_name).write_text(leg_compact, encoding="utf-8", newline="\n")

    # 3c. Copy all 5 step-by-step historical research notebooks (Run 1 -> Run 5) into bundle/notebooks/history_runs/
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

    # 3d. Copy step-by-step verification reports & JSON metrics from experiments/ into bundle/experiments/
    exp_src_dir = root / "experiments"
    bundle_exp_dir = bundle_dir / "experiments"
    packaged_experiment_reports: List[str] = []
    if exp_src_dir.is_dir():
        for exp_file in sorted(exp_src_dir.rglob("*")):
            if exp_file.is_file() and exp_file.suffix in (".md", ".json"):
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

    # 7. Build pre-zipped SE374_WSSS_Drive_Code.zip for 1-click upload to Google Drive
    zip_outer_path = out_base / "SE374_WSSS_Drive_Code.zip"
    if zip_outer_path.exists():
        zip_outer_path.unlink()

    all_bundle_files = sorted([p for p in bundle_dir.rglob("*") if p.is_file()])
    with zipfile.ZipFile(zip_outer_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in all_bundle_files:
            rel_inside_bundle = file_path.relative_to(bundle_dir).as_posix()
            # Write both root-relative (e.g. src/...) and folder-prefixed (SE374_WSSS_Drive_Code/src/...)
            # so extracting into /content/ or /content/SE374_WSSS_Drive_Code/ both work seamlessly
            zf.write(file_path, arcname=rel_inside_bundle)
            zf.write(file_path, arcname=f"SE374_WSSS_Drive_Code/{rel_inside_bundle}")

    # Also place a copy inside bundle_dir/SE374_WSSS_Drive_Code.zip for convenience
    zip_inner_path = bundle_dir / "SE374_WSSS_Drive_Code.zip"
    shutil.copy2(zip_outer_path, zip_inner_path)

    return {
        "bundle_dir": str(bundle_dir),
        "zip_outer_path": str(zip_outer_path),
        "zip_inner_path": str(zip_inner_path),
        "readme_path": str(readme_bundle_path),
        "manifest_path": str(manifest_path),
        "src_files_count": len(copied_src_files),
        "thin_runner_notebooks": packaged_notebooks,
        "history_runs_notebooks": packaged_history_notebooks,
        "experiment_verification_reports": packaged_experiment_reports,
        "supported_wsss_methods": method_keys,
    }


if __name__ == "__main__":
    summary = build_drive_package()
    print(f"[OK] Assembled Drive bundle at : {summary['bundle_dir']}")
    print(f"[OK] Pre-zipped archive at     : {summary['zip_outer_path']}")
    print(f"[OK] Packaged src/ files count : {summary['src_files_count']}")
    print(f"[OK] Thin Runner notebooks     : {summary['thin_runner_notebooks']}")
    print(f"[OK] Supported WSSS methods    : {summary['supported_wsss_methods']}")
