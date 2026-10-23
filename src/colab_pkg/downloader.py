"""
Dataset & Checkpoint Downloader Module for SE374.R11 BCSS-WSSS Pipeline.

Provides resilient multi-tier downloading (HuggingFace CDN primary mirror with
Google Drive gdown fallback and curl retry) and strict Patient-Level Anti-Leakage
sampling (Invariant 08).

All hyphens strictly use standard ASCII 0x2D ('-').
"""

from collections import defaultdict
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.request
import zipfile


# Official BCSS-WSSS Dataset Mirrors (Han et al., MedIA 2022)
GDRIVE_DATASET_ID: str = "1sCK8Cqsf4yCHzzIx98vitocES2L_tkjn"
HF_DATASET_URL: str = (
    "https://huggingface.co/datasets/nttduc/wsss-pathology/resolve/main/BCSS-WSSS.zip"
)

# Official Stage-1 ResNet38d Net_CAM Checkpoint Mirrors (Classification Gate Mechanism)
GDRIVE_STAGE1_ID: str = "118y3Q7i6lPOrw8OIFqt3tlP1PUM-i1Xc"
GDRIVE_STAGE1_ARML_ID: str = "1HiAI6yISiXZNokYgM2U5goic-I7GYG4H"
HF_STAGE1_URL: str = (
    "https://huggingface.co/longcris04/pseduo_mask_wsss/resolve/main/"
    "stage1_checkpoint_trained_on_bcss.pth"
)

# Official Stage-2 DeepLabV3+ ResNet-101 Checkpoint Mirrors (Semantic Segmentation)
GDRIVE_STAGE2_ID: str = "1cXm29ko8UevbaeW6YreSg-vuNaQiN_je"
GDRIVE_CKPT_ID: str = GDRIVE_STAGE2_ID
HF_STAGE2_URL: str = (
    "https://huggingface.co/longcris04/pseduo_mask_wsss/resolve/main/"
    "stage2_checkpoint_trained_on_bcss.pth"
)

# Official LUAD-HistoSeg Dataset Mirrors (Lung Adenocarcinoma WSSS Benchmark)
GDRIVE_LUAD_DATASET_ID: str = "1WyWfv-JHjLwkHOiN8blINhBOkKi--iA1"
HF_LUAD_DATASET_URL: str = (
    "https://huggingface.co/datasets/nttduc/wsss-pathology/resolve/main/LUAD-HistoSeg.zip"
)

# Official LUAD-HistoSeg Stage-1 and Stage-2 Checkpoint Mirrors
GDRIVE_LUAD_STAGE1_ID: str = "1deVn06uUGHBHCWxurn4b005Mr7FgUp8o"
HF_LUAD_STAGE1_URL: str = (
    "https://huggingface.co/longcris04/pseduo_mask_wsss/resolve/main/"
    "stage1_checkpoint_trained_on_luad.pth"
)
GDRIVE_LUAD_STAGE2_ID: str = "1HX-IrkKcaIjcOpe8fk9IMmjgaf4Ywk3D"
HF_LUAD_STAGE2_URL: str = (
    "https://huggingface.co/longcris04/pseduo_mask_wsss/resolve/main/"
    "stage2_checkpoint_trained_on_luad.pth"
)

# BCSS Whole-ROI Dataset Mirror (Nabil et al., HuggingFace nabil-m/bcss)
HF_BCSS_WHOLE_ROI_REPO: str = "nabil-m/bcss"

# Default Patient-Level Sanity Sampling Configuration (Invariant 08)
TARGET_PATIENTS: List[str] = ["TCGA-AQ-A04H", "TCGA-C8-A1HJ", "TCGA-AR-A0TQ"]
DEFAULT_MAX_SANITY_SAMPLES: int = 150
DEFAULT_MIN_SANITY_PATIENTS: int = 5


def resolve_split_dirs(
    base_search_dir: Union[str, Path] = Path("/content"),
    split: str = "test",
) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Searches for dataset split/img and split/mask directories under base_search_dir.

    Returns:
        Tuple of (root_dir, img_dir, mask_dir), or (None, None, None) if not found.
    """
    base_path = Path(base_search_dir)
    if not base_path.exists():
        return None, None, None

    # Direct check if base_search_dir itself contains split/img
    direct_img = base_path / split / "img"
    if direct_img.is_dir():
        return base_path, direct_img, base_path / split / "mask"

    candidates = sorted(base_path.glob(f"**/{split}/img"))
    if candidates:
        img_dir = candidates[0]
        mask_dir = img_dir.parent / "mask"
        root_dir = img_dir.parent.parent
        return root_dir, img_dir, mask_dir
    return None, None, None


def resolve_test_dirs(
    base_search_dir: Union[str, Path] = Path("/content"),
) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Searches for BCSS-WSSS test/img and test/mask directories under base_search_dir."""
    return resolve_split_dirs(base_search_dir=base_search_dir, split="test")


def is_valid_zip_archive(
    zip_path: Union[str, Path],
    min_valid_bytes: int = 1024,
) -> bool:
    """Verifies that a file exists, meets minimum byte size, and has a valid non-empty ZIP central directory."""
    zip_p = Path(zip_path)
    if not zip_p.is_file():
        return False
    try:
        if zip_p.stat().st_size < min_valid_bytes:
            return False
        if not zipfile.is_zipfile(zip_p):
            return False
        with zipfile.ZipFile(zip_p, "r") as zf:
            return len(zf.namelist()) > 0
    except Exception:
        return False


def is_valid_torch_checkpoint(
    ckpt_path: Union[str, Path],
    min_valid_bytes: int = 1024,
) -> bool:
    """Verifies that a PyTorch .pth checkpoint file exists, meets minimum byte size, and has valid archive/pickle structure."""
    ckpt_p = Path(ckpt_path)
    if not ckpt_p.is_file():
        return False
    try:
        if ckpt_p.stat().st_size < min_valid_bytes:
            return False
        if zipfile.is_zipfile(ckpt_p):
            with zipfile.ZipFile(ckpt_p, "r") as zf:
                return len(zf.namelist()) > 0
        import torch

        try:
            obj = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(ckpt_p, map_location="cpu")
        return isinstance(obj, dict) and len(obj) > 0
    except Exception:
        return False


def download_dataset_resilient(
    zip_path: Union[str, Path] = Path("/content/BCSS-WSSS.zip"),
    hf_url: str = HF_DATASET_URL,
    gdrive_id: str = GDRIVE_DATASET_ID,
    min_valid_bytes: int = 100 * 1024 * 1024,
) -> Path:
    """Downloads BCSS-WSSS.zip using HuggingFace CDN primary mirror with gdown and curl fallbacks."""
    zip_p = Path(zip_path)
    zip_p.parent.mkdir(parents=True, exist_ok=True)

    if zip_p.exists() and not is_valid_zip_archive(zip_p, min_valid_bytes=min_valid_bytes):
        print(f"[CẢNH BÁO] Phát hiện tệp nén lỗi hoặc chưa hoàn chỉnh tại {zip_p}. Đang xóa để tải lại...")
        zip_p.unlink()
    if zip_p.exists():
        return zip_p

    print(
        "--> Đang tải BCSS-WSSS.zip (2.96 GB) qua cổng HuggingFace CDN tốc độ cao "
        "(chống lỗi Google Drive Quota)..."
    )
    try:
        last_pct = [-1]

        def _progress(block_num: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                pct = int(block_num * block_size * 100 / total_size)
                if pct >= last_pct[0] + 10 and pct <= 100:
                    last_pct[0] = (pct // 10) * 10
                    mb_done = (block_num * block_size) / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(
                        f"    + Tiến độ tải BCSS-WSSS.zip: {last_pct[0]}% "
                        f"({mb_done:.0f}/{mb_total:.0f} MB)"
                    )

        urllib.request.urlretrieve(hf_url, str(zip_p), reporthook=_progress)
        if is_valid_zip_archive(zip_p, min_valid_bytes=min_valid_bytes):
            print("[OK] Đã tải thành công BCSS-WSSS.zip từ HuggingFace CDN Mirror!")
            return zip_p
        if zip_p.exists():
            zip_p.unlink()
    except Exception as exc_hf:
        print(f"[THÔNG BÁO] Chuyển sang nhánh dự phòng Google Drive ({exc_hf})...")

    try:
        import gdown

        gdown.download(id=gdrive_id, output=str(zip_p), quiet=False, fuzzy=True)
        if is_valid_zip_archive(zip_p, min_valid_bytes=min_valid_bytes):
            return zip_p
        if zip_p.exists():
            zip_p.unlink()
    except Exception as exc_gd:
        print(
            f"[CẢNH BÁO] Google Drive giới hạn lượt tải ({exc_gd}). "
            "Đang tải lại qua curl CDN..."
        )
        subprocess.run(
            ["curl", "-L", "--retry", "3", "-o", str(zip_p), hf_url],
            check=False,
        )

    return zip_p


def ensure_bcss_dataset(
    data_root: Union[str, Path] = Path("/content/BCSS-WSSS"),
    zip_path: Union[str, Path] = Path("/content/BCSS-WSSS.zip"),
    extract_dir: Union[str, Path] = Path("/content"),
    drive_cache_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path, Path]:
    """Ensures the BCSS-WSSS dataset is downloaded and extracted, returning (root_dir, test_img_dir, test_mask_dir)."""
    data_root_p = Path(data_root)
    extract_p = Path(extract_dir)
    zip_p = Path(zip_path)

    # 1. Check direct data_root first, then extract_dir
    root_found, test_img_dir, test_mask_dir = resolve_test_dirs(data_root_p)
    if root_found is None:
        root_found, test_img_dir, test_mask_dir = resolve_test_dirs(extract_p)

    # 2. Check optional Google Drive cache directory for pre-existing zip or folder
    if root_found is None and drive_cache_dir is not None:
        drive_cache_p = Path(drive_cache_dir)
        if drive_cache_p.exists():
            root_found, test_img_dir, test_mask_dir = resolve_test_dirs(drive_cache_p)
            if root_found is None:
                cached_zip = drive_cache_p / "BCSS-WSSS.zip"
                if is_valid_zip_archive(cached_zip, min_valid_bytes=100 * 1024 * 1024):
                    print(f"--> Phát hiện bản lưu BCSS-WSSS.zip trên Google Drive: {cached_zip}")
                    zip_p = cached_zip

    # 3. Download and extract if still not found
    if root_found is None:
        download_dataset_resilient(zip_p, HF_DATASET_URL, GDRIVE_DATASET_ID)
        if not zip_p.is_file():
            raise FileNotFoundError(f"Không thể tải tệp nén dữ liệu tại: {zip_p}")
        print(f"--> Đang giải nén bộ dữ liệu BCSS-WSSS vào {extract_p}...")
        extract_p.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_p, "r") as zf:
                zf.extractall(str(extract_p))
        except zipfile.BadZipFile:
            print(f"[CẢNH BÁO] Tệp nén {zip_p} bị hỏng khi giải nén. Đang xóa và tải lại...")
            zip_p.unlink(missing_ok=True)
            download_dataset_resilient(zip_p, HF_DATASET_URL, GDRIVE_DATASET_ID)
            with zipfile.ZipFile(zip_p, "r") as zf:
                zf.extractall(str(extract_p))
        root_found, test_img_dir, test_mask_dir = resolve_test_dirs(data_root_p)
        if root_found is None:
            root_found, test_img_dir, test_mask_dir = resolve_test_dirs(extract_p)

    if root_found is None or test_img_dir is None or test_mask_dir is None:
        raise FileNotFoundError("Không tìm thấy thư mục test/img sau khi giải nén!")

    total_imgs = len(list(test_img_dir.glob("*.png")))
    total_masks = len(list(test_mask_dir.glob("*.png"))) if test_mask_dir.exists() else 0
    print(f"[OK] Thư mục gốc dữ liệu: {root_found.resolve()}")
    print(f"[OK] Thư mục ảnh test   : {test_img_dir.resolve()}")
    print(f"[OK] Thư mục mask test  : {test_mask_dir.resolve()}")
    print(f"[OK] Tổng số ảnh test tìm thấy: {total_imgs} ảnh | Tổng số mask: {total_masks} mask")

    return root_found, test_img_dir, test_mask_dir


def download_checkpoint_resilient(
    target_path: Union[str, Path],
    hf_url: str,
    gdrive_ids: List[str],
    desc: str = "Pretrained Checkpoint",
    min_valid_bytes: int = 50 * 1024 * 1024,
) -> Path:
    """Downloads a model checkpoint with HuggingFace CDN primary mirror and Google Drive multi-ID fallback."""
    target_p = Path(target_path)
    target_p.parent.mkdir(parents=True, exist_ok=True)

    if target_p.exists() and not is_valid_torch_checkpoint(target_p, min_valid_bytes=min_valid_bytes):
        print(f"[CẢNH BÁO] Phát hiện checkpoint lỗi hoặc chưa hoàn chỉnh tại {target_p}. Đang xóa để tải lại...")
        target_p.unlink()
    if target_p.exists():
        return target_p

    print(
        f"--> Đang tải {desc} qua cổng HuggingFace CDN tốc độ cao "
        "(chống lỗi Google Drive Quota)..."
    )
    try:
        urllib.request.urlretrieve(hf_url, str(target_p))
        if is_valid_torch_checkpoint(target_p, min_valid_bytes=min_valid_bytes):
            print(f"[OK] Đã tải thành công {target_p.name} từ HuggingFace CDN Mirror!")
            return target_p
        if target_p.exists():
            target_p.unlink()
    except Exception as exc_hf:
        print(
            f"[THÔNG BÁO] Chuyển sang nhánh dự phòng Google Drive cho "
            f"{target_p.name} ({exc_hf})..."
        )

    for gid in gdrive_ids:
        try:
            import gdown

            gdown.download(id=gid, output=str(target_p), quiet=False, fuzzy=True)
            if is_valid_torch_checkpoint(target_p, min_valid_bytes=min_valid_bytes):
                return target_p
            if target_p.exists():
                target_p.unlink()
        except Exception as exc_gd:
            print(
                f"[THÔNG BÁO] Mirror Google Drive {gid} đang giới hạn lượt tải ({exc_gd}). "
                "Chuyển sang mirror kế tiếp..."
            )

    subprocess.run(
        ["curl", "-L", "--retry", "3", "-o", str(target_p), hf_url],
        check=False,
    )
    return target_p


def ensure_checkpoints(
    checkpoint_dir: Union[str, Path] = Path("/content/checkpoints"),
    drive_cache_dir: Optional[Union[str, Path]] = None,
    require_stage2: bool = True,
    min_valid_bytes: int = 1024,
) -> Dict[str, Optional[Path]]:
    """Ensures both Stage-1 (ResNet38d Net_CAM) and Stage-2 (DeepLabV3+) checkpoints are available."""
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    stage1_file = ckpt_dir / "stage1_checkpoint_trained_on_bcss.pth"
    stage2_file = ckpt_dir / "stage2_checkpoint_trained_on_bcss.pth"

    # Optional copy from Google Drive cache if available
    if drive_cache_dir is not None:
        drive_p = Path(drive_cache_dir)
        for fname, dst in [
            ("stage1_checkpoint_trained_on_bcss.pth", stage1_file),
            ("stage2_checkpoint_trained_on_bcss.pth", stage2_file),
        ]:
            cached_candidate = drive_p / fname
            if not is_valid_torch_checkpoint(dst, min_valid_bytes=min_valid_bytes) and is_valid_torch_checkpoint(
                cached_candidate, min_valid_bytes=50 * 1024 * 1024
            ):
                shutil.copy2(cached_candidate, dst)

    if not is_valid_torch_checkpoint(stage1_file, min_valid_bytes=min_valid_bytes):
        download_checkpoint_resilient(
            stage1_file,
            HF_STAGE1_URL,
            [GDRIVE_STAGE1_ARML_ID, GDRIVE_STAGE1_ID],
            "Pretrained Checkpoint Stage 1 ResNet38d Net_CAM",
        )
    if stage1_file.is_file():
        sz1_mb = stage1_file.stat().st_size / (1024 * 1024)
        print(f"[OK] Checkpoint Stage 1 đã sẵn sàng tại: {stage1_file.resolve()} ({sz1_mb:.2f} MB)")
    else:
        print(
            f"[CẢNH BÁO] Không tìm thấy Stage 1 Checkpoint tại {stage1_file}. "
            "Pipeline sẽ hoạt động với chế độ fallback an toàn."
        )

    if not is_valid_torch_checkpoint(stage2_file, min_valid_bytes=min_valid_bytes):
        download_checkpoint_resilient(
            stage2_file,
            HF_STAGE2_URL,
            [GDRIVE_STAGE2_ID],
            "Pretrained Checkpoint SOTA Stage 2 DeepLabV3+",
        )
    if stage2_file.is_file():
        sz2_mb = stage2_file.stat().st_size / (1024 * 1024)
        print(f"[OK] Checkpoint Stage 2 đã sẵn sàng tại: {stage2_file.resolve()} ({sz2_mb:.2f} MB)")
    elif require_stage2:
        raise FileNotFoundError(
            f"Không tìm thấy tệp checkpoint Stage 2 tại: {stage2_file.resolve()}"
        )

    return {
        "stage1": stage1_file if stage1_file.is_file() else None,
        "stage2": stage2_file if stage2_file.is_file() else None,
    }


def ensure_luad_dataset(
    data_root: Union[str, Path] = Path("/content/LUAD-HistoSeg"),
    zip_path: Union[str, Path] = Path("/content/LUAD-HistoSeg.zip"),
    extract_dir: Union[str, Path] = Path("/content"),
    drive_cache_dir: Optional[Union[str, Path]] = None,
    split: str = "test",
) -> Tuple[Path, Path, Path]:
    """Ensures the LUAD-HistoSeg dataset is downloaded and extracted, returning (root_dir, img_dir, mask_dir)."""
    data_root_p = Path(data_root)
    extract_p = Path(extract_dir)
    zip_p = Path(zip_path)

    root_found, img_dir, mask_dir = resolve_split_dirs(data_root_p, split=split)
    if root_found is None:
        root_found, img_dir, mask_dir = resolve_split_dirs(extract_p, split=split)

    if root_found is None and drive_cache_dir is not None:
        drive_cache_p = Path(drive_cache_dir)
        if drive_cache_p.exists():
            root_found, img_dir, mask_dir = resolve_split_dirs(drive_cache_p, split=split)
            if root_found is None:
                cached_zip = drive_cache_p / "LUAD-HistoSeg.zip"
                if is_valid_zip_archive(cached_zip, min_valid_bytes=50 * 1024 * 1024):
                    zip_p = cached_zip

    if root_found is None:
        download_dataset_resilient(
            zip_path=zip_p,
            hf_url=HF_LUAD_DATASET_URL,
            gdrive_id=GDRIVE_LUAD_DATASET_ID,
            min_valid_bytes=50 * 1024 * 1024,
        )
        if not zip_p.is_file():
            raise FileNotFoundError(f"Không thể tải tệp nén LUAD-HistoSeg tại: {zip_p}")
        print(f"--> Đang giải nén bộ dữ liệu LUAD-HistoSeg vào {extract_p}...")
        extract_p.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_p, "r") as zf:
                zf.extractall(str(extract_p))
        except zipfile.BadZipFile:
            print(f"[CẢNH BÁO] Tệp nén {zip_p} bị hỏng khi giải nén. Đang xóa và tải lại...")
            zip_p.unlink(missing_ok=True)
            download_dataset_resilient(
                zip_path=zip_p,
                hf_url=HF_LUAD_DATASET_URL,
                gdrive_id=GDRIVE_LUAD_DATASET_ID,
                min_valid_bytes=50 * 1024 * 1024,
            )
            with zipfile.ZipFile(zip_p, "r") as zf:
                zf.extractall(str(extract_p))
        root_found, img_dir, mask_dir = resolve_split_dirs(data_root_p, split=split)
        if root_found is None:
            root_found, img_dir, mask_dir = resolve_split_dirs(extract_p, split=split)

    if root_found is None or img_dir is None or mask_dir is None:
        raise FileNotFoundError(f"Không tìm thấy thư mục {split}/img sau khi giải nén LUAD-HistoSeg!")

    total_imgs = len(list(img_dir.glob("*.png")))
    total_masks = len(list(mask_dir.glob("*.png"))) if mask_dir.exists() else 0
    print(f"[OK] Thư mục gốc LUAD-HistoSeg: {root_found.resolve()}")
    print(f"[OK] Thư mục ảnh ({split}): {img_dir.resolve()} ({total_imgs} ảnh)")
    print(f"[OK] Thư mục mask ({split}): {mask_dir.resolve()} ({total_masks} mask)")

    return root_found, img_dir, mask_dir


def ensure_luad_checkpoints(
    checkpoint_dir: Union[str, Path] = Path("/content/checkpoints"),
    drive_cache_dir: Optional[Union[str, Path]] = None,
    require_stage2: bool = True,
    min_valid_bytes: int = 1024,
) -> Dict[str, Optional[Path]]:
    """Ensures both Stage-1 and Stage-2 checkpoints trained on LUAD-HistoSeg are available."""
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    stage1_file = ckpt_dir / "stage1_checkpoint_trained_on_luad.pth"
    stage2_file = ckpt_dir / "stage2_checkpoint_trained_on_luad.pth"

    if drive_cache_dir is not None:
        drive_p = Path(drive_cache_dir)
        for fname, dst in [
            ("stage1_checkpoint_trained_on_luad.pth", stage1_file),
            ("stage2_checkpoint_trained_on_luad.pth", stage2_file),
        ]:
            cached_candidate = drive_p / fname
            if not is_valid_torch_checkpoint(dst, min_valid_bytes=min_valid_bytes) and is_valid_torch_checkpoint(
                cached_candidate, min_valid_bytes=50 * 1024 * 1024
            ):
                shutil.copy2(cached_candidate, dst)

    if not is_valid_torch_checkpoint(stage1_file, min_valid_bytes=min_valid_bytes):
        download_checkpoint_resilient(
            stage1_file,
            HF_LUAD_STAGE1_URL,
            [GDRIVE_LUAD_STAGE1_ID],
            "Pretrained Checkpoint Stage 1 ResNet38d Net_CAM (LUAD)",
        )

    if not is_valid_torch_checkpoint(stage2_file, min_valid_bytes=min_valid_bytes):
        download_checkpoint_resilient(
            stage2_file,
            HF_LUAD_STAGE2_URL,
            [GDRIVE_LUAD_STAGE2_ID],
            "Pretrained Checkpoint Stage 2 DeepLabV3+ (LUAD)",
        )

    if stage2_file.is_file():
        sz2_mb = stage2_file.stat().st_size / (1024 * 1024)
        print(f"[OK] Checkpoint Stage 2 LUAD đã sẵn sàng tại: {stage2_file.resolve()} ({sz2_mb:.2f} MB)")
    elif require_stage2:
        raise FileNotFoundError(
            f"Không tìm thấy tệp checkpoint Stage 2 LUAD tại: {stage2_file.resolve()}"
        )

    return {
        "stage1": stage1_file if stage1_file.is_file() else None,
        "stage2": stage2_file if stage2_file.is_file() else None,
    }


def download_bcss_whole_roi_cohort(
    target_dir: Union[str, Path] = Path("/content/bcss_whole_roi"),
    repo_id: str = HF_BCSS_WHOLE_ROI_REPO,
    allow_mock: bool = False,
) -> Path:
    """Downloads or verifies the BCSS Whole-ROI cohort (nabil-m/bcss) with ground truth masks and metadata."""
    t_dir = Path(target_dir)
    t_dir.mkdir(parents=True, exist_ok=True)

    mask_dir = t_dir / "masks"
    rgb_dir = t_dir / "rgbs_colorNormalized"

    if mask_dir.is_dir() and rgb_dir.is_dir() and len(list(mask_dir.glob("*.png"))) > 0:
        print(f"[OK] Đã tìm thấy BCSS Whole-ROI dataset tại: {t_dir.resolve()}")
        return t_dir

    print(f"--> Đang tải BCSS Whole-ROI dataset từ HuggingFace repository: {repo_id}...")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(t_dir),
            ignore_patterns=["*.git*", "README.md"],
        )
        if not mask_dir.is_dir() or len(list(mask_dir.glob("*.png"))) == 0:
            raise FileNotFoundError(f"Tải thành công nhưng không tìm thấy tệp mặt nạ trong {mask_dir}")
        print(f"[OK] Tải thành công BCSS Whole-ROI dataset vào: {t_dir.resolve()}")
    except Exception as exc:
        if allow_mock:
            print(f"[THÔNG BÁO] Chế độ mock cho phép tiếp tục ({exc}).")
            mask_dir.mkdir(parents=True, exist_ok=True)
            rgb_dir.mkdir(parents=True, exist_ok=True)
            return t_dir
        raise RuntimeError(
            f"Không thể tải BCSS Whole-ROI cohort từ HuggingFace '{repo_id}': {exc}. "
            "Vui lòng cài đặt huggingface_hub và kiểm tra kết nối mạng."
        ) from exc

    return t_dir


def extract_patient_id(filename: str) -> str:
    """Extracts the TCGA patient identifier (TCGA-XX-XXXX) from a patch filename (Invariant 08)."""
    match = re.match(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return filename.split("_")[0]


def select_patient_files(
    test_img_dir: Union[str, Path],
    sanity_mode: bool = False,
    max_sanity_samples: int = DEFAULT_MAX_SANITY_SAMPLES,
    min_sanity_patients: int = DEFAULT_MIN_SANITY_PATIENTS,
    target_patients: Optional[List[str]] = None,
) -> Tuple[List[Path], Dict[str, List[Path]]]:
    """Selects evaluation patch files grouped strictly by Patient ID (Invariant 08).

    Args:
        test_img_dir: Directory containing test PNG patches.
        sanity_mode: If True, samples max_sanity_samples balanced across >= min_sanity_patients.
            If False, selects all 4,986 test patches.
        max_sanity_samples: Quota of patches in Sanity Mode (default 150).
        min_sanity_patients: Minimum distinct patients in Sanity Mode (default 5).
        target_patients: Priority test patient IDs.

    Returns:
        Tuple of (active_test_files, patient_to_files).
    """
    img_dir = Path(test_img_dir)
    target_pids = target_patients if target_patients is not None else list(TARGET_PATIENTS)

    all_test_files = sorted(list(img_dir.glob("*.png")))
    patient_to_files: Dict[str, List[Path]] = defaultdict(list)
    for p in all_test_files:
        pid = extract_patient_id(p.name)
        patient_to_files[pid].append(p)

    print(f"--> Tổng số mảng ảnh test thực tế: {len(all_test_files)} ảnh")
    print(f"--> Tổng số bệnh nhân trong tập test: {len(patient_to_files)} bệnh nhân")

    if sanity_mode:
        active_test_files: List[Path] = []
        num_target_groups = max(len(target_pids), min_sanity_patients)
        per_patient_quota = max(1, max_sanity_samples // num_target_groups)

        for pid in target_pids:
            if pid in patient_to_files:
                active_test_files.extend(patient_to_files[pid][:per_patient_quota])

        if len(active_test_files) < max_sanity_samples:
            for pid, pfiles in patient_to_files.items():
                if pid not in target_pids:
                    needed = min(per_patient_quota, max_sanity_samples - len(active_test_files))
                    active_test_files.extend(pfiles[:needed])
                    if len(active_test_files) >= max_sanity_samples:
                        break

        if len(active_test_files) < max_sanity_samples:
            selected_set = set(active_test_files)
            for p in all_test_files:
                if p not in selected_set:
                    active_test_files.append(p)
                    if len(active_test_files) >= max_sanity_samples:
                        break

        active_test_files = active_test_files[:max_sanity_samples]
        distinct_pids = len(set(extract_patient_id(p.name) for p in active_test_files))
        print(
            f"\n[CHẾ ĐỘ T4 FREE SANITY]: Đã chọn {len(active_test_files)} mảng ảnh thật "
            f"từ {distinct_pids} bệnh nhân test:"
        )
        sampled_patients: Dict[str, int] = defaultdict(int)
        for p in active_test_files:
            sampled_patients[extract_patient_id(p.name)] += 1
        for pid, count in sampled_patients.items():
            print(f"  * Bệnh nhân {pid}: {count} mảng ảnh")
    else:
        active_test_files = all_test_files
        print(
            f"\n[CHẾ ĐỘ FULL EVALUATION]: Chạy toàn bộ {len(active_test_files)} mảng ảnh thật "
            f"trên toàn bộ {len(patient_to_files)} bệnh nhân!"
        )

    return active_test_files, dict(patient_to_files)


def prepare_dataset_and_splits(
    data_root: Union[str, Path] = Path("/content/BCSS-WSSS"),
    zip_path: Union[str, Path] = Path("/content/BCSS-WSSS.zip"),
    extract_dir: Union[str, Path] = Path("/content"),
    sanity_mode: bool = False,
    max_sanity_samples: int = DEFAULT_MAX_SANITY_SAMPLES,
    min_sanity_patients: int = DEFAULT_MIN_SANITY_PATIENTS,
    target_patients: Optional[List[str]] = None,
    drive_cache_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """High-level step function that downloads/verifies dataset and selects patient files."""
    root_dir, test_img_dir, test_mask_dir = ensure_bcss_dataset(
        data_root=data_root,
        zip_path=zip_path,
        extract_dir=extract_dir,
        drive_cache_dir=drive_cache_dir,
    )
    active_test_files, patient_to_files = select_patient_files(
        test_img_dir=test_img_dir,
        sanity_mode=sanity_mode,
        max_sanity_samples=max_sanity_samples,
        min_sanity_patients=min_sanity_patients,
        target_patients=target_patients,
    )
    return {
        "data_root": root_dir,
        "test_img_dir": test_img_dir,
        "test_mask_dir": test_mask_dir,
        "active_test_files": active_test_files,
        "patient_to_files": patient_to_files,
    }


class ColabArtifactDownloader:
    """Compatibility helper for Colab notebooks downloading model checkpoints and assets."""

    def __init__(self, checkpoint_dir: Optional[Union[str, Path]] = None) -> None:
        self.checkpoint_dir = Path(checkpoint_dir or "checkpoints")

    def download_checkpoints(
        self,
        drive_cache_dir: Optional[Union[str, Path]] = None,
        require_stage2: bool = True,
    ) -> Tuple[str, str]:
        """Downloads/verifies stage 1 and stage 2 checkpoints.

        Returns:
            Tuple of (ckpt_stage1_path, ckpt_stage2_path).
        """
        paths = ensure_checkpoints(
            checkpoint_dir=self.checkpoint_dir,
            drive_cache_dir=drive_cache_dir,
            require_stage2=require_stage2,
        )
        return str(paths["stage1"]), str(paths["stage2"])

    def download_luad_checkpoints(
        self,
        drive_cache_dir: Optional[Union[str, Path]] = None,
        require_stage2: bool = True,
    ) -> Tuple[str, str]:
        """Downloads/verifies LUAD stage 1 and stage 2 checkpoints.

        Returns:
            Tuple of (ckpt_stage1_path, ckpt_stage2_path).
        """
        paths = ensure_luad_checkpoints(
            checkpoint_dir=self.checkpoint_dir,
            drive_cache_dir=drive_cache_dir,
            require_stage2=require_stage2,
        )
        return str(paths["stage1"]), str(paths["stage2"])

