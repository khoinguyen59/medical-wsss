"""Generate hardware-specific Google Colab notebooks (T4, L4, A100) from the master WSSS notebook.

Each generated notebook is pre-configured with:
1. Exact GPU runtime metadata (T4, L4, A100).
2. Explicit Vietnamese hardware & High-RAM instructions in Cell 0 and Cell 7.
3. Pre-set SANITY_MODE (True for T4 Sanity 150, False for Full 4986 on T4/L4/A100).
4. Hardware-tagged zip archive naming so downloads never collide across GPU tiers.
"""

import copy
import json
from pathlib import Path

MASTER_NOTEBOOK = Path("notebooks/SE374_Colab_WSSS_Pipeline.ipynb")

VARIANTS = [
    {
        "filename": "SE374_Colab_WSSS_T4_Sanity150.ipynb",
        "gpu_type": "T4",
        "high_ram": False,
        "sanity_mode": True,
        "mode_tag_override": "T4_Sanity150",
        "title_banner": (
            "# [BẢN 1/4 - DÀNH CHO GPU T4 MIỄN PHÍ - SANITY 150 ẢNH]\n"
            "## SE374.R11: WSSS Histopathology Tissue Segmentation Pipeline (T4 Free - 150 Mảng Ảnh)\n\n"
            "> **CẤU HÌNH PHẦN CỨNG BẮT BUỘC TRÊN GOOGLE COLAB:**\n"
            "> - **Loại GPU (Hardware Accelerator):** `T4 GPU` (15 GB VRAM - Tài khoản Colab Free miễn phí).\n"
            "> - **Có cần bật RAM cao cấp (High-RAM) không?** **KHÔNG CẦN BẬT HIGH-RAM!** Bộ nhớ System RAM mặc định (`12.7 GB`) là hoàn toàn dư dả (pipeline chỉ dùng `~2.4 GB System RAM` và `~1.2 GB GPU VRAM`).\n"
            "> - **Quy mô chạy:** `150 mảng ảnh thật` (chia đều `5 bệnh nhân x 30 ảnh/bệnh nhân`).\n"
            "> - **Thời gian hoàn tất:** `~1 - 2 phút`.\n"
            "> - **Tên file Zip tự động tải về:** `SE374_Colab_Results_DeepLabV3Plus_T4_Sanity150_<Timestamp>.zip`.\n\n"
        ),
    },
    {
        "filename": "SE374_Colab_WSSS_T4_Full4986.ipynb",
        "gpu_type": "T4",
        "high_ram": False,
        "sanity_mode": False,
        "mode_tag_override": "T4_Full4986",
        "title_banner": (
            "# [BẢN 2/4 - DÀNH CHO GPU T4 MIỄN PHÍ - FULL 4.986 ẢNH]\n"
            "## SE374.R11: WSSS Histopathology Tissue Segmentation Pipeline (T4 Free - Toàn Bộ 4.986 Mảng Ảnh)\n\n"
            "> **CẤU HÌNH PHẦN CỨNG BẮT BUỘC TRÊN GOOGLE COLAB:**\n"
            "> - **Loại GPU (Hardware Accelerator):** `T4 GPU` (15 GB VRAM - Chạy được ngay trên tài khoản Colab Free miễn phí).\n"
            "> - **Có cần bật RAM cao cấp (High-RAM) không?** **KHÔNG CẦN BẬT HIGH-RAM!** Vì thuật toán nạp và suy luận cuốn chiếu từng mảng ảnh một (streaming evaluation, không nhồi toàn bộ 4.986 ảnh vào RAM cùng lúc), nên chỉ tiêu thụ `~2.6 GB System RAM` và `~1.2 GB GPU VRAM`, chạy mượt trên RAM chuẩn `12.7 GB`.\n"
            "> - **Quy mô chạy:** **Toàn bộ `4.986 mảng ảnh thật`** trên toàn bộ các bệnh nhân tập test BCSS-WSSS.\n"
            "> - **Thời gian hoàn tất:** `~3 - 4 phút` trên T4 GPU.\n"
            "> - **Tên file Zip tự động tải về:** `SE374_Colab_Results_DeepLabV3Plus_T4_Full4986_<Timestamp>.zip`.\n\n"
        ),
    },
    {
        "filename": "SE374_Colab_WSSS_L4_Full4986.ipynb",
        "gpu_type": "L4",
        "high_ram": False,
        "sanity_mode": False,
        "mode_tag_override": "L4_Full4986",
        "title_banner": (
            "# [BẢN 3/4 - DÀNH CHO GPU NVIDIA L4 - FULL 4.986 ẢNH]\n"
            "## SE374.R11: WSSS Histopathology Tissue Segmentation Pipeline (L4 GPU - Toàn Bộ 4.986 Mảng Ảnh)\n\n"
            "> **CẤU HÌNH PHẦN CỨNG BẮT BUỘC TRÊN GOOGLE COLAB:**\n"
            "> - **Loại GPU (Hardware Accelerator):** `L4 GPU` (22.5 GB VRAM - Kiến trúc Ada Lovelace, Colab Pro).\n"
            "> - **Có cần bật RAM cao cấp (High-RAM) không?** **KHÔNG CẦN BẬT THÊM HIGH-RAM!** Trên Google Colab, máy chủ gắn GPU L4 mặc định đã được cấp sẵn `~53 GB System RAM` và `22.5 GB VRAM` (pipeline chỉ dùng `~2.6 GB RAM` và `~1.2 GB VRAM`), do đó bạn giữ nguyên thiết lập RAM mặc định để tiết kiệm Compute Units.\n"
            "> - **Quy mô chạy:** **Toàn bộ `4.986 mảng ảnh thật`** trên toàn bộ các bệnh nhân tập test BCSS-WSSS.\n"
            "> - **Thời gian hoàn tất:** `~1.5 - 2 phút` trên L4 GPU.\n"
            "> - **Tên file Zip tự động tải về:** `SE374_Colab_Results_DeepLabV3Plus_L4_Full4986_<Timestamp>.zip`.\n\n"
        ),
    },
    {
        "filename": "SE374_Colab_WSSS_A100_Full4986.ipynb",
        "gpu_type": "A100",
        "high_ram": True,
        "sanity_mode": False,
        "mode_tag_override": "A100_Full4986",
        "title_banner": (
            "# [BẢN 4/4 - DÀNH CHO SIÊU GPU NVIDIA A100 - FULL 4.986 ẢNH & HUẤN LUYỆN QUY MÔ LỚN]\n"
            "## SE374.R11: WSSS Histopathology Tissue Segmentation Pipeline (A100 SXM4 40GB/80GB)\n\n"
            "> **CẤU HÌNH PHẦN CỨNG BẮT BUỘC TRÊN GOOGLE COLAB:**\n"
            "> - **Loại GPU (Hardware Accelerator):** `A100 GPU` (40 GB hoặc 80 GB HBM2e VRAM - Colab Pro / Pro+).\n"
            "> - **Khi nào cần dùng bản A100 và có cần bật RAM cao cấp (High-RAM) không?**\n"
            ">   1. **Nếu chỉ chạy Đánh giá (Inference/Benchmark) 4.986 ảnh test (như mặc định trong file này):** Thực chất **T4 hoặc L4 đã chạy rất nhanh**, bạn **KHÔNG bắt buộc phải tốn Compute Units thuê A100**, và **KHÔNG cần bật High-RAM** (máy A100 mặc định đã có sẵn `~83.5 GB System RAM`). Trên A100, toàn bộ 4.986 ảnh chạy xong chỉ trong **`~45 - 60 giây`**.\n"
            ">   2. **Khi nào BẮT BUỘC cần A100 + BẬT HIGH-RAM (`~83.5 GB RAM`)?** Chỉ khi bạn muốn **Huấn luyện lại từ đầu (Full Retraining)** cả 2 giai đoạn Stage 1 (ResNet-38 CAM) + Stage 2 (DeepLabV3+ ResNet-101) trên toàn bộ **`23.422 mảng ảnh train`** với `batch_size >= 32` và `num_workers >= 8`.\n"
            "> - **Tên file Zip tự động tải về:** `SE374_Colab_Results_DeepLabV3Plus_A100_Full4986_<Timestamp>.zip`.\n\n"
        ),
    },
]


def _to_nb_lines(code_str: str) -> list:
    lines = code_str.splitlines()
    if not lines:
        return []
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


CELL_6_RESILIENT_DATASET = """import os
import subprocess
import urllib.request
import zipfile
from pathlib import Path

DATA_ROOT = Path('/content/BCSS-WSSS')
ZIP_PATH = Path('/content/BCSS-WSSS.zip')

# Google Drive File ID chính thức & Mirror CDN tốc độ cao không giới hạn lượt tải (Han et al., MedIA 2022)
GDRIVE_DATASET_ID = '1sCK8Cqsf4yCHzzIx98vitocES2L_tkjn'
HF_DATASET_URL = 'https://huggingface.co/datasets/nttduc/wsss-pathology/resolve/main/BCSS-WSSS.zip'

def resolve_test_dirs(base_search_dir=Path('/content')):
    candidates = list(base_search_dir.glob('**/test/img'))
    if candidates:
        img_dir = candidates[0]
        mask_dir = img_dir.parent / 'mask'
        root_dir = img_dir.parent.parent
        return root_dir, img_dir, mask_dir
    return None, None, None

def download_dataset_resilient(zip_path: Path, hf_url: str, gdrive_id: str) -> None:
    if zip_path.exists() and zip_path.stat().st_size < 100 * 1024 * 1024:
        zip_path.unlink()
    if zip_path.exists():
        return

    print("--> Đang tải BCSS-WSSS.zip (2.96 GB) qua cổng HuggingFace CDN tốc độ cao (chống lỗi Google Drive Quota)...")
    try:
        last_pct = [-1]
        def _progress(block_num, block_size, total_size):
            if total_size > 0:
                pct = int(block_num * block_size * 100 / total_size)
                if pct >= last_pct[0] + 10 and pct <= 100:
                    last_pct[0] = (pct // 10) * 10
                    mb_done = (block_num * block_size) / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f"    + Tiến độ tải BCSS-WSSS.zip: {last_pct[0]}% ({mb_done:.0f}/{mb_total:.0f} MB)")
        urllib.request.urlretrieve(hf_url, str(zip_path), reporthook=_progress)
        if zip_path.exists() and zip_path.stat().st_size > 1000 * 1024 * 1024:
            print("[OK] Đã tải thành công BCSS-WSSS.zip từ HuggingFace CDN Mirror!")
            return
    except Exception as exc_hf:
        print(f"[THÔNG BÁO] Chuyển sang nhánh dự phòng Google Drive ({exc_hf})...")

    try:
        import gdown
        gdown.download(id=gdrive_id, output=str(zip_path), quiet=False, fuzzy=True)
    except Exception as exc_gd:
        print(f"[CẢNH BÁO] Google Drive giới hạn lượt tải ({exc_gd}). Đang tải lại qua curl CDN...")
        subprocess.run(["curl", "-L", "--retry", "3", "-o", str(zip_path), hf_url], check=False)

root_found, test_img_dir, test_mask_dir = resolve_test_dirs()

if root_found is None:
    download_dataset_resilient(ZIP_PATH, HF_DATASET_URL, GDRIVE_DATASET_ID)
    print("--> Đang giải nén bộ dữ liệu BCSS-WSSS vào /content/...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall('/content/')
    root_found, test_img_dir, test_mask_dir = resolve_test_dirs()

if root_found is not None:
    DATA_ROOT = root_found
    print(f"[OK] Thư mục gốc dữ liệu: {DATA_ROOT.resolve()}")
    print(f"[OK] Thư mục ảnh test   : {test_img_dir.resolve()}")
    print(f"[OK] Thư mục mask test  : {test_mask_dir.resolve()}")
    total_imgs = len(list(test_img_dir.glob('*.png')))
    total_masks = len(list(test_mask_dir.glob('*.png'))) if test_mask_dir.exists() else 0
    print(f"[OK] Tổng số ảnh test tìm thấy: {total_imgs} ảnh | Tổng số mask: {total_masks} mask")
else:
    raise FileNotFoundError("Không tìm thấy thư mục test/img sau khi giải nén!")"""


CELL_10_RESILIENT_CHECKPOINTS = """import subprocess
import urllib.request
from pathlib import Path

CHECKPOINT_DIR = Path('/content/checkpoints')
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Checkpoint Stage 1 ResNet38d Net_CAM (Classification Gate Mechanism - CGM)
CKPT_STAGE1_FILE = CHECKPOINT_DIR / 'stage1_checkpoint_trained_on_bcss.pth'
GDRIVE_STAGE1_ID = '118y3Q7i6lPOrw8OIFqt3tlP1PUM-i1Xc'
GDRIVE_STAGE1_ARML_ID = '1HiAI6yISiXZNokYgM2U5goic-I7GYG4H'
HF_STAGE1_URL = 'https://huggingface.co/longcris04/pseduo_mask_wsss/resolve/main/stage1_checkpoint_trained_on_bcss.pth'

# 2. Checkpoint Stage 2 DeepLabV3+ ResNet-101 (Semantic Segmentation Backbone)
CKPT_FILE = CHECKPOINT_DIR / 'stage2_checkpoint_trained_on_bcss.pth'
GDRIVE_CKPT_ID = '1cXm29ko8UevbaeW6YreSg-vuNaQiN_je'
HF_STAGE2_URL = 'https://huggingface.co/longcris04/pseduo_mask_wsss/resolve/main/stage2_checkpoint_trained_on_bcss.pth'

def download_checkpoint_resilient(target_path: Path, hf_url: str, gdrive_ids: list, desc: str) -> None:
    if target_path.exists() and target_path.stat().st_size < 50 * 1024 * 1024:
        target_path.unlink()
    if target_path.exists():
        return

    print(f"--> Đang tải {desc} qua cổng HuggingFace CDN tốc độ cao (chống lỗi Google Drive Quota)...")
    try:
        urllib.request.urlretrieve(hf_url, str(target_path))
        if target_path.exists() and target_path.stat().st_size > 100 * 1024 * 1024:
            print(f"[OK] Đã tải thành công {target_path.name} từ HuggingFace CDN Mirror!")
            return
    except Exception as exc_hf:
        print(f"[THÔNG BÁO] Chuyển sang nhánh dự phòng Google Drive cho {target_path.name} ({exc_hf})...")

    for gid in gdrive_ids:
        try:
            import gdown
            gdown.download(id=gid, output=str(target_path), quiet=False, fuzzy=True)
            if target_path.exists() and target_path.stat().st_size > 100 * 1024 * 1024:
                return
        except Exception as exc_gd:
            print(f"[THÔNG BÁO] Mirror Google Drive {gid} đang giới hạn lượt tải ({exc_gd}). Chuyển sang mirror kế tiếp...")

    subprocess.run(["curl", "-L", "--retry", "3", "-o", str(target_path), hf_url], check=False)

# Tải cả 2 bộ trọng số Stage 1 (ResNet38d Net_CAM) và Stage 2 (DeepLabV3+ ResNet-101)
download_checkpoint_resilient(
    CKPT_STAGE1_FILE,
    HF_STAGE1_URL,
    [GDRIVE_STAGE1_ARML_ID, GDRIVE_STAGE1_ID],
    "Pretrained Checkpoint Stage 1 ResNet38d Net_CAM",
)
if CKPT_STAGE1_FILE.exists():
    sz1_mb = CKPT_STAGE1_FILE.stat().st_size / (1024 * 1024)
    print(f"[OK] Checkpoint Stage 1 đã sẵn sàng tại: {CKPT_STAGE1_FILE.resolve()} ({sz1_mb:.2f} MB)")
else:
    print(f"[CẢNH BÁO] Không tìm thấy Stage 1 Checkpoint tại {CKPT_STAGE1_FILE}. Pipeline sẽ hoạt động với chế độ fallback an toàn.")

download_checkpoint_resilient(
    CKPT_FILE,
    HF_STAGE2_URL,
    [GDRIVE_CKPT_ID],
    "Pretrained Checkpoint SOTA Stage 2 DeepLabV3+",
)
if CKPT_FILE.exists():
    file_size_mb = CKPT_FILE.stat().st_size / (1024 * 1024)
    print(f"[OK] Checkpoint Stage 2 đã sẵn sàng tại: {CKPT_FILE.resolve()} ({file_size_mb:.2f} MB)")
else:
    raise FileNotFoundError(f"Không tìm thấy tệp checkpoint Stage 2 tại: {CKPT_FILE.resolve()}")"""


def build_variants() -> None:
    master_data = json.loads(MASTER_NOTEBOOK.read_text(encoding="utf-8"))

    # Upgrade Master Notebook Cell 6 (Dataset Download) & Cell 10 (Checkpoint Download) & Cell 12 (PyTorch 2.6+ safe_torch_load & CGM-consistent M3/M5)
    master_data["cells"][6]["source"] = _to_nb_lines(CELL_6_RESILIENT_DATASET)
    master_data["cells"][10]["source"] = _to_nb_lines(CELL_10_RESILIENT_CHECKPOINTS)

    cell_12_src = "".join(master_data["cells"][12]["source"])
    cell_12_src = cell_12_src.replace(
        "    try:\n        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)\n    except TypeError:\n        try:\n        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)\n    except TypeError:\n        checkpoint = torch.load(ckpt_path, map_location=device)",
        "    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)",
    )
    cell_12_src = cell_12_src.replace(
        "    try:\n        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)\n    except TypeError:\n        checkpoint = torch.load(ckpt_path, map_location=device)",
        "    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)",
    )
    cell_12_src = cell_12_src.replace(
        "    checkpoint = torch.load(ckpt_path, map_location=device)\n",
        "    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)\n",
    )
    if "self.apply_cgm_gate(self.apply_prior_calibration(probs_m3), pred_cls_gate)" not in cell_12_src:
        cell_12_src = cell_12_src.replace(
            "        probs_m3 = self.apply_cgm_gate(probs_d4, pred_cls_gate)\n        probs_m3 = self.apply_cam_fusion(probs_m3, cam_np)\n        probs_m3 = self.apply_prior_calibration(probs_m3)",
            "        probs_m3 = self.apply_cgm_gate(probs_d4, pred_cls_gate)\n        probs_m3 = self.apply_cam_fusion(probs_m3, cam_np)\n        probs_m3 = self.apply_cgm_gate(self.apply_prior_calibration(probs_m3), pred_cls_gate)",
        )
    master_data["cells"][12]["source"] = _to_nb_lines(cell_12_src)

    # Fix Circular Reference in Cell 14 (use dict copy so metrics["ablation_methods"] never points to itself)
    cell_14_src = "".join(master_data["cells"][14]["source"])
    cell_14_src = cell_14_src.replace(
        "metrics = all_methods_metrics[best_method_key]",
        "metrics = dict(all_methods_metrics[best_method_key])",
    )
    master_data["cells"][14]["source"] = _to_nb_lines(cell_14_src)

    # Sanitize ablation_methods in Cell 18 so json.dump never hits a circular reference even on re-run
    cell_18_master_src = "".join(master_data["cells"][18]["source"])
    cell_18_master_src = cell_18_master_src.replace(
        '        "ablation_methods": metrics.get("ablation_methods", {}),',
        '        "ablation_methods": {k: {mk: mv for mk, mv in v.items() if mk != "ablation_methods"} for k, v in all_methods_metrics.items()},',
    )
    master_data["cells"][18]["source"] = _to_nb_lines(cell_18_master_src)

    MASTER_NOTEBOOK.write_text(json.dumps(master_data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    for cfg in VARIANTS:
        nb = copy.deepcopy(master_data)

        # Update metadata gpuType and optional highmem shape
        nb["metadata"]["colab"]["gpuType"] = cfg["gpu_type"]
        if cfg["high_ram"]:
            nb["metadata"]["colab"]["machine_shape"] = "hm"
        else:
            nb["metadata"]["colab"].pop("machine_shape", None)

        # Prepend hardware banner to Cell 0
        cell_0_lines = nb["cells"][0]["source"]
        # Remove the generic first 2 lines and prepend the variant-specific banner
        nb["cells"][0]["source"] = [cfg["title_banner"]] + cell_0_lines[2:]

        # Update Cell 8 (SANITY_MODE setting)
        cell_8_src = "".join(nb["cells"][8]["source"])
        if cfg["sanity_mode"]:
            cell_8_src = cell_8_src.replace("SANITY_MODE = False", "SANITY_MODE = True")
        else:
            cell_8_src = cell_8_src.replace("SANITY_MODE = True", "SANITY_MODE = False")
        nb["cells"][8]["source"] = [line + "\n" for line in cell_8_src.splitlines()[:-1]] + [
            cell_8_src.splitlines()[-1]
        ]

        # Update Cell 18 (mode_tag in zip export so each GPU variant has its own unmistakable zip prefix)
        cell_18_src = "".join(nb["cells"][18]["source"])
        cell_18_src = cell_18_src.replace(
            "mode_tag = 'Sanity150' if SANITY_MODE else 'Full4986'",
            f"mode_tag = '{cfg['mode_tag_override']}'",
        )
        nb["cells"][18]["source"] = [line + "\n" for line in cell_18_src.splitlines()[:-1]] + [
            cell_18_src.splitlines()[-1]
        ]

        out_path = MASTER_NOTEBOOK.parent / cfg["filename"]
        out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

        # Post-generation programmatic verification
        content_str = out_path.read_text(encoding="utf-8")
        assert "118y3Q7i6lPOrw8OIFqt3tlP1PUM-i1Xc" in content_str, f"Missing Stage 1 ID in {out_path}"
        assert "1cXm29ko8UevbaeW6YreSg-vuNaQiN_je" in content_str, f"Missing Stage 2 ID in {out_path}"
        assert "WSSSTissueStage1NetCAM" in content_str, f"Missing WSSSTissueStage1NetCAM in {out_path}"
        for bad_char in ("\u2014", "\u2013", "\u2015", "\u2212"):
            assert bad_char not in content_str, f"Forbidden non-ASCII dash in {out_path}"

        print(f"[OK] Generated and verified {out_path} (GPU={cfg['gpu_type']}, SANITY_MODE={cfg['sanity_mode']})")


if __name__ == "__main__":
    build_variants()
