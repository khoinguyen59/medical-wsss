# SE374.R11 - Gói Mã Nguồn Module Hóa Trên Google Drive (`src/colab_pkg`)

Thư mục `src/colab_pkg/` chứa toàn bộ mã nguồn thực nghiệm phân vùng mô bệnh học giám sát yếu (BCSS-WSSS) đã được tách rời khỏi Notebook theo kiến trúc Clean Modular Architecture. Bạn chỉ cần tải thư mục `src/colab_pkg` lên Google Drive (ví dụ: `/content/drive/MyDrive/SE374_WSSS_Code/colab_pkg`), sau đó gọi thực thi trong Notebook chỉ với 1-3 dòng lệnh.

## 1. Cấu Trúc Các Module Trong `src/colab_pkg/`

| Tên File | Chức Năng Cốt Lõi |
| :--- | :--- |
| `__init__.py` | Điểm xuất (entry-point) cung cấp toàn bộ hàm và lớp của gói `colab_pkg`. |
| `downloader.py` | Tải tự động bộ dữ liệu `BCSS-WSSS.zip` và 2 bộ trọng số `stage1_checkpoint_trained_on_bcss.pth`, `stage2_checkpoint_trained_on_bcss.pth` qua HuggingFace CDN kèm dự phòng Google Drive đa nhánh; kiểm định tính toàn vẹn tệp nén/checkpoint (`is_valid_zip_archive`, `is_valid_torch_checkpoint`); chọn mẫu chuẩn chống rò rỉ bệnh nhân (Invariant 08). |
| `model_loader.py` | Kiến trúc mạng `DeepLabV3Plus` (ResNet-101, 59.34M tham số), `WSSSTissueStage1NetCAM` (ResNet-38d, 105.09M tham số), `OtsuClinicalGating`, và `MultiMethodWSSSEngine` / `MultiMethodWSSSEngineV2` (5 phương pháp M1..M5 và 7 phương pháp thế hệ V2 M1..M7 tích hợp Full Dihedral D4 8-View TTA trên cả ảnh vuông lẫn chữ nhật + Beer-Lambert H&E Optical Density Calibration). |
| `evaluator.py` | Tính toán ma trận nhầm lẫn 4x4 (chuẩn WSSS-Tissue ROI) và 5x5 (kèm BACK), mIoU, mDice, Macro F1 (`Macro_F1_4cls`), Pixel Accuracy, VTP, TSR, kiểm định thống kê cặp Wilcoxon signed-rank & Student t-test, và xuất JSON/Markdown/CSV không đệ quy vòng (`allow_nan=False`). |
| `visualizer.py` | Trực quan hóa ảnh so sánh lâm sàng 5 cột 300 DPI, biểu đồ nhiệt ma trận nhầm lẫn 5x5 300 DPI, biểu đồ cột Ablation 300 DPI (5 hoặc 7 phương pháp), và đóng gói tệp `.zip` tự động tải về máy. |
| `colab_runner.py` | Bộ điều phối cấp cao `run_bcss_wsss_experiment(...)` (hỗ trợ `mode="smoke"|"quick"|"sanity"|"full"` và `include_v2=True`) cùng `run_streaming_evaluation(...)` giúp Notebook gọi chạy toàn bộ pipeline chỉ trong 1-3 dòng code. |

## 2. Cách Sử Dụng Trên Google Colab (Gọi Từ Google Drive)

```python
from google.colab import drive
import sys

drive.mount('/content/drive')
sys.path.insert(0, '/content/drive/MyDrive/SE374_WSSS_Code')

from colab_pkg import run_bcss_wsss_experiment

# Chạy toàn bộ 4.986 mảng ảnh test (hoặc đặt mode="sanity" / sanity_mode=True để chạy 150 ảnh nhanh):
results = run_bcss_wsss_experiment(mode="full", include_v2=True, mode_tag="T4_Full4986")
```
