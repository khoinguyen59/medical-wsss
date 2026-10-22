# Nhật Ký Thực Nghiệm Các Phiên Chạy Thật Trên Toàn Bộ Dữ Liệu (4.986 Ảnh Test)

Thư mục này lưu trữ các notebook Jupyter chứa đầy đủ log thực thi, kết quả và biểu đồ đối chứng của **3 lần chạy thật trên toàn bộ 4.986 ảnh test (23 bệnh nhân)** của tập BCSS-WSSS:

1. **`Run1_20261020_1022UTC_Stage2_DeepLabV3Plus_SingleCkpt_T4_Full4986.ipynb` (Lần 1)**:
   - Kiến trúc: Stage-2 DeepLabV3+ ResNet-101 đơn lẻ trên 4.986 ảnh test.
   - Kết quả: `mIoU_4cls = 68.71%`, `mDice_4cls = 81.15%`, `PA_4cls = 84.94%`.
2. **`Run2_20261020_1641UTC_DualCkpt_Stage1_CGM_Stage2_M1_to_M5_T4_Full4986.ipynb` (Lần 2)**:
   - Kiến trúc: Dual-Stage (Stage-1 ResNet-38d Net_CAM + Stage-2 DeepLabV3+ ResNet-101), đánh giá 5 phương pháp `M1..M5`.
   - Kết quả: `M5 Proposed SOTA Ensemble` đạt `mIoU_4cls = 70.49%`, `mDice_4cls = 82.51%`.
3. **`Run3_20261020_2055UTC_MultiScale_Pyramid_D4_M1_to_M7_T4_Full4986.ipynb` (Lần 3 - SOTA)**:
   - Kiến trúc: Thin Drive Runner v2 (`MultiMethodWSSSEngineV2`, FP16 Mixed Precision), đánh giá toàn diện 7 phương pháp `M1..M7`.
   - Kết quả: `M6 Multi-Scale Pyramid D4` đạt đỉnh `mIoU_4cls = 70.53%`, `mDice_4cls = 82.54%`, `TUM IoU = 79.37%`.
