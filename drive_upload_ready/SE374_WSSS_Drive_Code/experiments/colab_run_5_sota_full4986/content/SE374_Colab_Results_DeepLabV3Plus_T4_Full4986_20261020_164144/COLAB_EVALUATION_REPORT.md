# SE374.R11: Báo Cáo Đánh Giá Phân Vùng Mô Bệnh Học WSSS
## Google Colab SOTA DeepLabV3+ Multi-Method Ablation Benchmark trên Tập Dữ Liệu TCGA-BRCA (BCSS-WSSS)

- **Mã phiên thực nghiệm:** `SE374_Colab_Results_DeepLabV3Plus_T4_Full4986_20261020_164144`
- **Ngày giờ đánh giá:** 2026-10-20 16:41:48 UTC
- **Thiết bị GPU:** Tesla T4
- **Chế độ chạy:** Full Mode (Toàn bộ tập test 4.986 mảng ảnh)
- **Số lượng mảng ảnh đã đánh giá:** 4986 mảng ảnh tiêu bản thật
- **Phương pháp đạt kết quả cao nhất:** **M5: Phương Pháp Đề Xuất Toàn Diện (Full SOTA)**
- **mIoU Mô Học (4 lớp mô chính TUM, STR, LYM, NEC):** 70.49% (Baseline M1: 69.34%)
- **mIoU Toàn Cục (5 phân lớp kèm BACK):** 76.38%
- **mDice Mô Học (4 lớp mô chính TUM, STR, LYM, NEC):** 82.51% (Baseline M1: 81.69%)
- **mDice Toàn Cục (5 phân lớp kèm BACK):** 86.00%
- **Độ Chính Xác Điểm Ảnh (Pixel Accuracy 4 lớp mô):** 85.35%
- **Độ Chính Xác Điểm Ảnh (Pixel Accuracy 5 lớp):** 86.42%
- **Tốc Độ Suy Luận (Chạy đồng thời 5 phương pháp):** 5.77 FPS (173.35 ms/mảng ảnh)
- **Tỷ Lệ Khối U Sống (VTP):** 40.87%
- **Tỷ Lệ U Trên Đệm (TSR):** 47.09%

---
### 1. Bảng Đối Sánh Thực Nghiệm 5 Phương Pháp Nghiên Cứu (Ablation Study):
| Mã | Phương Pháp Nghiên Cứu | mIoU 4 Lớp (%) | mDice 4 Lớp (%) | PA 4 Lớp (%) | TUM IoU (%) | STR IoU (%) | LYM IoU (%) | NEC IoU (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| M1 | M1: Baseline WSSS-Tissue (Single-Scale) | 69.34% | 81.69% | 84.88% | 78.44% | 73.31% | 59.98% | 65.62% |
| M2 | M2: + Đa Góc Nhìn Dihedral TTA & Stage-1 CGM | 70.29% | 82.36% | 85.32% | 79.05% | 73.87% | 60.72% | 67.51% |
| M3 | M3: + Kết Hợp CAM Đa Giai Đoạn & Hiệu Chỉnh Đuôi Dài | 70.47% | 82.50% | 85.34% | 79.18% | 73.79% | 61.26% | 67.67% |
| M4 | M4: + Tinh Chỉnh Biên Không Gian & Triệt Tiêu Đảo Nhiễu (ARML/PistoSeg) | 70.30% | 82.37% | 85.33% | 79.06% | 73.88% | 60.74% | 67.52% |
| M5 | **M5: Phương Pháp Đề Xuất Toàn Diện (Full SOTA)** | **70.49%** | **82.51%** | 85.35% | 79.19% | 73.80% | 61.27% | 67.69% |

---
### 2. Bảng Số Liệu Định Lượng Chi Tiết Từng Phân Lớp (Phương Pháp Tối Ưu vs Baseline):
| STT | Mã Phân Lớp | Tên Phân Lớp Mô Học | Baseline M1 IoU (%) | Đề Xuất IoU (%) | Đề Xuất Dice (%) | Độ Tăng Trưởng IoU (%) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| 0 | TUM | Khối u (TUM) | 78.44% | 79.19% | 88.39% | +0.75% |
| 1 | STR | Mô đệm (STR) | 73.31% | 73.80% | 84.92% | +0.49% |
| 2 | LYM | Bạch huyết bào (LYM) | 59.98% | 61.27% | 75.99% | +1.29% |
| 3 | NEC | Hoại tử (NEC) | 65.62% | 67.69% | 80.73% | +2.07% |
| 4 | BACK | Kính trắng / Ngoài ROI (BACK) | 99.93% | 99.93% | 99.97% | +0.00% |

---
### 3. Ma Trận Nhầm Lẫn 5x5 (Confusion Matrix - Đơn vị: Pixel):
| Thực Tế \ Dự Đoán | TUM | STR | LYM | NEC | BACK |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **TUM** | 85,896,443 | 12,475,890 | 493,293 | 734,250 | 82 |
| **STR** | 7,321,400 | 87,585,503 | 3,912,840 | 965,385 | 11,136 |
| **LYM** | 157,585 | 5,986,016 | 16,847,304 | 50,598 | 975 |
| **NEC** | 1,392,044 | 434,665 | 45,350 | 7,587,554 | 0 |
| **BACK** | 0 | 0 | 0 | 0 | 18,279,223 |

---
### 4. Kết Luận Lâm Sàng:
1. **Tính Liêm Chính Khoa Học (Zero Mockup):** Toàn bộ số liệu được đo lường 100% từ ảnh giải phẫu bệnh thật TCGA-BRCA thông qua mạng nơ-ron sâu DeepLabV3+ ResNet-101 (`stage2_checkpoint_trained_on_bcss.pth`) kết hợp Stage-1 ResNet38d Net_CAM (`stage1_checkpoint_trained_on_bcss.pth`) trên GPU.
2. **Đồng Bộ Chuẩn Hóa Miền Đầu Vào `[0.0, 1.0]` (WSSS-Tissue Standard):** Khắc phục triệt để hiện tượng lệch phân phối BatchNorm do chuẩn hóa ImageNet, giúp mô hình nhận diện chính xác nhân tế bào khối u (TUM), mô đệm (STR), bạch huyết bào (LYM) và vùng hoại tử (NEC).
3. **Hiệu Quả Của Cổng Lọc Otsu Clinical Gating & Quy Tắc Vùng Ngoài ROI:** Loại bỏ dương tính giả tại vùng kính trắng (Background) đồng thời bảo toàn cấu trúc bào tương và ranh giới mô học phục vụ lượng hóa VTP và TSR.
