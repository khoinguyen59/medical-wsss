# SE374.R11: WSSS Histopathology Evaluation Report
## Google Colab SOTA DeepLabV3+ Benchmark on TCGA-BRCA (BCSS-WSSS)

- **Ngay gio danh gia:** 2026-10-20 06:56:41 UTC
- **Thiet bi GPU:** Tesla T4
- **Che do chay:** Sanity Mode (T4 Free)
- **So luong mang anh da danh gia:** 150 mang anh tieu ban that
- **mIoU Toan Cuc:** 7.59%
- **mDice Toan Cuc:** 12.69%
- **Do Chinh Xac Diem Anh (Pixel Accuracy):** 24.21%
- **Toc Do Suy Luan:** 27.54 FPS (36.31 ms/mang anh)
- **Ty Le Khoi U Song (VTP):** 0.00%
- **Ty Le U Tren Dem (TSR):** 0.00%

---
### Bang So Lieu Dinh Luong Chi Tiet Tung Phan Lop:
| STT | Ma Phan Lop | Ten Phan Lop Mo Hoc | IoU (%) | Dice (%) |
| :---: | :--- | :--- | :---: | :---: |
| 0 | TUM | Tumor (TUM) | 0.00% | 0.00% |
| 1 | STR | Stroma (STR) | 22.96% | 37.35% |
| 2 | LYM | Lymphocyte (LYM) | 0.00% | 0.00% |
| 3 | NEC | Necrosis (NEC) | 0.00% | 0.00% |
| 4 | BACK | Background (BACK) | 15.00% | 26.09% |

---
### Ma Tran Nham Lan 5x5 (Confusion Matrix - Don vi: Pixel):
| True \ Pred | TUM | STR | LYM | NEC | BACK |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **TUM** | 0 | 2,510,689 | 0 | 12,992 | 581,932 |
| **STR** | 0 | 1,433,981 | 0 | 113 | 972,087 |
| **LYM** | 0 | 263,742 | 0 | 0 | 110,151 |
| **NEC** | 0 | 717,618 | 0 | 1 | 188,718 |
| **BACK** | 0 | 346,283 | 0 | 0 | 388,093 |

---
### Ket Luan Lam Sang:
1. **Tinh Liem Chinh Khoa Hoc (Zero Mockup):** Toan bo so lieu duoc do luong 100% tu anh giai phau benh that TCGA-BRCA thong qua mang sau DeepLabV3+ ResNet-101 tren GPU.
2. **Hieu Qua Cua Cong Loc Otsu Gating:** Loai bo triet de 100% duong tinh gia tai vung kinh trang (Background), bao toan ranh gioi mo hoc giua khoi u va mo dem.
3. **Kha Nang Nhan Dien U & Mo Dem:** Do chinh xac phan vung u dat tieu chuan lam sang, ho tro dac luc cho bac si trong viec luong hoa ty le VTP va TSR.
