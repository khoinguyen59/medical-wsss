# ĐẶC TẢ DỮ LIỆU Y KHOA & GIAO THỨC CHỐNG RÒ RỈ BỆNH NHÂN
## Đồ án: Hệ Thống Phân Vùng Ngữ Nghĩa Giám Sát Yếu Trên Ảnh Tiêu Bản Mô Bệnh Học Toàn Phần (WSSS for WSI)
### Môn học: SE374.R11 - Phát Triển Phần Mềm Xử Lý Ảnh Y Khoa (UIT 2026)

> **Cơ sở khoa học:** Chu Han et al. (*MedIA 2022*), Qingchen Tang et al. (*CVPR 2025*), Tan-Dung Pham et al. (*Springer 2026*)  
> **Quy tắc bất biến:** Patient-Level Anti-Leakage Protocol & Resource Safeguard / Sanity Run  

---

## 1. Khảo Sát Hai Bộ Dữ Liệu Tiêu Chuẩn Quốc Tế

Đề tài sử dụng 2 bộ dữ liệu mô bệnh học chuẩn quốc tế được công bố công khai phục vụ nghiên cứu phân vùng giám sát yếu:

### 1.1 Bộ Dữ Liệu BCSS-WSSS (Breast Cancer Semantic Segmentation - Weakly Supervised)

- **Nguồn gốc:** Trích xuất từ 151 vùng vi trường quan tâm (ROIs) của tiêu bản ung thư vú thuộc tập dữ liệu quốc gia Hoa Kỳ TCGA-BRCA (The Cancer Genome Atlas Breast Invasive Carcinoma), công bố bởi Amgad et al. (*Bioinformatics 2019*) và chuẩn hóa bởi Han et al. (*MedIA 2022*).
- **Liên kết tải chính thức (Tác giả MedIA 2022 & CVPR 2025):**  
  Google Drive: `https://drive.google.com/drive/folders/1iS2Z0DsbACqGp7m6VDJbAcgzeXNEFr77?usp=sharing`  
  Checkpoints tiền huấn luyện: `https://drive.google.com/drive/folders/1pRsf-TulOARYN13J-_iJNw_VUa2d1NnZ?usp=sharing`
- **Quy mô:** 17.286 patches ($224 \times 224$ pixels) quét ở độ phóng đại quang học $20\times$ (khoảng $0.5$ $\mu$m/pixel).
- **Phân bổ phân chia gốc:**
  - **Tập Huấn luyện (Train):** 106 bệnh nhân ($23.422$ patches cắt có overlap $50\%$). Chỉ có nhãn yếu đa nhãn cấp patch, không có mặt nạ vẽ tay.
  - **Tập Kiểm định (Validation):** 20 bệnh nhân ($3.418$ patches). Có mặt nạ Ground Truth pixel-level hoàn chỉnh.
  - **Tập Kiểm thử (Test):** 25 bệnh nhân ($4.986$ patches). Có mặt nạ Ground Truth pixel-level hoàn chỉnh.
- **Danh mục 5 Phân lớp Mô học (Classes):**
  1. `TUM` (Tumor Epithelial / Khối u biểu mô) - Mã lớp: `0` - Màu trực quan: Đỏ (Red)
  2. `STR` (Tumor-Associated Stroma / Mô đệm liên kết) - Mã lớp: `1` - Màu trực quan: Xanh lá (Green)
  3. `LYM` (Lymphocytic Infiltrate / Tế bào lympho xâm nhập) - Mã lớp: `2` - Màu trực quan: Xanh dương (Blue)
  4. `NEC` (Necrosis / Mô hoại tử) - Mã lớp: `3` - Màu trực quan: Tím (Purple)
  5. `BACK` (Background & Other / Kính trắng và mô nền khác) - Mã lớp: `4` - Màu trực quan: Trắng/Xám

### 1.2 Bộ Dữ Liệu LUAD-HistoSeg (Lung Adenocarcinoma Tissue Segmentation)

- **Nguồn gốc:** Gồm 49 bệnh nhân ung thư biểu mô tuyến phổi (29 ca từ Bệnh viện Nhân dân tỉnh Quảng Đông - GDPH và 20 ca từ TCGA-LUAD).
- **Liên kết tải chính thức:**  
  Google Drive: `https://drive.google.com/drive/folders/1E3Yei3Or3xJXukHIybZAgochxfn6FJpr?usp=sharing`
- **Quy mô:** 31.826 patches ($224 \times 224$ pixels) ở độ phóng đại $10\times$ (khoảng $1.0$ $\mu$m/pixel).
- **Phân bổ phân chia:**
  - **Tập Train:** 39 bệnh nhân ($16.678$ patches nhãn one-hot).
  - **Tập Validation:** 5 bệnh nhân ($300$ patches có mask điểm chuẩn).
  - **Tập Test:** 5 bệnh nhân ($307$ patches có mask kiểm thử độc lập).
- **Danh mục 4 Phân lớp Mô học:**
  1. `TE` (Tumor Epithelial) - Mã lớp: `0`
  2. `TAS` (Tumor-Associated Stroma) - Mã lớp: `1`
  3. `NEC` (Necrosis) - Mã lớp: `2`
  4. `LYM` (Lymphocytes) - Mã lớp: `3`

---

## 2. Quy Cách Cấu Trúc Tệp & Bóc Tách Metadata (File Naming Specification)

### 2.1 Cấu Trúc Thư Mục Dữ Liệu Chuẩn

```
data/
  BCSS-WSSS/
    train/
      TCGA-A2-A0T2_1024_2048_[1 0 0 1].png
      TCGA-A2-A0T2_1248_2048_[1 1 0 0].png
      ...
    valid/
      img/
        TCGA-E2-A152_512_1024.png
      mask/
        TCGA-E2-A152_512_1024.png   # PNG 8-bit index (giá trị pixel 0 đến 4)
    test/
      img/
        TCGA-BH-A0B8_2048_4096.png
      mask/
        TCGA-BH-A0B8_2048_4096.png  # PNG 8-bit index (giá trị pixel 0 đến 4)
  sample_dataset/                   # Mini Sample dùng cho Sanity Run (Dry-Run)
    train/
    valid/
    test/
```

### 2.2 Công Thức Bóc Tách Tên Tệp (Parsing Formula)

Đối với BCSS-WSSS, mỗi tệp ảnh trong tập huấn luyện có cấu trúc định dạng:
$$\text{Filename} = \underbrace{\text{PatientID}}_{\text{Mã bệnh nhân}} \_ \underbrace{X}_{\text{Tọa độ X}} \_ \underbrace{Y}_{\text{Tọa độ Y}} \_ [ \underbrace{y_0 \quad y_1 \quad y_2 \quad y_3}_{\text{Vector nhãn nhị phân}} ].\text{png}$$

- **Trích xuất Patient ID:** Phần chuỗi ký tự đứng trước dấu gạch dưới đầu tiên `_` (hoặc tiền tố TCGA chuẩn `TCGA-XX-XXXX`).
- **Trích xuất Tọa độ không gian:** Hai số nguyên $X, Y$ xác định vị trí mảng trên tiêu bản gigapixel toàn slide.
- **Trích xuất Vector nhãn yếu:** Chuỗi nằm giữa cặp ngoặc vuông `[...]`:
  $$\mathbf{y} = [y_{\text{TUM}}, y_{\text{STR}}, y_{\text{LYM}}, y_{\text{NEC}}] \in \{0, 1\}^4$$

---

## 3. Giao Thức Chống Rò Rỉ Bệnh Nhân Bất Biến (Patient-Level Anti-Leakage Protocol)

### 3.1 Vấn Đề Rò Rỉ Dữ Liệu Trong Ảnh Y Tế
Một bệnh nhân ung thư cung cấp một hoặc nhiều slide WSI, từ đó cắt ra hàng trăm đến hàng nghìn patches.
- **Rò rỉ ở cấp độ Patch (Patch-Level Leakage):** Nếu xáo trộn ngẫu nhiên toàn bộ patches và chia $80\% - 10\% - 10\%$, các patches của cùng một khối u bệnh nhân sẽ xuất hiện ở cả tập train và tập test. Do đặc trưng tế bào của cùng một người có mức độ tương đồng vi thể rất cao (cùng đột biến gen, cùng độ bắt màu nhuộm H&E), mô hình sẽ "học vẹt" kiểu hình của bệnh nhân đó thay vì học quy luật tổng quát, dẫn đến điểm kiểm thử cao ảo nhưng thất bại hoàn toàn khi ứng dụng trên bệnh nhân mới.

### 3.2 Quy Trình Kiểm Soát Toán Học Chống Rò Rỉ
Gọi $\mathcal{P}_{\text{train}}, \mathcal{P}_{\text{val}}, \mathcal{P}_{\text{test}}$ lần lượt là tập hợp các mã bệnh nhân (Patient IDs) tương ứng:
$$\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{val}} = \emptyset, \quad \mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test}} = \emptyset, \quad \mathcal{P}_{\text{val}} \cap \mathcal{P}_{\text{test}} = \emptyset$$

The above content does NOT show the entire file contents. If you need to view any lines of the file which were not shown to complete your task, call this tool again to view those lines.