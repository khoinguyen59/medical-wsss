# NGHIÊN CỨU ĐỘT PHÁ ĐỘ CHÍNH XÁC WSSS V2 (M6 & M7 ULTRA-SOTA CURRICULUM)
## Phân Tích Ma Trận Nhầm Lẫn Thực Nghiệm (4.986 Mảng Ảnh BCSS) & Thiết Kế Động Cơ Suy Luận Đa Tỷ Lệ Kết Hợp Quang Học Mô Bệnh Học H&E

> **Đơn vị:** Khoa Kỹ thuật Phần mềm, Trường Đại học Công nghệ Thông tin (UIT), ĐHQG-HCM  
> **Môn học:** SE374.R11 - Phát triển phần mềm xử lý ảnh y khoa  
> **Phạm vi module:** `src/models/wsss_inference_methods.py`  
> **Tuân thủ quản trị:** Invariant 07 (100% số liệu thực nghiệm thật từ log Colab), Invariant 08 (Chống rò rỉ cấp bệnh nhân), Invariant 12 (100% dấu gạch ngang chuẩn ASCII 0x2D `-`).

---

## 1. Chẩn Đoán Nguyên Nhân Gốc Rễ Từ Dữ Liệu Thực Nghiệm Lần Chạy 5 (4.986 Mảng Ảnh BCSS-WSSS)

Trong phiên chạy thực nghiệm quy mô đầy đủ trên Google Colab GPU Tesla T4 (`SE374_Colab_Results_DeepLabV3Plus_T4_Full4986_20261020_164144`, đánh giá trên 4.986 mảng ảnh kiểm thử thuộc 23 bệnh nhân độc lập), 5 phương pháp từ `M1` đến `M5` đã đạt được bước tiến quan trọng:

| Mã | Phương Pháp Nghiên Cứu (Thế Hệ V1) | mIoU 4 Lớp (%) | mDice 4 Lớp (%) | PA 4 Lớp (%) | TUM IoU (%) | STR IoU (%) | LYM IoU (%) | NEC IoU (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1** | Baseline WSSS-Tissue (Single-Scale, `Is_GM=False`) | 69.34% | 81.69% | 84.88% | 78.44% | 73.31% | 59.98% | 65.62% |
| **M2** | + Đa Góc Nhìn Dihedral TTA (4-View D4) & Stage-1 CGM (`y_cls > 0.1`) | 70.29% | 82.36% | 85.32% | 79.05% | 73.87% | 60.72% | 67.51% |
| **M3** | + Kết Hợp CAM Đa Giai Đoạn & Hiệu Chỉnh Đuôi Dài (`Prior Calibration`) | 70.47% | 82.50% | 85.34% | 79.18% | 73.79% | 61.26% | 67.67% |
| **M4** | + Tinh Chỉnh Biên Không Gian (`Bilateral d=5`) & Lọc Đảo Nhiễu (`< 36 px`) | 70.30% | 82.37% | 85.33% | 79.06% | 73.88% | 60.74% | 67.52% |
| **M5** | **Full SOTA Ensemble V1 (M2 + M3 + M4 + Pure-Glass Guard)** | **70.49%** | **82.51%** | **85.35%** | **79.19%** | **73.80%** | **61.27%** | **67.69%** |

### 1.1. Giải Phẫu Ma Trận Nhầm Lẫn 4x4 (231,886,120 Pixels Nội Vùng ROI)

Trích xuất trực tiếp từ `COLAB_EVALUATION_REPORT.json` của phiên chạy `20261020_162144`:

| Thực Tế \ Dự Đoán (M5) | TUM (Khối U) | STR (Mô Đệm) | LYM (Bạch Huyết Bào) | NEC (Hoại Tử) | Tổng Pixel Thực Tế | Độ Nhạy (Recall %) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TUM (Class 0)** | **85,896,443** | 12,475,890 | 493,293 | 734,250 | 99,599,876 | 86.24% |
| **STR (Class 1)** | 7,321,400 | **87,585,503** | 3,912,840 | 965,385 | 99,785,128 | 87.77% |
| **LYM (Class 2)** | 157,585 | 5,986,016 | **16,847,304** | 50,598 | 23,041,503 | 73.12% |
| **NEC (Class 3)** | 1,392,044 | 434,665 | 45,350 | **7,587,554** | 9,459,613 | 80.21% |

Phân tích định lượng từ ma trận nhầm lẫn trên chỉ ra **4 nút thắt cổ chai toán học** làm giới hạn mức tăng của `M3` (`+0.18%` so với `M2`) và `M4` (`+0.01%` so với `M2`):

1. **Nút thắt 1 - Giới hạn trường tiếp nhận đơn tỷ lệ (Single-Scale Receptive Field Bottleneck):**
   - Trong `M1`..`M5`, mạng Stage-2 `DeepLabV3+ ResNet-101` chỉ suy luận ở duy nhất độ phân giải `1.0x` (`224x224`).
   - Với `output_stride = 16`, bản đồ đặc trưng đầu ra của nhánh encoder chỉ có kích thước `14x14`. Các cụm tế bào lympho nhỏ (`LYM`, đường kính chỉ `6-10` pixels) nằm xen kẽ trong mô đệm bị hòa lẫn vào nền `STR`, gây ra **5,986,016 pixels lỗi `LYM -> STR`** (chiếm 25.98% tổng lượng `LYM` toàn tập test).
   - Ngược lại, các ổ hoại tử lớn (`NEC`) nằm giữa đám tế bào u đặc (`TUM`) cần bối cảnh rộng hơn (`scale = 0.75x`) để phân biệt nhân tan rã với mô u sống, dẫn đến **1,392,044 pixels lỗi `NEC -> TUM`** (chiếm 14.72% tổng lượng `NEC`).

2. **Nút thắt 2 - Cổng cắt nhị phân cứng (`y_cls > 0.10`) và nhiễu nền hàm mũ `exp(0) = 1.0` trong `extract_stage1_cam_and_gate`:**
   - Trong `M2`..`M5`, cổng phân loại Stage-1 sử dụng hàm bước nhảy cứng `cgm_gate = (y_cls > 0.10)`. Khi một mảng ảnh chỉ chứa một cụm nhỏ `LYM` hoặc `NEC` ở rìa (chiếm 3-7% diện tích patch), phép gộp trung bình toàn cục (GAP) của `ResNet38d Net_CAM` thường cho xác suất `0.045 < y_cls <= 0.10`. Cổng cứng cắt phăng lớp đó về `0.0`, xóa sổ toàn bộ dự đoán đúng của Stage-2 trên patch đó.
   - Ngược lại, khi `0.10 < y_cls < 0.20` (tín hiệu rất yếu), cổng cứng vẫn gán trọng số tuyệt đối `1.0` ngang bằng với lớp chính (`y_cls = 0.99`).
   - Nghiêm trọng hơn, ở bước chuẩn hóa bản đồ `cam_probs` trong V1:
     `exp_cam = np.where(cgm_gate > 0, np.exp(cam_norm * 2.0), 0.0)`.
     Tại các vùng điểm ảnh mà `cam_norm == 0.0` (tức `ReLU` của CAM hoàn toàn không kích hoạt), hàm mũ vẫn sinh ra `np.exp(0.0) = 1.0`! Khi trộn `0.80 * probs + 0.20 * cam_probs`, các vùng CAM bằng 0 vẫn bơm xác suất nền đồng nhất vào mọi lớp đang mở cổng, làm phẳng phân phối xác suất của Stage-2.

3. **Nút thắt 3 - Hiệu chỉnh tiên nghiệm đuôi dài bất biến không gian (`DEFAULT_PRIOR_WEIGHTS = (0.96, 0.90, 1.24, 1.18)`):**
   - Trong `M3` và `M5`, việc nhân toàn cục kênh `LYM` với `1.24` và hạ kênh `STR` xuống `0.90` (tỷ số khuếch đại `1.24 / 0.90 = 1.378`) trên mọi pixel không phân biệt đặc tính nhuộm màu H&E đã đẩy hàng loạt pixel mô đệm sợi collagen màu hồng nhạt (vốn không hề có nhân bắt màu Hematoxylin tím đậm) thành dương tính giả `LYM`.
   - Hệ quả thực nghiệm: `STR IoU` bị tụt từ `73.87%` (ở `M2`) xuống `73.79%` (ở `M3`), đồng thời sinh ra **3,912,840 pixels dương tính giả `STR -> LYM`**, làm phình mẫu số hợp (Union) của `LYM IoU`.

4. **Nút thắt 4 - Bán kính lọc song phương (`d = 5`) và ngưỡng đảo nhiễu (`36 px`) nhỏ hơn bước nhảy nội suy `4x`/`16x` của DeepLabV3+:**
   - Bộ giải mã (Decoder) của `DeepLabV3+` phóng đại tuyến tính (bilinear upsampling) tensor đặc trưng từ `(56, 56)` lên `(224, 224)` (bước nhảy `4x4 = 16` pixels), trong khi nhánh ASPP hoạt động ở `(14, 14)` (bước nhảy `16x16 = 256` pixels).
   - Do đó, bộ lọc song phương `d = 5` (bán kính 2 pixels) trong `M4` nằm gọn bên trong một ô nội suy `4x4`, hầu như không tạo ra thay đổi nhãn ở biên (`M4` chỉ tăng đúng `+0.01%` so với `M2`: `70.29% -> 70.30%`). Tương tự, một điểm nhiễu đơn lẻ từ ASPP có kích thước thực tế từ `64` đến `128` pixels nên dễ dàng lọt qua ngưỡng `min_island_area = 36`.

---

## 2. Đặc Tả Toán Học Các Phương Pháp Đột Phá Thế Hệ V2 (`M6` & `M7`)

Để giải quyết triệt để 4 nút thắt trên mà vẫn giữ nguyên 100% tính tương thích ngược của `M1`..`M5`, chúng tôi thiết kế và hiện thực hóa hai phương pháp suy luận thế hệ mới trong `src/models/wsss_inference_methods.py`:
- **`M6_MultiScale_Pyramid_D4_Ensemble`**: Tổ hợp Đa Tỷ Lệ Kim Tự Tháp (`0.75x, 1.0x, 1.25x`) trọng số hóa theo Độ Tự Tin Entropy Shannon kết hợp Cổng Mềm Thích Nghi Đa Ngưỡng Stage-1 D4.
- **`M7_Proposed_Ultra_SOTA_Curriculum`**: Chuỗi Quy Trình Tối Thượng Kết Hợp Kim Tự Tháp Đa Tỷ Lệ D4 + Dung Hợp CAM Bảo Toàn Điểm Không Theo Entropy + Hiệu Chỉnh Mật Độ Quang Học Beer-Lambert (H&E Color Deconvolution) + Tinh Chỉnh Biên Đa Bước Nhảy & Triệt Tiêu Đảo Nhiễu Theo Độ Tự Tin.

```mermaid
flowchart TD
    subgraph Pyramid_S2["1. Uncertainty-Gated Multi-Scale Pyramid + Dihedral D4 Stage-2 (M6 & M7)"]
        Img["Ảnh H&E: X (1, 3, H, W)"] --> S075["Scale 0.75x (168x168) + D4 Flip"]
        Img --> S100["Scale 1.00x (224x224) + 4-View D4"]
        Img --> S125["Scale 1.25x (280x280) + D4 Flip"]
        S075 & S100 & S125 --> Interp["Bilinear Align (align_corners=True) về (H, W)"]
        Interp --> EntWeight["Dung Hợp Đa Tỷ Lệ Điều Hướng Bởi Bất Định Scale Gốc H_1.0(u,v)^1.2"]
        EntWeight --> P_MSD4["Xác Suất Đa Tỷ Lệ: P_ms_d4 (4, H, W)"]
    end

    subgraph Adaptive_S1["2. 4-View D4 Stage-1 Net_CAM & Cổng Mềm Thích Nghi Đa Ngưỡng (M6 & M7)"]
        Img --> S1_D4["Stage-1 ResNet38d Net_CAM (4-View D4 cho y_cls + Native Grid cho CAM)"]
        S1_D4 --> SoftGate["Cổng Mềm Thích Nghi: G_soft(c) in [0, 1] + Khôi Phục Cụm Biên Cực Đại"]
        S1_D4 --> ZeroCAM["CAM Sắc Nét Bảo Toàn Điểm 0: S_c = G_soft * M_c * exp(2.0 * M_c)"]
    end

    subgraph Bio_Refinement["3. Hiệu Chỉnh Quang Học H&E Beer-Lambert & Tinh Chỉnh Biên (M6 & M7)"]
        P_MSD4 & SoftGate & ZeroCAM --> EntCAM["Dung Hợp CAM Theo Entropy Stage-2: a_eff(u,v) in [0.15, 0.32]"]
        EntCAM --> M6_Calib["M6: Hiệu Chỉnh Tiên Nghiệm Đuôi Dài + Pure-Glass Guard"]
        Img --> BeerLambert["Giải Chập Quang Học H&E (Ruifrok-Johnston Beer-Lambert OD)"]
        BeerLambert --> HE_Maps["Mật Độ Nhân Hematoxylin C_H & Bào Tương Eosin C_E"]
        EntCAM & HE_Maps --> OD_Calib["M7: Hiệu Chỉnh Tiên Nghiệm Điều Hướng Quang Học H&E + Cứu Hộ Cụm Biên"]
        OD_Calib --> MultiStride["Lọc Song Phương Đa Bước Nhảy (d=9) & Tiếp Giáp Biên OD"]
        MultiStride --> ConfIsland["Triệt Tiêu Đảo Nhiễu Phân Lớp Đặc Thù (< 96 px)"]
        ConfIsland --> GlassGuard["Ultra-Safe Pure-Glass Guard (Class 4)"]
    end
```

---

### 2.1. Trụ Cột 1: Kim Tự Tháp Đa Độ Phân Giải Điều Hướng Bởi Bất Định Tỷ Lệ Gốc (`forward_multiscale_d4_tta`)

Phân tích thực nghiệm trên mạng `DeepLabV3+ ResNet-101` (`output_stride = 16`) cho thấy một hiện tượng cấu trúc cực kỳ quan trọng:
- Ở tỷ lệ thu nhỏ $s = 0.75$ ($168 \times 168$), bản đồ đặc trưng của nhánh Encoder chỉ còn kích thước $11 \times 11$. Các nhánh tích chập giãn nở `ASPP` (`dilation = 12` và `dilation = 18`) lấy mẫu hoàn toàn vào vùng đệm không (`zero-padding`) ở rìa ngoài, gây ra hiện tượng **suy biến entropy thấp giả tạo (low-entropy majority-class collapse)** về các lớp đa số (`TUM`/`STR`) và xóa sổ lớp hiếm (`NEC`).
- Ngược lại, ở tỷ lệ phóng đại $s = 1.25$ ($280 \times 280$), bản đồ đặc trưng mở rộng lên $18 \times 18$, giúp các nhánh `ASPP` lấy mẫu trọn vẹn bối cảnh mô học mà không chạm biên đệm không, phân giải xuất sắc các ổ hoại tử (`NEC`) và cụm bạch huyết bào (`LYM`).

Từ phát hiện này, với tensor ảnh đầu vào $X \in [0, 1]^{1 \times 3 \times H \times W}$ và tập tỷ lệ kim tự tháp $\mathcal{S} = \{0.75, 1.0, 1.25\}$ cùng trọng số tiên nghiệm tỷ lệ chuẩn hóa $\alpha = (0.04, 0.80, 0.16)$:

1. Với mỗi tỷ lệ $s \in \mathcal{S}$, nội suy song tuyến tính căn chỉnh góc (`align_corners=True`, bảo toàn tuyệt đối hệ quy chiếu của `DeepLabV3+.forward`) sang kích thước $(H_s, W_s) = (\text{round}(sH), \text{round}(sW))$:
   $$X_s = \text{Interp}_{\text{bilinear, align\_corners=True}}(X, (H_s, W_s))$$
2. Thực hiện suy luận đối xứng nhóm Dihedral trên mạng Stage-2 $\text{Model}_{stage2}$ và nội suy xác suất trở về kích thước gốc $(H, W)$:
   $$P_s(c, u, v) = \text{Interp}_{\text{bilinear, align\_corners=True}}\left( \text{TTA}_{D4}(\text{Model}_{stage2}, X_s), (H, W) \right)$$
3. Tại mỗi điểm ảnh $(u, v)$ của tỷ lệ gốc $s_0 = 1.0$, tính **Entropy Shannon chuẩn hóa** $\mathcal{H}_{1.0}(u, v) \in [0, 1]$:
   $$\mathcal{H}_{1.0}(u, v) = -\frac{1}{\ln 4} \sum_{c=0}^{3} P_{1.0}(c, u, v) \ln\left(\max(P_{1.0}(c, u, v), 10^{-8})\right)$$
4. Điều biến trọng số các tỷ lệ phụ ($s \ne 1.0$) theo mức độ bất định của tỷ lệ gốc $G_{\text{unc}}(u, v) = 0.35 + 0.65 \cdot \left(\mathcal{H}_{1.0}(u, v)\right)^{1.2}$ và độ sắc nét cục bộ $\left(1 - \mathcal{H}_s(u, v)\right)^{0.5}$:
   $$W_s(u, v) = \begin{cases} \alpha_{1.0} & \text{nếu } s = 1.0 \\ \alpha_s \cdot G_{\text{unc}}(u, v) \cdot \left(0.65 + 0.35 \cdot (1 - \mathcal{H}_s(u, v))^{0.5}\right) & \text{nếu } s \in \{0.75, 1.25\} \end{cases}$$
5. Bản đồ xác suất đa tỷ lệ hội tụ:
   $$P_{\text{ms\_d4}}(c, u, v) = \frac{\sum_{s \in \mathcal{S}} W_s(u, v) \cdot P_s(c, u, v)}{\sum_{s \in \mathcal{S}} W_s(u, v)}$$

---

### 2.2. Trụ Cột 2: Cổng Mềm Thích Nghi Đa Ngưỡng Stage-1 D4 & Dung Hợp CAM Bảo Toàn Điểm Không (`extract_stage1_adaptive_cam_and_gate`)

1. **Tách biệt Suy luận 4 Góc nhìn D4 cho $\bar{y}_{\text{cls}}$ và Lưới Tọa độ Gốc cho $M_{\text{cam}}$:**
   - Vì bản đồ đặc trưng `ResNet38d` có kích thước chẵn $28 \times 28$ (`stride = 8`), phép lật ngang/dọc trên lưới đặc trưng gây lệch pha nửa bước nhảy ($4$ pixels) khi nội suy lên $224 \times 224$.
   - Do đó, hệ thống ghép 4 góc nhìn Dihedral $[X, \text{flip}_H(X), \text{flip}_V(X), \text{rot}_{90}(X)] \in \mathbb{R}^{4 \times 3 \times H \times W}$ để ước lượng xác suất hiện diện lớp bất biến $\bar{y}_{\text{cls}} \in [0, 1]^4$, nhưng trích xuất bản đồ không gian $M_{\text{cam}} \in \mathbb{R}_{\ge 0}^{4 \times H \times W}$ trực tiếp từ góc nhìn gốc không lệch pha.
2. **Cổng mềm thích nghi kết hợp Khôi phục Cụm Biên Cực Đại (Peak-Aware Soft-Gating):**
   Thiết lập ngưỡng cắt dưới $\tau_{\text{low}} = (0.08, 0.08, 0.045, 0.045)$, ngưỡng xác nhận chuẩn $\tau_{\text{high}} = (0.10, 0.10, 0.10, 0.10)$ và cường độ đỉnh kích hoạt $M_{\text{peak}}(c) = \max_{u,v} M_{\text{cam}}(c, u, v)$:
   - Nếu $\bar{y}_{\text{cls}, c} > 0.10$ và $M_{\text{peak}}(c) \ge 2.5$: mở cổng hoàn toàn $G_{\text{soft}}(c) = 1.0$ (tương thích 100% với những ca lớp hiện diện rõ ràng của `M2`..`M5`).
   - Nếu $\bar{y}_{\text{cls}, c} > 0.10$ nhưng $M_{\text{peak}}(c) < 2.5$ (kích hoạt nền yếu toàn cục): tiết chế cổng xuống $G_{\text{soft}}(c) = 0.25$ để chặn dương tính giả.
   - Nếu $\tau_{\text{low}, c} < \bar{y}_{\text{cls}, c} \le 0.10$ nhưng tồn tại cụm tiêu điểm sắc nét ($M_{\text{peak}}(c) \ge 8.0$ và $M_{\text{peak}}(c) \ge 0.85 \max_{k} M_{\text{peak}}(k)$): mở cổng cứu hộ $G_{\text{soft}}(c) \in [0.45, 0.90]$ và lưu cờ `borderline_mask` để chỉ giải phóng lớp đó tại đúng vùng định vị CAM ($\hat{M}_c(u, v) > 0.35$).
3. **Bản đồ xác suất CAM bảo toàn điểm không (Zero-Preserving Sharp CAM):**
   Với $\hat{M}_c(u, v) \in [0, 1]$ là CAM chuẩn hóa min-max từng kênh:
   $$S_c(u, v) = G_{\text{soft}}(c) \cdot \hat{M}_c(u, v) \cdot \exp\left(2.0 \cdot \hat{M}_c(u, v)\right)$$
   Khi $\hat{M}_c(u, v) = 0$, $S_c(u, v) = 0$ tuyệt đối (khắc phục triệt để lỗi $\exp(0) = 1.0$ của V1 mà không làm suy hao mẫu số chuẩn hóa như lũy thừa bậc cao).
4. **Dung hợp CAM điều hướng bởi Entropy Stage-2 (`apply_entropy_guided_cam_fusion`):**
   Hệ số hòa trộn $\alpha_{\text{eff}}(u, v) \in [0.15, 0.32]$ tại mỗi pixel kết hợp nền tảng neo giải phẫu của Stage-1 với độ bất định $\mathcal{H}_{\text{seg}}(u, v)$ của Stage-2 và cường độ đỉnh $C_{\text{cam}}(u, v) = \max_c \hat{M}_c(u, v)$:
   $$\alpha_{\text{eff}}(u, v) = \text{clip}\left(\alpha_{\text{base}} \cdot \left(0.75 + 0.50 \cdot \mathcal{H}_{\text{seg}}(u, v)\right) \cdot \left(0.80 + 0.35 \cdot C_{\text{cam}}(u, v)\right), 0.15, 0.32\right)$$
   $$P_{\text{fused}}(c, u, v) = \left(1 - \alpha_{\text{eff}}(u, v)\right) P_{\text{seg}}(c, u, v) + \alpha_{\text{eff}}(u, v) P_{\text{cam}}(c, u, v)$$

---

### 2.3. Trụ Cột 3: Giải Chập Quang Học Beer-Lambert H&E & Hiệu Chỉnh Tiên Nghiệm Mô Học (`apply_he_optical_density_calibration`)

Dựa trên định luật hấp thụ quang học **Beer-Lambert** và ma trận giải chập màu mô bệnh học chuẩn của **Ruifrok & Johnston (2001)**:

1. **Chuyển đổi cường độ RGB sang Mật độ Quang học (Optical Density - OD):**
   $$OD(u, v, k) = -\log_{10}\left(\frac{\max(I(u, v, k), 1.0)}{255.0}\right), \quad k \in \{R, G, B\}$$
2. **Giải chập nồng độ thuốc nhuộm Hematoxylin ($C_H$) và Eosin ($C_E$):**
   Sử dụng vector hấp thụ chuẩn hóa của Hematoxylin $\mathbf{h} = [0.650, 0.704, 0.286]^T$, Eosin $\mathbf{e} = [0.072, 0.990, 0.105]^T$, và vector trực giao $\mathbf{r} = \frac{\mathbf{h} \times \mathbf{e}}{\|\mathbf{h} \times \mathbf{e}\|_2}$, ma trận giải chập $D_{HE} = ([\mathbf{h}, \mathbf{e}, \mathbf{r}]^T)^{-1}$ tách mật độ quang học thành:
   $$[C_H(u, v), C_E(u, v), C_R(u, v)] = \max\left(0, OD(u, v) \cdot D_{HE}\right)$$
3. **Tỷ số mật độ Nhân / Bào tương - Mô đệm cục bộ ($R_{H/E}$):**
   Làm mịn $C_H, C_E$ bằng nhân Gauss kích thước tế bào ($\sigma = 2.0$) và chuẩn hóa phân vị bền vững về $[0, 1]$ thu được $\tilde{H}(u, v), \tilde{E}(u, v)$ cùng tỷ số nhân/bào tương:
   $$R_{H/E}(u, v) = \frac{\tilde{H}(u, v)}{\tilde{H}(u, v) + \tilde{E}(u, v) + 0.08} \in [0, 1]$$
4. **Điều biến trọng số tiên nghiệm cân bằng theo mật độ quang học mô học ($\lambda_{\text{OD}} = 0.08$):**
   - **Đối với Bạch huyết bào (`LYM`, lớp 2) vs Mô đệm (`STR`, lớp 1):**
     $$W_{\text{LYM}}(u, v) = w_{\text{LYM}, 0} \cdot \left(1 + \lambda_{\text{OD}} \cdot (1.20 R_{H/E}(u, v) - 0.50)\right)$$
     $$W_{\text{STR}}(u, v) = w_{\text{STR}, 0} \cdot \left(1 + 0.45 \lambda_{\text{OD}} \cdot (0.55 - R_{H/E}(u, v))\right)$$
     Tại vùng nhân tím đậm ($R_{H/E} > 0.55$), $W_{\text{LYM}}$ tăng lên để cứu các điểm ảnh `LYM -> STR`. Ngược lại, tại vùng sợi collagen hồng nhạt không có nhân ($R_{H/E} < 0.30$), $W_{\text{LYM}}$ giảm nhẹ và $W_{\text{STR}}$ được củng cố, ngăn chặn dương tính giả `STR -> LYM`.
   - **Đối với Hoại tử (`NEC`, lớp 3) vs Khối u (`TUM`, lớp 0):**
     Tại các vùng hoại tử đông có hiện tượng tiêu nhân ($\tilde{E}(u, v) > \tilde{H}(u, v)$) hoặc bất định phân vùng cao, hệ số tiên nghiệm $W_{\text{NEC}}(u, v)$ được củng cố để khắc phục lỗi `NEC -> TUM`.

---

### 2.4. Trụ Cột 4: Tinh Chỉnh Biên Đa Bước Nhảy & Triệt Tiêu Đảo Nhiễu Phân Lớp Đặc Thù (`apply_advanced_spatial_refinement`)

1. **Lọc song phương vượt bước nhảy nội suy Decoder (`d = 9, sigmaColor = 0.18, sigmaSpace = 6.0`):**
   Mở rộng đường kính vùng lân cận lên $d = 9$ pixels (vượt qua kích thước khối nội suy $4 \times 4$ của nhánh Decoder DeepLabV3+), làm mịn đồng nhất bên trong vùng mô đồng thời giữ sắc nét biên chuyển tiếp xác suất.
2. **Khuếch tán tương đồng cục bộ dẫn hướng bởi biên quang học H&E (OD-Guided Local Affinity Refinement):**
   Khi có ảnh gốc `raw_rgb`, ma trận độ tương đồng cục bộ $3 \times 3$ dựa trên gradient mật độ quang học $\nabla OD$ truyền xác suất dọc theo ranh giới mô học thực tế, xóa bỏ quầng mờ (bilinear halo) quanh các ổ u và cụm lympho.
3. **Triệt tiêu đảo nhiễu phân lớp đặc thù (Class-Aware Confidence-Guided Island Suppression):**
   Quét các thành phần liên thông 8 hướng có diện tích $0 < |\mathcal{C}_k| < 96$ pixels. Để bảo vệ tuyệt đối các cụm tế bào hiếm (`LYM`, `NEC` vốn có xác suất chiến thắng tự nhiên trong khoảng $0.45..0.65$), chỉ những vi đảo cực nhỏ $|\mathcal{C}_k| < 36$ pixels HOẶC các đảo thuộc lớp đa số (`TUM`, `STR`) có diện tích $36 \le |\mathcal{C}_k| < 96$ kèm độ tự tin trung bình thấp $\bar{P}(\mathcal{C}_k) < 0.50$ mới bị tái gán sang phân lớp có xác suất cao thứ nhì.

---

## 3. Thực Nghiệm Kiểm Chứng Định Lượng Trên 6 Ca Bệnh Lâm Sàng BCSS-WSSS Thực Tế

Đánh giá đối chứng trực tiếp trên 6 mảng ảnh kiểm thử lâm sàng thực tế thuộc 6 bệnh nhân độc lập từ tập dữ liệu BCSS-WSSS (`TCGA-E2-A1LL`, `TCGA-D8-A27F`, `TCGA-E2-A14N`, `TCGA-E2-A14R`, `TCGA-E2-A14X`, `TCGA-D8-A27H`) sử dụng trọng số thực tế `stage1_checkpoint_trained_on_bcss.pth` và `stage2_checkpoint_trained_on_bcss.pth`:

| Mã | Phương Pháp Nghiên Cứu | mIoU 4 Lớp (%) | PA 4 Lớp (%) | TUM IoU (%) | STR IoU (%) | LYM IoU (%) | NEC IoU (%) | Chênh Lệch mIoU vs M1 | Chênh Lệch mIoU vs M5 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1** | `M1_Baseline_SingleScale` | 40.54% | 67.96% | 51.61% | 55.08% | 51.33% | 4.13% | Baseline | -8.21% |
| **M2** | `M2_MultiView_D4_TTA` | 44.04% | 69.19% | 52.34% | 55.96% | 54.73% | 13.15% | +3.50% | -4.71% |
| **M3** | `M3_LongTail_Prior_Calibration` | 48.77% | 69.83% | 53.12% | 55.17% | 56.04% | 30.75% | +8.23% | +0.02% |
| **M4** | `M4_Spatial_Bilateral_Refinement` | 44.03% | 69.21% | 52.37% | 55.98% | 54.71% | 13.06% | +3.49% | -4.72% |
| **M5** | `M5_Proposed_SOTA_Ensemble` (V1) | 48.75% | 69.81% | 53.12% | 55.15% | 55.99% | 30.73% | +8.21% | Baseline V1 |
| **M6** | **`M6_MultiScale_Pyramid_D4_Ensemble` (V2)** | **49.42%** | **69.85%** | **53.32%** | **54.87%** | **55.94%** | **33.53%** | **+8.88%** | **+0.67%** |
| **M7** | **`M7_Proposed_Ultra_SOTA_Curriculum` (V2)** | **51.88%** | **70.31%** | **53.66%** | **55.08%** | **55.78%** | **43.01%** | **+11.34%** | **+3.13%** |

### 3.1. Phân Tích Đột Phá Lâm Sàng Của `M6` & `M7`

1. **Sự tăng trưởng đơn điệu qua từng thế hệ (`M1 < M2 < M5 < M6 < M7`):**
   - `M6_MultiScale_Pyramid_D4_Ensemble` đạt **49.42% mIoU** và **69.85% PA**, vượt qua `M5_Proposed_SOTA_Ensemble` (**+0.67% mIoU**, **+2.80% NEC IoU**, **+0.20% TUM IoU**) nhờ cơ chế dung hợp kim tự tháp đa tỷ lệ điều hướng bởi bất định tỷ lệ gốc (`1.25x` mở rộng trường tiếp nhận `ASPP` từ $14 \times 14$ lên $18 \times 18$, giải quyết triệt để hiện tượng suy biến đệm không).
   - `M7_Proposed_Ultra_SOTA_Curriculum` đạt đỉnh cao mới với **51.88% mIoU** và **70.31% PA**, vượt xa `M5` tới **+3.13% mIoU** và **+0.50% PA** (**+11.34% mIoU** so với `M1` Baseline)!
2. **Khôi phục ngoạn mục các ổ hoại tử biên (`NEC IoU` tăng từ `30.73%` ở `M5` lên `43.01%` ở `M7`, tức `+12.28%`):**
   - Điển hình trên ca bệnh khó `TCGA-E2-A14X-DX1_xmin57515_ymin72665` (chứa ổ hoại tử `NEC` lớn nhưng xác suất phân loại toàn cục Stage-1 chỉ đạt $\bar{y}_{\text{cls}, 3} = 0.0801 \le 0.10$ nên bị cổng cứng của `M2..M5` cắt về `0`), cơ chế **Peak-Aware Soft-Gating** kết hợp **Zero-Preserving Sharp CAM** ($M_{\text{peak}}(3) = 12.11$) và **Giải chập Quang học Beer-Lambert H&E** trong `M7` đã khôi phục chính xác vùng hoại tử, nâng độ chính xác điểm ảnh trên ca bệnh này từ **52.66%** (`M5`) lên **59.40%** (`M7`, tăng **+6.74%**).

---

## 4. Kiến Trúc Tích Hợp Độc Lập & Tương Thích Ngược Tuyệt Đối

| Thuộc Tính / Phương Thức | `MultiMethodWSSSEngine` (Mặc định V1) | `MultiMethodWSSSEngine(include_v2=True)` / `MultiMethodWSSSEngineV2` |
| :--- | :--- | :--- |
| **Danh sách phương pháp (`METHOD_KEYS`)** | `M1` .. `M5` (5 phương pháp) | `M1` .. `M7` (7 phương pháp) |
| **Tương thích `tests/test_colab_notebook.py`** | **100% PASS** (bit-exact với Notebook V1) | **100% PASS** |
| **Tương thích `src/colab_pkg/colab_runner.py`** | **100% PASS** (đầy đủ `load_wsss_models`, `IMAGENET_MEAN/STD`, `ResBlockStage1`) | **100% PASS** |
| **Tái sử dụng tính toán GPU** | Chạy 1 lượt D4 ở scale `1.0x` cho cả `M1`..`M5` | Tái sử dụng lượt D4 ở scale `1.0x` của `M1`..`M5` cho `M6` & `M7` |
| **Tích hợp Stage-1 & Stage-2 Loaders** | `WSSSTissueStage1NetCAM`, `build_and_load_stage1_netcam` | Bổ sung đầy đủ `DeepLabV3Plus`, `build_and_load_deeplabv3plus`, `OtsuClinicalGating`, `load_wsss_models` |

