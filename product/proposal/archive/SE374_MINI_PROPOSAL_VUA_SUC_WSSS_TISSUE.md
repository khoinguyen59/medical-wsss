---
title: "Mini Proposal: Phân Vùng Ngữ Nghĩa Giám Sát Yếu Trên Ảnh Tiêu Bản Mô Bệnh Học Dựa Trên Mạng Nơ-ron Đa Tầng Và Cổng Phân Loại (Weakly-Supervised Histopathology Segmentation via MLPS and CGM)"
author:
  - "[Họ và tên Sinh viên 1 - MSSV: ...]"
  - "[Họ và tên Sinh viên 2 - MSSV: ...]"
date: "Tháng 10/2026"
---

# Mini Proposal

## A. Information

- **Project / Tên đề tài:** Weakly-Supervised Histopathology Tissue Segmentation via Multi-Layer Pseudo-Supervision and Classification Gate Mechanism (Phần Mềm Hỗ Trợ Phân Vùng Ngữ Nghĩa Giám Sát Yếu Trên Ảnh Tiêu Bản Mô Bệnh Học Dựa Trên Mạng Nơ-ron Đa Tầng Và Cổng Phân Loại)
- **Course / Môn học:** SE374.R11 - Phát Triển Phần Mềm Xử Lý Ảnh Y Khoa (Học kỳ 1, Năm học 2026-2027)
- **Faculty / Khoa:** Khoa Kỹ thuật Phần mềm (Software Engineering), Trường Đại học Công nghệ Thông tin - ĐHQG-HCM (UIT)
- **Duration / Thời gian thực hiện:** 10/2026 - 12/2026 (10 tuần)
- **Group / Nhóm:** [Nhóm Sinh Viên SE374]
- **Members / Thành viên:**
  1. [Họ và tên Sinh viên 1] - MSSV: [MSSV 1] - Email: [Email UIT 1]
  2. [Họ và tên Sinh viên 2] - MSSV: [MSSV 2] - Email: [Email UIT 2]
- **Supervisor / Giảng viên hướng dẫn:** TS. Bùi Cao Doanh (Giảng viên phụ trách môn học SE374.R11, UIT)

---

## B. Content

### 1. Mini Problem Formulation

#### 1.1 Project Description
Đề tài tập trung nghiên cứu, tái lập và hoàn thiện giải pháp phân vùng ngữ nghĩa giám sát yếu (Weakly-Supervised Semantic Segmentation - WSSS) cho ảnh mô bệnh học ung thư dựa trên kiến trúc hai giai đoạn tách rời: Bộ sinh nhãn giả chú ý lũy tiến (Progressive Dropout Attention - PDA) và Mạng phân vùng giám sát giả đa tầng kết hợp Cổng phân loại (Multi-Layer Pseudo-Supervision and Classification Gate Mechanism - MLPS & CGM). Trong thực hành giải phẫu bệnh, việc gán nhãn thủ công từng điểm ảnh (pixel-level mask) trên các tiêu bản mô học đòi hỏi hàng trăm giờ làm việc của bác sĩ và gây tốn kém chi phí y tế rất lớn [Bejnordi et al., 2017; Han et al., 2022]. Để giải quyết bài toán này một cách vừa sức, chắc chắn và khả thi trong khuôn khổ 10 tuần, nhóm kế thừa trực tiếp mã nguồn mở đã kiểm chứng từ công trình của Han et al. (*Medical Image Analysis 2022*), tập trung vào bộ dữ liệu chuẩn BCSS-WSSS (Ung thư vú). Nhóm đề xuất cải tiến nhẹ ở khâu điều hòa ranh giới nhãn giả bằng cơ chế trộn mẫu mô (Tissue Data Mixing) và đóng gói thành một ứng dụng phần mềm phân tích mô học trực quan, hỗ trợ bác sĩ chẩn đoán nhanh tỷ lệ mô u sống và diện tích tổn thương.

#### 1.2 Assumptions, Constraints, Scope & Out of Scope
- **Assumptions (Giả định khoa học):**
  - Dữ liệu đầu vào là các mảng ảnh mô bệnh học cắt lát $224 \times 224$ pixels từ tiêu bản nhuộm H&E ở độ phóng đại chuẩn hóa $20\times$ ($0.5\,\mu\text{m/pixel}$) [Amgad et al., 2019].
  - Trong quá trình huấn luyện, mô hình chỉ được cung cấp nhãn phân loại nhị phân đa lớp ở cấp độ mảng ảnh ($y \in \{0, 1\}^C$), tuyệt đối không sử dụng mặt nạ vẽ tay cấp điểm ảnh [Han et al., 2022].
- **Constraints (Ràng buộc kỹ thuật & học thuật):**
  - **Ràng buộc chống rò rỉ bệnh nhân (Patient-Level Anti-Leakage):** Quá trình chia tách tập dữ liệu huấn luyện, kiểm định và kiểm thử (`train / val / test`) bắt buộc phân chia triệt để theo Mã định danh bệnh nhân (Patient ID). Không trộn lẫn các mảng ảnh của cùng một bệnh nhân giữa các tập dữ liệu.
  - **Ràng buộc phần cứng vừa sức:** Toàn bộ pipeline huấn luyện và đánh giá được thiết kế để vận hành trơn tru trên 1 GPU đơn phổ thông (NVIDIA RTX 3060/4060 từ 8GB đến 12GB VRAM), hoàn toàn không đòi hỏi cụm máy chủ đa GPU hay các mô hình nền tảng tỷ tham số.
- **Scope (Phạm vi đồ án):**
  - Kế thừa và tái hiện mã nguồn thực nghiệm của WSSS-Tissue trên bộ dữ liệu chuẩn quốc tế BCSS-WSSS (151 vùng mô u, 5 phân lớp: Biểu mô u, Mô đệm, Lympho bào, Hoại tử, và Khác).
  - Nghiên cứu và thử nghiệm cải tiến ở khâu sinh mặt nạ giả (thử nghiệm cơ chế lọc ngưỡng thích ứng và trộn mẫu biên).
  - Xây dựng phần mềm giao diện Web ứng dụng (Web-based Clinical Demo) cho phép bác sĩ tải ảnh mảng, hiển thị bản đồ nhiệt CAM, dự đoán mặt nạ phân tầng đa lớp và tự động tính tỷ lệ diện tích các thành phần mô.
- **Out of Scope (Ngoài phạm vi):**
  - Không huấn luyện lại các mô hình nền tảng (Foundation Models) quy mô gigapixel hàng trăm triệu tham số (như UNI, CONCH) do vượt quá năng lực tính toán và thời gian của đồ án môn học.
  - Không can thiệp vào phần cứng máy quét tiêu bản kính hiển vi quang học.

#### 1.3 Existing Approaches & Research Gap
- **Hạn chế của phương pháp Class Activation Maps (CAM) truyền thống:** Các giải pháp phân loại yếu kinh điển (CAM, Grad-CAM, HistoSegNet) thường chỉ kích hoạt được vùng trung tâm tế bào có tính phân biệt cao nhất (discriminative regions), dẫn đến mặt nạ giả bị co ngót và bỏ sót các dải ranh giới mô đệm và lympho xâm lấn [Chan et al., 2019; Han et al., 2022].
- **Thách thức của các mô hình SOTA phức tạp:** Các công trình gần đây (HisynSeg 2025, ConStruct 2026, PathMamba 2024) tuy đạt chỉ số mIoU cao hơn nhưng đòi hỏi khối lượng tài nguyên tính toán lớn, thời gian huấn luyện dài ngày và phụ thuộc vào các trọng số Foundation Model khó cấu hình trên máy trạm sinh viên.
- **Khoảng trống thực tiễn:** Thiếu một phiên bản phần mềm gọn nhẹ, dễ triển khai cục bộ, trực quan hóa rõ ràng cơ chế hoạt động của từng tầng mạng nơ-ron (từ bản đồ đặc trưng CAM đến cổng phân loại CGM) phục vụ học tập và ứng dụng thực hành.

#### 1.4 Research Questions & Hypotheses
- **RQ1 (Nghiên cứu Giải thuật AI):** Liệu việc áp dụng mạng ResNet-38 kết hợp cơ chế Progressive Dropout Attention (PDA) và Cổng phân loại CGM có bảo đảm khả năng tái lập chỉ số mIoU $\ge 64.0\%$ trên bộ dữ liệu BCSS-WSSS khi chỉ sử dụng 1 GPU máy trạm đơn hay không?
  - *Giả thuyết H1:* Cơ chế PDA buộc mạng phải chú ý đến các vùng đặc trưng thứ cấp ngoài vùng lõi, kết hợp cùng CGM triệt tiêu hoàn toàn bóng mờ dương tính giả, giúp mô hình đạt mIoU $\ge 64.2\%$ tương đương công bố của Han et al. (*MedIA 2022*).
- **RQ2 (Ứng dụng Kỹ thuật Phần mềm):** Làm thế nào để đóng gói toàn bộ pipeline 2 giai đoạn phức tạp thành một công cụ phần mềm dễ dùng, hỗ trợ bác sĩ xem nhanh kết quả phân vùng trong thời gian $<2$ giây/mảng ảnh?
  - *Giả thuyết H2:* Việc xuất mô hình sang TorchScript và xây dựng giao diện tương tác bằng FastAPI kết hợp Web UI gọn nhẹ sẽ tối ưu hóa thời gian suy luận xuống dưới $1.5$ giây/ảnh, đáp ứng tốt nhu cầu thử nghiệm lâm sàng.

#### 1.5 Objectives & Potential Contributions
- **Mục tiêu 1:** Tái lập thành công 100% pipeline mã nguồn mở WSSS-Tissue trên BCSS-WSSS, đạt mIoU $\ge 64.20\%$ và mDice $\ge 77.10\%$, bảo đảm tính ổn định qua 5 random seeds.
- **Mục tiêu 2:** Thử nghiệm cải tiến khâu tiền xử lý hoặc điều hòa ranh giới nhãn giả, hướng tới cải thiện thêm $+1.0\%$ đến $+1.5\%$ mIoU so với bản gốc.
- **Mục tiêu 3:** Phát triển ứng dụng Web hoàn chỉnh hỗ trợ bác sĩ tải ảnh, xem phân tầng mô học và tự động tính toán tỷ lệ tải lượng u (Viable Tumor Percentage).

---

### 2. Related Work

| Phương Pháp / Công Trình | Xuất Bản / Tác Giả | Bản Chất Kỹ Thuật | Điểm Mạnh | Hạn Chế |
| :--- | :--- | :--- | :--- | :--- |
| **HistoSegNet** | Chan et al., *ICCV 2019* | Phân loại patch ảnh kết hợp Grad-CAM để sinh mặt nạ giả phân tầng mô. | Đơn giản, tốc độ suy luận nhanh. | Mặt nạ CAM bị co ngót nghiêm trọng; mIoU chỉ đạt $51.20\%$. |
| **WSSS-Tissue (Nền tảng chọn lọc)** | Han et al., *MedIA 2022* | Pipeline 2 giai đoạn: Giai đoạn 1 huấn luyện ResNet-38 với Progressive Dropout (PDA); Giai đoạn 2 huấn luyện DeepLabV3+ bằng Giám sát giả đa tầng (MLPS) kết hợp Cổng phân loại (CGM). | **Mã nguồn hoàn chỉnh, cấu hình tối ưu:** Giảm $32.7\times$ thời gian gán nhãn; mIoU đạt **$64.20\%$**, mDice **$77.10\%$** trên BCSS. | Chưa tích hợp bộ trộn dữ liệu biên mô học. |
| **PistoSeg / HisynSeg** | Fang et al., *AAAI 2023 / IEEE TMI 2025* | Tổng hợp ảnh mô học ghép (Mosaic / Bézier curves) từ các patch thuần nhất để tạo ranh giới giả định. | Ranh giới cong tự nhiên, nâng mIoU lên $70.8\%$. | Thuật toán tổng hợp ảnh tương đối phức tạp, thời gian huấn luyện dài. |
| **ConStruct** | Le et al., *MIDL 2026* | Chưng cất cấu trúc từ Foundation Model CONCH sang SegFormer qua Adapters. | SOTA kỷ lục $70.96\%$ mIoU. | Phụ thuộc hoàn toàn vào trọng số CONCH thương mại/hạn chế truy cập. |

---

### 3. Data

Đồ án tập trung toàn diện vào bộ dữ liệu chuẩn mực được sử dụng rộng rãi nhất trong lĩnh vực:

- **Tên bộ dữ liệu:** **BCSS-WSSS (Breast Cancer Semantic Segmentation - Weakly Supervised)**.
- **Nguồn gốc:** Trích xuất từ kho dữ liệu quốc tế TCGA-BRCA, chuẩn hóa bởi Amgad et al. (*Bioinformatics 2019*) và Han et al. (*MedIA 2022*).
- **Quy mô:** 151 vùng mô quan tâm (ROIs) cắt từ các tiêu bản giải phẫu bệnh ung thư vú ở độ phóng đại $20\times$. Tổng cộng $17.286$ mảng ảnh kích thước $224 \times 224$ pixels.
- **Danh mục 5 phân lớp mô học:**
  1. *Biểu mô u (Tumor Epithelium - TUM):* Vùng tế bào ác tính xâm lấn.
  2. *Mô đệm (Stroma - STR):* Vùng mô liên kết nâng đỡ vi môi trường u.
  3. *Lympho bào (Inflammatory / Lymphocytic Infiltrate - LYM):* Tế bào miễn dịch thâm nhiễm.
  4. *Hoại tử (Necrosis - NEC):* Mô tế bào chết do thiếu máu nuôi dưỡng.
  5. *Khác (Other normal / background - OTR):* Mô mỡ, mạch máu và chất nền bình thường.
- **Phân chia dữ liệu chống rò rỉ bệnh nhân (Patient-Level Split):**
  - **Tập Huấn luyện (Train):** 106 ca bệnh ($23.422$ patches với nhãn phân loại nhị phân đa lớp cấp patch).
  - **Tập Kiểm định (Val):** 20 ca bệnh ($3.418$ patches có mặt nạ chuẩn pixel ground truth).
  - **Tập Kiểm thử (Test):** 25 ca bệnh ($4.986$ patches có mặt nạ chuẩn pixel ground truth của bác sĩ).
  - *Nguyên tắc bất biến:* Không có bất kỳ mảng ảnh nào của cùng một bệnh nhân xuất hiện ở nhiều tập khác nhau.

---

### 4. Evaluation

#### 4.1 Tiêu chí Đo lường Lâm sàng & Kỹ thuật
- **Đo lường giá trị thực tế lâm sàng:**
  - **Tiết kiệm thời gian đọc ca bệnh:** Giảm thời gian khoanh vùng tiêu bản từ $\sim 180$ phút xuống $\le 6$ phút cho mỗi 100 mảng ảnh ($>30\times$ speedup).
  - **Độ trễ suy luận ứng dụng:** Thời gian phản hồi suy luận trên giao diện Web $<1.5$ giây cho mỗi mảng ảnh $224 \times 224$ px.
  - **Tự động định lượng tải lượng u:** Tự động tính toán diện tích u ($mm^2$) và tỷ lệ phần trăm mô u sống (Viable Tumor Score).
- **Đo lường định lượng trong phòng thí nghiệm:**
  - **Mean Intersection over Union (mIoU):**
    $$\text{mIoU} = \frac{1}{C} \sum_{c=1}^C \frac{|P_c \cap G_c|}{|P_c \cup G_c|}$$
  - **Mean Dice Similarity Coefficient (mDice):**
    $$\text{mDice} = \frac{1}{C} \sum_{c=1}^C \frac{2 |P_c \cap G_c|}{|P_c| + |G_c|}$$
  - Đánh giá trên từng phân lớp cụ thể: IoU_TUM, IoU_STR, IoU_LYM, IoU_NEC.

#### 4.2 Giao thức Thực nghiệm Khách quan
- Cố định **5 Random Seeds** (`seeds = [42, 100, 2024, 2025, 2026]`), báo cáo kết quả trung bình và độ lệch chuẩn ($\mu \pm \sigma$).
- 100% số liệu thực nghiệm được lưu vết trong thư mục `experiments/logs/`.

---

### 5. Baselines & Benchmark Evidence

Bảng số liệu đối chuẩn thực nghiệm trên tập kiểm thử BCSS-WSSS (trích dẫn nguyên bản từ bài báo Han et al., *MedIA 2022*):

| Phương Pháp Thực Nghiệm | Dạng Giám Sát | Kiến Trúc Mô Hình | BCSS-WSSS (mIoU %) | BCSS-WSSS (mDice %) | Nguồn Kiểm Chứng |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Fully Supervised Baseline** | Giám sát đầy đủ (Pixel masks) | DeepLabV3+ | $72.80$ | $84.10$ | Han et al., *MedIA 2022* |
| **HistoSegNet** | Giám sát yếu (Patch labels) | CNN + Grad-CAM | $51.20$ | $64.30$ | Chan et al., *ICCV 2019* |
| **CAM Baseline (bn7)** | Giám sát yếu (Patch labels) | ResNet-38 | $58.40$ | $71.20$ | Han et al., *MedIA 2022* |
| **WSSS-Tissue (MLPS Full - Core)** | Giám sát yếu (Patch labels) | ResNet-38 + DeepLabV3+ | **$64.20$** | **$77.10$** | Han et al., *MedIA 2022* |
| **Mục tiêu đồ án vừa sức** | Giám sát yếu (Patch labels) | WSSS-Tissue + Tissue Mixing | **$\ge 65.50$** | **$\ge 78.50$** | Đề xuất nhóm SE374 |

---

### 6. Proposed Method

Nhóm đề xuất giải pháp trọng tâm, khả thi và bám sát mã nguồn mở chính thức: **Mô hình Phân vùng Ngữ nghĩa Giám sát Yếu Hai Giai đoạn (WSSS-Tissue Pipeline)** kết hợp **Giao diện Web Hỗ trợ Chẩn đoán Phân tầng Mô học (Pathology Assistant Web App)**.

#### 6.1 Sơ Đồ Kiến Trúc Quy Trình Hai Giai Đoạn

![Hình 1: Sơ đồ kiến trúc phương pháp phân vùng ngữ nghĩa giám sát yếu WSSS-Tissue (Trích nguồn chính thức: Chu Han et al., Medical Image Analysis 2022, DOI: 10.1016/j.media.2022.102487)](product/proposal/fig1_wsss_tissue_workflow.png)

Kiến trúc phương pháp (Hình 1) được tổ chức thành 2 giai đoạn tách rời rõ ràng:

1. **Giai đoạn 1: Sinh Mặt Nạ Giả Bằng Chú Ý Lũy Tiến (Weakly-supervised Pseudo Mask Generation - Hình 1a):**
   - Mạng trích xuất đặc trưng là ResNet-38 tiếp nhận mảng ảnh $x$.
   - Áp dụng cơ chế **Progressive Dropout Attention (PDA - ký hiệu D)**: thay vì chỉ kích hoạt một vài nốt tế bào nổi trội, mạng loại bỏ có chủ đích các vùng kích hoạt cực đại ở các bước lặp trước để buộc mô hình mở rộng trường chú ý sang toàn bộ diện tích mô liên quan.
   - Trích xuất nhãn giả đa tầng ở 3 mức phân giải khác nhau: tầng $b4\_5$, tầng $b5\_2$, và tầng chuẩn hóa $bn7$.
2. **Giai đoạn 2: Huấn Luyện Mạng Phân Vùng Đa Tầng Và Cổng Phân Loại (Pseudo-supervised Tissue Semantic Segmentation - Hình 1b):**
   - Mạng phân vùng ngữ nghĩa DeepLabV3+ tiếp nhận mảng ảnh đầu vào và dự đoán bản đồ xác suất cho từng phân lớp.
   - Tích hợp **Cổng Phân Loại (Classification Gate Mechanism - CGM - ký hiệu G)**: lấy vector xác suất nhãn cấp mảng $\hat{y}_k$ từ Giai đoạn 1 làm ngưỡng điều kiện. Nếu một loại mô (ví dụ: hoại tử) không xuất hiện trong mảng ảnh ($y_k = 0$), CGM sẽ ép xác suất vùng tương ứng về 0, loại bỏ hoàn toàn hiện tượng dương tính giả.
   - Hàm mất mát giám sát giả đa tầng kết hợp: $\mathcal{L}_{total} = \mathcal{L}_{bn7} + \alpha \mathcal{L}_{b5\_2} + \beta \mathcal{L}_{b4\_5}$.
3. **Cải tiến Trọng tâm Vừa sức của Nhóm:**
   - Bổ sung kỹ thuật trộn mẫu mô học (Tissue Patch Mixing) nhẹ nhàng trong khâu nạp dữ liệu của Giai đoạn 2 để làm mượt các đường biên tiếp giáp giữa mô u và mô đệm.

#### 6.2 Kiến Trúc Phần Mềm Ứng Dụng (Software Deliverable)
- **Tầng Backend:** Dịch vụ API viết bằng FastAPI, nạp trọng số mô hình đã huấn luyện (TorchScript), cung cấp 2 endpoint chính: `/api/predict_cam` và `/api/predict_segmentation`.
- **Tầng Frontend:** Giao diện Web HTML5/CSS/JavaScript thuần, cho phép kéo thả ảnh mảng mô học, hiển thị song song ảnh gốc H&E, bản đồ nhiệt chú ý CAM, và mặt nạ phân tầng đa màu sắc.
- **Module Báo cáo:** Tự động tính toán diện tích u ($mm^2$) và tỷ lệ mô u sống, hiển thị biểu đồ tròn phân bố mô học.

---

### 7. Expected Results

#### 7.1 Kết Quả Định Lượng Kỳ Vọng
- Tái lập thành công Baseline WSSS-Tissue trên BCSS-WSSS với mIoU $\ge 64.20\%$, mDice $\ge 77.10\%$.
- Cải tiến nhẹ đạt mIoU $\ge 65.50\%$ ($+1.3\%$ cải thiện so với baseline).
- Thời gian suy luận $<1.5$ giây/ảnh trên GPU NVIDIA đơn.

#### 7.2 Sản Phẩm Bàn Giao
1. Mã nguồn huấn luyện và kiểm thử sạch sẽ, có tài liệu hướng dẫn chạy lệnh trong `experiments/`.
2. Ứng dụng Web tương tác chạy cục bộ cho phép bác sĩ thử nghiệm phân vùng mô bệnh học.
3. Báo cáo kỹ thuật cuối kỳ đầy đủ số liệu thực nghiệm và slide thuyết trình 16:9.

#### 7.3 Kế Hoạch Dự Phòng
- Nếu phần mở rộng trộn dữ liệu không tăng hiệu năng như kỳ vọng, nhóm sẽ dùng 100% pipeline gốc của WSSS-Tissue (đã kiểm chứng $64.2\%$ mIoU) làm sản phẩm bảo vệ chính, tập trung hoàn thiện giao diện phần mềm và phân tích lỗi y khoa chuyên sâu. Đồ án bảo đảm an toàn tuyệt đối về mặt tiến độ.

---

## References

1. **Chu Han, Jiatai Lin, Jinhai Mai, et al.** "Multi-Layer Pseudo-Supervision for Histopathology Tissue Semantic Segmentation using Patch-level Classification Labels." *Medical Image Analysis (MedIA)*, vol. 80, p. 102487, 2022. DOI: `10.1016/j.media.2022.102487`.
2. **Mohamed Amgad, Habiba Elfandy, et al.** "Structured crowdsourcing enables pan-cancer analysis of data quality and usability in image segmentation." *Bioinformatics*, vol. 35, no. 18, pp. 3468-3477, 2019. DOI: `10.1093/bioinformatics/btz083`.
3. **Loris Nanni, et al.** "HistoSegNet: Semantic Segmentation of Histological Images using Deep Learning." *ICCV Workshops*, 2019. DOI: `10.1109/ICCVW.2019.00100`.
4. **Zijie Fang, Yifeng Wang, et al.** "HisynSeg: Weakly-Supervised Histopathological Image Segmentation via Image-Mixing Synthesis and Consistency Regularization." *IEEE Transactions on Medical Imaging (TMI)*, vol. 44, no. 4, pp. 1765-1782, 2025. DOI: `10.1109/TMI.2024.3520129`.
5. **Babak Ehteshami Bejnordi, et al.** "Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer (CAMELYON16)." *JAMA*, vol. 318, no. 22, pp. 2199-2210, 2017. DOI: `10.1001/jama.2017.14580`.

---

## Appendix: Kế Hoạch Tiến Độ 10 Tuần Thực Tế (Gantt Chart Vừa Sức)

| Tuần | Hạng Mục Công Việc Cụ Thể | Sản Phẩm Đầu Ra | Phụ Trách |
| :---: | :--- | :--- | :---: |
| **T01** | Nghiên cứu bài báo WSSS-Tissue, thiết lập môi trường PyTorch, hoàn thiện bản Mini Proposal. | Tài liệu nghiên cứu, File Mini Proposal. | [Cả nhóm] |
| **T02** | Nạp bộ dữ liệu BCSS-WSSS, lập trình module kiểm soát phân chia chống rò rỉ bệnh nhân (`patient_split.py`). | Bộ dữ liệu sạch, Unit test pass 100%. | [Sinh viên 2] |
| **T03** | Chạy Sanity Run Giai đoạn 1 trên batch nhỏ để kiểm tra GPU VRAM, khởi chạy huấn luyện ResNet-38 với PDA. | Script huấn luyện Stage 1 chạy ổn định. | [Sinh viên 1] |
| **T04** | Hoàn thành huấn luyện Giai đoạn 1, sinh và xuất toàn bộ nhãn giả đa tầng (MLPS: $b4\_5$, $b5\_2$, $bn7$). | Tập nhãn giả `pseudo_masks/` hoàn chỉnh. | [Sinh viên 1] |
| **T05** | Khởi chạy huấn luyện Giai đoạn 2: Mạng DeepLabV3+ kết hợp Cổng phân loại CGM trên tập nhãn giả. | Checkpoint mô hình phân vùng Stage 2. | [Sinh viên 1] |
| **T06** | Thử nghiệm cải tiến khâu trộn dữ liệu biên mô học (Tissue Mixing) để tinh lọc ranh giới. | Bảng đối chứng thực nghiệm sơ bộ. | [Sinh viên 1] |
| **T07** | Xây dựng API dịch vụ bằng FastAPI và giao diện Web tương tác hiển thị CAM và mặt nạ phân tầng. | Ứng dụng Web Demo chạy cục bộ. | [Sinh viên 2] |
| **T08** | Lập trình tính năng tính toán tải lượng u ($mm^2$) và xuất bảng tổng kết kết quả chẩn đoán mô học. | Tính năng định lượng hoạt động chính xác. | [Sinh viên 2] |
| **T09** | Chạy đánh giá định lượng trên 5 Random Seeds ($\mu \pm \sigma$), phân tích các ca bệnh dự đoán sai. | Bảng số liệu mIoU, Dice đối chuẩn chi tiết. | [Cả nhóm] |
| **T10** | Hoàn thiện Báo cáo tổng kết đồ án, đóng gói mã nguồn và chuẩn bị Slide thuyết trình Marp 16:9. | Bộ hồ sơ đồ án cuối kỳ hoàn chỉnh. | [Cả nhóm] |
