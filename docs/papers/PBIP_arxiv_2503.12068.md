# Báo Cáo Nghiên Cứu Toàn Văn: PBIP (CVPR 2025 / arXiv:2503.12068)
## Prototype-Based Image Prompting for Weakly Supervised Histopathological Image Segmentation

> **Mã bài báo:** `arXiv:2503.12068v1 [cs.CV]`  
> **Hội nghị công bố:** **IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2025)**  
> **Tác giả:** Qingchen Tang*, Lei Fan*, Maurice Pagnucco, Yang Song  
> **Đơn vị nghiên cứu:** University of New South Wales (UNSW Sydney, Australia)  
> **Mã nguồn chính thức:** [GitHub QingchenTang/PBIP](https://github.com/QingchenTang/PBIP)  
> **Chủ đề môn học SE374.R11 liên quan:** **Topic 6: Weakly-supervised segmentation for whole slide images**  
> **File PDF gốc:** [`docs/papers/PBIP_arxiv_2503.12068.pdf`](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/Y_KHOA/docs/papers/PBIP_arxiv_2503.12068.pdf)  

---

## 1. Tóm Tắt & Động Lực Nghiên Cứu (Abstract & Motivation)

### 1.1. Thách Thức Trong Phân Vùng Ảnh Bệnh Học Giám Sát Yếu
- **Hạn chế của CAM truyền thống:** Các phương pháp dựa trên Bản đồ kích hoạt lớp (Class Activation Maps - CAM) thường chỉ kích hoạt những vùng có tính phân biệt cao nhất (most discriminative regions), dẫn đến việc mặt nạ dự đoán bị co hẹp, thiếu sót ranh giới thực tế của mô u.
- **Thách thức của các mô hình Text-Vision (CLIP):** Việc đưa prompt dạng văn bản (text prompts) vào ảnh mô bệnh học gặp thất bại lớn vì:
  1. **Tính đồng nhất liên lớp (Inter-class Homogeneity):** Các loại mô khác nhau (mô u vs mô lành, chất nền liên kết vs mô hoại tử) có màu sắc và cấu trúc sợi nhuộm H&E rất giống nhau, khiến CAM dễ kích hoạt nhầm.
  2. **Tính dị thể nội lớp (Intra-class Heterogeneity):** Trong cùng một loại mô u, hình thái tế bào, mức độ biệt hóa và độ đậm nhạt của chất nhuộm thay đổi rất mạnh giữa các lát cắt. Mô tả bằng từ ngữ thông thường không thể nắm bắt được các sắc thái vi thể phức tạp này.

### 1.2. Đột Phá Kỹ Thuật Của PBIP (CVPR 2025)
- Thay vì dùng prompt văn bản, PBIP đề xuất **Nhắc ảnh dựa trên nguyên mẫu (Prototype-Based Image Prompting)**:
  - Xây dựng một **Ngân hàng ảnh (Image Bank)** từ tập huấn luyện thông qua thuật toán phân cụm (Clustering), trích xuất nhiều đặc trưng nguyên mẫu (Prototypes) cho mỗi lớp mô để bắt trọn tính dị thể nội lớp.
  - Sử dụng **Matching Loss** giữa đặc trưng ảnh đầu vào và các nguyên mẫu đặc thù của từng lớp mô thông qua học tương phản (Contrastive Learning).
  - Kết hợp với backbone hiện đại **SegFormer** và bộ mã hóa thị giác y tế **MedCLIP / PLIP**, tạo ra mặt nạ giả (Pseudo-Masks) sắc nét vượt bậc.

---

## 2. Kiến Trúc Mô Hình & Phương Pháp Luận Cốt Lõi

PBIP bao gồm hai mạng thành phần chính kết hợp với Image Bank:

1. **Ngân hàng Ảnh (Image Bank):**
   - Gom nhóm ảnh huấn luyện theo nhãn, phân thành $K$ phân lớp bằng K-Means clustering.
   - Chọn $N_K$ ảnh tiêu biểu nhất cho mỗi phân lớp để trích xuất đặc trưng nguyên mẫu $\mathbf{P} \in \mathbb{R}^{K \times D}$ thông qua image encoder của MedCLIP/PLIP.
2. **Mạng Phân Loại Đa Tầng (Classification Network - ClassNet):**
   - Tiếp nhận ảnh đầu vào $\mathbf{X}$ và các nguyên mẫu $\mathbf{P}$, sinh mặt nạ giả phân cấp $\mathbf{M}$ qua các tầng phân giải của SegFormer.
3. **Mạng So Khớp Đặc Trưng Ảnh (Image Feature Matching Network - ImgMatchNet):**
   - Tách biệt vùng tiền cảnh (Foreground) và hậu cảnh (Background) từ mặt nạ sơ bộ $\mathbf{M}$.
   - Tính toán hàm mất mát tương đồng $\mathcal{L}_{\text{SIM}}$ để ép các đặc trưng tiền cảnh phải trùng khớp chặt chẽ với nguyên mẫu cùng lớp.

### Hàm Mất Mát Tổng Thể:
$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{CLS}} + \beta \cdot \mathcal{L}_{\text{SIM}}$$

---

## 3. Kết Quả Thực Nghiệm SOTA Trên Các Bộ Dữ Liệu Chuẩn

PBIP thiết lập kỷ lục SOTA mới trên cả 4 bộ dữ liệu giải phẫu bệnh quốc tế:

| Phương Pháp | Giám sát | BCSS-WSSS (mIoU %) | BCSS-WSSS (Dice %) | LUAD-HistoSeg (mIoU %) | LUAD-HistoSeg (Dice %) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| HistoSegNet (ICCV 2019) | Yếu (Weak) | $32.19 \pm 5.8$ | $48.70 \pm 6.8$ | $45.32 \pm 4.3$ | $62.37 \pm 4.3$ |
| CLIMS (CVPR 2022) | Yếu (Weak) | $38.98 \pm 2.4$ | $56.09 \pm 2.6$ | $39.45 \pm 3.1$ | $56.58 \pm 3.0$ |
| OEEN (2023) | Yếu (Weak) | $59.93 \pm 0.5$ | $74.95 \pm 0.4$ | $72.54 \pm 0.2$ | $84.08 \pm 0.2$ |
| WSSS-Tissue / MLPS (2022) | Yếu (Weak) | $65.17 \pm 1.7$ | $78.91 \pm 1.1$ | $73.54 \pm 2.1$ | $84.75 \pm 1.4$ |
| TPRO (2023) | Yếu (Weak) | $63.92 \pm 0.4$ | $77.99 \pm 0.3$ | $74.14 \pm 0.8$ | $85.14 \pm 0.5$ |
| Proto2Seg (2024) | Yếu (Weak) | $67.86 \pm 0.9$ | $81.00 \pm 0.9$ | $74.91 \pm 1.2$ | $85.97 \pm 1.2$ |
| **PBIP (CVPR 2025 - SOTA)** | **Yếu (Weak)** | **$69.54 \pm 0.6$** | **$82.03 \pm 0.5$** | **$76.44 \pm 0.5$** | **$86.64 \pm 0.3$** |

- **Trên BCSS-WSSS:** PBIP đạt **$69.54\%$ mIoU** và **$82.03\%$ Dice score**, vượt trội hơn WSSS-Tissue hơn $4.3\%$ mIoU.
- **Trên LUAD-HistoSeg:** Đạt **$76.44\%$ mIoU** và **$86.64\%$ Dice score**.

---

## 4. Giá Trị Kế Thừa Cho Đồ Án Môn Học SE374.R11

1. **Kết hợp hoàn hảo giữa LowMagWSS (UIT) và PBIP (CVPR 2025):**
   - Nhóm có thể kết hợp: Mô phỏng lưu trữ độ phóng đại thấp của TS. Bùi Cao Doanh (`LowMagWSS`) + Cơ chế trích xuất mặt nạ nguyên mẫu của `PBIP` để xây dựng giải pháp phân vùng tổn thương ung thư vừa tiết kiệm dung lượng lưu trữ, vừa đạt độ chính xác SOTA.
2. **Dữ liệu nhẹ, hoàn toàn khả thi trên phần cứng học tập:**
   - Bộ dữ liệu `BCSS-WSSS` và `LUAD-HistoSeg` sử dụng các patch kích thước $256\times 256$ hoặc $512\times 512$, không yêu cầu nạp toàn bộ slide nặng gigabyte vào VRAM, loại bỏ hoàn toàn nguy cơ tràn bộ nhớ (OOM).
