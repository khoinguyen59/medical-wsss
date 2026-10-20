Gemini
Phân Vùng Ngữ Nghĩa Giám Sát Yếu và Phân Tầng Kiểu Hình Mô Bệnh Học Trên Ảnh Tiêu Bản Toàn Phần: Đột Phá Công Nghệ, Đối Chuẩn SOTA và Hệ Sinh Thái Mã Nguồn Mở (2024-2026)
Tổng Quan Kiến Trúc và Động Lực Học Thuật Giai Đoạn 2024-2026

Phân vùng ngữ nghĩa mức pixel (pixel-level semantic segmentation) trên ảnh tiêu bản toàn phần (Whole Slide Images - WSI) nhuộm Hematoxylin và Eosin (H&E) đóng vai trò nền tảng trong việc định lượng vi môi trường khối u (Tumor Microenvironment - TME), xác định tỷ lệ xâm nhiễm tế bào lympho (lymphocytic infiltrate), đánh giá hoại tử (necrosis) và phân tầng tiên lượng bệnh nhân ung thư. Tuy nhiên, độ phân giải gigapixel của WSI (thường vượt mức 
20.000
×
20.000
20.000×20.000 đến 
100.000
×
100.000
100.000×100.000 pixel ở độ phóng đại 
40
×
40×) kết hợp với tính chất hình thái học đặc thù tạo ra rào cản lớn về chi phí gán nhãn. Việc chuyển đổi từ gán nhãn đa giác dày đặc (dense pixel-level masks) sang nhãn mức mảng ảnh (patch-level tags) hoặc mức tiêu bản (slide-level labels) giúp cắt giảm hơn 95% thời gian làm việc của bác sĩ giải phẫu bệnh, thúc đẩy sự bùng nổ của bài toán Phân vùng ngữ nghĩa giám sát yếu (Weakly-Supervised Semantic Segmentation - WSSS).   

Khác biệt hoàn toàn với ảnh tự nhiên (như PASCAL VOC hay MS-COCO) nơi tiền cảnh và hậu cảnh có ranh giới hình học tách biệt rõ ràng, ảnh mô bệnh học chịu sự chi phối đồng thời của hai hiện tượng đối nghịch là tính đồng nhất liên lớp (inter-class homogeneity) và tính dị biệt nội lớp (intra-class heterogeneity). Các lát cắt mô được cấu thành từ sự lặp lại bất quy tắc của các tế bào và cấu trúc mô đệm làm cho đặc trưng hình thái giữa các loại mô khác nhau trở nên tương đồng về màu sắc và kết cấu, khuếch đại độ bất định tại ranh giới chuyển tiếp. Ngược lại, cùng một kiểu hình mô học (chẳng hạn biểu mô ác tính ở các cấp độ biệt hóa Gleason hoặc mô đệm phản ứng) lại biểu hiện biến thiên hình thái lớn giữa các vùng không gian và giữa các cơ sở y tế do sai lệch quy trình nhuộm (stain variation). Do đó, các thuật toán Bản đồ kích hoạt lớp (Class Activation Maps - CAM) truyền thống vốn chỉ tập trung vào vùng phân biệt mạnh nhất (most discriminative regions) bị suy giảm hiệu năng nghiêm trọng khi áp dụng trực tiếp lên WSI, dẫn đến hiện tượng kích hoạt cục bộ không đầy đủ (under-activation) hoặc tràn kích hoạt sang mô lành lân cận (over-activation).   

Trong giai đoạn từ năm 2024 đến năm 2026, cộng đồng thị giác máy tính và hình ảnh y sinh quốc tế tại các diễn đàn CVPR, MICCAI, IEEE TMI, MedIA, MIDL và ISBI đã thực hiện cuộc dịch chuyển toàn diện nhằm giải quyết nghịch lý này. Thay vì chỉ tinh chỉnh hậu xử lý bằng trường ngẫu nhiên có điều kiện (Conditional Random Fields - CRF) như thế hệ HistoSegNet trước đây, các kiến trúc giai đoạn 2024-2026 tập trung vào bốn trụ cột đột phá: tái định vị kích hoạt không gian và khử nhiễu nhân quả phản thực tế trên CAM; chuyển đổi bài toán giám sát yếu thành giám sát đầy đủ thông qua tổng hợp trộn mẫu (Mosaic và đường cong Bézier) kết hợp điều chuẩn tự giám sát; chưng cất cấu trúc từ các Mô hình nền tảng đa phương thức và tự giám sát (Foundation Models như CONCH, UNI, Virchow, DINOv3, SAM, MedCLIP) xuống các bộ giải mã phân vùng chi tiết; và ứng dụng Mô hình không gian trạng thái chọn lọc (Selective State Space Models - Mamba) để nắm bắt ngữ cảnh toàn cục độ phức tạp tuyến tính 
𝑂
(
𝑁
)
O(N) trên chuỗi hàng chục nghìn mảng ảnh WSI.   

Phần 1: Bảng Ma Trận Đối Chuẩn SOTA (Giai Đoạn 2024-2026)

Để thiết lập hệ quy chiếu định lượng khách quan theo chuẩn mực của IEEE Transactions on Medical Imaging (IEEE TMI) và Medical Image Analysis (MedIA), hệ thống khảo sát dưới đây tổng hợp 30 công trình và cấu hình mô hình trọng điểm xuất bản hoặc được đối chuẩn trực tiếp trong giai đoạn 2024-2026. Các tập dữ liệu đối chuẩn cốt lõi trong ma trận này bao gồm BCSS / BCSS-WSSS, LUAD-HistoSeg và WSSS4LUAD. Tập dữ liệu BCSS và BCSS-WSSS (Breast Cancer Semantic Segmentation) bao gồm 23.422 mảng ảnh huấn luyện chỉ có nhãn mức ảnh, 3.418 ảnh kiểm định và 4.986 ảnh kiểm tra có nhãn mức pixel kích thước 
224
×
224
224×224 được trích xuất từ 151 tiêu bản WSI ung thư vú H&E với 4 lớp tiền cảnh là Khối u (TUM - 4.738 mẫu đơn lớp), Mô đệm (STR - 2.903 mẫu), Thâm nhiễm lympho (LYM - 679 mẫu) và Hoại tử (NEC - 1.058 mẫu). Tập dữ liệu LUAD-HistoSeg (Lung Adenocarcinoma Histopathology Segmentation) được trích xuất từ 54 WSI ung thư biểu mô tuyến phổi, gồm 16.678 mảng ảnh huấn luyện giám sát yếu và 607 ảnh kiểm định/kiểm tra (300 validation, 307 test) có nhãn mức pixel cho 4 kiểu hình: Biểu mô khối u (TE - 1.574 mẫu), Hoại tử (NEC - 787 mẫu), Tế bào lympho (LYM - 42 mẫu) và Mô đệm liên quan khối u (TAS - 2.192 mẫu). Bên cạnh đó, bộ đối chuẩn WSSS4LUAD chứa 10.091 mảng ảnh huấn luyện có nhãn hiện diện mô và 120 ảnh kiểm định/kiểm tra (40 validation, 80 test, kích thước từ 
200
×
200
200×200 đến 
5000
×
5000
5000×5000 pixel) bao gồm 3 lớp là Khối u (TUM - 1.181 mẫu), Mô đệm (STR - 1.680 mẫu) và Mô bình thường (NOM - 1.832 mẫu).   

Ma Trận Tổng Hợp 30 Mô Hình và Cấu Hình Đối Chuẩn WSSS Mô Bệnh Học (2024-2026)

Bảng dữ liệu dưới đây trình bày chi tiết từng công trình nghiên cứu và cấu hình đối chuẩn, tuân thủ nguyên tắc ghi nhận chính xác các số liệu phần trăm (% mIoU, % Dice, chỉ số IoU từng lớp mô học) và ghi chú NR (Not Reported) đối với các thông số không được tác giả công bố trong ấn phẩm gốc.   

STT	Tên Mô Hình / Bài Báo & Tác Giả Chính	Hội Nghị / Tạp Chí & Năm	Đường Dẫn Kiểm Chứng (DOI / arXiv / URL)	Dạng Thức Giám Sát	Cơ Chế Giải Thuật Cốt Lõi	Bộ Dữ Liệu Thực Nghiệm	Kết Quả Định Lượng Chính Xác (% mIoU, % Dice, Chỉ Số Lớp)	Chi Phí Phần Cứng & Điện Toán
1	

C2RM-Seg: Causal Counterfactual Reasoning with Structural-Semantic Priors... (Zhenbing Liu et al.)

	

MICCAI 2026 / arXiv 2026

	

arXiv:2606.25508 / MICCAI PDF

[cite: 10, 24]

	

Nhãn mức mảng ảnh (Patch-level labels)

	

Suy luận nhân quả phản thực tế (C2RM) loại bỏ nhiễu nhuộm/nền + Kiến trúc Dual-Path (ResNeSt-200e + Frozen DINOv3 ViT-B/16 với Cross-Path Gating) + Hàm mất mát Uncertainty-Gated Margin (UGM)

	

BCSS-WSSS, LUAD-HistoSeg

	

BCSS-WSSS: 72.17% mIoU (Dice: NR, lớp: NR)




LUAD-HistoSeg: 79.62% mIoU, 18.07 HD95 (Dice: NR, lớp: NR)

	

Đóng băng DINOv3 ViT-B/16, chỉ huấn luyện Semantic Adapter và nhánh ResNeSt-200e (VRAM/Giờ: NR)


2	

XWIL: Weakly Supervised Histopathology Tissue Segmentation (OpenReview / Han et al. benchmark)

	

OpenReview 2026 (ICLR / MIDL track)

	

OpenReview PDF

[cite: 22]

	

Nhãn hiện diện mô mức mảng ảnh (
224
×
224
224×224)

	

Tối ưu hóa học giám sát yếu tương tác chéo (XWIL) với kiểm định thống kê hiệu chỉnh Holm (
𝑝
𝐻
𝑜
𝑙
𝑚
≤
0.0018
p
Holm
	​

≤0.0018)

	

BCSS-WSSS, LUAD-HistoSeg, WSSS4LUAD

	

BCSS-WSSS: 71.96% mIoU (TUM, STR, NEC đạt đỉnh)




LUAD-HistoSeg: 81.10% mIoU (TE, NEC, TAS đạt đỉnh)




WSSS4LUAD: 
80.49
±
0.09
%
80.49±0.09% mIoU (Dice: NR)

	NR
3	

SIA-WSSS: Synchronous Inhibition and Activation for Weakly Supervised... (Jiansong Fan, Xiang Pan et al.)

	

MICCAI 2025

[cite: 27, 28]

	

10.1007/978-3-032-05141-7_56 / MICCAI PDF

[cite: 5, 29]

	

Nhãn mức mảng ảnh (Image-level labels)

	

Đồ thị có hướng động trong Regularized Focus Mechanism (RFM) + Discriminative Activation Module (DAM) + Region Self-modulation Module (RSM) trên ViT-B

	

BCSS-WSSS, LUAD-HistoSeg

	

BCSS-WSSS: 73.4% mIoU, 79.9% mDice




LUAD-HistoSeg: 78.5% mIoU, 85.5% mDice (Chỉ số từng lớp: NR)

	

Backbone ViT-B, Adam, batch size 16, 
𝑙
𝑟
=
10
-
4
lr=10
-4
 (Loại GPU/Giờ: NR)


The above content does NOT show the entire file contents. If you need to view any lines of the file which were not shown to complete your task, call this tool again to view those lines.