# Nghiên cứu Toàn diện về Phân đoạn Ngữ nghĩa Giám sát Yếu (WSSS) cho Ảnh Toàn Tiêu bản trong Giải phẫu Bệnh Kỹ thuật số

## Các Công trình Tiêu biểu và Cơ chế Phương pháp luận Tiên tiến

Phân đoạn ngữ nghĩa mô học trên ảnh toàn tiêu bản (Whole Slide Images -
WSI) giữ vai trò hạt nhân trong giải phẫu bệnh kỹ thuật số
(Computational Pathology - CPath), cung cấp cơ sở định lượng cho việc
xác định bờ diện cắt ung thư, giải mã vi môi trường miễn dịch và dự đoán
tiên lượng sống còn^1^. Dù vậy, việc thiết lập nhãn chi tiết ở cấp độ
điểm ảnh (pixel-level annotations) trên các tệp ảnh gigapixel đòi hỏi
hàng trăm giờ làm việc của các bác sĩ giải phẫu bệnh giàu kinh nghiệm,
dẫn đến chi phí bất khả thi đối với các tập dữ liệu lớn^2^. Phân đoạn
ngữ nghĩa giám sát yếu (Weakly-Supervised Semantic Segmentation - WSSS)
khắc phục rào cản này bằng cách suy diễn ranh giới phân đoạn dày đặc chỉ
từ các nhãn cấp độ ảnh hoặc mảnh cắt (slide/patch-level labels)^1^.

Trong giai đoạn 2024-2026, lĩnh vực này đã trải qua sự chuyển dịch mô
hình căn bản. Các phương pháp truyền thống vốn dựa trên bản đồ kích hoạt
lớp (Class Activation Mapping - CAM) tạo ra bởi các mạng phân loại tiêu
chuẩn đang được thay thế bằng những khung giải thuật tích hợp cơ chế gợi
ý nguyên mẫu đa phương thức (prototype prompting), neo giữ hình học vật
lý, và sự kế thừa không gian đặc trưng trích xuất từ các mô hình nền
tảng bệnh học (Pathology Foundation Models) quy mô tỷ tham số^1^.

### Tiến bộ trong Tinh chỉnh Bản đồ Kích hoạt Lớp và Sinh Mặt nạ Giả lập

Các kỹ thuật CAM kinh điển gặp phải hai giới hạn cố hữu khi áp dụng vào
mô bệnh học: xu hướng chỉ kích hoạt các vùng mang tính phân biệt cao
nhất (discriminative regions) khiến mặt nạ bị co cụm cục bộ, và hiện
tượng rò rỉ kích hoạt sang các mô liên kết xung quanh do sự đồng nhất
hình thái giữa các lớp mô (inter-class homogeneity) kết hợp với tính
biến thiên nội tại của tế bào ung thư (intra-class heterogeneity)^1^. Để
giải quyết triệt để sự suy giảm độ chính xác này, các công trình từ năm
2024 đến năm 2026 đã đưa ra các cơ chế tinh chỉnh tiên tiến.

Khung làm việc Tái định vị Kích hoạt và Học tương hỗ (Activation
Relocalization and Mutual Learning - ARML) giải quyết sự co cụm của CAM
bằng cách đưa vào cơ chế tái định vị kích hoạt (Activation
Relocalization Scheme - ARS)^2^. Cơ chế này áp dụng phép gộp trung bình
không gian (Spatial Average Pooling - SAP) kết hợp với các tầng tích
chập chuẩn hóa nhằm tái cân bằng trọng số kích hoạt trên toàn bộ chiều
kênh và không gian, ép buộc mạng nơ-ron mở rộng vùng chú ý sang các diện
tích mô biên^2^. Trong giai đoạn phân đoạn tiếp theo, ARML thiết lập cơ
chế học tương hỗ giữa hai mạng song song kết hợp các hàm mất mát triệt
tiêu nhiễu theo điểm (point-wise noise suppression loss), ngăn chặn việc
mô hình học vẹt các sai lệch ngẫu nhiên có trong nhãn giả thô^2^.

Phương pháp Ức chế và Kích hoạt Đồng bộ (Synchronous Inhibition and
Activation) tiếp cận vấn đề thông qua việc triệt tiêu hiện tượng kích
hoạt sai lệch ở các vùng biên có độ tự tin thấp^6^. Công trình này tích
hợp Mô-đun Lọc Vùng (Region Filtration Module - RFM) song song với
Mô-đun Kích hoạt Phân biệt (Discriminative Activation Module - DAM)^6^.
Trong khi RFM cắt tỉa các vùng kích hoạt lan tỏa không đặc hiệu vào mô
lành, DAM đồng thời bù đắp và kích hoạt lại các tiểu thùy u bị bỏ sót,
tạo ra các đường phân định sắc nét tương thích với cấu trúc giải
phẫu^6^.

Khác biệt với việc xử lý trực tiếp trên các mảnh cắt thực tế có nhãn
không hoàn chỉnh, mô hình PistoSeg đề xuất giải pháp tổng hợp tập dữ
liệu nhân tạo^3^. Phương pháp này trích xuất các mảnh cắt thuần nhất một
loại mô (single tissue type) rồi ghép nối lại với nhau theo các đường
cắt ngẫu nhiên để tự động sinh ra các nhãn điểm ảnh chuẩn xác tuyệt
đối^3^. Mạng phân đoạn ban đầu được huấn luyện có giám sát hoàn toàn
trên tập dữ liệu tổng hợp này, tạo tiền đề cho một mô-đun tinh chỉnh mặt
nạ giả lập dựa trên sự tương đồng không gian biểu diễn ẩn giữa CAM và
nhãn tổng hợp thông qua các tầng chiếu Query-Key^3^.

Khung hợp nhất Đồng vận Vật lý - Ngữ nghĩa (PhysPro) đưa các tiên nghiệm
vật lý quang học và đặc tính hấp thụ thuốc nhuộm H&E bất biến vào quá
trình học biểu diễn^4^. Bằng cách sử dụng các ràng buộc vật lý để neo
giữ không gian ngữ nghĩa, PhysPro ngăn chặn hiện tượng tách ghép sai
lệch giữa biến dị hình thái và nhiễu nhãn giả, thiết lập kỷ lục mới trên
tập kiểm chuẩn BCSS-WSSS với chỉ số mean Dice đạt 84.3% và mIoU đạt
72.2%^4^.

Nghiên cứu Khuếch tán Đồ thị Nhận biết Ranh giới (ProBAG) giải quyết sự
thiếu liên tục của nhãn giả bằng cách xây dựng biểu diễn đồ thị từ các
siêu điểm ảnh (superpixels)^7^. Quá trình khuếch tán không gian trên đồ
thị được hướng dẫn bởi các nguyên mẫu lai kết hợp giữa hình ảnh và văn
bản bệnh học, cho phép lan truyền nhãn chính xác dọc theo các ranh giới
màng đáy của biểu mô khối u^7^.

Nghiên cứu LowMagWSS đánh giá tác động của việc giảm dung lượng lưu trữ
WSI đối với bài toán WSSS^10^. Nhóm tác giả chứng minh rằng việc áp dụng
kỹ thuật siêu phân giải định hướng bệnh học (STAR-RL) để tái tạo ảnh từ
độ phóng đại 10x và 20x lên 40x mang lại sự cải thiện vượt bậc đối với
các cấu trúc tế bào lympho thâm nhiễm (+5.09 IoU so với nội suy song
tuyến), đồng thời khẳng định các chỉ số đo lường hình ảnh thông thường
như PSNR hay SSIM không phản ánh tuyến tính chất lượng phân đoạn mô học
hạ nguồn^10^.

### Cơ chế Gợi ý Nguyên mẫu Thị giác và Tương phản Đa phương thức

Sự xuất hiện của kỹ thuật Gợi ý Nguyên mẫu (Prototype Prompting) đã giải
quyết điểm nghẽn lớn nhất của CAM: sự thiếu hụt ngữ cảnh toàn cục khi
phân tích độc lập từng mảnh cắt nhỏ^1^. Thay vì dựa vào mô tả văn bản
thuần túy vốn gặp khó khăn trong việc phân biệt các sắc thái vi thể phức
tạp, mô hình PBIP (Prototype-Based Image Prompting) thiết lập cơ chế gợi
ý trực tiếp bằng hình ảnh^1^.

PBIP tự động xây dựng một ngân hàng ảnh nguyên mẫu từ tập huấn luyện
bằng cách lọc các mảnh mô đơn lớp, loại bỏ nền kính, và tiến hành gom
cụm K-means thành ${width="0.19726596675415572in"
height="0.27476268591426073in"} lớp con cho mỗi phân loại mô học nhằm mô
hình hóa đầy đủ tính đa dạng nội lớp^1^. Khi phân tích một ảnh đầu vào
${width="0.19422681539807524in"
height="0.2705304024496938in"}, bộ mã hóa MedCLIP trích xuất các vectơ
đặc trưng đa tầng, đồng thời mạng ClassNet (với backbone SegFormer)
trích xuất bản đồ đặc trưng phân cấp
${width="0.20019575678040244in"
height="0.27884405074365703in"}^1^. Điểm tự tin của từng điểm ảnh đối
với lớp mô học ${width="0.12858158355205598in"
height="0.2785925196850394in"} được xác định thông qua độ tương đồng
Cosine trung bình giữa vectơ đặc trưng tại điểm đó và các vectơ nguyên
mẫu ${width="0.5093318022747156in"
height="0.3000174978127734in"} thuộc ngân hàng ảnh^1^:

S_c(x, y) = \max_k \cos(F(x, y), P_c^k){width="6.458333333333333in"
height="0.9148064304461943in"}

Một mô-đun phân ngưỡng thích ứng sẽ tính toán ngưỡng động
$\tau_c${width="0.10405839895013123in"
height="0.23586504811898512in"} dựa trên hàm mật độ phân phối cường độ
của bản đồ kích hoạt để phân tách không gian tiền cảnh và hậu cảnh^1^.
Hàm mất mát tương đồng đối nghịch được thiết lập để tối đa hóa tính đồng
nhất của các điểm ảnh tiền cảnh với nguyên mẫu mục tiêu
($\mathcal{L}_{\text{pos}}${width="0.45757436570428695in"
height="0.25651902887139105in"}) và triệt tiêu kích hoạt sai lệch tại
hậu cảnh ($\mathcal{L}_{\text{neg}}${width="0.4658213035870516in"
height="0.25724409448818897in"})^1^:

\mathcal{L}_{\text{proto}} = \mathcal{L}_{\text{pos}} + \lambda \mathcal{L}_{\text{neg}}{width="6.458333333333333in"
height="0.5866688538932633in"}

Nhờ cơ chế định hướng hình ảnh phân cấp này, PBIP hạn chế tối đa sự nhầm
lẫn giữa mô đệm xơ hóa và biểu mô ung thư biệt hóa kém, vượt qua các
phương pháp WSSS dùng gợi ý văn bản trên nhiều tập dữ liệu mô bệnh học
lớn^1^.

### Khai thác Biểu diễn Không gian từ các Mô hình Nền tảng Bệnh học

Giai đoạn 2024-2026 đánh dấu bước thoái trào của việc sử dụng các mạng
ResNet hoặc Vision Transformer tiền huấn luyện trên ImageNet tự
nhiên^5^. Thay vào đó, các mô hình nền tảng giải phẫu bệnh tự giám sát
(Self-Supervised Learning - SSL) được huấn luyện trên hàng trăm nghìn
tiêu bản WSI lâm sàng đã cung cấp các không gian biểu diễn đặc trưng có
khả năng phân tách mô bệnh học vượt trội^5^. Việc đóng băng (freezing)
hoặc tinh chỉnh nhẹ (fine-tuning) các bộ mã hóa này làm nền tảng trích
xuất đặc trưng cho các bộ giải mã WSSS giúp đẩy nhanh tốc độ hội tụ và
cải thiện đáng kể độ sắc nét của mặt nạ phân đoạn^12^.

+----------+------+-------+-------+------+-------------------+
| **Mô hình Nền      | **Tổ chức  | **Kiến trúc  | **Phương     | **Quy mô   | **Trạng thái Trọng số & Bản quyền**  |
| tảng**             | Phát       | & Số lượng   | pháp & Mục   | Dữ liệu    |                                      |
|                    | triển**    | Tham số**    | tiêu SSL**   | Tiền Huấn  |                                      |
|                    |            |              |              | luyện**    |                                      |
+----------+------+-------+-------+------+-------------------+
| **UNI**            | Mahmood    | ViT-Large/16 | DINOv2       | Mass-100K: | Hugging Face (MahmoodLab/UNI), Giấy  |
|                    | Lab        | (303M tham   | (DINO + iBOT | \>100,000  | phép CC-BY-NC-ND-4.0                 |
| \[cite: 12\]       | (Harvard / | số, chiều    | masked-image | WSIs,      |                                      |
|                    | BWH)       | nhúng 1024)  | modeling +   | \>100M     |                                      |
|                    |            |              | điều hòa     | patches    |                                      |
|                    |            |              | KoLeo)       | H&E và IHC |                                      |
+----------+------+-------+-------+------+-------------------+
| **CONCH**          | Mahmood    | Thị giác:    | CoCa (Kết    | 1.17 triệu | Hugging Face (MahmoodLab/CONCH),     |
|                    | Lab        | ViT-B/16     | hợp          | cặp        | Quyền học thuật hạn chế              |
| \[cite: 14\]       | (Harvard / | (90M); Văn   | Contrastive  | ảnh-chú    |                                      |
|                    | BWH)       | bản: CoCa    | Image-Text   | thích giải |                                      |
|                    |            | Transformer  | và           | phẫu bệnh  |                                      |
|                    |            | (110M)       | Captioning   | từ PubMed  |                                      |
|                    |            |              | Loss)        | Central    |                                      |
+----------+------+-------+-------+------+-------------------+
| **Virchow /        | Paige AI / | ViT-Giant/14 | DINOv2 tự    | 1.5M đến   | Hugging Face (paige-ai/Virchow),     |
| Virchow2**         | MSKCC      | & ViT-H/14   | giám sát cải | 3.0M WSIs  | Apache-2.0 (Yêu cầu đăng ký)         |
|                    |            | (632M tham   | tiến trên    | lâm sàng   |                                      |
| \[cite: 15, 16\]   |            | số, SwiGLU,  | cụm siêu     | đa dạng từ |                                      |
|                    |            | LayerScale)  | tính toán    | Memorial   |                                      |
|                    |            |              |              | Sloan      |                                      |
|                    |            |              |              | Kettering  |                                      |
+----------+------+-------+-------+------+-------------------+
| **Prov-GigaPath**  | Providence | Mã hóa       | DINOv2 cấp   | Prov-Path: | Hugging Face                         |
|                    | Health /   | patch:       | patch +      | 1.3 tỷ     | (prov-gigapath/prov-gigapath),       |
| \[cite: 13, 17\]   | Microsoft  | ViT-Giant    | LongNet cấp  | patches    | Apache-2.0                           |
|                    |            | (1.13B); Mã  | slide (ngữ   | (256x256)  |                                      |
|                    |            | hóa slide:   | cảnh siêu    | từ 171,189 |                                      |
|                    |            | LongNet      | dài)         | WSIs thuộc |                                      |
|                    |            | (85M)        |              | 31 cơ quan |                                      |
+----------+------+-------+-------+------+-------------------+
| **GigaPath-Flash** | Xu et al.  | Mã hóa       | Tối ưu hóa   | Tương      | Hugging Face                         |
|                    |            | patch:       | suy luận     | thích      | (prov-gigapath/prov-gigapath-flash), |
| \[cite: 18\]       |            | ViT-Small/16 | nhẹ, chưng   | trọng số   | Apache-2.0                           |
|                    |            | (22M); Mã    | cất kiến     | Prov-Path, |                                      |
|                    |            | hóa slide:   | thức hai cấp | giảm 75%   |                                      |
|                    |            | LongNet-12L  | độ           | tiêu thụ   |                                      |
|                    |            | (21M)        |              | bộ nhớ GPU |                                      |
+====================+============+==============+==============+============+======================================+

## Kho Mã Nguồn Mở và Trọng số Mô hình Tiền Huấn Luyện

Việc xây dựng một trạm làm việc phân đoạn WSSS hoàn chỉnh đòi hỏi sự kết
hợp giữa các kho mã nguồn thuật toán chuyên biệt và các thư viện xử lý
hạ tầng ảnh gigapixel đã được kiểm chứng. Bảng dưới đây tổng hợp các tài
nguyên mã nguồn mở, trạng thái phân phối trọng số và vai trò kỹ thuật
của từng công cụ trong chuỗi xử lý.

+----------+-----------------+-----------+----------+-----------+
| **Tên Dự án / Mô  | **Định danh Kho Lưu trữ**       | **Ngăn xếp Công     | **Trạng thái      | **Vai trò Kiến trúc |
| hình**            |                                 | nghệ Cốt lõi**      | Trọng số & Khả    | trong Hệ thống**    |
|                   |                                 |                     | năng Tái lập**    |                     |
+----------+-----------------+-----------+----------+-----------+
| **PBIP**          | GitHub: QingchenTang/PBIP       | PyTorch, SegFormer, | Trọng số công     | Sinh bản đồ CAM và  |
|                   |                                 | MedCLIP Feature     | khai cho BCSS và  | khởi tạo mặt nạ giả |
| \[cite: 19\]      |                                 | Engine              | LUAD; tái lập đầy | lập ban đầu^1^      |
|                   |                                 |                     | đủ pipeline CAM   |                     |
+----------+-----------------+-----------+----------+-----------+
| **PistoSeg**      | GitHub: Vison307/PistoSeg       | PyTorch, Single     | Mã nguồn huấn     | Tiền huấn luyện     |
|                   |                                 | Tissue Synthesis,   | luyện hoàn chỉnh; | mạng phân đoạn trên |
| \[cite: 3\]       |                                 | Q-K Refinement      | cung cấp quy      | dữ liệu tổng hợp    |
|                   |                                 |                     | trình sinh ảnh    | không nhiễu^3^      |
|                   |                                 |                     | tổng hợp          |                     |
+----------+-----------------+-----------+----------+-----------+
| **LowMagWSS**     | GitHub: Dung-Dx/LowMagWSS       | PyTorch, STAR-RL    | Cung cấp trọng số | Tối ưu hóa tốc độ   |
|                   |                                 | Super Resolution,   | STAR-RL và kịch   | xử lý slide qua     |
| \[cite: 10\]      |                                 | WSSS-Tissue         | bản đối chuẩn     | phân đoạn ở độ      |
|                   |                                 |                     | 10x/20x/40x       | phóng đại thấp^10^  |
+----------+-----------------+-----------+----------+-----------+
| **WSSS-Tissue**   | GitHub: ChuHan89/WSSS-Tissue    | Multi-layer         | Trọng số công     | Kiểm chuẩn đường    |
|                   |                                 | Pseudo-supervision, | khai qua liên kết | ống tạo mặt nạ giả  |
| \[cite: 20\]      |                                 | Progressive Dropout | Drive; đóng vai   | lập truyền          |
|                   |                                 |                     | trò đối chuẩn cơ  | thống^20^           |
|                   |                                 |                     | sở                |                     |
+----------+-----------------+-----------+----------+-----------+
| **WSSS4LUAD       | GitHub: yzouag/WSSS4LUAD        | MMSegmentation,     | Cung cấp          | Đường ống tiêu      |
| Baseline**        |                                 | ScaleNet101,        | checkpoint phân   | chuẩn xử lý tập dữ  |
|                   |                                 | ResNet38, PSPNet    | loại patch và mô  | liệu ung thư phổi   |
| \[cite: 22\]      |                                 |                     | hình phân đoạn    | Grand-Challenge^22^ |
|                   |                                 |                     | PSPNet            |                     |
+----------+-----------------+-----------+----------+-----------+
| **UNI             | Hugging Face: MahmoodLab/UNI    | timm, PyTorch,      | Tệp               | Trích xuất vector   |
| Checkpoint**      |                                 | ViT-L/16 Engine     | pytorch_model.bin | đặc trưng cấp patch |
|                   |                                 |                     | chính thức (cần   | 1024 chiều chất     |
| \[cite: 12\]      |                                 |                     | ký thỏa thuận sử  | lượng cao^12^       |
|                   |                                 |                     | dụng)             |                     |
+----------+-----------------+-----------+----------+-----------+
| **CONCH           | Hugging Face: MahmoodLab/CONCH  | OpenCLIP, timm,     | Trọng số chính    | Trích xuất đặc      |
| Checkpoint**      |                                 | ViT-B/16 + Text     | thức (yêu cầu xác | trưng đa phương     |
|                   |                                 | Transformer         | thực tài khoản tổ | thức và phân loại   |
| \[cite: 14\]      |                                 |                     | chức nghiên cứu)  | zero-shot^14^       |
+----------+-----------------+-----------+----------+-----------+
| **Virchow         | Hugging Face: paige-ai/Virchow  | timm, ViT-H/14,     | Tệp               | Backbone trích xuất |
| Checkpoint**      |                                 | 632M params         | model.safetensors | đặc trưng mô bệnh   |
|                   |                                 |                     | tải trực tiếp qua | học lâm sàng chuyên |
| \[cite: 15\]      |                                 |                     | Hugging Face Hub  | sâu^15^             |
+----------+-----------------+-----------+----------+-----------+
| **Prov-GigaPath** | Hugging Face:                   | DINOv2 Tile         | Trọng số chính    | Mã hóa ngữ cảnh     |
|                   | prov-gigapath/prov-gigapath     | Encoder + LongNet   | thức              | trọn vẹn của hàng   |
| \[cite: 17\]      |                                 | Slide Encoder       | pytorch_model.bin | chục nghìn patch    |
|                   |                                 |                     | và                | trên toàn slide^13^ |
|                   |                                 |                     | slide_encoder.pth |                     |
+----------+-----------------+-----------+----------+-----------+
| **TIAToolbox**    | GitHub:                         | OpenSlide, PyTorch, | Đầy đủ mã nguồn   | Động cơ đọc WSI,    |
|                   | tissueimageanalytics/tiatoolbox | Zarr Caching, Dask, | thư viện chuẩn    | trích xuất patch    |
| \[cite: 23, 24\]  |                                 | Shapely             | CPath; có sẵn các | song song và ghép   |
|                   |                                 |                     | mô hình phân đoạn | nối mặt nạ WSI^24^  |
+===================+=================================+=====================+===================+=====================+

Quá trình tải và sử dụng các mô hình nền tảng cấp patch được tối ưu hóa
thông qua thư viện timm kết hợp với kho lưu trữ Hugging Face Hub:

> Python
>
> import timm\
> import torch\
> from torchvision import transforms\
> from PIL import Image\
> \
> \# Khởi tạo mô hình nền tảng Virchow từ Hugging Face Hub\
> virchow_model = timm.create_model(\"hf_hub:paige-ai/Virchow\",
> pretrained=True)\
> virchow_model.eval().cuda()\
> \
> \# Chuẩn hóa kích thước đầu vào theo tiêu chuẩn ViT\
> preprocess_pipeline = transforms.Compose(\[\
> transforms.Resize((224, 224),
> interpolation=transforms.InterpolationMode.BICUBIC),\
> transforms.ToTensor(),\
> transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224,
> 0.225)),\
> \])\
> \
> \# Trích xuất vector biểu diễn không gian 1280 chiều\
> patch_img = Image.open(\"tumor_region_patch.png\").convert(\"RGB\")\
> tensor_input = preprocess_pipeline(patch_img).unsqueeze(0).cuda()\
> \
> with torch.no_grad():\
> patch_embeddings = virchow_model(tensor_input)

Đối với công đoạn suy luận trên toàn bộ ảnh tiêu bản gigapixel, động cơ
SemanticSegmentor thuộc thư viện TIAToolbox cho phép thực thi cơ chế cửa
sổ trượt (sliding window) với độ trễ tối thiểu, đồng thời tận dụng bộ
nhớ đệm dạng khối Zarr để ngăn ngừa hiện tượng cạn kiệt RAM hệ
thống^24^:

> Python
>
> from tiatoolbox.models.engine.semantic_segmentor import
> SemanticSegmentor, IOSegmentorConfig\
> \
> \# Định cấu hình độ phân giải xử lý và kích thước cửa sổ trượt\
> segmentation_config = IOSegmentorConfig(\
> input_resolutions=\[{\"units\": \"mpp\", \"resolution\": 0.5}\], \# Xử
> lý ở độ phân giải 20x\
> output_resolutions=\[{\"units\": \"mpp\", \"resolution\": 0.5}\],\
> patch_input_shape=\[1024, 1024\],\
> patch_output_shape=\[1024, 1024\],\
> stride_shape=\[768, 768\] \# Chồng lấn 256 pixel triệt tiêu biên\
> )\
> \
> \# Khởi tạo động cơ phân đoạn với mô hình WSSS đã huấn luyện\
> wsi_segmentor = SemanticSegmentor(\
> model=\"fcn_resnet50_unet-bcss\",\
> auto_get_mask=True \# Tự động loại bỏ nền kính qua Otsu\
> )\
> \
> \# Kích hoạt suy luận phân đoạn trên tiêu bản SVS\
> wsi_segmentor.run(\
> imgs=\[\"clinical_biopsy_specimen.svs\"\],\
> save_dir=\"wsi_segmentation_outputs\",\
> patch_mode=False\
> )

## Bộ Dữ liệu Công khai và Tiêu chuẩn Đánh giá

Việc kiểm chuẩn các giải pháp WSSS đòi hỏi các tập dữ liệu có sự phân
tách rõ ràng: nhãn cấp ảnh hoặc mảnh cắt phục vụ huấn luyện giám sát
yếu, và nhãn mặt nạ điểm ảnh chuẩn (ground truth) được thẩm định nghiêm
ngặt bởi các chuyên gia giải phẫu bệnh để phục vụ việc đánh giá độc
lập^1^.

Trong các nghiên cứu mô bệnh học vi thể, hiện tượng mất cân bằng diện
tích giữa các cấu trúc diễn ra rất phổ biến. Điển hình như ổ tế bào
lympho thâm nhiễm hoặc các vệt hoại tử tế bào vi thể thường chỉ chiếm
dưới 5% tổng diện tích mô trên tiêu bản nhưng lại chứa đựng ý nghĩa tiên
lượng sống còn^10^. Do đó, việc đánh giá hiệu năng phân đoạn WSSS không
thể dựa trên độ chính xác điểm ảnh thông thường mà phải sử dụng các chỉ
số hình thái chuyên biệt^4^.

Chỉ số Giao trên Hợp Trung bình (Mean Intersection over Union - mIoU)
phản ánh mức độ trùng khớp không gian giữa tập điểm ảnh dự đoán
${width="0.2132163167104112in"
height="0.27718175853018373in"} và tập điểm ảnh thực nghiệm
${width="0.24403215223097113in"
height="0.279919072615923in"} trên toàn bộ
${width="0.16851159230096238in"
height="0.273830927384077in"} phân lớp mô học^4^:

\text{mIoU} = \frac{1}{C} \sum_{c=1}^C \frac{|P_c \cap G_c|}{|P_c \cup G_c|}{width="6.458333333333333in"
height="0.909834864391951in"}

Hệ số Tương đồng Dice Trung bình (Mean Dice Similarity Coefficient -
mDice) nhấn mạnh vào sự cân bằng giữa độ nhạy và độ chính xác phân đoạn,
đồng thời tạo ra hình phạt nghiêm khắc hơn đối với các lỗi dự đoán sai
lệch vùng biên^4^:

\text{mDice} = \frac{1}{C} \sum_{c=1}^C \frac{2 |P_c \cap G_c|}{|P_c| + |G_c|}{width="6.458333333333333in"
height="0.909834864391951in"}

Chỉ số Giao trên Hợp có Trọng số Tần suất (Frequency-Weighted IoU -
FwIoU) giải quyết bài toán mất cân bằng phân phối diện tích bằng cách
gán trọng số cho từng lớp tỷ lệ thuận với số lượng điểm ảnh thực tế của
lớp đó trên toàn tiêu bản^10^:

\text{FwIoU} = \frac{1}{\sum_{c=1}^C |G_c|} \sum_{c=1}^C |G_c| \frac{|P_c \cap G_c|}{|P_c \cup G_c|}{width="6.458333333333333in"
height="0.909834864391951in"}

## Kiến trúc Phần mềm Trạm Làm việc Lâm sàng và Lộ trình Triển khai

Chuyển đổi một mô hình phân đoạn WSSS từ nghiên cứu thực nghiệm sang
phần mềm ứng dụng lâm sàng đòi hỏi một kiến trúc hướng dịch vụ mạnh mẽ,
giải quyết triệt để sự mất cân đối giữa khối lượng tính toán gigapixel
nặng nề và nhu cầu phản hồi tương tác mượt mà của bác sĩ bệnh học trên
giao diện web.

### Thiết kế Phân tầng Hệ thống và Luồng Xử lý Dữ liệu

Hệ thống trạm làm việc được tổ chức thành bốn phân tầng chức năng độc
lập nhằm đảm bảo tính mở rộng và khả năng chịu tải cao trong môi trường
bệnh viện:

1.  **Phân tầng Lưu trữ và Động cơ Cắt lát Kim tự tháp (Storage & Tiling
    Engine)**: Tiếp nhận các tệp slide thô định dạng SVS hoặc NDPI từ hệ
    thống máy quét tiêu bản^17^. Thư viện libvips kết hợp với OpenSlide
    chịu trách nhiệm chuyển đổi tệp WSI thành cấu trúc kim tự tháp đa độ
    phân giải Deep Zoom Image (DZI) hoặc chuẩn IIIF (International Image
    Interoperability Framework)^28^. Các lát cắt ảnh kích thước chuẩn
     \times 224${width="0.9046227034120735in"
    height="0.25746937882764653in"} pixel được đánh chỉ mục theo hệ tọa
    độ cấp độ phóng đại {level}/{x}\_{y}.png và phân phối tĩnh thông qua
    cụm lưu trữ đối tượng MinIO hoặc AWS S3, cho phép truy xuất ngẫu
    nhiên với độ trễ mạng dưới 150ms^29^.

2.  **Phân tầng Điều phối Tác vụ Nặng và Suy luận GPU (Asynchronous
    Processing Farm)**: Việc phân đoạn toàn bộ tiêu bản chứa hàng chục
    nghìn mảnh cắt tiêu tốn từ vài chục giây đến vài phút trên phần cứng
    GPU, do đó không thể xử lý trực tiếp trên chu kỳ phản hồi HTTP của
    máy chủ web. Cổng giao tiếp API (FastAPI) tiếp nhận yêu cầu phân
    tích và phát sinh thông điệp vào hàng đợi Redis. Các worker Celery
    phân tán trang bị card đồ họa sẽ lấy tác vụ, sử dụng TIAToolbox để
    thực thi suy luận cửa sổ trượt^24^. Toàn bộ ma trận xác suất phân
    loại đa lớp đầu ra được nén khối và lưu trữ bền vững dưới định dạng
    mảng đa chiều Zarr trên ổ đĩa NVMe, hỗ trợ đọc ngẫu nhiên cục bộ mà
    không cần nạp toàn bộ ma trận gigapixel vào bộ nhớ RAM^24^.

3.  **Phân tầng Cổng Dịch vụ và Định lượng Bệnh học (API Gateway &
    Analytics Layer)**: Cung cấp các giao diện lập trình RESTful phục vụ
    việc truy vấn dữ liệu lát cắt ảnh, trích xuất ma trận xác suất theo
    vùng quan tâm (ROI), tính toán các chỉ số diện tích thâm nhiễm tế
    bào học, và chuyển đổi đường viền phân đoạn thành các chuỗi đa giác
    vector GeoJSON^31^.

4.  **Phân tầng Giao diện Tương tác Phía Khách (Interactive Clinical Web
    UI)**: Xây dựng trên nền tảng React và TypeScript, sử dụng thư viện
    OpenSeadragon 6.x làm khung nhìn phóng to thu nhỏ chuyên dụng^29^.
    Mặt nạ phân đoạn đa lớp được dựng đè trực tiếp lên lớp ảnh vi thể
    H&E bằng bộ kết xuất tăng tốc phần cứng WebGL (OpenSeadragonGL) hoặc
    canvas đa lớp độ nét cao (CanvasOverlayHd)^29^. Giao diện tích hợp
    thanh điều khiển độ trong suốt (opacity slider) từ 0% đến 100%, bảng
    chuyển đổi hiển thị từng lớp mô riêng biệt, và các công cụ đo đạc
    kích thước vi thể trực quan^29^.

### Phương pháp Luận Định lượng Diện tích U Xâm lấn và Lympho Thâm nhiễm (sTILs)

Tính năng quan trọng nhất của trạm làm việc là chuyển đổi các bản đồ
phân đoạn điểm ảnh thành những dấu ấn sinh học có giá trị điều trị thực
tế^33^. Quy trình định lượng tỷ lệ tế bào lympho thâm nhiễm mô đệm
(stromal Tumor-Infiltrating Lymphocytes - sTILs) được thiết lập tuân thủ
nghiêm ngặt theo khuyến cáo của Nhóm Công tác Dấu ấn Miễn dịch Ung thư
Quốc tế (International Immuno-Oncology Biomarker Working Group) và tiêu
chuẩn của Trường Bệnh học Hoa Kỳ (College of American Pathologists -
CAP)^27^.

Quy trình bắt đầu bằng việc xác định không gian Giường khối u (Tumor
Bed - ký hiệu $\Omega_{\text{bed}}${width="0.3797736220472441in"
height="0.2602154418197725in"}), bao gồm toàn bộ diện tích khối u biểu
mô xâm lấn cùng với mạng lưới mô đệm xen kẽ giữa các đám tế bào ung
thư^27^. Hệ thống loại bỏ hoàn toàn khỏi vùng tính toán các diện tích
hoại tử, nang thoái hóa, các cấu trúc tiểu thùy bình thường và vùng mô
đệm nằm ngoài ranh giới xâm lấn của khối u^27^.

Tỷ lệ diện tích sTILs được tính bằng phần trăm diện tích bị xâm chiếm
bởi các tế bào lympho thâm nhiễm trên tổng diện tích khoang mô đệm khả
thi bên trong giường khối u^27^:

\text{sTILs (\%)} = \frac{\iint_{(x,y) \in \Omega_{\text{bed}}} \mathbb{I}(\hat{Y}(x,y) = \text{LYM}) \, dx dy}{\iint_{(x,y) \in \Omega_{\text{bed}}} (\mathbb{I}(\hat{Y}(x,y) = \text{LYM}) + \mathbb{I}(\hat{Y}(x,y) = \text{STR})) \, dx dy} \times 100{width="6.458333333333333in"
height="0.7805686789151356in"}

Biểu diễn dưới dạng tích phân rời rạc trên ma trận phân đoạn
$\hat{Y} \in \{0, 1, 2, 3, 4\}^{H \times W}${width="0.7547736220472441in"
height="0.27255686789151357in"} xuất ra từ mô hình:

\text{sTILs (\%)} = \frac{\text{Area}(\text{sTILs})}{\text{Area}(\text{Tumor Stroma})} \times 100{width="6.458333333333333in"
height="0.855145450568679in"}

Trong đó $\mathbb{I}(\cdot)${width="0.3098961067366579in"
height="0.25480314960629924in"} là hàm chỉ thị toán học nhận giá trị 1
khi điều kiện thỏa mãn và 0 trong trường hợp ngược lại. Song song với
chỉ số sTILs, hệ thống xác định Chỉ số Xâm lấn Biểu mô Khối u (Tumor
Extent Index - TEI) và Tỷ lệ Mô đệm trên Khối u (Stroma-to-Tumor Ratio -
STR) nhằm phản ánh mức độ phản ứng xơ hóa quanh u:

\text{TSR (\%)} = \frac{\text{Area}(\text{Tumor})}{\text{Area}(\text{Tumor}) + \text{Area}(\text{Stroma})} \times 100{width="6.458333333333333in"
height="0.7805686789151356in"}

Low ($< 10\%$), Intermediate (-50\%$), High ($> 50\%$){width="6.458333333333333in"
height="0.7805686789151356in"}

Dựa trên giá trị sTILs liên tục tính toán được, ca bệnh được tự động
phân nhóm vào ba tầng tiên lượng lâm sàng tiêu chuẩn: mức độ thấp (dưới
10%), mức độ trung bình (từ 10% đến 39%), và mức độ cao (từ 40% trở lên,
tương ứng với thể ung thư vú ưu thế lympho - LPBC)^27^. Nhóm có tỷ lệ
sTILs cao là dấu chỉ điểm sinh học mạnh mẽ cho khả năng đạt đáp ứng hoàn
toàn trên mô học (pathologic Complete Response - pCR) sau hóa trị tân bổ
trợ đối với bệnh nhân ung thư vú thể tam âm (TNBC) hoặc HER2-dương
tính^36^.

### Tích hợp Xuất Báo cáo Kết quả Bệnh học Cấu trúc Chuẩn CAP

Nhằm đảm bảo tính tích hợp vào quy trình vận hành thực tế của khoa Giải
phẫu bệnh, hệ thống trang bị công cụ tự động biên dịch báo cáo chẩn đoán
tóm tắt tuân theo các giao thức Báo cáo Tóm tắt (Synoptic Reporting
Protocols) của Trường Bệnh học Hoa Kỳ (CAP)^35^. Báo cáo được định dạng
bằng ngôn ngữ HTML/CSS và biên dịch sang tài liệu PDF độ phân giải cao
thông qua thư viện WeasyPrint.

Tài liệu báo cáo bao gồm ba khối thông tin tiêu chuẩn y khoa bắt buộc:

- **Siêu dữ liệu Ca bệnh và Truy xuất Nguồn gốc**: Mã số tiếp nhận bệnh
  phẩm (Accession Number), mã định danh bệnh nhân được ẩn danh hóa theo
  tiêu chuẩn HIPAA/GDPR, ngày sinh thiết, ngày quét quang học, loại
  thiết bị quét vi thể, độ phân giải quang học (Microns Per Pixel -
  MPP), và hàm băm (hash) xác thực phiên bản mô hình AI thực thi nhằm
  phục vụ kiểm toán lâm sàng.

- **Các Mục Kiểm tra Chẩn đoán Tóm tắt theo Quy chuẩn CAP (CAP Cancer
  Protocol Checklist)**: Phương thức lấy bệnh phẩm (sinh thiết lõi hoặc
  phẫu thuật cắt bỏ), phân loại mô bệnh học theo Tổ chức Y tế Thế giới
  (WHO), độ mô học Nottingham cải tiến (tạo ống tuyến, đa hình thái
  nhân, số lượng phân bào), kích thước vùng u lớn nhất đo được trên vi
  thể, và tình trạng diện cắt phẫu thuật^35^.

- **Kết quả Phân tích Định lượng AI và Dấu vết Vi thể**: Bảng tổng hợp
  các chỉ số định lượng bao gồm diện tích khối u xâm lấn tuyệt đối
  (^2${width="0.44259951881014875in"
  height="0.2599398512685914in"}), tỷ lệ sTILs (%) kèm phân nhóm tiên
  lượng, tỷ lệ diện tích hoại tử, và tỷ lệ mô đệm^27^. Đính kèm trong
  báo cáo là bản đồ nhiệt (heatmap) tổng quan phân bố các lớp mô trên
  toàn tiêu bản cùng các ảnh vi thể quang trường cao (High-Power Field -
  HPF) chụp tại các độ phóng đại 20x và 40x ghi nhận những ổ thâm nhiễm
  miễn dịch dày đặc nhất có đè ranh giới phân đoạn^33^. Báo cáo kết thúc
  bằng phần xác nhận, phê duyệt và ký số của bác sĩ giải phẫu bệnh phụ
  trách^39^.

### Lộ trình Kỹ thuật Triển khai Dự án Capstone 10-12 Tuần

Lộ trình phát triển được thiết lập thành bốn giai đoạn nối tiếp nhau,
tuân thủ các nguyên tắc kỹ thuật phần mềm y tế và kiểm soát chất lượng
thuật toán.

+-------+-------+---------------------------+---------+-------+
| **Giai đoạn | **Mục tiêu  | **Nhiệm vụ Kỹ thuật Chi tiết**                       | **Sản phẩm      | **Kế hoạch  |
| & Thời      | Kỹ thuật    |                                                      | Chuyển giao**   | Quản trị    |
| gian**      | Trọng tâm** |                                                      |                 | Rủi ro Kỹ   |
|             |             |                                                      |                 | thuật**     |
+-------+-------+---------------------------+---------+-------+
| **Giai đoạn | Thiết lập   | \- Cấu hình môi trường tính toán PyTorch, CUDA,      | \- Kho dữ liệu  | **Rủi ro:** |
| 1: Tuần 1   | hạ tầng dữ  | OpenSlide và TIAToolbox^28^.                         | patch đã tiền   | Tràn dung   |
| - 3**      | liệu và     |                                                      | xử lý.          | lượng lưu   |
|             | đường ống   | \- Tải và lập chỉ mục hai tập dữ liệu BCSS-WSSS và   |                 | trữ cục bộ  |
|             | trích xuất  | LUAD-HistoSeg^7^.                                    | \- Module trích | khi trích   |
|             | đặc trưng   |                                                      | xuất embedding  | xuất hàng   |
|             | nền tảng    | \- Xây dựng đường ống cắt patch tự động ở các độ     | tối ưu hóa bộ   | triệu patch |
|             |             | phân giải 20x và 40x có lọc nền kính qua Otsu^10^.   | nhớ GPU.        | rời rạc.    |
|             |             |                                                      |                 |             |
|             |             | \- Tích hợp mô hình nền tảng Paige-AI Virchow và UNI | \- Báo cáo đối  | **Giải      |
|             |             | qua timm để tạo kho lưu trữ vector đặc trưng         | chuẩn hiệu năng | pháp:** Lưu |
|             |             | patch^12^.                                           | đọc WSI giữa    | trữ ma trận |
|             |             |                                                      | OpenSlide và    | điểm ảnh    |
|             |             |                                                      | libvips.        | dưới định   |
|             |             |                                                      |                 | dạng Zarr   |
|             |             |                                                      |                 | hoặc HDF5   |
|             |             |                                                      |                 | nén         |
|             |             |                                                      |                 | khối^24^.   |
+-------+-------+---------------------------+---------+-------+
| **Giai đoạn | Huấn luyện  | \- Cài đặt khung làm việc PBIP: thiết lập ngân hàng  | \- Checkpoint   | **Rủi ro:** |
| 2: Tuần 4   | mô hình     | ảnh nguyên mẫu bằng thuật toán K-means^1^.           | mô hình phân    | Nhãn giả    |
| - 6**      | WSSS và     |                                                      | đoạn WSSS hoàn  | giai đoạn 1 |
|             | hoàn thiện  | \- Áp dụng hàm mất mát tương đồng đối nghịch         | chỉnh.          | chứa nhiều  |
|             | cơ chế tinh | [x]{width="0.38270450568678915in"  |                 | vùng nhiễu  |
|             | chỉnh nhãn  | height="0.2296227034120735in"} và mô-đun ngưỡng động | \- Mã nguồn     | làm trôi    |
|             | giả         | $\tau_c${width="0.10405839895013123in" | pipeline sinh   | ranh giới   |
|             |             | height="0.23586504811898512in"}^1^.                  | CAM và tinh     | mô đệm.     |
|             |             |                                                      | chỉnh mặt nạ    |             |
|             |             | \- Huấn luyện mô hình phân đoạn giai đoạn 2          | giả lập.        | **Giải      |
|             |             | (SegFormer-B2 hoặc Mask2Former) trên các nhãn giả đã |                 | pháp:**     |
|             |             | lọc nhiễu^1^.                                        | \- Báo cáo phân | Tích hợp    |
|             |             |                                                      | tích bóc tách   | mô-đun      |
|             |             | \- Kiểm chuẩn hiệu năng thuật toán trên tập test     | (ablation       | triệt tiêu  |
|             |             | BCSS-WSSS theo các chỉ số mIoU, mDice, FwIoU^4^.     | study) đánh giá | nhiễu theo  |
|             |             |                                                      | đóng góp của    | điểm của    |
|             |             |                                                      | từng thành      | ARML hoặc   |
|             |             |                                                      | phần.           | ràng buộc   |
|             |             |                                                      |                 | vật lý      |
|             |             |                                                      |                 | PhysPro^2^. |
+-------+-------+---------------------------+---------+-------+
| **Giai đoạn | Xây dựng    | \- Xây dựng backend FastAPI: Endpoints quản lý       | \- Ứng dụng web | **Rủi ro:** |
| 3: Tuần 7   | trạm làm    | slide, truyền lát cắt DZI, khởi tạo tác vụ phân      | hiển thị mượt   | Trình duyệt |
| - 9**      | việc Web và | tích^29^.                                            | mà tiêu bản WSI | bị quá tải  |
|             | luồng suy   |                                                      | từ 1.25x đến    | bộ nhớ và   |
|             | luận không  | \- Cấu hình Celery và Redis quản lý luồng tính toán  | 40x^29^.        | tụt khung   |
|             | đồng bộ     | GPU worker xử lý tiêu bản ngầm.                      |                 | hình khi    |
|             |             |                                                      | \- Động cơ      | hiển thị    |
|             |             | \- Tích hợp Zarr store lưu trữ ma trận xác suất phân | truyền tải lát  | lớp phủ     |
|             |             | đoạn sau suy luận^24^.                               | cắt DZI độ trễ  | gigapixel.  |
|             |             |                                                      | thấp            |             |
|             |             | \- Phát triển giao diện React tích hợp OpenSeadragon | (\<150ms/tile). | **Giải      |
|             |             | 6.x và plugin OpenSeadragonGL^29^.                   |                 | pháp:** Áp  |
|             |             |                                                      | \- Hệ thống phủ | dụng cơ chế |
|             |             | \- Hoàn thiện tính năng đổi độ trong suốt lớp phủ    | mặt nạ phân     | WebGL chỉ   |
|             |             | phân đoạn và ẩn/hiện từng loại mô^29^.               | đoạn thời gian  | nạp và dựng |
|             |             |                                                      | thực trên trình | các mảnh    |
|             |             |                                                      | duyệt.          | mặt nạ nằm  |
|             |             |                                                      |                 | trong khung |
|             |             |                                                      |                 | nhìn hiển   |
|             |             |                                                      |                 | thị hiện    |
|             |             |                                                      |                 | tại^29^.    |
+-------+-------+---------------------------+---------+-------+
| **Giai đoạn | Định lượng  | \- Hiện thực hóa thuật toán tự động bóc tách giường  | \- Module xuất  | **Rủi ro:** |
| 4: Tuần 10  | sTILs, tích | khối u và tính toán sTILs, TEI^27^.                  | báo cáo PDF     | Sai lệch    |
| - 12**     | hợp báo cáo |                                                      | chuẩn CAP có    | tọa độ giữa |
|             | CAP PDF và  | \- Xây dựng module xuất file PDF báo cáo bệnh học    | chữ ký số và    | ảnh vi thể  |
|             | đóng gói hệ | tóm tắt chuẩn CAP bằng WeasyPrint^35^.               | nhật ký kiểm    | chụp phóng  |
|             | thống       |                                                      | toán^35^.       | to trên báo |
|             |             | \- Tích hợp tính năng tự động trích xuất các khung   |                 | cáo PDF và  |
|             |             | nhìn vi thể điển hình đính kèm vào báo cáo.          | \- Bản phân     | tọa độ      |
|             |             |                                                      | phối Docker     | slide gốc.  |
|             |             | \- Thực hiện kiểm thử tích hợp (integration tests)   | Compose triển   |             |
|             |             | và kiểm thử giao diện người dùng.                    | khai môi trường | **Giải      |
|             |             |                                                      | sản xuất bằng   | pháp:**     |
|             |             | \- Đóng gói toàn bộ ngăn xếp dịch vụ bằng Docker     | một câu lệnh.   | Chuẩn hóa   |
|             |             | Compose (FastAPI, React, Celery, Redis, MinIO).      |                 | toàn bộ các |
|             |             |                                                      | \- Bộ tài liệu  | phép biến   |
|             |             |                                                      | kỹ thuật hoàn   | đổi hình    |
|             |             |                                                      | chỉnh và báo    | học thông   |
|             |             |                                                      | cáo nghiệm thu  | qua ma trận |
|             |             |                                                      | dự án.          | Affine      |
|             |             |                                                      |                 | không gian  |
|             |             |                                                      |                 | tuyệt       |
|             |             |                                                      |                 | đối^31^.    |
+=============+=============+======================================================+=================+=============+

## Kết luận

Phân đoạn ngữ nghĩa giám sát yếu trên ảnh toàn tiêu bản đại diện cho một
bước tiến quan trọng trong việc tháo gỡ điểm nghẽn dữ liệu của giải phẫu
bệnh kỹ thuật số^2^. Việc kết hợp các biểu diễn không gian ẩn sâu sắc từ
các mô hình nền tảng như UNI, Virchow hoặc Prov-GigaPath với cơ chế gợi
ý nguyên mẫu trực quan của PBIP và các ràng buộc vật lý của PhysPro cho
phép đạt được độ chính xác phân đoạn tiệm cận các giải pháp học có giám
sát đầy đủ^1^. Khi các thuật toán này được tích hợp vào một kiến trúc
phần mềm hướng dịch vụ vững chắc--kết hợp khả năng truyền tải lát cắt
ảnh đa tỷ lệ của OpenSeadragon, động cơ tính toán phân tán của
TIAToolbox và quy chuẩn định lượng dấu ấn sinh học sTILs theo tiêu chuẩn
CAP--hệ thống sẽ tạo ra một trạm làm việc lâm sàng hiện đại, thu hẹp
khoảng cách giữa đổi mới học máy lý thuyết và thực tiễn chẩn đoán ung
thư học^24^.

#### Nguồn trích dẫn

1.  Prototype-Based Image Prompting for Weakly Supervised,
    [https://openaccess.thecvf.com/content/CVPR2025/papers/Tang_Prototype-Based_Image_Prompting_for_Weakly_Supervised_Histopathological_Image_Segmentation_CVPR_2025_paper.pdf](https://openaccess.thecvf.com/content/CVPR2025/papers/Tang_Prototype-Based_Image_Prompting_for_Weakly_Supervised_Histopathological_Image_Segmentation_CVPR_2025_paper.pdf)

2.  Weakly Supervised Histological Tissue Segmentation with Activation,
    [https://papers.miccai.org/miccai-2024/paper/1156_paper.pdf](https://papers.miccai.org/miccai-2024/paper/1156_paper.pdf)

3.  Weakly-Supervised Semantic Segmentation for Histopathology,
    [https://mmlab-sigs.github.io/groups/zwang/publications/6106.FangZ.pdf](https://mmlab-sigs.github.io/groups/zwang/publications/6106.FangZ.pdf)

4.  Physics-Grounded Weakly Supervised Histopathological Tissue,
    [https://papers.miccai.org/miccai-2026/paper/1322_paper.pdf](https://papers.miccai.org/miccai-2026/paper/1322_paper.pdf)

5.  Performance of 19 pathology foundation models on 31 weakly,
    [https://www.researchgate.net/figure/Performance-of-19-pathology-foundation-models-on-31-weakly-supervised-prediction-tasks-a_fig2_396094020](https://www.researchgate.net/figure/Performance-of-19-pathology-foundation-models-on-31-weakly-supervised-prediction-tasks-a_fig2_396094020)

6.  Synchronous Inhibition and Activation for Weakly Supervised,
    [https://papers.miccai.org/miccai-2025/paper/5096_paper.pdf](https://papers.miccai.org/miccai-2025/paper/5096_paper.pdf)

7.  ProBAG: Prototype-Guided Boundary-Aware Graph Diffusion for,
    [https://openreview.net/pdf?id=Nd2bHza845](https://openreview.net/pdf?id=Nd2bHza845)

8.  \[Literature Review\] Superpixel Boundary Correction for Weakly,
    [https://www.themoonlight.io/en/review/superpixel-boundary-correction-for-weakly-supervised-semantic-segmentation-on-histopathology-images](https://www.themoonlight.io/en/review/superpixel-boundary-correction-for-weakly-supervised-semantic-segmentation-on-histopathology-images)

9.  ProBAG: Prototype-Guided Boundary-Aware Graph Diffusion \... -
    arXiv,
    [https://arxiv.org/pdf/2608.11765](https://arxiv.org/pdf/2608.11765)

10. Toward Efficient Weakly Supervised Semantic Segmentation Using,
    [https://arxiv.org/html/2607.10783v1](https://arxiv.org/html/2607.10783v1)

11. Toward Efficient Weakly Supervised Semantic Segmentation Using,
    [https://arxiv.org/abs/2607.10783](https://arxiv.org/abs/2607.10783)

12. MahmoodLab/UNI - Hugging Face,
    [https://huggingface.co/MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI)

13. A whole-slide foundation model for digital pathology from
    real-world,
    [https://www.researchgate.net/publication/380788897_A_whole-slide_foundation_model_for_digital_pathology_from_real-world_data](https://www.researchgate.net/publication/380788897_A_whole-slide_foundation_model_for_digital_pathology_from_real-world_data)

14. MahmoodLab/CONCH - Hugging Face,
    [https://huggingface.co/MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/CONCH)

15. paige-ai/Virchow - Hugging Face,
    [https://huggingface.co/paige-ai/Virchow](https://huggingface.co/paige-ai/Virchow)

16. Image Feature Extraction Models - Hugging Face,
    [https://huggingface.co/models?pipeline_tag=image-feature-extraction](https://huggingface.co/models?pipeline_tag=image-feature-extraction)

17. prov-gigapath/prov-gigapath - Hugging Face,
    [https://huggingface.co/prov-gigapath/prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath)

18. prov-gigapath/prov-gigapath-flash - Hugging Face,
    [https://huggingface.co/prov-gigapath/prov-gigapath-flash](https://huggingface.co/prov-gigapath/prov-gigapath-flash)

19. QingchenTang/PBIP: Prototype-Based Image Prompting for Weakly,
    [https://github.com/QingchenTang/PBIP](https://github.com/QingchenTang/PBIP)

20. GitHub - ChuHan89/WSSS-Tissue,
    [https://github.com/ChuHan89/WSSS-Tissue](https://github.com/ChuHan89/WSSS-Tissue)

21. Causal spectral segmenter: counterfactual graph reasoning for,
    [https://academic.oup.com/bioinformatics/article/42/9/btag674/8789989](https://academic.oup.com/bioinformatics/article/42/9/btag674/8789989)

22. yzouag/WSSS4LUAD: this repo is for Grand Challenge \... - GitHub,
    [https://github.com/yzouag/WSSS4LUAD](https://github.com/yzouag/WSSS4LUAD)

23. Computational Pathology Toolbox developed by TIA Centre \... -
    GitHub,
    [https://github.com/tissueimageanalytics/tiatoolbox](https://github.com/tissueimageanalytics/tiatoolbox)

24. semantic_segmentor - TIA Toolbox 2.1.3 Documentation,
    [https://tia-toolbox.readthedocs.io/en/latest/\_autosummary/tiatoolbox.models.engine.semantic_segmentor.html](https://tia-toolbox.readthedocs.io/en/latest/_autosummary/tiatoolbox.models.engine.semantic_segmentor.html)

25. tiatoolbox: a computational pathology toolbox for whole-slide
    images,
    [https://hysenlabs.com/projects/tissueimageanalytics-tiatoolbox](https://hysenlabs.com/projects/tissueimageanalytics-tiatoolbox)

26. Semantic Segmentation Models - TIA Toolbox 2.1.2 Documentation,
    [https://tia-toolbox.readthedocs.io/en/v2.1.2/\_notebooks/jnb/06-semantic-segmentation.html](https://tia-toolbox.readthedocs.io/en/v2.1.2/_notebooks/jnb/06-semantic-segmentation.html)

27. Assessment of Tumor-Infiltrating Lymphocytes in Triple-Negative,
    [https://www.mdpi.com/2075-4418/15/19/2492](https://www.mdpi.com/2075-4418/15/19/2492)

28. TIAToolbox as an end-to-end library for advanced tissue image,
    [https://pmc.ncbi.nlm.nih.gov/articles/PMC9509319/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9509319/)

29. OpenSeadragon,
    [https://openseadragon.github.io/](https://openseadragon.github.io/)

30. IIIF/awesome-iiif at engineering.q42.nl - GitHub,
    [https://github.com/IIIF/awesome-iiif?ref=engineering.q42.nl](https://github.com/IIIF/awesome-iiif?ref=engineering.q42.nl)

31. AnchorR: A QuPath and R interface for collaborative exploration of,
    [https://www.biorxiv.org/content/10.64898/2026.08.05.742985v1.full-text](https://www.biorxiv.org/content/10.64898/2026.08.05.742985v1.full-text)

32. rendeirolab/wsidata-viewer · GitHub - GitHub,
    [https://github.com/rendeirolab/wsidata-viewer](https://github.com/rendeirolab/wsidata-viewer)

33. Scoring of tumor-infiltrating lymphocytes: From visual estimation
    to,
    [https://www.researchgate.net/publication/326256963_Scoring_of_tumor-infiltrating_lymphocytes_From_visual_estimation_to_machine_learning](https://www.researchgate.net/publication/326256963_Scoring_of_tumor-infiltrating_lymphocytes_From_visual_estimation_to_machine_learning)

34. Automated scoring methods for quantitative interpretation of Tumour,
    [https://pmc.ncbi.nlm.nih.gov/articles/PMC11440723/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11440723/)

35. CAP_Foundation_Cancer_Proto,
    [https://www.slideshare.net/slideshow/capfoundationcancerprotocolsjune2023pdf/259266017](https://www.slideshare.net/slideshow/capfoundationcancerprotocolsjune2023pdf/259266017)

36. Interobserver variability in the assessment of stromal tumor,
    [https://iris.unito.it/bitstream/2318/1801942/4/nihms-1717487.pdf](https://iris.unito.it/bitstream/2318/1801942/4/nihms-1717487.pdf)

37. A Computational Tumor-Infiltrating Lymphocyte Assessment Method,
    [https://pmc.ncbi.nlm.nih.gov/articles/PMC8318866/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8318866/)

38. The pathology report as a boundary object: From clinical,
    [https://www.parapathology.com/clippings/the-pathology-report-as-a-boundary-object-from-clinical-communication-to-computational-representatio](https://www.parapathology.com/clippings/the-pathology-report-as-a-boundary-object-from-clinical-communication-to-computational-representatio)

39. VoiceOver PRO -- pathology reporting layer that sits above your,
    [https://www.voicebrook.com/voiceover-pro](https://www.voicebrook.com/voiceover-pro)

40. How Structured Data is Powering AI in Pathology,
    [https://digitalpathologyplace.com/podcast/why-structured-reporting-is-the-future-of-pathology-mtuitive-on-workflow-data-compliance-with-peter-otoole/](https://digitalpathologyplace.com/podcast/why-structured-reporting-is-the-future-of-pathology-mtuitive-on-workflow-data-compliance-with-peter-otoole/)

41. Whole Slide Image Registration - TIA Toolbox 2.1.3 Documentation,
    [https://tia-toolbox.readthedocs.io/en/stable/\_notebooks/jnb/10-wsi-registration.html](https://tia-toolbox.readthedocs.io/en/stable/_notebooks/jnb/10-wsi-registration.html)
