# Phân Vùng Ngữ Nghĩa và Hỗ Trợ Chẩn Đoán Ảnh Tiêu Bản Toàn Phần Bằng Học Giám Sát Yếu: Chiến Lược Nghiên Cứu và Triển Khai Phần Mềm Lâm Sàng

Ảnh tiêu bản giải phẫu bệnh toàn phần (Whole Slide Image - WSI) là dữ
liệu số hóa gigapixel mô tả toàn diện vi môi trường tế bào và mô học ở
độ phóng đại cao (\times${width="0.3810761154855643in"
height="0.2563604549431321in"} hoặc
\times${width="0.3810761154855643in"
height="0.2563604549431321in"}), với kích thước thông thường vượt mức
,000 \times 50,000${width="1.4521456692913386in"
height="0.2583147419072616in"} điểm ảnh và dung lượng tệp từ 1 đến 3 GB
mỗi tiêu bản^1^. Việc xây dựng các hệ thống trí tuệ nhân tạo hỗ trợ phân
vùng ngữ nghĩa (semantic segmentation) và chẩn đoán mô bệnh học gặp phải
rào cản lớn do chi phí gán nhãn thủ công mức pixel đòi hỏi hàng chục giờ
làm việc của các bác sĩ giải phẫu bệnh giàu kinh nghiệm, đồng thời tồn
tại độ biến thiên nhận định liên chuyên gia đáng kể^1^. Để giải quyết
điểm nghẽn này, phương pháp học giám sát yếu (Weakly Supervised
Learning - WSL) kết hợp học đa thực thể (Multiple Instance Learning -
MIL) và phân vùng ngữ nghĩa giám sát yếu (Weakly Supervised Semantic
Segmentation - WSSS) dựa trên nhãn chẩn đoán cấp tiêu bản (slide-level
labels) đã trở thành hướng đi chủ đạo trong giải phẫu bệnh tính toán^4^.

Quá trình chuyển dịch từ phân loại cấp tiêu bản sang bản địa hóa và phân
vùng chuẩn xác cấp pixel đặt ra nhiều thách thức lý thuyết lẫn kỹ thuật
phần mềm. Báo cáo phân tích chuyên sâu các giải pháp thuật toán tiết
kiệm tài nguyên tính toán, kiến trúc hệ thống phần mềm đáp ứng thực tiễn
lâm sàng, các công trình khoa học có thiết kế tối ưu, cùng các kho dữ
liệu mở kèm tập đặc trưng trích xuất sẵn nhằm hỗ trợ việc xây dựng đồ án
nghiên cứu đạt chuẩn khoa học và có tính ứng dụng cao.

## Hướng Tiếp Cận Thuật Toán Tối Ưu Cho Nghiên Cứu Tiết Kiệm Tài Nguyên

Nghịch lý lớn nhất đối với các nhóm nghiên cứu học thuật khi tiếp cận
WSI là sự chênh lệch giữa quy mô dữ liệu gigapixel và giới hạn bộ nhớ
phần cứng GPU tiêu chuẩn^1^. Việc huấn luyện một mạng nơ-ron tích chập
(CNN) hoặc Vision Transformer (ViT) nguyên khối từ đầu theo cơ chế lan
truyền ngược toàn phần trên toàn bộ tiêu bản là bất khả thi^1^. Chiến
lược giải quyết hiệu quả nhất hiện nay là áp dụng quy trình tách rời hai
giai đoạn (decoupled two-stage framework), kết hợp giữa việc trích xuất
biểu diễn ngoại tuyến thông qua các mô hình nền tảng bệnh học (Pathology
Foundation Models - PFMs) và tối ưu hóa các bộ tổng hợp MIL gọn nhẹ^8^.

Trong giai đoạn tiền xử lý, tiêu bản WSI được phân đoạn tự động để loại
bỏ nền kính quang học, sau đó được phân cắt thành hàng nghìn lát cắt nhỏ
(patches hoặc tiles) với kích thước cố định như
 \times 256${width="0.9046227034120735in"
height="0.25746937882764653in"} hoặc
 \times 512${width="0.9046227034120735in"
height="0.25746937882764653in"} pixel^3^. Thay vì sử dụng các mạng trích
xuất đặc trưng huấn luyện trên ảnh tự nhiên như ImageNet vốn thiếu độ
nhạy với hình thái nhân tế bào và chất đệm vi thể, các nghiên cứu tiên
tiến hiện nay tận dụng các mô hình nền tảng được huấn luyện tự giám sát
(Self-Supervised Learning - SSL) trên hàng chục triệu lát cắt mô bệnh
học như UNI, Prov-GigaPath, Virchow, CTransPath hoặc Phikon^10^. Bằng
cách đóng băng trọng số của mô hình nền tảng và trích xuất véc-tơ đặc
trưng ngoại tuyến (offline feature extraction), một tiêu bản gigapixel
phức tạp được rút gọn thành một ma trận đặc trưng
 \in \mathbb{R}^{N \times D}${width="1.0044488188976377in"
height="0.27203849518810147in"}, trong đó
${width="0.19542104111986003in"
height="0.27219378827646545in"} biểu diễn số lượng lát mô và
${width="0.18326881014873142in"
height="0.2647211286089239in"} là số chiều nhúng đặc trưng (thường là
768 hoặc 1024)^9^. Toàn bộ pha huấn luyện phân loại và phân vùng sau đó
chỉ diễn ra trên không gian véc-tơ này, cho phép huấn luyện hoàn tất
trong vài chục phút trên các dòng GPU phổ thông mà không tiêu tốn tài
nguyên tính toán lặp lại^7^.

Để chuyển đổi từ dự đoán nhị phân cấp tiêu bản sang phân vùng ngữ nghĩa
chi tiết, hướng tiếp cận thông minh là tạo lập mặt nạ giả định
(pseudo-mask generation) thông qua cơ chế bản đồ kích hoạt lớp (Class
Activation Mapping - CAM) hoặc trọng số chú ý (attention weights) của
mạng MIL^5^. Tuy nhiên, các phương pháp CAM cổ điển thường gặp hiện
tượng chỉ kích hoạt các vùng lõi có tính phân biệt cao nhất
(discriminative cores) mà bỏ qua các ranh giới mô thâm nhiễm^15^. Nhằm
khắc phục hạn chế này mà không cần gán nhãn pixel, các nghiên cứu gần
đây áp dụng cơ chế tổng hợp hình học và ràng buộc nhất quán đặc
trưng^15^. Tiêu biểu là việc khai thác các lát mô đơn thành phần
(homogeneous patches) để tổng hợp thành các tiêu bản nhân tạo thông qua
phép biến đổi ngẫu nhiên dạng lưới Mosaic hoặc đường cong Bézier^15^. Vì
vị trí ghép nối được kiểm soát hoàn toàn bằng thuật toán, hệ thống tự
động sở hữu nhãn phân vùng mức pixel hoàn hảo cho dữ liệu tổng hợp, từ
đó chuyển bài toán học giám sát yếu thành bài toán học có giám sát đầy
đủ trên tập dữ liệu mô phỏng^15^.

  -------------------------------------------------------------
  **Phương pháp**  **Yêu cầu nhãn dữ  **Cơ chế trích    **Chi phí phần **Độ phân giải phân vùng**
                   liệu**             xuất đặc trưng**  cứng huấn      
                                                        luyện**        
  -------- --------- --------- ------- --------------------------
  **Phân vùng toàn Mặt nạ đa giác cấp Huấn luyện        Cực cao (Yêu   Cực cao (Mức ranh giới tế bào và mô học)^1^
  phần (Fully      pixel trên toàn bộ end-to-end trên   cầu cụm GPU    
  Supervised)**    tiêu bản^1^        các mảng ảnh      chuyên dụng đa 
                                      lớn^1^            nhân)^1^       

  **Học đa thực    Chỉ cần nhãn chẩn  ResNet-50 tiền    Rất thấp (Chạy Thấp (Khoanh vùng thô theo kích thước lát cắt
  thể cổ điển      đoán cấp tiêu bản  huấn luyện trên   tốt trên GPU    \times 512${width="0.9046227034120735in"
  (AB-MIL,         (Slide-level)^4^   tập ImageNet^9^   cá nhân)^7^    height="0.25746937882764653in"})^15^
  Max-Pooling)**                                                       

  **Phân vùng giám Nhãn cấp tiêu bản  Mô hình nền tảng  Thấp đến trung Cao (Mặt nạ tái tạo tiệm cận mức có giám sát)^16^
  sát yếu kết hợp  kết hợp tri thức   bệnh học (UNI,    bình (Khởi tạo 
  Foundation Model tự giám sát^16^    Virchow,          đặc trưng      
  (WSSS + PFM)**                      CTransPath)^10^   ngoại          
                                                        tuyến)^8^      
  -------------------------------------------------------------

## Các Công Trình Tiêu Biểu Với Kiến Trúc Trực Quan Và Hiệu Quả Vượt Trội

Việc lựa chọn mô hình tham chiếu đóng vai trò quyết định đến độ tin cậy
khoa học của đồ án. Các công trình nổi bật trong lĩnh vực giải phẫu bệnh
tính toán không chỉ đạt độ chính xác chẩn đoán vượt trội mà còn sở hữu
tư duy thiết kế toán học mạch lạc và khả năng giải thích hình thái rõ
ràng.

### CLAM: Cơ Chế Chú Ý Có Cổng Kết Hợp Phân Cụm Ràng Buộc

Được công bố trên tạp chí *Nature Biomedical Engineering*, mô hình CLAM
(Clustering-constrained Attention Multiple Instance Learning) của
Mahmood và cộng sự đã thiết lập một chuẩn mực trong phân loại và khoanh
vùng tổn thương WSI^6^. Kiến trúc cốt lõi của CLAM dựa trên mạng nơ-ron
chú ý có cổng (Gated-Attention Mechanism), cho phép gán cho mỗi lát mô
${width="0.10047790901137357in"
height="0.2612412510936133in"} một trọng số chú ý chuẩn hóa
${width="0.19444444444444445in"
height="0.2708333333333333in"} phản ánh mức độ đóng góp của lát mô đó
vào quyết định lâm sàng^6^:

a_i = \frac{\exp(w^T (\tanh(V h_i^T) \odot \text{sigm}(U h_i^T)))}{\sum_{j=1}^N \exp(w^T (\tanh(V h_j^T) \odot \text{sigm}(U h_j^T)))}{width="6.458333333333333in"
height="0.8700601487314086in"}

Trong công thức trên,  \in \mathbb{R}^D${width="0.7735454943132108in"
height="0.2578488626421697in"} là véc-tơ biểu diễn của lát mô
${width="0.10047790901137357in"
height="0.2612412510936133in"}, các ma trận
, U \in \mathbb{R}^{L \times D}${width="1.3085936132983378in"
height="0.2561800087489064in"} và véc-tơ
 \in \mathbb{R}^{L \times 1}${width="0.7703991688538933in"
height="0.25680008748906386in"} là các tham số học được, trong khi
$\odot${width="0.1667749343832021in"
height="0.25711176727909013in"} đại diện cho tích phần tử Hadamard^6^.
Đột phá của CLAM nằm ở việc tích hợp thêm một hàm mất mát phân cụm không
giám sát ở cấp độ thực thể (instance-level clustering loss)^6^. Mô hình
chủ động chọn ra các lát mô có điểm chú ý cao nhất và thấp nhất trong
một slide để gán nhãn giả định, sau đó ép không gian nhúng phải phân
tách rõ rệt các cụm mô bệnh học dương tính và âm tính^6^. Nhờ cơ chế
này, bản đồ nhiệt do CLAM tạo ra có khả năng phân biệt sắc nét ranh giới
giữa mô u và mô đệm lành tính xung quanh, khắc phục triệt để hiện tượng
mô hình học vẹt các đặc trưng giả tạo^6^.

### HisynSeg và PistoSeg: Phân Vùng Dựa Trên Tổng Hợp Hình Học

Khác biệt với các phương pháp cố gắng làm mịn bản đồ kích hoạt CAM bằng
các hàm hậu xử lý phức tạp, các công trình HisynSeg và PistoSeg giải
quyết bài toán WSSS bằng tư duy mô phỏng hình học trực quan^15^. Các tác
giả nhận định rằng mô bệnh học có tính đồng nhất cao ở các vùng mô
nhỏ^15^. Bằng cách tự động tách các lát cắt chứa thuần nhất một loại mô
(như biểu mô ung thư, mô đệm xơ hóa hoặc mô hoại tử), hệ thống sử dụng
thuật toán tạo đường phân cách Bézier ngẫu nhiên để ghép các lát mô này
lại với nhau^15^.

Quá trình này tạo ra hàng nghìn mẫu ảnh ghép có sẵn đường ranh giới phân
vùng chính xác ở cấp độ điểm ảnh^15^. Một mạng phân vùng ngữ nghĩa tiêu
chuẩn (như DeepLabV3+ với xương sống EfficientNet) sau đó được huấn
luyện trực tiếp trên tập dữ liệu mô phỏng này^15^. Đồng thời, các ảnh
bệnh học thực tế chưa có nhãn được đưa vào thông qua hàm mất mát điều
hòa tính nhất quán đặc trưng (feature consistency regularization) nhằm
đảm bảo mô hình không bị quá khớp với các đường cắt ghép nhân tạo^15^.
Kết quả thử nghiệm chứng minh phương pháp này vượt qua các thuật toán
WSSS truyền thống và thu hẹp đáng kể khoảng cách hiệu năng với các mô
hình học có giám sát đầy đủ^15^.

### EP-SAM: Tinh Chỉnh Mô Hình Phân Vùng Đa Năng Với Bộ Nhớ Tối Thiểu

Công trình EP-SAM (Enhanced Prompt Segment Anything Model) mang lại giải
pháp đột phá trong việc tận dụng mô hình nền tảng thị giác tổng quát SAM
cho phân vùng mô bệnh học giám sát yếu^20^. Mô hình SAM vốn sở hữu khả
năng nhận diện đường viền vật thể chính xác nhưng thiếu hiểu biết chuyên
sâu về mặt ngữ nghĩa bệnh học^20^. EP-SAM giải quyết vấn đề bằng cách
dùng các tín hiệu kích hoạt từ mô hình học giám sát yếu để tự động tạo
ra các điểm gợi ý (point prompts) và khung gợi ý (box prompts) đưa vào
SAM^20^. Toàn bộ mạng SAM gốc được giữ cố định trọng số, chỉ có các bộ
điều hợp ngữ nghĩa trọng lượng nhẹ (lightweight adapters) được cập
nhật^20^. Kiến trúc thông minh này cho phép thực thi toàn bộ quy trình
huấn luyện phân vùng WSI phức tạp chỉ với 12 GB bộ nhớ GPU, tạo điều
kiện thuận lợi cho việc nghiên cứu trên phần cứng máy trạm phổ
thông^20^.

+-------+-------+----------+---------+-------+
| **Công      | **Năm công  | **Nguyên lý cốt    | **Tài nguyên    | **Ưu điểm   |
| trình**     | bố**        | lõi**              | phần cứng**     | vượt trội** |
+-------+-------+----------+---------+-------+
| **CLAM**    | 2021        | Chú ý có cổng kết  | 1 GPU thông     | Mã nguồn    |
|             |             | hợp ràng buộc phân | thường (Chạy    | hoàn thiện, |
| \[cite: 6,  |             | cụm đa thực thể^6^ | trên đặc trưng  | bản đồ      |
| 21\]        |             |                    | nhúng)^7^       | nhiệt trực  |
|             |             |                    |                 | quan, phân  |
|             |             |                    |                 | tách ranh   |
|             |             |                    |                 | giới rõ     |
|             |             |                    |                 | ràng^6^     |
+-------+-------+----------+---------+-------+
| **HisynSeg  | 2024        | Tạo ảnh ghép       | GPU tầm trung   | Phân vùng   |
| /           |             | Mosaic/Bézier biến | (Huấn luyện     | đa thành    |
| PistoSeg**  |             | bài toán WSL thành | DeepLabV3+)^15^ | phần vi môi |
|             |             | có giám sát^15^    |                 | trường u mà |
| \[cite: 15, |             |                    |                 | không phụ   |
| 17\]        |             |                    |                 | thuộc CAM   |
|             |             |                    |                 | thô^15^     |
+-------+-------+----------+---------+-------+
| **EP-SAM**  | 2024        | Dùng tín hiệu giám | Tối thiểu 12 GB | Ranh giới   |
|             |             | sát yếu làm Prompt | VRAM GPU^20^    | phân vùng   |
| \[cite:     |             | tự động điều hướng |                 | sắc nét,    |
| 20\]        |             | SAM^20^            |                 | chuyển giao |
|             |             |                    |                 | tri thức từ |
|             |             |                    |                 | SAM xuất    |
|             |             |                    |                 | sắc^20^     |
+-------+-------+----------+---------+-------+
| **SSHR**    | 2024        | Hiệu chỉnh phân    | GPU thông       | Rút ngắn    |
|             |             | cấp đơn giai đoạn  | thường^23^      | thời gian   |
| \[cite:     |             | (Single-stage      |                 | huấn luyện  |
| 23\]        |             | rectification)^23^ |                 | từ 2 đến 5  |
|             |             |                    |                 | lần so với  |
|             |             |                    |                 | WSSS hai    |
|             |             |                    |                 | giai        |
|             |             |                    |                 | đoạn^23^    |
+=============+=============+====================+=================+=============+

## Kho Dữ Liệu Mở và Khung Mã Nguồn Nền Tảng Khởi Đầu

Để triển khai đồ án thành công trong thời gian ngắn mà không tiêu tốn
tài nguyên tải và xử lý hàng terabyte dữ liệu ảnh thô, các nhóm nghiên
cứu nên khai thác các bộ dữ liệu đối sánh đã được tiền trích xuất đặc
trưng bởi cộng đồng khoa học quốc tế.

### CAMELYON16: Bộ Dữ Liệu Đối Sánh Chuẩn Cho Phát Hiện Di Căn

CAMELYON16 là tập dữ liệu kinh điển và bắt buộc phải có trong các nghiên
cứu giải phẫu bệnh tính toán, bao gồm 399 tiêu bản mô hạch bạch huyết
nhuộm H&E từ các bệnh nhân ung thư vú được quét ở độ phóng đại
\times${width="0.3810761154855643in"
height="0.2563604549431321in"} tại Trung tâm Y tế Đại học Radboud và
Trung tâm Y tế Đại học Utrecht^14^.

Đặc tính y khoa của bộ dữ liệu này mang tính thử thách cao: nhiều tiêu
bản dương tính chỉ chứa các ổ vi di căn (micrometastases) chiếm diện
tích dưới 1% tổng thể tích mô của lát cắt^14^. Điểm thuận lợi đặc biệt
cho nghiên cứu giám sát yếu là tập dữ liệu cung cấp nhãn chẩn đoán nhị
phân cấp tiêu bản cho quá trình huấn luyện, đồng thời đi kèm mặt nạ ranh
giới tổn thương cấp pixel được vẽ tỉ mỉ bởi các chuyên gia giải phẫu
bệnh^19^. Điều này cho phép nhóm nghiên cứu huấn luyện mô hình bằng nhãn
yếu nhưng có thể đánh giá mức độ chính xác của phân vùng thông qua chỉ
số diện tích dưới đường cong ROC mức pixel (Pixel-level AUC) và đường
cong FROC (Free-response ROC)^4^. Hiện nay, kho lưu trữ Hugging Face và
thư viện mã nguồn mở torchmil đã chia sẻ sẵn toàn bộ tập véc-tơ đặc
trưng nhúng của CAMELYON16 trích xuất bởi mô hình nền tảng UNI
(kaczmarj/camelyon16-uni, dung lượng 17,3 GB), cho phép nạp trực tiếp
vào mô hình huấn luyện mà không cần tải hàng trăm gigabyte ảnh gốc^24^.

### Các Phân Nhóm Tiêu Biểu Từ The Cancer Genome Atlas (TCGA)

Kho lưu trữ ung thư học The Cancer Genome Atlas cung cấp hàng chục nghìn
tiêu bản mô bệnh học mở từ nhiều vị trí giải phẫu^3^. Hai bài toán phân
loại phân nhóm mô học tiêu chuẩn thường được lựa chọn trong các đồ án
gồm:

- **TCGA-NSCLC (Non-Small Cell Lung Cancer):** Phân loại phân nhóm ung
  thư phổi giữa Ung thư biểu mô tuyến (Lung Adenocarcinoma - LUAD) và
  Ung thư biểu mô tế bào vảy (Lung Squamous Cell Carcinoma - LUSC)^3^.

- **TCGA-RCC (Renal Cell Carcinoma):** Phân loại ba phân nhóm ung thư
  biểu mô tế bào thận gồm thể tế bào trong (KIRC), thể nhú (KIRP), và
  thể kỵ màu (KICH)^3^.

Cả hai bài toán trên đều có sẵn quy trình xử lý và bộ đặc trưng trích
xuất chuẩn trong kho mã nguồn của dự án CLAM và gói mở rộng Phikon từ
Owkin, giúp nhóm nghiên cứu dễ dàng tái lập kết quả cơ sở chỉ sau vài
lệnh thực thi^7^.

### WSSS4LUAD và BCSS: Chuyên Biệt Cho Phân Vùng Vi Môi Trường Khối U

Khi định hướng nghiên cứu yêu cầu phân vùng đa lớp phức tạp trong vi môi
trường khối u thay vì bài toán nhị phân, hai tập dữ liệu sau là lựa chọn
tối ưu:

- **WSSS4LUAD:** Bộ dữ liệu giải đấu chuyên biệt về phân vùng ngữ nghĩa
  giám sát yếu trên ung thư biểu mô tuyến ở phổi, cung cấp các mảng mô
  lớn với nhãn phân loại đa lớp (biểu mô ung thư, mô đệm liên kết và mô
  hoại tử) phục vụ việc tinh chỉnh mặt nạ giả định^1^.

- **BCSS (Breast Cancer Semantic Segmentation):** Tập dữ liệu phân vùng
  mô bệnh học ung thư vú chi tiết với các nhãn phân đoạn vi môi trường u
  hoàn chỉnh, thường được sử dụng làm chuẩn đối sánh đánh giá năng lực
  tái tạo ranh giới mô của các mô hình WSSS^15^.

+--------+-------+-------+-------+----------+
| **Bộ dữ liệu / | **Loại mô   | **Nhiệm vụ  | **Kích      | **Nguồn trích xuất |
| Tài nguyên**   | bệnh học**  | khoa học    | thước &     | đặc trưng có sẵn** |
|                |             | chính**     | Trạng thái  |                    |
|                |             |             | dữ liệu**   |                    |
+--------+-------+-------+-------+----------+
| **CAMELYON16** | Hạch bạch   | Phát hiện   | 399 WSI; có | Hugging Face       |
|                | huyết ung   | di căn,     | sẵn nhãn    | (camelyon16-uni) / |
| \[cite: 8,     | thư vú^24^  | định vị ổ   | pixel kiểm  | torchmil           |
| 24\]           |             | vi di       | thử^14^     |                    |
|                |             | căn^14^     |             | \[cite: 24, 25\]   |
+--------+-------+-------+-------+----------+
| **TCGA-NSCLC** | Mô sinh     | Phân loại   | \>1.000     | Tích hợp trong     |
|                | thiết phổi  | phân nhóm   | WSI; nhãn   | CLAM /             |
| \[cite: 3, 8\] | (FFPE)^3^   | phụ LUAD vs | chẩn đoán   | HistoSSLscaling^7^ |
|                |             | LUSC^3^     | cấp tiêu    |                    |
|                |             |             | bản^3^      |                    |
+--------+-------+-------+-------+----------+
| **TCGA-RCC**   | Mô ung thư  | Phân loại 3 | \>800 WSI;  | Tích hợp trong cấu |
|                | thận^3^     | phân nhóm   | nhãn giải   | hình mẫu của       |
| \[cite: 3,     |             | KIRC, KIRP, | phẫu bệnh   | CLAM^7^            |
| 12\]           |             | KICH^3^     | học^3^      |                    |
+--------+-------+-------+-------+----------+
| **WSSS4LUAD**  | Ung thư     | Phân vùng   | Hàng nghìn  | Mã nguồn và dữ     |
|                | biểu mô     | đa cấu trúc | mảng mô;    | liệu từ ban tổ     |
| \[cite: 1,     | tuyến ở     | vi môi      | nhãn yếu    | chức Grand         |
| 17\]           | phổi^1^     | trường u^1^ | kèm test    | Challenge^1^       |
|                |             |             | pixel^1^    |                    |
+--------+-------+-------+-------+----------+
| **BCSS**       | Mô học ung  | Phân vùng   | 151 vùng    | GitHub PistoSeg /  |
|                | thư vú^15^  | ngữ nghĩa   | quan tâm    | HisynSeg^17^       |
| \[cite: 15,    |             | chất đệm,   | (ROI) WSI   |                    |
| 17\]           |             | hoại tử,    | gán nhãn    |                    |
|                |             | biểu mô^15^ | chi         |                    |
|                |             |             | tiết^15^    |                    |
+================+=============+=============+=============+====================+

## Kiến Trúc Phần Mềm Và Các Tính Năng Được Đánh Giá Cao Trong Lâm Sàng

Một thuật toán đạt chỉ số AUC cao trong môi trường thử nghiệm vẫn có thể
thất bại khi ứng dụng vào thực tế nếu không tương thích với quy trình
chẩn đoán của khoa Giải phẫu bệnh^28^. Các bác sĩ chuyên khoa làm việc
dưới áp lực thời gian lớn và đối mặt với trách nhiệm pháp lý cao đối với
từng kết luận vi thể^28^. Việc thiết kế phần mềm cần tập trung vào các
chức năng tối ưu hóa thời gian và cung cấp thông tin định lượng khách
quan.

### Sàng Lọc và Phân Luồng Ca Bệnh Tự Động (AI Triage & Prioritization)

Trong ca làm việc điển hình, một bác sĩ giải phẫu bệnh phải xử lý từ 50
đến hơn 100 ca xét nghiệm, trong đó phần lớn là các mẫu sinh thiết định
kỳ âm tính hoặc lành tính^28^. Tính năng phân luồng tự động hoạt động
ngầm ngay khi tiêu bản được máy quét tải lên máy chủ lưu trữ hình
ảnh^28^. Mô hình học giám sát yếu thực hiện suy luận nhanh để tính toán
xác suất ác tính^28^. Các ca bệnh có nguy cơ cao hoặc phát hiện thấy tổn
thương nghi ngờ di căn sẽ được tự động đẩy lên đầu danh sách hàng đợi
(worklist)^28^. Điều này giúp bệnh nhân có tiên lượng nặng được chẩn
đoán sớm hơn, đồng thời hệ thống đóng vai trò như một mắt đọc độc lập
thứ hai (second-read support) giúp giảm thiểu sai sót do quá tải thị
giác^28^.

### Kiểm Soát Chất Lượng Ảnh Tiền Phân Tích (Pre-analytic Quality Control)

Từ 5% đến 20% tiêu bản số hóa gặp phải các lỗi kỹ thuật trong quy trình
nhuộm hoặc quét ảnh, bao gồm nếp gấp mô, bọt khí dưới lá kính
(coverslip), dị vật bám trên tiêu bản và các vùng mô bị mất nét
(out-of-focus)^29^. Nếu không được kiểm soát, các vùng ảnh lỗi này có
thể khiến mô hình AI nhận diện sai lệch^28^. Phần mềm lâm sàng tiêu
chuẩn tích hợp các công cụ tự động như HistoQC ngay tại cổng tiếp nhận
dữ liệu^32^. Tiêu bản không đạt chuẩn chất lượng sẽ bị gắn cờ cảnh báo
và chuyển lại cho kỹ thuật viên phòng xét nghiệm để lau chùi hoặc quét
lại trước khi chuyển tới bàn làm việc của bác sĩ^28^.

### Kiến Trúc Hiển Thị Đa Độ Phân Giải Và Lớp Phủ Bản Đồ Chú Ý Tương Tác

Bác sĩ giải phẫu bệnh đưa ra kết luận dựa trên việc đối chiếu hình thái
nhân và liên kết mô học, do đó mô hình AI bắt buộc phải cung cấp bằng
chứng trực quan minh bạch^6^. Về mặt kỹ thuật phần mềm, việc hiển thị
ảnh gigapixel đòi hỏi hệ thống xây dựng cấu trúc kim tự tháp hình ảnh
(Image Pyramid) với định dạng Deep Zoom Image (DZI) hoặc OME-TIFF^33^.
Phía máy trạm sử dụng thư viện trình xem OpenSeadragon tương tác với máy
chủ thông qua giao diện lập trình ứng dụng viết bằng FastAPI và thư viện
chuyên dụng OpenSlide^28^. Hệ thống phân chia tiêu bản thành các mảnh
nhỏ kích thước  \times 512${width="0.9046227034120735in"
height="0.25746937882764653in"} pixel và chỉ nạp đúng tọa độ cùng mức
thu phóng mà bác sĩ đang quan sát^28^.

Lớp phủ bản đồ chú ý (Attention Heatmap) từ mô hình MIL hoặc mặt nạ phân
vùng WSSS được hiển thị dưới dạng một lớp phủ bán trong suốt (alpha
overlay) đè lên cấu trúc nhuộm H&E gốc^6^. Phần mềm cần trang bị thanh
trượt độ trong suốt tức thời (alpha blending slider) để bác sĩ có thể
bật/tắt lớp chỉ báo AI trong tích tắc^7^. Khi nghi ngờ một ổ vi di căn,
bác sĩ có thể phóng to trực tiếp lên vật kính
\times${width="0.3810761154855643in"
height="0.2563604549431321in"} tại đúng tọa độ AI gợi ý để quan sát chi
tiết nhân quái, tỷ lệ nhân trên nguyên sinh chất và các phân bào bất
thường^6^.

### Đo Lường Định Lượng Tải Lượng Khối U (Quantitative Burden Metrics)

Chẩn đoán ung thư hiện đại đòi hỏi các thông số định lượng chính xác
phục vụ xếp giai đoạn bệnh theo hệ thống TNM của Ủy ban Liên hợp Ung thư
Hoa Kỳ (AJCC)^2^. Thay vì để bác sĩ tự ước lượng diện tích u bằng mắt
hoặc đo thủ công bằng các công cụ thước đo thô sơ, phần mềm AI tự động
tính toán tổng diện tích khối u
(^2${width="0.4619149168853893in"
height="0.2729494750656168in"}), tỷ lệ phần trăm mô ung thư trên tổng
diện tích lát cắt, và đường kính lớn nhất của ổ xâm lấn^14^. Trong bài
toán ung thư vú di căn hạch, hệ thống tự động phân loại kích thước tổn
thương thành đại di căn (macrometastasis,
$> 2$ mm{width="0.33344160104986875in"
height="0.2570275590551181in"} mm), vi di căn (micrometastasis,
.2 - 2$ mm{width="0.6787106299212599in"
height="0.2588899825021872in"} mm) hoặc tế bào u biệt lập (isolated
tumor cells - ITC, $< 0.2$ mm{width="0.5359153543307087in"
height="0.260906605424322in"} mm)^14^. Việc cung cấp các chỉ số khách
quan này giúp chuẩn hóa biên bản chẩn đoán và giảm bớt thời gian tính
toán thủ công^28^.

### Tiêu Chuẩn Thẩm Định Lâm Sàng Theo Hướng Dẫn CAP

Để một công trình phần mềm y tế được giới chuyên môn công nhận và chuẩn
bị cho quá trình thử nghiệm lâm sàng, quy trình thẩm định cần tuân thủ
Hướng dẫn Thẩm định WSI của Hội Bác sĩ Giải phẫu bệnh Hoa Kỳ (College of
American Pathologists - CAP Guidelines)^2^.

Theo khuyến cáo cập nhật của CAP, tập dữ liệu thẩm định độc lập phải bao
gồm tối thiểu 60 ca bệnh phản ánh đầy đủ phổ bệnh học dự kiến triển
khai^2^. Độ tương đồng chẩn đoán (diagnostic concordance) giữa phương
pháp đọc trên ảnh số có AI hỗ trợ và phương pháp đọc truyền thống qua
kính hiển vi quang học phải đạt từ 95% trở lên^2^. Mọi trường hợp sai
lệch dưới 95% đều bắt buộc phải tiến hành phân tích nguyên nhân kỹ thuật
hoặc lỗi nhận định lâm sàng^34^. Đồng thời, để loại bỏ hiệu ứng thiên
lệch do trí nhớ của chuyên gia khi tham gia thử nghiệm đối chứng, nghiên
cứu phải thiết lập một khoảng thời gian rửa trôi (washout period) kéo
dài ít nhất 2 tuần giữa các lần đọc cùng một ca bệnh trên hai phương
thức khác nhau^2^.

  ------------------------------------
  **Thành phần phần **Vai trò trong   **Công nghệ triển  **Lợi ích đối với
  mềm**             quy trình lâm     khai khuyến nghị** bác sĩ giải phẫu
                    sàng**                               bệnh**
  --------- --------- --------- ---------
  **Hệ thống tiền   Loại bỏ tiêu bản  HistoQC, Python    Tránh gián đoạn
  lọc QC**          mờ, bọt khí, lỗi  OpenCV, lọc màu    mạch đọc, giảm
                    mô gấp trước chẩn HSV^28^            thiểu sai sót do
                    đoán^29^                             ảnh hỏng^28^

  **Phân luồng ca   Ưu tiên các ca    Bộ phân loại MIL   Giảm thời gian
  bệnh (Triage)**   nguy cơ cao hoặc  nhẹ                chờ đợi cho bệnh
                    chứa vi tổn       (CLAM/AB-MIL)^8^   nhân nặng, hạn
                    thương lên đầu                       chế bỏ sót^28^
                    hàng đợi^28^                         

  **Trình xem       Hiển thị mượt mà  OpenSeadragon,     Tái lập cảm giác
  Gigapixel**       tiêu bản phóng    FastAPI,           điều khiển tương
                    đại cao không có  OpenSlide,         đương kính hiển
                    độ trễ^29^        Libvips^28^        vi quang học^2^

  **Lớp phủ bản đồ  Trực quan hóa tọa WebGL Canvas       Minh bạch hóa cơ
  chú ý**           độ nghi ngờ kèm   Overlay, Shader    chế AI, định
                    thanh trượt độ    colormap^7^        hướng tập trung
                    mờ^6^                                thị giác nhanh^6^

  **Mô-đun định     Tính diện tích u, Phép toán hình học Tự động hóa xếp
  lượng**           kích thước ổ di   điểm ảnh, tích     giai đoạn TNM,
                    căn và tải lượng  phân diện tích     chuẩn hóa biên
                    mô bệnh^14^       mô^28^             bản giải phẫu^14^
  ------------------------------------

## Lộ Trình Triển Khai Đồ Án Nghiên Cứu Khả Thi

Một đồ án nghiên cứu thành công đòi hỏi sự cân bằng giữa tính mới về mặt
giải thuật và độ hoàn thiện về mặt kỹ thuật phần mềm. Nhóm nghiên cứu có
thể chia quy trình triển khai thành bốn giai đoạn tuần tự, tận dụng tối
đa các tài nguyên mở và hạ tầng tính toán sẵn có:

Giai đoạn đầu tiên tập trung vào việc thiết lập dữ liệu nền tảng và môi
trường lập trình. Thay vì tải toàn bộ kho dữ liệu ảnh WSI thô, nhóm
nghiên cứu tải tập đặc trưng trích xuất sẵn của bộ dữ liệu CAMELYON16
bằng mô hình nền tảng UNI từ Hugging Face hoặc thư viện torchmil^24^.
Đồng thời, lưu trữ một vài tệp WSI gốc định dạng .svs để phục vụ xây
dựng giao diện hiển thị kim tự tháp ảnh^33^. Môi trường làm việc được
thiết lập chuẩn hóa với PyTorch, OpenSlide-Python, Py-vips và
FastAPI^28^.

Giai đoạn thứ hai dành cho việc phát triển và hoàn thiện lõi thuật toán
AI. Nhóm nghiên cứu triển khai mô hình cơ sở CLAM hoặc AB-MIL trên tập
đặc trưng có sẵn^6^. Đóng góp khoa học của đồ án có thể được mở rộng
thông qua việc tích hợp cơ chế tổng hợp hình học Mosaic/Bézier từ
PistoSeg hoặc xây dựng hàm mất mát điều hòa chú ý phản thực tế
(counterfactual attention regularization) để tinh chỉnh độ sắc nét của
ranh giới tổn thương^17^. Hiệu năng của thuật toán được đo lường bằng
hai tiêu chí: chỉ số AUC cấp tiêu bản cho bài toán chẩn đoán và chỉ số
FROC hoặc Pixel-AUC đối chiếu với nhãn ranh giới pixel có sẵn của tập
kiểm thử CAMELYON16^4^.

Giai đoạn thứ ba chuyển dịch sang kỹ thuật phần mềm với việc xây dựng
một nguyên mẫu ứng dụng hoàn chỉnh. Máy chủ dịch vụ (Backend) sử dụng
FastAPI cung cấp các điểm cuối phục vụ ảnh gạch DZI
(/tile/{z}/{x}/{y}.png) từ tệp WSI gốc và một điểm cuối trả về ma trận
trọng số nhiệt của mô hình^33^. Giao diện người dùng (Frontend) được
phát triển bằng React hoặc Vue tích hợp OpenSeadragon, xây dựng đầy đủ
thanh danh sách ca bệnh phân luồng theo độ nguy cơ, cửa sổ kính hiển vi
số đa độ phóng đại, thanh trượt alpha blending cho lớp phủ nhiệt và bảng
tóm tắt diện tích tải lượng khối u^6^.

Giai đoạn cuối cùng là kiểm thử và thẩm định thực nghiệm. Hệ thống được
đo lường thời gian đáp ứng (latency) trong các thao tác nạp ảnh và suy
luận AI, hướng tới mục tiêu thời gian xử lý dưới 1 giây cho mỗi tiêu bản
khi đã có ma trận đặc trưng. Kết quả nghiên cứu và giao diện phần mềm
được đóng gói thành báo cáo khoa học hoàn chỉnh, đối chiếu các tiêu chí
tương đồng chẩn đoán dựa trên khung hướng dẫn của Hội Bác sĩ Giải phẫu
bệnh Hoa Kỳ nhằm khẳng định giá trị thực tiễn của công trình trước hội
đồng chuyên môn^2^.

#### Nguồn trích dẫn

1.  WSSS4LUAD: Grand Challenge on Weakly-supervised Tissue,
    [https://arxiv.org/html/2204.06455v2](https://arxiv.org/html/2204.06455v2)

2.  Validation of Whole Slide Imaging Diagnosis for Breast Core Needle,
    [http://www.jmatonline.com/PDF/109-PB-353-364-03887.pdf](http://www.jmatonline.com/PDF/109-PB-353-364-03887.pdf)

3.  Scaling Self-Supervised Learning for Histopathology with Masked,
    [https://www.medrxiv.org/content/10.1101/2023.07.21.23292757v1.full.pdf](https://www.medrxiv.org/content/10.1101/2023.07.21.23292757v1.full.pdf)

4.  arXiv:2004.05024v4 \[eess.IV\] 11 May 2021,
    [https://arxiv.org/pdf/2004.05024](https://arxiv.org/pdf/2004.05024)

5.  Transformer based multiple instance learning for weakly supervised,
    [https://arxiv.org/html/2205.08878v1](https://arxiv.org/html/2205.08878v1)

6.  (PDF) Data Efficient and Weakly Supervised Computational,
    [https://www.researchgate.net/publication/340826474_Data_Efficient_and_Weakly_Supervised_Computational_Pathology_on_Whole_Slide_Images](https://www.researchgate.net/publication/340826474_Data_Efficient_and_Weakly_Supervised_Computational_Pathology_on_Whole_Slide_Images)

7.  mahmoodlab/CLAM: Open source tools for computational pathology,
    [https://github.com/mahmoodlab/clam](https://github.com/mahmoodlab/clam)

8.  Feature Re-Embedding: Towards Foundation Model-Level \... - arXiv,
    [https://arxiv.org/html/2402.17228v4](https://arxiv.org/html/2402.17228v4)

9.  RETHINKING MULTIPLE-INSTANCE LEARNING FROM FEATURE,
    [https://proceedings.iclr.cc/paper_files/paper/2025/file/463a91da3c832bd28912cd0d1b8d9974-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2025/file/463a91da3c832bd28912cd0d1b8d9974-Paper-Conference.pdf)

10. Pathology Foundation Models - arXiv,
    [https://arxiv.org/pdf/2407.21317](https://arxiv.org/pdf/2407.21317)

11. Pathology Foundation Models - PMC,
    [https://pmc.ncbi.nlm.nih.gov/articles/PMC11799676/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11799676/)

12. Scaling Self-Supervised Learning for Histopathology with Masked,
    [https://www.medrxiv.org/content/10.1101/2023.07.21.23292757v1.full-text](https://www.medrxiv.org/content/10.1101/2023.07.21.23292757v1.full-text)

13. README.md - owkin/HistoSSLscaling - GitHub,
    [https://github.com/owkin/HistoSSLscaling/blob/main/README.md](https://github.com/owkin/HistoSSLscaling/blob/main/README.md)

14. kaczmarj/metastasis-abmil-128um-uni - Hugging Face,
    [https://huggingface.co/kaczmarj/metastasis-abmil-128um-uni](https://huggingface.co/kaczmarj/metastasis-abmil-128um-uni)

15. HisynSeg: Weakly-Supervised Histopathological Image \... - arXiv,
    [https://arxiv.org/html/2412.20924v1](https://arxiv.org/html/2412.20924v1)

16. Prototype-Based Image Prompting for Weakly Supervised \... - arXiv,
    [https://arxiv.org/html/2503.12068v1](https://arxiv.org/html/2503.12068v1)

17. Weakly-Supervised Semantic Segmentation for Histopathology,
    [https://mmlab-sigs.github.io/groups/zwang/publications/6106.FangZ.pdf](https://mmlab-sigs.github.io/groups/zwang/publications/6106.FangZ.pdf)

18. Weakly Supervised Teacher-Student Framework with Progressive,
    [https://www.xiahepublishing.com/2771-165X/JCTP-2025-00055](https://www.xiahepublishing.com/2771-165X/JCTP-2025-00055)

19. Domain Adaptation Without the Compute Burden for Efficient Whole,
    [https://openreview.net/forum?id=gNyqM50ZY2](https://openreview.net/forum?id=gNyqM50ZY2)

20. EP-SAM: Weakly Supervised Histopathology Segmentation \... - arXiv,
    [https://arxiv.org/abs/2410.13621](https://arxiv.org/abs/2410.13621)

21. Data-efficient and weakly supervised computational pathology on,
    [https://pmc.ncbi.nlm.nih.gov/articles/PMC8711640/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8711640/)

22. Interpretability and visualization a,b, For RCC (a) and NSCLC (b),
    [https://www.researchgate.net/figure/Interpretability-and-visualization-a-b-For-RCC-a-and-NSCLC-b-subtyping-a_fig4_349707547](https://www.researchgate.net/figure/Interpretability-and-visualization-a-b-For-RCC-a-and-NSCLC-b-subtyping-a_fig4_349707547)

23. Single-Stage Hierarchical Rectification for Weakly Supervised,
    [https://arxiv.org/abs/2606.20250](https://arxiv.org/abs/2606.20250)

24. CAMELYON16 dataset - torchmil,
    [https://torchmil.readthedocs.io/en/latest/api/datasets/camelyon16mil_dataset/](https://torchmil.readthedocs.io/en/latest/api/datasets/camelyon16mil_dataset/)

25. kaczmarj/camelyon16-uni · Datasets at Hugging Face,
    [https://huggingface.co/datasets/kaczmarj/camelyon16-uni](https://huggingface.co/datasets/kaczmarj/camelyon16-uni)

26. Scaling Self-Supervised Learning for Histopathology with Masked,
    [https://www.researchgate.net/publication/372674514_Scaling_Self-Supervised_Learning_for_Histopathology_with_Masked_Image_Modeling](https://www.researchgate.net/publication/372674514_Scaling_Self-Supervised_Learning_for_Histopathology_with_Masked_Image_Modeling)

27. Counterfactual Attention Regularization for Multiple Instance
    Learning,
    [https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/9801.pdf](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/9801.pdf)

28. AI for Digital Pathology - Scnsoft,
    [https://www.scnsoft.com/healthcare/artificial-intelligence/digital-pathology](https://www.scnsoft.com/healthcare/artificial-intelligence/digital-pathology)

29. Technical and Diagnostic Issues in Whole Slide Imaging \... -
    Frontiers,
    [https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2022.918580/full](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2022.918580/full)

30. Deep-learning triage of 3D pathology datasets for comprehensive,
    [https://www.biorxiv.org/content/10.1101/2025.07.20.665804v1.full-text](https://www.biorxiv.org/content/10.1101/2025.07.20.665804v1.full-text)

31. Artificial Intelligence in Digital Pathology for Prostate Cancer
    Detection,
    [https://jcp.bmj.com/content/79/10/669](https://jcp.bmj.com/content/79/10/669)

32. Semantic Segmentation Based Quality Control of Histopathology,
    [https://arxiv.org/html/2410.03289v2](https://arxiv.org/html/2410.03289v2)

33. Entwicklung einer Lernsoftware für das Fach "Pathologie" - SerWisS,
    [https://serwiss.bib.hs-hannover.de/files/1972/BA_Jannes_Neemann.pdf](https://serwiss.bib.hs-hannover.de/files/1972/BA_Jannes_Neemann.pdf)

34. EARLY ONLINE RELEASE,
    [https://triangle-octagon-8fxh.squarespace.com/s/105858_arpa2020-0723-cp.pdf](https://triangle-octagon-8fxh.squarespace.com/s/105858_arpa2020-0723-cp.pdf)

35. Whole-slide imaging at primary pathological diagnosis - Ovid,
    [https://www.ovid.com/00041332-201711000-00003](https://www.ovid.com/00041332-201711000-00003)
