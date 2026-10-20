# Báo Cáo Kỹ Thuật: Hệ Sinh Thái Phần Mềm Và Các Mô Hình Tiên Tiến Phục Vụ Xử Lý, Phân Vùng Ảnh Tiêu Bản Gigapixel (WSI) Trong Giải Phẫu Bệnh Số

## Cấu Trúc Dữ Liệu Kim Tự Tháp Và Thách Thức Kỹ Thuật Của Ảnh Toàn Tiêu Bản

Ảnh tiêu bản toàn phần (Whole Slide Image - WSI) đại diện cho một trong
những phương thức dữ liệu có kích thước vật lý và dung lượng số lớn nhất
trong chẩn đoán y khoa hiện đại^1^. Một tiêu bản mô bệnh học thông
thường sau khi quét quang học ở độ phóng đại 20x hoặc 40x thường sở hữu
độ phân giải ma trận dao động từ
,000 \times 20,000${width="1.4521456692913386in"
height="0.2583147419072616in"} đến hơn
,000 \times 100,000${width="1.6665627734033246in"
height="0.2569280402449694in"} pixel, tương đương với kích thước tệp từ
1 GB đến 5 GB dưới dạng nén và vượt quá 40 GB khi giải mã ma trận thô
vào bộ nhớ RAM^1^. Khác với các định dạng chẩn đoán hình ảnh truyền
thống như DICOM trong chụp cắt lớp vi tính (CT) hay cộng hưởng từ (MRI)
vốn tải được trực tiếp toàn bộ khối thể tích vào bộ nhớ đồ họa, WSI
không thể đưa nguyên vẹn vào VRAM của GPU^3^.

Để giải quyết giới hạn vật lý này, các nhà sản xuất máy quét tiêu bản tổ
chức dữ liệu theo cấu trúc kim tự tháp đa tầng (multi-resolution
pyramid)^5^. Tầng cơ sở (Level 0) bảo toàn độ phân giải nguyên bản cao
nhất với hệ số kích thước điểm ảnh vật lý (Microns Per Pixel - MPP) xấp
xỉ 0.25 đến 0.5 $\mu m/\text{pixel}${width="0.6471358267716535in"
height="0.27137904636920385in"}^7^. Các tầng kế tiếp được giảm mẫu hình
học theo cấp số nhân (hệ số giảm mẫu 2x, 4x, 16x, 32x hoặc 64x) nhằm
phục vụ tác vụ duyệt tổng quan và hiển thị thời gian thực^9^. Định dạng
tệp này đòi hỏi các giải pháp phần mềm phải trừu tượng hóa việc đọc dữ
liệu thông qua cơ chế giải nén theo vùng logic (dynamic region tiling),
trong đó chỉ những mảnh ảnh nhỏ (patches hoặc tiles kích thước
 \times 256${width="0.9046227034120735in"
height="0.25746937882764653in"} hoặc
 \times 512${width="0.9046227034120735in"
height="0.25746937882764653in"} pixel) tại tọa độ không gian xác định
mới được nạp vào bộ nhớ tính toán^9^.

Quy trình phân tích tự động một tiêu bản WSI hoàn chỉnh tuân theo mô
hình xử lý ba giai đoạn nghiêm ngặt^9^:

1.  Tiền xử lý phân lập vùng mô học (tissue segmentation) nhằm loại bỏ
    nền kính quang học và các dị vật nhân tạo ở tầng phân giải thấp, sau
    đó tạo lưới tọa độ để cắt ảnh ở tầng phân giải mục tiêu^9^.

2.  Nạp dữ liệu song song và mã hóa đặc trưng cục bộ bằng mạng nơ-ron
    học sâu (feature extraction)^9^.

3.  Tổng hợp thông tin toàn slide phục vụ phân loại tổn thương thông qua
    học yếu hoặc suy luận dày đặc (dense prediction) nhằm phân lập từng
    cá thể nhân tế bào^9^.

## Các Kho Lưu Trữ Mã Nguồn Mở Phục Vụ Chu Trình Xử Lý Và Phân Vùng WSI

### CLAM: Học Đa Thực Thể Ràng Buộc Phân Cụm Dưới Giám Sát Yếu

Kho lưu trữ [CLAM](https://github.com/mahmoodlab/CLAM) do
Mahmood Lab tại Trường Y Harvard và Bệnh viện Phụ nữ Brigham phát triển,
công bố trên tạp chí *Nature Biomedical Engineering*, đóng vai trò là
một trong những chuẩn mực kỹ thuật trong giải phẫu bệnh tính toán^9^. Hệ
thống giải quyết triệt để rào cản chi phí gán nhãn bằng cách chỉ yêu cầu
một nhãn chẩn đoán duy nhất ở cấp độ toàn bộ tiêu bản mà không đòi hỏi
bất kỳ vùng khoanh tổn thương thủ công (ROI) nào^1^. Chu trình thực thi
của CLAM được tổ chức thành các tập lệnh tuần tự hóa: tiền xử lý nhị
phân hóa mặt nạ mô ở tầng phân giải thấp, xuất tọa độ điểm cắt vào định
dạng HDF5 (create_patches_fp.py), và trích xuất véc-tơ đặc trưng 1024
chiều bằng mô hình ResNet50 cắt tầng (extract_features_fp.py)^9^.

Kiến trúc mô hình của CLAM tích hợp mạng học đa thực thể (MIL) dựa trên
cơ chế chú ý (attention backbone) kết hợp hàm mất mát phân cụm có giám
sát (clustering-constrained loss)^9^. Ràng buộc này ép buộc không gian
đặc trưng phải tách biệt rõ rệt giữa các mảnh có trọng số đóng góp chẩn
đoán cao nhất và thấp nhất^15^. Đầu ra của CLAM cung cấp bản đồ nhiệt
chú ý (attention heatmap) có độ phân giải tương đương tầng Level 0, giúp
bác sĩ giải phẫu bệnh dễ dàng kiểm chứng lại các cụm tế bào ác tính^1^.

Đối với các đồ án kỹ thuật phần mềm y tế, CLAM là mã nguồn tham khảo
tiêu chuẩn nhờ kiến trúc mô-đun hóa rõ ràng, cấu trúc lưu trữ dữ liệu
trung gian chuẩn mực bằng HDF5 và PyTorch tensor, cho phép sinh viên
nhanh chóng tái tạo pipeline phân loại phân nhóm ung thư (subtyping) và
trực quan hóa bản đồ chú ý chẩn đoán^9^.

### HoVer-Net: Phân Vùng Cá Thể Và Phân Loại Nhân Tế Bào Đồng Thời

Mã nguồn [HoVer-Net](https://github.com/vqdang/hover_net)
do Trung tâm Phân tích Hình ảnh Mô (TIA Centre) tại Đại học Warwick phát
triển, công bố trên tạp chí *Medical Image Analysis*, là giải pháp hàng
đầu cho bài toán phân vùng cá thể (instance segmentation) ở cấp độ vi
thể^13^. Vấn đề nan giải nhất khi phân tích nhân tế bào trên các lát cắt
mô học nhuộm Hematoxylin & Eosin (H&E) là hiện tượng các nhân tế bào nằm
chen chúc hoặc đè lên nhau, khiến các mô hình phân đoạn ngữ nghĩa như
U-Net hoặc thuật toán Watershed cổ điển thường xuyên làm dính ranh giới
giữa các tế bào^13^.

HoVer-Net giải quyết vấn đề này bằng cách thiết kế nhánh giải mã khoảng
cách ngang và dọc (Horizontal and Vertical distance branches) song song
với nhánh phân đoạn nhị phân và nhánh phân loại tế bào^13^. Bằng việc
hồi quy gradient khoảng cách chuẩn hóa từ mỗi pixel thuộc nhân tới khối
tâm của chính nhân đó, mô hình tạo ra các đường phân cách biên sắc nét
ngay cả khi các tế bào tiếp xúc mật thiết^13^. Repository cung cấp sẵn
tập lệnh xử lý toàn diện cho tệp WSI gigapixel (run_wsi.sh), sử dụng kỹ
thuật cửa sổ trượt có lưu bộ đệm và tự động hậu xử lý nối liền các đa
giác bị cắt ngang giữa các lát^13^. Kết quả xuất ra ở định dạng GeoJSON
chuẩn hóa hoặc tệp tương thích trực tiếp với QuPath^13^.

Mã nguồn này thích hợp cho các đồ án tập trung vào phân tích định lượng
vi môi trường khối u (Tumor Microenvironment - TME), đo lường tỷ lệ tế
bào lympho xâm nhập khối u (TILs), hoặc phân tích hình thái học chất
nhiễm sắc^13^.

### Slideflow: Nền Tảng Học Sâu Toàn Trình Tích Hợp Hiển Thị Thời Gian Thực

Framework
[Slideflow](https://github.com/slideflow/slideflow) do
James Dolezal tại Viện Ung thư Geisinger phát triển và công bố trên *BMC
Bioinformatics*, đại diện cho một nền tảng công nghệ phần mềm hoàn chỉnh
cho giải phẫu bệnh số^20^. Slideflow loại bỏ sự phân mảnh giữa các công
cụ nghiên cứu bằng cách kết nối thông suốt từ khâu tiền xử lý, chuẩn hóa
màu nhuộm (stain normalization: Macenko, Vahadane, Reinhard), huấn luyện
mô hình phân loại đa nhiệm, đến khâu triển khai giao diện người dùng
tương tác^20^.

Điểm nổi bật của Slideflow là khả năng trừu tượng hóa phần cứng và tối
ưu hóa I/O: hệ thống hỗ trợ cả backend PyTorch lẫn TensorFlow, sử dụng
luồng nạp dữ liệu siêu tốc bằng TFRecords và tệp nhị phân được lưu trữ
trực tiếp trên RAM disk hoặc ổ cứng NVMe^20^. Slideflow tự động phát
hiện định dạng tệp và chuyển đổi giữa backend giải mã cuCIM (ưu tiên tốc
độ trên GPU) hoặc Libvips (hỗ trợ toàn diện các định dạng mở rộng như
.mrxs, .ndpi, .scn)^20^. Điểm độc đáo nhất là công cụ máy trạm tương tác
**Slideflow Studio**, cho phép nạp các mô hình đã huấn luyện để suy luận
và kết xuất bản đồ nhiệt phân loại theo thời gian thực khi người dùng
lướt chuột qua các vùng trên tiêu bản gigapixel^20^.

Slideflow sở hữu mức độ phù hợp toàn diện cho các đồ án kỹ thuật phần
mềm cần xây dựng một ứng dụng y tế có độ hoàn thiện công nghiệp, kết nối
từ tầng tiền xử lý đến tầng giao diện người dùng chuyên nghiệp^20^.

### WSInfer: Tối Ưu Hóa Tốc Độ Suy Luận Và Đóng Gói Mô Hình Chuẩn Hóa

Kho lưu trữ [WSInfer](https://github.com/SBU-BMI/wsinfer)
do Khoa Tin học Y sinh thuộc Đại học Stony Brook phát triển và công bố
trên *npj Precision Oncology*, tập trung chuyên biệt vào bài toán tối ưu
hóa hiệu năng suy luận (inference runtime) và phân phối mô hình trên quy
mô lớn^8^. Khi chuyển giao các nghiên cứu AI sang môi trường bệnh viện,
các mô hình học sâu thường gặp rào cản về tốc độ xử lý chậm và sự phụ
thuộc phức tạp vào môi trường lập trình Python^8^.

WSInfer giải quyết vấn đề này bằng cách chuẩn hóa các mô hình phân loại
lát cắt dưới định dạng tuần tự hóa TorchScript, cho phép thực thi suy
luận đồ thị tĩnh mà không cần truy cập vào mã nguồn Python ban đầu của
mô hình^25^. Hệ thống đạt tốc độ xử lý trung bình từ 20 đến 30 giây cho
toàn bộ một tiêu bản WSI kích thước lớn (diện tích mô trên 170
^2${width="0.44259951881014875in"
height="0.2599398512685914in"}) trên phần cứng GPU thông dụng^8^. Dự án
đi kèm kho mô hình tiền huấn luyện phong phú (WSInfer Model Zoo) trên
Hugging Face bao gồm các tác vụ phân loại ung thư vú, ung thư phổi, phân
độ mô học tuyến tiền liệt và nhận diện tế bào lympho^8^. Kết quả suy
luận được lưu dưới dạng CSV, HDF5 và GeoJSON, đồng thời tích hợp trực
tiếp vào phần mềm QuPath qua tiện ích mở rộng bản địa^8^.

WSInfer thích hợp cho các đồ án công nghệ thông tin y tế tập trung vào
kỹ thuật triển khai hệ thống (MLOps, microservices, container hóa bằng
Docker) phục vụ xử lý hàng loạt tiêu bản mà không đòi hỏi sinh viên phải
đầu tư tài nguyên tự huấn luyện mô hình nền^25^.

### Trident: Bộ Công Cụ Tiền Xử Lý Phân Tán Cho Các Mô Hình Nền Tảng

Kho mã nguồn
[Trident](https://github.com/mahmoodlab/trident) được
Mahmood Lab công bố nhằm giải quyết bài toán xử lý dữ liệu tiêu bản quy
mô siêu lớn phục vụ các mô hình nền tảng (Foundation Models)^11^. Hệ
thống cho phép thiết lập quy trình xử lý khép kín chỉ bằng một dòng lệnh
duy nhất: tự động phân vùng mô học bằng thuật toán HEST hoặc GrandQC để
lọc bỏ triệt để dị vật và vết mực bút lông, tính toán tọa độ lưới cắt,
và trích xuất embedding bằng 33 bộ mã hóa cấp lát cắt (như UNI, CONCH,
Virchow, Prov-GigaPath) kết hợp cùng 6 bộ mã hóa cấp độ toàn slide hiện
đại (như Titan, GigaPath, PRISM)^11^.

Trident cung cấp cơ chế phân phối tải tính toán song song trên nhiều
GPU, hệ thống khóa an toàn tệp (.lock) cho phép tự động tiếp tục tiến
trình (smart resume) khi xảy ra sự cố gián đoạn hệ thống, và pipeline
đệm dữ liệu thông minh hai tầng để nạp ảnh mượt mà từ các cụm lưu trữ
mạng (NAS/SAN)^11^. Toàn bộ đường biên mô học và tọa độ được xuất ra
định dạng GeoJSON chuẩn hóa, sẵn sàng nạp vào QuPath để kiểm định chất
lượng^11^.

Đây là tài nguyên tham khảo hữu ích cho các đề tài cần thiết kế tầng xử
lý dữ liệu backend quy mô lớn trong môi trường máy chủ phân tán^11^.

### SAM2-Path: Tinh Chỉnh Mô Hình Nền Tảng Phân Vùng Tự Động

Mã nguồn
[SAM2-PATH](https://github.com/simzhangbest/SAM2PATH) kế
thừa kiến trúc Segment Anything 2 (SAM 2) của Meta và điều chỉnh chuyên
biệt cho bài toán phân vùng ngữ nghĩa đa lớp (multi-class semantic
segmentation) trong mô học bệnh học^27^. Mặc dù SAM 2 sở hữu khả năng
phân vùng hình học tổng quát ấn tượng trên ảnh tự nhiên, mô hình này
thiếu hụt tri thức bệnh học chuyên sâu và đòi hỏi người dùng phải liên
tục đưa vào các điểm tương tác (prompt clicks)^28^.

SAM2-Path khắc phục nhược điểm trên bằng cách bổ sung một bộ mã hóa tri
thức bệnh học chuyên dụng (sử dụng trọng số từ mô hình nền tảng UNI)
chạy song song với bộ mã hóa Hiera của SAM 2^28^. Các đặc trưng này được
hợp nhất thông qua một tầng căn chỉnh số chiều^28^. Điểm đột phá là việc
ứng dụng mạng nơ-ron Kolmogorov-Arnold (KAN) để tự động sinh ra các
token gợi ý (learnable prompt tokens) tương ứng với từng nhóm cấu trúc
mô^28^. Nhờ đó, SAM2-Path vận hành hoàn toàn tự động, loại bỏ sự phụ
thuộc vào thao tác nhấp chuột thủ công của con người mà vẫn đạt độ chính
xác cao trên các tập dữ liệu mô bệnh học tiêu chuẩn như BCSS, CRAG và
GlaS^28^.

Kho lưu trữ này mang giá trị học thuật cao, phù hợp cho các đồ án tốt
nghiệp định hướng nghiên cứu chuyên sâu về các kiến trúc học sâu thị
giác máy tính thế hệ mới^28^.

+-----------------------------------+--------+---------+--------------------------+---------+
| **Tên Repository & Liên Kết**                                       | **Nhiệm Vụ Cốt | **Kiến Trúc Mô  | **Dữ Liệu Vào / Định Dạng Ra**                     | **Đánh Giá Mức  |
|                                                                     | Lõi**          | Hình**          |                                                    | Độ Phù Hợp Đồ   |
|                                                                     |                |                 |                                                    | Án**            |
+-----------------------------------+--------+---------+--------------------------+---------+
| [CLAM](https://github.com/mahmoodlab/CLAM)            | Phân loại WSI, | Attention-MIL,  | WSI (SVS/NDPI)                                     | Rất cao; mẫu    |
|                                                                     | học yếu, tạo   | ResNet50/ViT    | [x]{width="0.1891272965879265in" | mực cho bài     |
| \[cite: 9\]                                                         | bản đồ chú     | backbone^9^     | height="0.2311548556430446in"} HDF5, Tensor .pt,   | toán phân loại  |
|                                                                     | ý^9^           |                 | Heatmaps^9^                                        | nhãn slide^9^   |
+-----------------------------------+--------+---------+--------------------------+---------+
| [HoVer-Net](https://github.com/vqdang/hover_net)      | Phân vùng cá   | Preact-ResNet50 | Patch/WSI                                          | Rất cao; chuẩn  |
|                                                                     | thể và phân    | đa nhánh (NP,   | [x]{width="0.1891272965879265in" | mực cho định    |
| \[cite: 13\]                                                        | loại nhân tế   | HV, TP)^13^     | height="0.2311548556430446in"} GeoJSON, Đa giác    | lượng vi môi    |
|                                                                     | bào^13^        |                 | QuPath^13^                                         | trường khối     |
|                                                                     |                |                 |                                                    | u^13^           |
+-----------------------------------+--------+---------+--------------------------+---------+
| [Slideflow](https://github.com/slideflow/slideflow)   | Chu trình toàn | CNN, ViT, MIL,  | Đa định dạng                                       | Tuyệt vời; hoàn |
|                                                                     | diện, huấn     | GANs, SSL^20^   | [x]{width="0.1891272965879265in" | chỉnh nhất cho  |
| \[cite: 20\]                                                        | luyện và hiển  |                 | height="0.2311548556430446in"} TFRecords, Parquet, | sản phẩm phần   |
|                                                                     | thị thực^20^   |                 | GUI live^20^                                       | mềm trạm^20^    |
+-----------------------------------+--------+---------+--------------------------+---------+
| [WSInfer](https://github.com/SBU-BMI/wsinfer)         | Tối ưu hóa suy | TorchScript     | Thư mục WSI                                        | Rất cao; tối ưu |
|                                                                     | luận, phân     | Model Zoo       | [x]{width="0.1891272965879265in" | cho kiến trúc   |
| \[cite: 24\]                                                        | phối mô hình   | (ResNet,        | height="0.2311548556430446in"} GeoJSON, CSV,       | backend         |
|                                                                     | TorchScript^8^ | Inception)^25^  | QuPath tiles^8^                                    | microservice^8^ |
+-----------------------------------+--------+---------+--------------------------+---------+
| [Trident](https://github.com/mahmoodlab/trident)      | Tiền xử lý tự  | 33 Patch        | SVS/TIFF/Zarr                                      | Rất cao; phù    |
|                                                                     | động quy mô    | Encoders (UNI,  | [x]{width="0.1891272965879265in" | hợp cho tầng    |
| \[cite: 11\]                                                        | lớn, trích     | CONCH) + 6      | height="0.2311548556430446in"} HDF5, GeoJSON       | Data Pipeline   |
|                                                                     | xuất           | Slide           | contours^11^                                       | phân tán^11^    |
|                                                                     | embedding^11^  | Encoders^11^    |                                                    |                 |
+-----------------------------------+--------+---------+--------------------------+---------+
| [SAM2-PATH](https://github.com/simzhangbest/SAM2PATH) | Phân vùng ngữ  | SAM 2 + UNI     | Image tiles                                        | Cao; thích hợp  |
|                                                                     | nghĩa cấu trúc | Pathology       | [x]{width="0.1891272965879265in" | cho đề tài      |
| \[cite: 27, 28\]                                                    | mô tự động^28^ | Encoder + KAN   | height="0.2311548556430446in"} Mặt nạ phân đoạn đa | nghiên cứu giải |
|                                                                     |                | Prompts^28^     | lớp^28^                                            | thuật mới^28^   |
+=====================================================================+================+=================+====================================================+=================+

## Các Thư Viện Và Framework Mã Nguồn Mở Phổ Biến Cho Dữ Liệu Khối Gigapixel

Việc lập trình thao tác với ảnh WSI đòi hỏi các thư viện có khả năng làm
việc với hệ tọa độ logic phân cấp và giải mã hình ảnh theo yêu cầu nhằm
giải quyết bài toán tắc nghẽn bộ nhớ^3^.

### OpenSlide và OpenSlide Python

Thư viện
[OpenSlide](https://github.com/openslide/openslide) được
viết bằng ngôn ngữ C cùng lớp vỏ bọc
[OpenSlide-Python](https://github.com/openslide/openslide-python),
đóng vai trò là hạ tầng nền tảng được sử dụng rộng rãi nhất trong cộng
đồng khoa học máy tính y tế^3^. Thư viện này thiết lập một tầng trừu
tượng hóa cho phép đọc thống nhất hàng loạt định dạng WSI độc quyền của
các nhà sản xuất máy quét tiêu bản như Aperio (.svs), Hamamatsu (.ndpi),
Leica (.scn), MIRAX (.mrxs) và Philips TIFF^3^. Giao diện lập trình cốt
lõi read_region(location, level, size) cho phép đọc chính xác một lát
cắt hình chữ nhật ở tầng phân giải tùy ý mà chỉ cần nạp và giải nén các
khối điểm ảnh tương ứng từ đĩa cứng^3^.

Mặc dù là thành phần phụ thuộc cơ sở trong hầu hết các nghiên cứu,
OpenSlide vận hành hoàn toàn trên CPU, dẫn đến tình trạng nghẽn cổ chai
I/O khi cần trích xuất đồng thời hàng chục nghìn lát cắt phục vụ cho
việc huấn luyện mạng nơ-ron sâu^3^. Trong đồ án kỹ thuật phần mềm,
OpenSlide là thư viện cơ bản bắt buộc phải nắm vững để xây dựng các thao
tác đọc ghi dữ liệu ban đầu^3^.

### cuCIM (RAPIDS - NVIDIA)

Nhằm khắc phục triệt để giới hạn hiệu năng của CPU trong xử lý hình ảnh
y sinh, NVIDIA phát triển thư viện
[cuCIM](https://github.com/rapidsai/cucim) thuộc hệ sinh
thái RAPIDS và Clara^4^. cuCIM cung cấp khả năng giải nén và xử lý hình
ảnh đa chiều (N-Dimensional) trực tiếp bằng bộ tăng tốc đồ họa GPU^4^.
Điểm giá trị của cuCIM là việc cung cấp giao diện lập trình tương thích
tương đương với OpenSlide và scikit-image (cucim.skimage), hỗ trợ cấu
trúc mảng CuPy, giúp việc chuyển đổi mã nguồn từ môi trường CPU sang GPU
không đòi hỏi tái cấu trúc phức tạp^3^.

Với việc tích hợp plugin tăng tốc cuslide2 và công nghệ GPUDirect
Storage (GDS), cuCIM có thể nạp các khối điểm ảnh nén JPEG hoặc JPEG2000
trực tiếp từ ổ cứng thể rắn NVMe vào bộ nhớ VRAM của GPU mà không cần
sao chép qua bộ nhớ máy chủ (host memory)^4^. Thử nghiệm đo kiểm cho
thấy cuCIM đạt tốc độ xử lý nhanh hơn OpenSlide tới 6 lần trong các tác
vụ trích xuất lát cắt đa luồng^3^. Đây là công cụ tối ưu cho các đồ án
phần mềm y tế yêu cầu thông lượng xử lý cao hoặc chạy trên máy chủ có
cấu hình card đồ họa chuyên dụng^4^.

### TIAToolbox

Được thiết kế và duy trì bởi TIA Centre,
[TIAToolbox](https://github.com/TissueImageAnalytics/tiatoolbox)
là bộ công cụ cấp cao chuẩn hóa toàn bộ chu trình giải phẫu bệnh tính
toán trong một giao diện lập trình Python trực quan^32^. Thư viện này
bao bọc các thao tác phức tạp như: đọc ảnh đa định dạng, chuẩn hóa màu
sắc mô học theo các thuật toán Ruifrok, Macenko hoặc Vahadane, tự động
chia lưới cắt lát ảnh, và tích hợp sẵn các mô hình phân loại mô hoặc
phát hiện tế bào^32^.

TIAToolbox tích hợp cơ chế quản lý bộ nhớ đệm thông minh (smart tile
caching) và tự động lựa chọn backend giải mã tối ưu tùy theo môi trường
phần cứng^33^. Nhờ thiết kế hướng đối tượng trực quan và tài liệu hướng
dẫn chi tiết, thư viện này là sự lựa chọn phù hợp cho sinh viên cần
triển khai nhanh mô-đun tiền xử lý và suy luận trong đồ án mà không phải
tự viết lại các thuật toán hình học cơ bản^32^.

### HistomicsTK

Bộ công cụ
[HistomicsTK](https://github.com/DigitalSlideArchive/HistomicsTK)
được phát triển bởi tập đoàn Kitware phối hợp cùng hệ sinh thái Digital
Slide Archive (DSA), tập trung vào các phương pháp phân tích định lượng
và xử lý tín hiệu hình ảnh y sinh^35^. Điểm nổi bật của HistomicsTK là
việc cung cấp các thuật toán giải phẫu bệnh kinh điển đạt độ chính xác
quang học cao, tiêu biểu là kỹ thuật phân tách màu nhuộm (Color
Deconvolution) dựa trên định luật hấp thụ ánh sáng Beer-Lambert^37^.
Thuật toán này cho phép tách biệt rành mạch mật độ quang học của phẩm
nhuộm Hematoxylin (bắt màu nhân tế bào) và Eosin (bắt màu tế bào chất),
tạo tiền đề vững chắc cho các thuật toán phân đoạn hình thái học hạt
nhân^37^.

Về mặt công nghệ phần mềm, HistomicsTK được đóng gói theo tiêu chuẩn
giao diện dòng lệnh Slicer CLI, cho phép các hàm xử lý Python được đóng
gói thành các Docker container độc lập^35^. Giao diện web của hệ sinh
thái DSA có thể tự động đọc thông số đầu vào của thuật toán và kích hoạt
xử lý bất đồng bộ^35^. Đây là kiến trúc phù hợp cho các đồ án hướng đến
việc xây dựng hệ thống phần mềm xử lý ảnh y tế dạng dịch vụ đám mây^35^.

### MONAI Pathology

Là phân hệ chuyên sâu nằm trong dự án
[MONAI](https://github.com/Project-MONAI/MONAI) do NVIDIA
và King\'s College London đồng sáng lập, MONAI Pathology đại diện cho
chuẩn kiến trúc học sâu chuyên nghiệp dành cho nghiên cứu y tế^40^. Phân
hệ này cung cấp các cấu trúc dữ liệu chuyên biệt như WSIReader (tự động
điều phối linh hoạt giữa backend OpenSlide và cuCIM), GridPatchDataset
phục vụ việc nạp dữ liệu lát cắt theo cơ chế trì hoãn (lazy loading),
cùng các lớp tăng cường dữ liệu y khoa (Stain Jittering, Gaussian Blur,
Random Affine)^40^.

Một trong những đóng góp kỹ thuật quan trọng của MONAI Pathology là cơ
chế suy luận lát cắt mô-đun (Modular Patch Inference) kết hợp ghép nối
hòa trộn biên (sliding-window blending)^40^. Khi mô hình học sâu thực
hiện phân vùng trên từng lát cắt đơn lẻ, ranh giới giữa các lát thường
xuất hiện các vết đứt gãy hoặc sai lệch tín hiệu^40^. Thuật toán ghép
nối của MONAI áp dụng trọng số hàm Gauss để làm mượt vùng chồng lấn giữa
các lát lân cận, tái tạo bản đồ phân vùng liên tục trên toàn bộ WSI^40^.
Framework này là lựa chọn thích hợp cho các đồ án định hướng phát triển
mô hình học sâu chuyên nghiệp trên nền tảng PyTorch^41^.

+-----------------------------------------+-------+-------+---------+-------+
| **Thư Viện Phần Mềm & Mã Nguồn**                                                | **Ngôn Ngữ  | **Tăng Tốc  | **Chức Năng Kỹ  | **Mức Độ    |
|                                                                                 | & Backend** | Phần Cứng** | Thuật Nổi Bật** | Phức Tạp    |
|                                                                                 |             |             |                 | Triển       |
|                                                                                 |             |             |                 | Khai**      |
+-----------------------------------------+-------+-------+---------+-------+
| [OpenSlide](https://github.com/openslide/openslide)               | C, Lớp vỏ   | CPU đa      | Đọc ngẫu nhiên  | Rất thấp;   |
|                                                                                 | Python^3^   | luồng       | tọa độ Level 0, | thư viện    |
| \[cite: 3, 31\]                                                                 |             |             | hỗ trợ định     | chuẩn cơ sở |
|                                                                                 |             |             | dạng đa hãng^3^ |             |
+-----------------------------------------+-------+-------+---------+-------+
| [cuCIM](https://github.com/rapidsai/cucim)                        | C++, CUDA,  | GPU Native  | Giải nén TIFF   | Trung bình; |
|                                                                                 | Python^4^   | (CUDA,      | siêu tốc trên   | yêu cầu GPU |
| \[cite: 6\]                                                                     |             | GDS)^4^     | VRAM, API       | NVIDIA^6^   |
|                                                                                 |             |             | scikit-image^3^ |             |
+-----------------------------------------+-------+-------+---------+-------+
| [TIAToolbox](https://github.com/TissueImageAnalytics/tiatoolbox)  | Python^33^  | CPU / GPU   | Chuẩn hóa màu   | Thấp; API   |
|                                                                                 |             | gián tiếp   | nhuộm, tự động  | hướng đối   |
| \[cite: 32\]                                                                    |             |             | tạo lưới tile,  | tượng hoàn  |
|                                                                                 |             |             | nạp ảnh^32^     | chỉnh^32^   |
+-----------------------------------------+-------+-------+---------+-------+
| [HistomicsTK](https://github.com/DigitalSlideArchive/HistomicsTK) | Python,     | CPU /       | Phân tách màu   | Trung bình; |
|                                                                                 | C++^35^     | OpenMP /    | quang học       | phụ thuộc   |
| \[cite: 35\]                                                                    |             | GPU         | Beer-Lambert,   | hệ sinh     |
|                                                                                 |             |             | chuẩn Slicer    | thái        |
|                                                                                 |             |             | CLI^35^         | DSA^35^     |
+-----------------------------------------+-------+-------+---------+-------+
| [MONAI Pathology](https://github.com/Project-MONAI/MONAI)         | PyTorch,    | CPU / GPU   | WSI lazy        | Trung bình; |
|                                                                                 | Python^41^  | (CUDA)^40^  | loader, hòa     | yêu cầu kỹ  |
| \[cite: 40\]                                                                    |             |             | trộn biên       | năng        |
|                                                                                 |             |             | trượt, tiền xử  | PyTorch^42^ |
|                                                                                 |             |             | lý y sinh^40^   |             |
+=================================================================================+=============+=============+=================+=============+

## Ý Tưởng Đột Phá Nhằm Giảm Thiểu Chi Phí Gán Nhãn Dữ Liệu Y Khoa

Chi phí gán nhãn thủ công trong giải phẫu bệnh số là một trở ngại lớn:
việc yêu cầu bác sĩ giải phẫu bệnh vẽ viền tỉ mỉ hàng nghìn nhân tế bào
hoặc khoanh vùng ranh giới khối u trên các lát cắt gigapixel tốn kém
nhiều thời gian và dễ phát sinh sự sai lệch chủ quan giữa các chuyên
gia^1^. Để vượt qua thách thức này, cộng đồng nghiên cứu quốc tế đã phát
triển các phương pháp luận giúp tối ưu hóa hiệu quả dữ liệu.

### Học Đa Thực Thể Ràng Buộc Phân Cụm Dưới Giám Sát Yếu (Weakly-Supervised MIL)

Phương pháp học đa thực thể tiếp cận bài toán bằng cách chỉ sử dụng duy
nhất một nhãn tổng thể cấp slide (ví dụ: bệnh nhân mang khối u ác tính
hay lành tính, hoặc có đột biến gen xác định), vốn là thông tin có sẵn
trong bệnh án điện tử bệnh viện^12^. Toàn bộ tiêu bản WSI được xem như
một \"túi\" (bag) chứa tập hợp
${width="0.19542104111986003in"
height="0.27219378827646545in"} lát cắt (instances)
$\{x_1, x_2, \dots, x_N\}${width="1.9453127734033246in"
height="0.27095472440944884in"}^14^. Sau khi mỗi lát cắt được chiếu vào
một véc-tơ đặc trưng  \in \mathbb{R}^D${width="1.0135640857392827in"
height="0.2726137357830271in"}, mô hình sử dụng cơ chế chú ý định lượng
(Gated Attention Mechanism) để tính toán mức độ quan trọng tương đối
${width="0.20681430446194227in"
height="0.2688593613298338in"} của từng lát cắt^9^:

a_k = \frac{\exp(w^T (\tanh(V h_k^T) \odot \text{sigm}(U h_k^T)))}{\sum_{j=1}^N \exp(w^T (\tanh(V h_j^T) \odot \text{sigm}(U h_j^T)))}{width="6.458333333333333in"
height="1.075376202974628in"}

Trong đó ${width="0.1814238845144357in"
height="0.26850721784776904in"},
${width="0.18956146106736657in"
height="0.25976924759405073in"}, và
${width="0.189669728783902in"
height="0.2599179790026247in"} là các ma trận trọng số có thể học được,
còn $\odot${width="0.1667749343832021in"
height="0.25711176727909013in"} là phép nhân phần tử^9^. Biểu diễn cô
đọng của toàn bộ tiêu bản được hình thành từ phép tổng tích lũy có trọng
số {\text{slide}} = \sum_{k=1}^N a_k h_k${width="1.4181878827646544in"
height="0.29197944006999127in"}.

Các mô hình như CLAM bổ sung một hàm mất mát phân cụm nhị phân ép buộc
các lát cắt có trọng số chú ý cao nhất phải tách biệt với các lát cắt có
trọng số thấp trong không gian tiềm ẩn^9^. Nhờ cơ chế này, mô hình vừa
đưa ra chẩn đoán chính xác cấp bệnh phẩm, vừa tự động định vị và làm nổi
bật các ổ ung thư trên bản đồ nhiệt mà không cần bất kỳ đường viền phân
vùng thủ công nào từ bác sĩ^1^.

### Ứng Dụng Mô Hình Nền Tảng Học Tự Giám Sát (Foundation Models)

Sự phát triển của học tự giám sát (Self-Supervised Learning - SSL) cho
phép các mô hình thị giác trích xuất tri thức hình thái học từ hàng
triệu đến hàng tỷ lát cắt WSI hoàn toàn không có nhãn^2^.

- **CTransPath:** Được đào tạo trên hơn 15 triệu lát cắt từ cơ sở dữ
  liệu TCGA và PAIP, kết hợp cấu trúc CNN và Transformer cùng kỹ thuật
  học tương phản (contrastive learning) để nắm bắt đặc trưng hình thái
  mô học^43^.

- **UNI (Harvard / Mahmood Lab):** Mô hình ViT-L/16 huấn luyện theo
  phương pháp DINOv2 trên hơn 100 triệu lát cắt thu thập từ 100.000 tiêu
  bản thuộc 20 cơ quan giải phẫu khác nhau^28^.

- **Prov-GigaPath (Microsoft & Providence Health):** Được huấn luyện
  trên tập dữ liệu quy mô 1,3 tỷ lát cắt từ 171.189 tiêu bản WSI lâm
  sàng^2^. Kiến trúc này kết hợp bộ mã hóa lát cắt DINOv2 cùng bộ mã hóa
  toàn slide dựa trên kiến trúc LongNet^2^, cho phép mô hình tiếp nhận
  và xử lý ngữ cảnh cực dài lên tới hàng chục nghìn lát cắt đồng thời
  trong một lượt nạp^2^.

Khi áp dụng vào các bài toán chuyên biệt trong đồ án phần mềm y tế, các
mô hình nền tảng này đóng vai trò là bộ trích xuất đặc trưng cố định
(frozen feature extractor)^43^. Kỹ sư chỉ cần huấn luyện một bộ phân
loại tuyến tính đơn giản (Linear Probe) hoặc một mạng MIL gọn nhẹ trên
vài chục mẫu bệnh phẩm là đã có thể đạt độ chính xác tương đương hoặc
vượt trội so với các mô hình CNN truyền thống được huấn luyện có giám
sát từ đầu trên hàng nghìn ca bệnh^2^.

### Tinh Chỉnh Mô Hình Thị Giác Nền Tảng Cho Phân Vùng Tương Tác

Phương pháp phân vùng truyền thống đòi hỏi việc xây dựng các tập dữ liệu
gán nhãn đường viền khổng lồ^47^. Các nhóm nghiên cứu hiện chuyển đổi
bài toán sang việc thích ứng các mô hình nền tảng thị giác như Segment
Anything (SAM và SAM 2) thông qua cơ chế tương tác người-máy
(Human-in-the-loop)^28^.

Thay vì yêu cầu chuyên gia giải phẫu bệnh vẽ toàn bộ chu vi phức tạp của
nhân tế bào hoặc ống tuyến, các mô hình như SAM-Path hay NuClick (tích
hợp trong MONAI Label) cho phép bác sĩ chỉ cần thực hiện một cú nhấp
chuột đơn lẻ (single click prompt) vào vùng trung tâm của cấu trúc^29^.
Mạng nơ-ron sẽ tự động suy đoán gradient ranh giới và tạo ra đường viền
khép kín chính xác trong vài mili-giây^48^. Thao tác này giúp giảm hơn
90% thời gian tạo nhãn phân vùng vi thể mà vẫn đảm bảo độ chính xác giải
phẫu^48^.

### Cơ Chế Vòng Lặp Học Chủ Động (Active Learning with Human-in-the-Loop)

Thay vì gán nhãn dữ liệu một cách ngẫu nhiên hoặc dàn trải, cơ chế học
chủ động (Active Learning) sử dụng các độ đo độ bất định (entropy hoặc
khoảng cách biên phân loại) của mô hình để tự động sàng lọc trong kho
WSI gigapixel những vùng mô mà thuật toán khó dự đoán nhất^40^. Hệ thống
sẽ gửi yêu cầu can thiệp tới bác sĩ để hiệu chỉnh duy nhất các vùng mô
này thông qua giao diện trạm làm việc^49^. Dữ liệu sau khi hiệu chỉnh
được đưa vào hàng đợi tái huấn luyện liên tục (continuous training
loop)^49^. Cách tiếp cận này giúp mô hình đạt điểm hội tụ sớm với số
lượng mẫu cần gán nhãn ít hơn đáng kể so với phương pháp thu thập dữ
liệu thụ động.

## Các Giải Pháp Phần Mềm Trạm Làm Việc Lâm Sàng Và Công Cụ Gán Nhãn Trực Quan

Khoảng cách giữa các mô hình học sâu và ứng dụng thực tiễn trong bệnh
viện được thu hẹp nhờ các trạm làm việc (clinical workstations) và phần
mềm duyệt ảnh WSI^8^. Các hệ thống này cần đảm bảo khả năng duyệt ảnh
gigapixel mượt mà, đồng thời hỗ trợ kết xuất đồng thời hàng triệu đối
tượng hình học và bản đồ nhiệt trong suốt^35^.

### QuPath (Trạm Làm Việc Chuyên Dụng Cho Máy Trạm Cục Bộ)

Phần mềm [QuPath](https://github.com/qupath/qupath) là
giải pháp mã nguồn mở phổ biến hàng đầu dành cho phân tích ảnh hiển vi
và giải phẫu bệnh số, được phát triển tại Đại học Queen\'s Belfast và
Đại học Edinburgh bằng ngôn ngữ Java/JavaFX theo giấy phép GPL-3.0^50^.
QuPath sở hữu công cụ kết xuất kim tự tháp (pyramidal rendering engine)
cục bộ, cho phép phóng to, thu nhỏ và di chuyển trên các tệp ảnh
gigapixel với độ trễ tối thiểu^34^.

Kiến trúc dữ liệu của QuPath xây dựng trên mô hình phân cấp đối tượng
(Object Hierarchy): một vùng mô lớn (Annotation) có thể chứa các ô gạch
phân loại (Tiles), và bên trong chứa hàng trăm nghìn cá thể nhân hoặc tế
bào (Detections) với đầy đủ thông số đo lường hình học và cường độ màu
sắc^25^. QuPath có khả năng mở rộng thông qua các đoạn mã kịch bản
Groovy và hệ sinh thái tiện ích cắm ngoài phong phú^34^. Nổi bật trong
số đó là **qupath-extension-wsinfer**, cho phép người dùng nạp mô hình
từ WSInfer Model Zoo và chạy suy luận trực tiếp trên vùng tiêu bản đang
hiển thị qua giao diện đồ họa^25^. Tương tự,
**qupath-extension-monailabel** kết nối QuPath với máy chủ MONAI Label
để cung cấp khả năng phân vùng tế bào bán tự động thông qua công cụ nhấp
chuột tương tác^48^.

QuPath là nền tảng máy trạm cục bộ thích hợp cho các đồ án tập trung vào
nghiên cứu giải thuật, kiểm chứng nhãn và tích hợp AI dưới dạng plugin
phía máy khách^25^.

### Digital Slide Archive (DSA) & HistomicsUI (Nền Tảng Đám Mây Cấp Doanh Nghiệp)

Hệ thống [[Digital Slide
Archive]{.underline}](https://github.com/DigitalSlideArchive/digitalslidearchive.info)
kết hợp cùng giao diện người dùng
[HistomicsUI](https://github.com/DigitalSlideArchive/histomicsui)
là giải pháp Web Client-Server cấp doanh nghiệp do Kitware phát
triển^35^. DSA sử dụng nền tảng quản lý dữ liệu Girder (Python), cơ sở
dữ liệu MongoDB và máy chủ ô gạch WSI hiệu năng cao large_image, đóng
gói toàn bộ dịch vụ thông qua Docker Compose^37^.

HistomicsUI được xây dựng bằng Vue.js và thư viện hiển thị bản đồ GeoJS,
tối ưu hóa để hiển thị hàng triệu chú thích dạng vector (GeoJSON) và các
lớp bản đồ nhiệt trong suốt phủ lên ảnh WSI mà không gây nghẽn trình
duyệt web của người dùng^35^. Điểm giá trị nhất của hệ thống là cơ chế
tích hợp thuật toán Slicer CLI: các kỹ sư có thể đóng gói mô hình AI
thành các Docker container độc lập và đăng ký với DSA^35^. Giao diện web
sẽ tự động sinh biểu mẫu nhập tham số, điều phối tiến trình chạy nền
trên máy chủ, và hiển thị kết quả phân vùng trực tiếp lên tiêu bản của
bác sĩ^35^.

DSA và HistomicsUI là kiến trúc phù hợp cho các đồ án kỹ thuật phần mềm
hướng đến việc triển khai hệ thống phân tán, hỗ trợ nhiều người dùng
cộng tác qua môi trường mạng và tuân thủ các tiêu chuẩn y tế doanh
nghiệp^35^.

### Cytomine (Nền Tảng Cộng Tác Đa Trung Tâm Trên Nền Web)

Phần mềm [Cytomine](https://github.com/cytomine/) do Đại
học Liège phát triển, hướng tới mục tiêu xây dựng một nền tảng quản lý
và phân tích hình ảnh y sinh học cộng tác thông qua trình duyệt web^54^.
Hệ thống được tổ chức hoàn toàn theo kiến trúc hướng dịch vụ vi mô
(Microservices) gồm các container Docker đảm nhiệm các vai trò riêng
biệt: dịch vụ lõi quản lý dữ liệu người dùng, máy chủ quản lý ảnh tiêu
bản PIMS (Python Image Management Server)^55^, và cơ sở dữ liệu quan hệ
PostgreSQL tích hợp phần mở rộng không gian PostGIS nhằm lưu trữ hiệu
quả tọa độ hình học của hàng triệu đường biên tế bào^54^.

Cytomine cung cấp thư viện máy khách chính thức bằng Python
(Cytomine-python-client), cho phép các nhà phát triển kết nối trực tiếp
với máy chủ thông qua REST API^56^. Các kỹ sư AI có thể viết mã nạp ảnh
về máy trạm để suy luận, sau đó đẩy các nhãn phân loại, đường bao tổn
thương và phép đo định lượng ngược trở lại máy chủ để hiển thị tức thì
trên giao diện của các bác sĩ giải phẫu bệnh^56^.

Cytomine phù hợp cho các dự án nhóm có quy mô nhiều sinh viên, trong đó
có sự phân định rõ ràng giữa nhóm phát triển hạ tầng đám mây và nhóm
phát triển giải thuật học máy y sinh^54^.

### Slideflow Studio (Giao Diện Suy Luận Tương Tác Cấp Lát Cắt Thời Gian Thực)

Đi kèm trong gói mã nguồn Slideflow, [[Slideflow
Studio]{.underline}](https://github.com/slideflow/slideflow) cung cấp
giao diện tương tác thời gian thực trên màn hình máy tính thông qua công
nghệ kết xuất đồ họa bằng OpenGL và tăng tốc GPU^20^. Khác với các phần
mềm hiển thị tiêu bản thông thường vốn chỉ hiển thị các kết quả đã được
tính toán từ trước (pre-computed), Slideflow Studio cho phép bác sĩ rê
chuột qua bất kỳ vùng mô học nào trên tiêu bản gigapixel để mô hình AI
thực thi suy luận trực tiếp tại vùng đó^23^.

Giao diện sẽ lập tức hiển thị phân phối xác suất phân loại, kích hoạt
các lớp đặc trưng nội tại của mạng nơ-ron (layer activations), và kết
xuất bản đồ nhiệt chú ý cục bộ^20^. Đây là công cụ hữu ích trong giai
đoạn nghiệm thu và trình diễn sản phẩm (demo), giúp các bác sĩ và hội
đồng đánh giá chuyên môn có thể tương tác trực quan với thuật toán^23^.

+-------------------------------------+--------+------------+-------------+-------+
| **Nền Tảng Trạm Làm Việc & Liên Kết**                                   | **Kiến Trúc   | **Công Nghệ Frontend & | **Phương Thức Tích Hợp  | **Môi       |
|                                                                         | Phần Mềm**    | Backend**              | Mô Hình AI**            | Trường Ứng  |
|                                                                         |               |                        |                         | Dụng Tối    |
|                                                                         |               |                        |                         | Ưu**        |
+-------------------------------------+--------+------------+-------------+-------+
| [QuPath](https://github.com/qupath/qupath)                | Ứng dụng      | Java, JavaFX, OpenCV,  | QuPath Extensions       | Nghiên cứu  |
|                                                                         | Desktop độc   | Bio-Formats^31^        | (WSInfer, MONAI Label), | cá nhân,    |
| \[cite: 50, 51\]                                                        | lập^52^       |                        | Groovy scripts^25^      | phân tích   |
|                                                                         |               |                        |                         | cục bộ máy  |
|                                                                         |               |                        |                         | trạm^8^     |
+-------------------------------------+--------+------------+-------------+-------+
| [[HistomicsUI                                                           | Web           | Vue.js, GeoJS, Python  | Container hóa Docker    | Bệnh viện,  |
| (DSA)]{.underline}](https://github.com/DigitalSlideArchive/histomicsui) | Client-Server | Girder, MongoDB^37^    | qua Slicer CLI, Girder  | hệ thống    |
|                                                                         | phân tán^35^  |                        | REST API^35^            | web y tế    |
| \[cite: 53\]                                                            |               |                        |                         | cấp doanh   |
|                                                                         |               |                        |                         | nghiệp^35^  |
+-------------------------------------+--------+------------+-------------+-------+
| [Cytomine](https://github.com/cytomine/)                  | Web           | Bootstrap, Docker,     | REST API qua            | Hội đồng    |
|                                                                         | Microservices | PIMS,                  | Cytomine-python-client, | chẩn đoán   |
| \[cite: 54\]                                                            | tập trung^54^ | PostgreSQL/PostGIS^54^ | Batch jobs^54^          | từ xa, đào  |
|                                                                         |               |                        |                         | tạo đa      |
|                                                                         |               |                        |                         | trung       |
|                                                                         |               |                        |                         | tâm^54^     |
+-------------------------------------+--------+------------+-------------+-------+
| [[Slideflow                                                             | Desktop GUI   | Python, OpenGL,        | Nạp trực tiếp mô hình   | Thử nghiệm  |
| Studio]{.underline}](https://github.com/slideflow/slideflow)            | thời gian     | PyTorch/TensorFlow,    | phân loại / MIL đã huấn | giải thuật, |
|                                                                         | thực^23^      | cuCIM^20^              | luyện^20^               | trình diễn  |
| \[cite: 20\]                                                            |               |                        |                         | tương tác   |
|                                                                         |               |                        |                         | trực        |
|                                                                         |               |                        |                         | tiếp^23^    |
+=========================================================================+===============+========================+=========================+=============+

## Đề Xuất Định Hướng Kiến Trúc Kỹ Thuật Toàn Diện Cho Đồ Án Kỹ Thuật Phần Mềm Y Tế

Để hiện thực hóa một hệ thống phần mềm giải phẫu bệnh số có khả năng vận
hành ổn định, mã nguồn mở và đáp ứng các tiêu chuẩn kỹ thuật y tế, kiến
trúc hệ thống nên được chia tách thành ba tầng độc lập theo mô hình vi
dịch vụ (Microservices).

### Tầng Quản Trị Dữ Liệu Và Đọc Ảnh Gigapixel (Storage & Tiling Engine)

Hệ thống sử dụng **OpenSlide** làm backend cơ sở để đảm bảo khả năng
tương thích với đa dạng định dạng WSI từ các dòng máy quét khác nhau,
đồng thời cấu hình chuyển đổi sang **cuCIM** khi triển khai trên các máy
chủ có trang bị GPU NVIDIA nhằm tối ưu hóa thông lượng đọc tệp^3^.

Quá trình tiền xử lý được tự động hóa bằng cách kế thừa các mô-đun lọc
không gian màu của **CLAM** hoặc **TIAToolbox**: tách biệt vùng mô sống
khỏi kính tiêu bản ở tầng phân giải thấp (hệ số giảm mẫu 32x), tính toán
lưới cắt các lát có kích thước
 \times 256${width="0.9046227034120735in"
height="0.25746937882764653in"} pixel tại độ phóng đại 20x (MPP xấp xỉ
0.5 $\mu m/\text{pixel}${width="0.6471358267716535in"
height="0.27137904636920385in"}), và lưu trữ tọa độ dưới dạng bảng dữ
liệu HDF5 hoặc cơ sở dữ liệu quan hệ^7^. Việc lưu trữ tập trung tọa độ
giúp loại bỏ nhu cầu ghi hàng chục nghìn lát cắt rời rạc ra ổ cứng, giảm
thiểu chi phí lưu trữ tệp và phân mảnh hệ thống tệp tin^9^.

### Tầng Dịch Vụ Suy Luận Học Sâu Phân Tán (AI Inference Microservice)

Tầng tính toán học sâu được đóng gói thành một vi dịch vụ độc lập sử
dụng framework FastAPI hoặc gRPC, tách biệt hoàn toàn khỏi máy chủ ứng
dụng web^35^.

- **Trích xuất đặc trưng nền tảng:** Thay vì tự huấn luyện các mạng tích
  chập từ đầu, hệ thống nạp các trọng số đã được tiền huấn luyện của mô
  hình nền tảng y tế **CTransPath** hoặc **UNI** làm bộ trích xuất đặc
  trưng cố định^43^. Các lát cắt mô học được chuyển thành các véc-tơ đặc
  trưng cô đọng^9^.

- **Phân loại và định vị tổn thương cấp slide:** Kế thừa mạng nơ-ron học
  đa thực thể có cơ chế chú ý của **CLAM (Attention-MIL)** để gom tụ các
  véc-tơ đặc trưng thành xác suất chẩn đoán bệnh phẩm, đồng thời tính
  toán trọng số chú ý để kết xuất bản đồ nhiệt tổn thương^9^.

- **Phân vùng tế bào và cấu trúc mô vi thể:** Khi người dùng yêu cầu
  phân tích chi tiết vi môi trường khối u trong một vùng quan tâm (ROI),
  dịch vụ sẽ kích hoạt mô hình **HoVer-Net** hoặc **WSInfer** để dự đoán
  ranh giới nhân tế bào và phân loại các phân nhóm (tế bào biểu mô ung
  thư, tế bào viêm, tế bào mô đệm)^8^. Toàn bộ tọa độ đường biên đa giác
  được chuẩn hóa sang định dạng chuẩn GeoJSON^8^.

### Tầng Trạm Làm Việc Trực Quan Hóa (Presentation & Clinical Workstation Tier)

Tùy thuộc vào mục tiêu sản phẩm của đồ án, việc triển khai giao diện
người dùng có thể thực hiện theo một trong hai định hướng:

1.  *Định hướng Web Y tế Doanh nghiệp (Enterprise Web Solution):* Tận
    dụng hạ tầng máy chủ của **Digital Slide Archive (DSA)** kết hợp
    giao diện **HistomicsUI**^35^. Giao diện Web hiển thị ảnh tiêu bản
    thông qua OpenSeadragon hoặc GeoJS, gọi API tới dịch vụ suy luận AI
    và kết xuất các lớp bản đồ nhiệt cũng như đa giác tế bào trực tiếp
    lên trình duyệt bằng WebGL^35^. Hướng đi này giúp đồ án chứng minh
    năng lực thiết kế phần mềm phân tán, quản lý người dùng và khả năng
    vận hành qua mạng diện rộng^37^.

2.  *Định hướng Tích hợp Tiện ích Chuyên sâu (Clinical Plugin
    Integration):* Phát triển sản phẩm dưới dạng một Tiện ích mở rộng
    (Plugin/Extension) viết bằng Java cho phần mềm **QuPath**^25^.
    Plugin đóng vai trò làm cầu nối giao tiếp: gửi vùng ảnh bác sĩ chỉ
    định tới dịch vụ AI backend qua REST API, sau đó chuyển đổi kết quả
    JSON trả về thành các đối tượng đồ họa phân cấp bản địa của
    QuPath^25^. Hướng tiếp cận này tận dụng được toàn bộ các công cụ đo
    lường giải phẫu bệnh học sẵn có của QuPath, cho phép bác sĩ vừa xem
    kết quả AI, vừa chỉnh sửa trực tiếp các đường viền sai sót để phục
    vụ hội chẩn chuyên môn^25^.

#### Nguồn trích dẫn

1.  CLAM-Mahmood Lab,
    [http://clam.mahmoodlab.org/](http://clam.mahmoodlab.org/)

2.  A whole-slide foundation model for digital pathology from
    real-world,
    [https://www.researchgate.net/publication/380788897_A_whole-slide_foundation_model_for_digital_pathology_from_real-world_data](https://www.researchgate.net/publication/380788897_A_whole-slide_foundation_model_for_digital_pathology_from_real-world_data)

3.  cuCIM - A GPU image I/O and processing library - Zenodo,
    [https://zenodo.org/records/5151998/files/RAPIDS_cuCIM-SciPy2021-final.pdf?download=1](https://zenodo.org/records/5151998/files/RAPIDS_cuCIM-SciPy2021-final.pdf?download=1)

4.  Multi Dimensional Image Processing - NVIDIA Developer,
    [https://developer.nvidia.com/cucim-for-image-io-processing](https://developer.nvidia.com/cucim-for-image-io-processing)

5.  NVIDIA cuCIM Documentation,
    [https://docs.nvidia.com/cucim/latest/](https://docs.nvidia.com/cucim/latest/)

6.  cuCIM - RAPIDS GPU-accelerated image processing library - GitHub,
    [https://github.com/rapidsai/cucim](https://github.com/rapidsai/cucim)

7.  Prov-GigaPath \| Model Catalog - Microsoft Foundry,
    [https://ai.azure.com/catalog/models/Prov-GigaPath](https://ai.azure.com/catalog/models/Prov-GigaPath)

8.  Open and reusable deep learning for pathology with WSInfer \... -
    PMC,
    [https://pmc.ncbi.nlm.nih.gov/articles/PMC10781748/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10781748/)

9.  mahmoodlab/CLAM: Open source tools for computational pathology,
    [https://github.com/mahmoodlab/clam](https://github.com/mahmoodlab/clam)

10. Working with Whole Slide Images in Python with cuCIM - Medium,
    [https://medium.com/@soroush_oskouei/working-with-whole-slide-images-in-python-with-cucim-quick-guide-to-using-cucim-for-efficient-0ca84f7a1b00](https://medium.com/@soroush_oskouei/working-with-whole-slide-images-in-python-with-cucim-quick-guide-to-using-cucim-for-efficient-0ca84f7a1b00)

11. GitHub - mahmoodlab/TRIDENT: Toolkit for large-scale whole-slide,
    [https://github.com/mahmoodlab/trident/](https://github.com/mahmoodlab/trident/)

12. Data-efficient and weakly supervised computational pathology on,
    [https://pmc.ncbi.nlm.nih.gov/articles/PMC8711640/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8711640/)

13. vqdang/hover_net: Simultaneous Nuclear Instance Segmentation,
    [https://github.com/vqdang/hover_net](https://github.com/vqdang/hover_net)

14. GitHub - mahmoodlab/HistoFL: Federated Learning for,
    [https://github.com/mahmoodlab/HistoFL](https://github.com/mahmoodlab/HistoFL)

15. A Weakly Supervised Approach for HPV Status Prediction in,
    [https://www.preprints.org/manuscript/202511.0812](https://www.preprints.org/manuscript/202511.0812)

16. GitHub - mahmoodlab/PORPOISE: Pan-Cancer Integrative Histology,
    [https://github.com/mahmoodlab/PORPOISE](https://github.com/mahmoodlab/PORPOISE)

17. mahmoodlab/TOAD: AI-based pathology predicts origins for cancers,
    [https://github.com/mahmoodlab/TOAD](https://github.com/mahmoodlab/TOAD)

18. HistoNeXt: dual-mechanism feature pyramid network for cell nuclear,
    [https://pmc.ncbi.nlm.nih.gov/articles/PMC11706021/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11706021/)

19. OGevaertLab/NucSegAI - Hugging Face,
    [https://huggingface.co/OGevaertLab/NucSegAI](https://huggingface.co/OGevaertLab/NucSegAI)

20. GitHub - slideflow/slideflow: Powerful, open-source AI tools for,
    [https://github.com/slideflow/slideflow](https://github.com/slideflow/slideflow)

21. Artificial intelligence-based morphologic classification and \... -
    PMC,
    [https://pmc.ncbi.nlm.nih.gov/articles/PMC11177984/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11177984/)

22. James Dolezal jamesdolezal - GitHub,
    [https://github.com/jamesdolezal](https://github.com/jamesdolezal)

23. Slideflow: deep learning for digital histopathology with real-time,
    [https://pmc.ncbi.nlm.nih.gov/articles/PMC10967068/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10967068/)

24. WSInfer: deep learning inference on whole slide images - GitHub,
    [https://github.com/SBU-BMI/wsinfer](https://github.com/SBU-BMI/wsinfer)

25. WSInfer \| (Para) Pathology Notes,
    [https://www.parapathology.com/computational-digital-and-mathematical-pathology/wsinfer](https://www.parapathology.com/computational-digital-and-mathematical-pathology/wsinfer)

26. User Guide -- WSInfer documentation - Read the Docs,
    [https://wsinfer.readthedocs.io/en/latest/user_guide.html](https://wsinfer.readthedocs.io/en/latest/user_guide.html)

27. SAM2-PATH: A better segment anything model for semantic,
    [https://www.catalyzex.com/paper/sam2-path-a-better-segment-anything-model-for](https://www.catalyzex.com/paper/sam2-path-a-better-segment-anything-model-for)

28. SAM2-Path: A better segment anything model for semantic \... -
    arXiv,
    [https://arxiv.org/html/2408.03651v1](https://arxiv.org/html/2408.03651v1)

29. cvlab-stonybrook/SAMPath: Repository for \"SAM-Path: A Segment,
    [https://github.com/cvlab-stonybrook/SAMPath](https://github.com/cvlab-stonybrook/SAMPath)

30. Official implementation of FPNuNet: A Frequency-Aware \... - GitHub,
    [https://github.com/LuluQin766/FPNuNet](https://github.com/LuluQin766/FPNuNet)

31. lingxitong/Awesome-AI4DigitalPathology - GitHub,
    [https://github.com/lingxitong/Awesome-AI4DigitalPathology](https://github.com/lingxitong/Awesome-AI4DigitalPathology)

32. Computational Pathology Toolbox developed by TIA Centre \... -
    GitHub,
    [https://github.com/tissueimageanalytics/tiatoolbox](https://github.com/tissueimageanalytics/tiatoolbox)

33. tiatoolbox: a computational pathology toolbox for whole-slide
    images,
    [https://hysenlabs.com/projects/tissueimageanalytics-tiatoolbox](https://hysenlabs.com/projects/tissueimageanalytics-tiatoolbox)

34. 18 Free and Open-source Whole Slide Imaging Pathology Projects,
    [https://medevel.com/os-whole-slide-imaging-projects-1800/](https://medevel.com/os-whole-slide-imaging-projects-1800/)

35. HistomicsTK - bio.tools,
    [https://bio.tools/histomicstk](https://bio.tools/histomicstk)

36. Digital Slide Archive,
    [https://digitalslidearchive.github.io/digital_slide_archive/](https://digitalslidearchive.github.io/digital_slide_archive/)

37. Documentation for installing and maintaining the Digital Slide,
    [https://github.com/DigitalSlideArchive/digitalslidearchive.info](https://github.com/DigitalSlideArchive/digitalslidearchive.info)

38. histomicstk.preprocessing.color_deconvolution - Digital Slide
    Archive,
    [https://digitalslidearchive.github.io/HistomicsTK/histomicstk.preprocessing.color_deconvolution.html](https://digitalslidearchive.github.io/HistomicsTK/histomicstk.preprocessing.color_deconvolution.html)

39. HistoKit: Fast and Accurate Tissue and Artifact Detection and Data,
    [https://www.scitepress.org/Papers/2026/146464/146464.pdf](https://www.scitepress.org/Papers/2026/146464/146464.pdf)

40. Milestones · Project-MONAI/MONAI - GitHub,
    [https://github.com/Project-MONAI/MONAI/milestones](https://github.com/Project-MONAI/MONAI/milestones)

41. MONAI - Medical Open Network for AI,
    [https://project-monai.github.io/](https://project-monai.github.io/)

42. Project-MONAI/tutorials - GitHub,
    [https://github.com/Project-MONAI/tutorials](https://github.com/Project-MONAI/tutorials)

43. GitHub - Xiyue-Wang/TransPath,
    [https://github.com/Xiyue-Wang/TransPath](https://github.com/Xiyue-Wang/TransPath)

44. jamesdolezal/CTransPath - Hugging Face,
    [https://huggingface.co/jamesdolezal/CTransPath](https://huggingface.co/jamesdolezal/CTransPath)

45. UNI/README_old.md at main · mahmoodlab/UNI - GitHub,
    [https://github.com/mahmoodlab/UNI/blob/main/README_old.md](https://github.com/mahmoodlab/UNI/blob/main/README_old.md)

46. Prov-GigaPath - GitHub,
    [https://github.com/prov-gigapath/prov-gigapath](https://github.com/prov-gigapath/prov-gigapath)

47. (PDF) Segment anything in medical images - ResearchGate,
    [https://www.researchgate.net/publication/377596529_Segment_anything_in_medical_images](https://www.researchgate.net/publication/377596529_Segment_anything_in_medical_images)

48. MONAI Label Pathology App - QuPath Nuclei Segmentation Example,
    [https://github.com/Project-MONAI/tutorials/blob/main/monailabel/monailabel_pathology_nuclei_segmentation_QuPath.ipynb](https://github.com/Project-MONAI/tutorials/blob/main/monailabel/monailabel_pathology_nuclei_segmentation_QuPath.ipynb)

49. MONAI Label is an intelligent open source image labeling \... -
    GitHub,
    [https://github.com/project-monai/monailabel](https://github.com/project-monai/monailabel)

50. QuPath,
    [https://qupath.github.io/](https://qupath.github.io/)

51. QuPath - Open-source bioimage analysis for research - GitHub,
    [https://github.com/qupath/qupath](https://github.com/qupath/qupath)

52. QuPath · bio.tools,
    [https://bio.tools/qupath](https://bio.tools/qupath)

53. DigitalSlideArchive/HistomicsUI: A Girder plugin for \... - GitHub,
    [https://github.com/DigitalSlideArchive/histomicsui](https://github.com/DigitalSlideArchive/histomicsui)

54. Cytomine ULiège - GitHub,
    [https://github.com/cytomine-uliege](https://github.com/cytomine-uliege)

55. cytomine/Cytomine-pims: Python Image Management Server - GitHub,
    [https://github.com/cytomine/pims](https://github.com/cytomine/pims)

56. Import image failure using the Cytomine Python Client API \... -
    GitHub,
    [https://github.com/cytomine/Cytomine-python-client/issues/36](https://github.com/cytomine/Cytomine-python-client/issues/36)
