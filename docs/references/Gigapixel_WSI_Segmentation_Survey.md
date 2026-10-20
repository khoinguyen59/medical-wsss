# Phân đoạn ngữ nghĩa và phân tích giám sát yếu, hiệu quả dữ liệu cho ảnh giải phẫu bệnh toàn tiêu bản Gigapixel: Khảo sát chuyên sâu

## Bối cảnh nghiên cứu và các hệ hình khái niệm trong phân tích WSI Gigapixel

Quá trình số hóa tiêu bản mô bệnh học đã thúc đẩy sự hình thành của các
ảnh toàn tiêu bản (Whole Slide Images - WSI) với độ phân giải siêu cao,
thường dao động từ ,000 \times 40,000${width="1.452364391951006in"
height="0.2583530183727034in"} đến hơn
,000 \times 100,000${width="1.6665627734033246in"
height="0.2569280402449694in"} điểm ảnh ở độ phóng đại
\times${width="0.3810761154855643in"
height="0.2563604549431321in"} hoặc
\times${width="0.3810761154855643in"
height="0.2563604549431321in"}^1^. Kích thước gigapixel này ngăn cản
việc đưa trực tiếp toàn bộ dữ liệu vào bộ nhớ của các bộ xử lý đồ họa
hiện đại, đòi hỏi quy trình tiền xử lý phân rã tiêu bản thành hàng chục
nghìn mảnh cắt nhỏ (tiles hoặc patches)^2^.

Trong thực hành lâm sàng, việc phân đoạn ngữ nghĩa ở cấp độ điểm ảnh
nhằm xác định chính xác ranh giới của các cấu trúc vi mô--chẳng hạn như
biểu mô ác tính, chất đệm khối u, vùng thâm nhiễm lympho bào hay các ổ
hoại tử--đóng vai trò quyết định đối với việc đánh giá vi môi trường
khối u (Tumor Microenvironment - TME)^5^. Tuy nhiên, việc tạo lập các
nhãn phân đoạn mức pixel (pixel-level dense masks) trên các ma trận
gigapixel đòi hỏi hàng trăm giờ làm việc tỉ mỉ từ các chuyên gia giải
phẫu bệnh^5^. Khoảng cách biểu diễn giữa nhãn chẩn đoán mức tiêu bản sẵn
có trong hồ sơ bệnh án và yêu cầu phân đoạn cấu trúc vi mô dày đặc đã
trở thành nút thắt trung tâm của ngành giải phẫu bệnh tính toán
(Computational Pathology - CPath), tạo động lực trực tiếp cho các hệ
hình học giám sát yếu (Weakly-Supervised Learning) và học hiệu quả dữ
liệu (Data-Efficient Learning) phát triển^2^.

Học đa thể hiện (Multiple Instance Learning - MIL) từ lâu đã đóng vai
trò là khung lý thuyết tiêu chuẩn để giải quyết bài toán giám sát yếu
cấp tiêu bản^2^. Trong không gian giải bài toán của MIL, mỗi WSI được mô
hình hóa như một \"túi\" (bag)
 = \{x_1, x_2, \dots, x_N\}${width="1.9338123359580053in"
height="0.27031824146981626in"} chứa
${width="0.19726596675415572in"
height="0.27476268591426073in"} thể hiện cục bộ đại diện cho các tile,
liên kết với một nhãn chẩn đoán gộp
${width="0.17209208223972003in"
height="0.26846347331583553in"}^10^. Các kỹ thuật gộp dựa trên cơ chế
chú ý (Attention-based Pooling), tiêu biểu như ABMIL hay mạng phân cụm
có ràng buộc CLAM, tính toán trọng số quan trọng tương đối
${width="0.20681430446194227in"
height="0.2688593613298338in"} cho từng thể hiện để tổng hợp thành
vector biểu diễn toàn slide ${width="0.34375in"
height="0.27359689413823274in"}^10^:

a_k = \frac{\exp(w^T (\tanh(V h_k^T) \odot \text{sigm}(U h_k^T)))}{\sum_{j=1}^N \exp(w^T (\tanh(V h_j^T) \odot \text{sigm}(U h_j^T)))}, \quad z = \sum_{k=1}^N a_k h_k{width="6.458333333333333in"
height="0.9148064304461943in"}

Mặc dù các giá trị trọng số chú ý này có thể được ánh xạ ngược trở lại
tọa độ không gian để xây dựng bản đồ nhiệt (heatmaps) phục vụ định vị
vùng nghi ngờ ung thư^2^, cơ chế này bộc lộ những hạn chế cố hữu khi mở
rộng sang phân đoạn ngữ nghĩa đa lớp dày đặc^14^. Trọng số chú ý của MIL
phản ánh đóng góp thống kê của từng tile đối với nhãn toàn cục hơn là
ranh giới hình thái học tế bào thực tế, dẫn đến việc thiếu hụt độ sắc
nét ở đường biên mô và thường xuyên bỏ sót các cấu trúc nền tảng không
mang tính phân biệt cao đối với nhãn slide^2^.

Để trích xuất mặt nạ phân đoạn chi tiết mà chỉ dựa vào nhãn phân loại
cấp hình ảnh hoặc cấp patch, các phương pháp phân đoạn ngữ nghĩa giám
sát yếu (Weakly-Supervised Semantic Segmentation - WSSS) truyền thống đã
áp dụng Bản đồ kích hoạt lớp (Class Activation Maps - CAM) và
Grad-CAM^5^. Bằng cách chiếu trọng số của lớp phân loại tuyến tính lên
bản đồ đặc trưng trước lớp gộp toàn cục (Global Average Pooling), CAM
ước tính phản ứng không gian đối với từng lớp mô học cụ thể^5^:

M_c(x, y) = \sum_k w_k^c f_k(x, y){width="6.458333333333333in"
height="0.8998906386701663in"}

Tuy nhiên, việc triển khai CAM thuần túy trong ảnh mô bệnh học thường
vấp phải nghịch lý định vị phân biệt: mô hình phân loại có xu hướng tối
ưu hóa bằng cách chỉ tập trung kích hoạt các vùng lõi có tính phân biệt
cao nhất (chẳng hạn như các đám tế bào ung thư dị sản rõ rệt), trong khi
bỏ qua các vùng viền chuyển tiếp, mô đệm lân cận hoặc các cấu trúc xơ
hóa^5^. Hiện tượng này trở nên đặc biệt nghiêm trọng do hai đặc trưng
sinh học cố hữu là tính đồng nhất liên lớp (inter-class
homogeneity)--khi các mô có nguồn gốc khác nhau nhưng lại biểu hiện phổ
màu nhuộm và kết cấu tương tự dưới kính hiển vi--và tính không đồng
nhất nội lớp (intra-class heterogeneity), thể hiện qua sự đa dạng hình
thái của cùng một loại tổn thương ác tính trong lòng khối u^5^. Kết quả
là các mặt nạ giả (pseudo-masks) tạo ra từ CAM thường bị phân mảnh,
thiếu hụt diện tích hoặc xâm lấn sai lệch sang các cấu trúc bình
thường^5^.

Nhằm khắc phục sự suy thoái ranh giới của CAM, các cấu trúc tinh chế
nhãn giả lặp dựa trên mô hình giáo viên - học sinh (Teacher-Student
framework) đã được áp dụng rộng rãi^16^. Trong cơ chế này, mạng giáo
viên được ổn định hóa bằng phương pháp trung bình động lũy thừa
(Exponential Moving Average - EMA) để cung cấp các phân phối không gian
mượt mà, kết hợp với các bộ lọc ngưỡng tin cậy thích ứng nhằm loại bỏ
các điểm ảnh không chắc chắn^18^. Mạng học sinh sau đó được tối ưu hóa
đồng thời bằng hàm mất mát giám sát trên nhãn giả đã tinh chế và hàm mất
mát nhất quán đối xứng^18^:

\mathcal{L}_{\text{sim}} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_k \exp(\text{sim}(z_i, z_k)/\tau)}{width="6.458333333333333in"
height="0.5866688538932633in"}

Để hoàn thiện tính liên tục hình thái, các trường ngẫu nhiên có điều
kiện dày đặc (Dense Conditional Random Fields - Dense CRF) cùng các
thuật toán phân cụm không gian thường xuyên được tích hợp làm lớp tinh
chỉnh hậu xử lý, ràng buộc các điểm ảnh có vị trí không gian kề cận và
giá trị màu sắc tương đồng phải thuộc về cùng một thực thể mô^9^.

Một hướng tiếp cận mang tính đột phá khác là chuyển hóa căn bản bài toán
giám sát yếu thành bài toán giám sát đầy đủ thông qua các kỹ thuật tổng
hợp dữ liệu và trộn ảnh cấu trúc^14^. Bằng cách khai thác các patch mô
học thuần nhất chỉ chứa một loại mô duy nhất (single-tissue patches),
các phương pháp như PistoSeg và HisynSeg áp dụng phép biến đổi ghép
Mosaic hoặc tạo mặt nạ phân chia ngẫu nhiên bằng đường cong Bézier để
kết hợp nhiều mô hình thái khác nhau vào một hình ảnh nhân tạo^14^. Do
nguồn gốc của từng vùng thành phần đã được định danh chính xác, hệ thống
tự động thiết lập được mặt nạ phân đoạn mức điểm ảnh chuẩn xác mà không
cần thông qua giai đoạn sinh CAM sơ bộ^14^. Mô hình phân đoạn dày đặc
được huấn luyện trực tiếp trên tập dữ liệu tổng hợp này, kết hợp với các
cơ chế chính quy hóa tính nhất quán đặc trưng trên ảnh thực để triệt
tiêu các hiện tượng giả tạo đường viền (stitching artifacts)^14^.

Song song với việc tận dụng nhãn cấp độ slide hoặc patch, các hình thức
giám sát thưa (sparse supervision) như giám sát dựa trên điểm chấm
(point supervision) hoặc nét vẽ (scribble supervision) đã chứng minh
tiềm năng vượt trội trong việc cân bằng giữa chi phí gán nhãn và độ
chính xác phân đoạn^22^. Thay vì phân định toàn bộ ranh giới phức tạp,
chuyên gia chỉ cần đánh dấu vị trí trung tâm hoặc phác thảo một đường
nét đại diện^23^. Các thuật toán lan truyền nhãn dựa trên sơ đồ Voronoi,
phân cụm k-means và học tương phản cấp độ điểm ảnh sẽ khai thác gradient
màu sắc tự nhiên giữa nhân tế bào và tế bào chất để mở rộng nhãn thưa
thành các vùng mặt nạ hoàn chỉnh^23^. Xu hướng này tiếp tục được khuếch
đại thông qua việc chuyển giao các mô hình nền tảng phân đoạn thị giác
như Segment Anything Model (SAM), nơi các tương tác điểm hoặc hộp thưa
từ chuyên gia được ánh xạ trực tiếp thành các phân đoạn mô bệnh học có
độ chính xác tiệm cận với việc phân định thủ công^25^.

## Các công trình ảnh hưởng và bước đột phá công nghệ

Những bước tiến gần đây trong phân đoạn mô bệnh học giám sát yếu tập
trung vào việc tái cấu trúc không gian biểu diễn của nhãn giả và áp dụng
các cơ chế dẫn đường dựa trên nguyên mẫu thị giác^5^. Đáng chú ý, khung
làm việc Prototype-Based Image Prompting (PBIP) đã giải quyết triệt để
sự thất bại của các gợi ý văn bản trong việc nắm bắt các biến thiên vi
thể bằng cách xây dựng ngân hàng nguyên mẫu hình ảnh thông qua phân cụm
đặc trưng từ tập huấn luyện^5^. Bằng việc kết hợp mạng phân loại đa tầng
SegFormer và mạng đối sánh đặc trưng hình ảnh dưới ràng buộc hàm mất mát
tương phản, PBIP kích hoạt đồng bộ toàn bộ phạm vi mô thay vì chỉ tập
trung vào các điểm dị sản cực trị^5^.

Một bước phát triển tiếp theo là công trình ConStruct, giới thiệu chiến
lược chưng cất cấu trúc kết hợp hai mô hình nền tảng đông cứng^9^.
ConStruct sử dụng mô hình thị giác - ngôn ngữ CONCH để trích xuất ngữ
nghĩa bệnh lý phong phú, đồng thời tận dụng bộ mã hóa SegFormer MiT-B1
để bảo toàn cấu trúc ranh giới không gian^9^. Bằng cách chỉ huấn luyện
một tập hợp các adapter trọng lượng nhẹ chiếm khoảng
$< 1\%${width="0.452257217847769in"
height="0.2614610673665792in"} tổng tham số mô hình kết hợp với bộ lọc
Dense CRF, ConStruct đã thiết lập kỷ lục hiệu năng mới trên chuẩn kiểm
thử BCSS-WSSS^9^.

Ở khía cạnh tinh chế kích hoạt nội tại, khung làm việc ARML (Activation
Relocalization and Mutual Learning) đã đề xuất phương án mở rộng ma trận
chú ý không gian để tái phân bố các tín hiệu kích hoạt từ vùng
discriminative sang toàn bộ diện tích mô, kết hợp học tương hỗ giữa hai
nhánh mạng để triệt tiêu nhiễu nhãn giả^16^. Cùng hướng đi đó, mô hình
SIA-WSSS tích hợp cơ chế đồng bộ hóa thông tin ức chế và kích hoạt, ngăn
ngừa hiện tượng kích hoạt lan truyền không kiểm soát sang các cấu trúc
mô liên kết lân cận^28^.

Sự xuất hiện của các mô hình không gian trạng thái (State Space Models -
SSM), đặc biệt là kiến trúc Mamba với độ phức tạp tính toán tuyến tính
$\mathcal{O}(N)${width="0.5387379702537183in"
height="0.26936898512685914in"} theo chiều dài chuỗi, đã tạo nên một
bước ngoặt lớn cho bài toán phân tích WSI^11^. Mô hình PathMamba là công
trình tiên phong đưa Mamba vào bài toán phân đoạn mô bệnh học giám sát
yếu đa lớp^11^. Kiến trúc này kết hợp học tương phản với khối
Contrastive Mamba Block (DC-Mamba), triển khai quét chọn lọc (selective
scan) bốn hướng để trích xuất thông tin phi nhân quả ở cả mức độ hạt cục
bộ và toàn cục^11^. Quá trình này được tối ưu hóa thông qua hệ thống
phương trình trạng thái rời rạc hóa có tham số phụ thuộc trực tiếp vào
tín hiệu đầu vào:

h\\'(t) = A h(t) + B x(t), \quad y(t) = C h(t){width="6.458333333333333in"
height="0.5866688538932633in"}

Trong đó các ma trận $\bar{A}, \bar{B}${width="0.8614359142607174in"
height="0.2709361329833771in"} được biến điệu động bởi mạng nơ-ron theo
ngữ cảnh của từng tile, cho phép mô hình nén và tích lũy thông tin không
gian dài hạn mà không gặp phải rào cản bùng nổ bộ nhớ^29^. Kết hợp với
hàm mất mát Deep Contrast Supervised Loss, PathMamba phân tách rõ rệt
các ranh giới mô học phức tạp mà chỉ cần giám sát bằng nhãn phân loại
patch^11^.

Đối với bài toán tổng hợp toàn slide, mô hình MambaMIL đã khắc phục hiện
tượng mất liên kết hình học 2D khi làm phẳng không gian bằng khối
Sequence Reordering Mamba (SRMamba), tái sắp xếp các chuỗi thể hiện dựa
trên độ tương đồng tương quan để tối ưu hóa khả năng mô hình hóa bối
cảnh lâm sàng quy mô lớn^10^. Tiếp nối thành công này, mô hình
2DMambaMIL mở rộng cơ chế quét trạng thái sang không gian hai chiều tự
nhiên, bảo toàn toàn vẹn tính topo học của lát cắt mô học^29^.

Hiệu quả của các phương pháp phân đoạn ngữ nghĩa giám sát yếu và hiệu
quả dữ liệu thường được định chuẩn thông qua độ đo trung bình chỉ số
tương đồng Dice (mDice) và tỷ lệ giao trên hợp (mIoU) trên các tập dữ
liệu công khai tiêu chuẩn như
[BCSS-WSSS](https://github.com/ChuHan89/WSSS-Tissue) và
[LUAD-HistoSeg](https://github.com/ChuHan89/WSSS-Tissue)^6^.
Bảng dưới đây đối sánh hiệu năng phân đoạn giữa các phương pháp giám sát
yếu tiêu biểu và các đường cơ sở giám sát đầy đủ:

  ------------------------------------------------------------------
  **Phương pháp**   **Loại kiến trúc**  **Dạng giám     **BCSS-WSSS   **BCSS-WSSS   **LUAD-HistoSeg   **Điểm nổi bật kỹ     **Nguồn
                                        sát**           (mIoU %)**    (mDice %)**   (mIoU %)**        thuật**               trích
                                                                                                                            dẫn**
  --------- ---------- -------- ------- ------- --------- ----------- -----
  **Fully           DeepLabV3+ / UNet   Giám sát đầy đủ 72.80 - 74.50 84.10 - 85.60 76.20 - 78.40     Huấn luyện trực tiếp  ^7^
  Supervised                            (Pixel-level)                                                 trên mặt nạ vẽ tay    
  Baseline**                                                                                          chuẩn của chuyên gia  

  **WSSS-Tissue**   CNN + Attention     Giám sát yếu    64.20         77.10         67.80             Multi-layer           ^7^
                    Dropout             (Patch-level)                                                 pseudo-supervision,   
                                                                                                      rớt vùng chú ý để ép  
                                                                                                      học đường biên        

  **PistoSeg**      CNN + Feature       Giám sát yếu    70.75         81.90         71.40             Ghép ảnh Mosaic từ mô ^34^
                    Consistency         (Patch-level)                                                 thuần nhất, chuyển    
                                                                                                      đổi thành giám sát    
                                                                                                      dày                   

  **HisynSeg**      DeepLabV3+          Giám sát yếu    70.82         82.15         71.95             Tổng hợp mặt nạ       ^15^
                    (EfficientNet-b6)   (Patch-level)                                                 Bézier kết hợp lọc    
                                                                                                      ảnh chân thực và ràng 
                                                                                                      buộc nhất quán        

  **PathMamba**     Dual-Granularity    Giám sát yếu    68.90         80.40         70.85             Quét chọn lọc 4 hướng ^11^
                    Mamba SSM           (Patch-level)                                                 phi nhân quả kết hợp  
                                                                                                      Deep Contrast         
                                                                                                      Supervised Loss       

  **PBIP**          SegFormer + Visual  Giám sát yếu    70.15         81.80         72.30             Ngân hàng nguyên mẫu  ^5^
                    Bank                (Image-level)                                                 hình ảnh triệt tiêu   
                                                                                                      tính bất đồng nhất    
                                                                                                      nội lớp               

  **ConStruct**     CONCH + SegFormer   Giám sát yếu    **70.96**     **82.83**     \-               Chưng cất cấu trúc từ ^9^
                    Adapters            (Image-level)                                                 Foundation Model, chỉ 
                                                                                                      huấn luyện 3.7% tham  
                                                                                                      số                    
  ------------------------------------------------------------------

Khoảng cách hiệu năng phân đoạn giữa các phương pháp tiếp cận giám sát
yếu tiên tiến nhất như ConStruct hay HisynSeg đối với mô hình giám sát
đầy đủ truyền thống hiện đã được thu hẹp xuống dưới
$< 5\%${width="0.2855905511811024in"
height="0.26417104111986in"} mIoU^7^. Điều này minh chứng rằng sự kết
hợp giữa các cơ chế tổng hợp biên nhân tạo và việc trích xuất đặc trưng
sâu từ các mô hình nền tảng có khả năng bù đắp hoàn toàn sự thiếu hụt
của các đường viền gán nhãn thủ công^9^.

## Mô hình nền tảng, kiến trúc mạng và ngăn xếp công nghệ

Sự phát triển của giải phẫu bệnh tính toán giai đoạn gần đây được định
hình bởi bước chuyển dịch từ việc sử dụng các mạng ResNet tiền huấn
luyện trên tập dữ liệu tự nhiên ImageNet sang các Mô hình Nền tảng Giải
phẫu bệnh (Pathology Foundation Models - PFMs) quy mô lớn^36^. Các mô
hình này được huấn luyện thông qua tự học không giám sát
(Self-Supervised Learning - SSL) hoặc học đa phương thức thị giác - ngôn
ngữ trên hàng trăm triệu hình ảnh mô học, mang lại khả năng tổng quát
hóa vượt bậc^38^.

Trong kiến trúc tổng thể của việc phân tích WSI gigapixel, quy trình
trích xuất và phân đoạn diễn ra thông qua sự phối hợp phân tầng chặt chẽ
giữa các thành phần mô hình chuyên biệt. Tiêu bản kích thước gigapixel
ban đầu được chia tách thành lưới các tile mô học ở các mức phóng đại
chuẩn hóa^3^. Các tile này được đưa qua các bộ mã hóa nền tảng cấp độ
patch--chẳng hạn như UNI2-h, Virchow 2 hay H-optimus-0--để chuyển đổi
cấu trúc điểm ảnh thành các vector nhúng cô đọng mang đặc trưng hình
thái học tinh vi^38^. Tiếp theo, tập hợp hàng vạn vector nhúng cùng với
tọa độ không gian tương ứng được nạp vào bộ mã hóa cấp độ toàn tiêu bản
(slide-level encoder) như cấu trúc LongNet trong Prov-GigaPath hoặc kiến
trúc TITAN để tổng hợp bối cảnh mô bệnh học toàn cục^3^. Cuối cùng, các
đặc trưng ngữ cảnh này được giải mã thông qua các đầu đọc phân đoạn dày
đặc (dense decoders) như ViT-Adapter, SegFormer hoặc Mask2Former để tái
cấu trúc thành bản đồ phân đoạn ngữ nghĩa ở cấp độ điểm ảnh^5^.

Đi đầu trong phân khúc mô hình nền tảng toàn slide là hệ thống
[Prov-GigaPath](https://github.com/prov-gigapath/prov-gigapath)
do Providence và Microsoft phát triển^3^. Prov-GigaPath tích hợp bộ mã
hóa tile ViT-G khổng lồ với bộ mã hóa slide xây dựng trên kiến trúc
LongNet 12 lớp, sử dụng cơ chế chú ý mở rộng (dilated attention) để xử
lý các chuỗi cực dài lên tới hàng chục nghìn vector nhúng tile^3^. Gần
đây, biến thể tối ưu GigaPath-Flash sử dụng DINOv2-ViT-S/16 kết hợp bộ
mã hóa slide 384 chiều đã cung cấp giải pháp tính toán hiệu quả cao cho
các hạ tầng giới hạn tài nguyên^3^.

Ở cấp độ phân tích hình thái tế bào siêu chi tiết, hệ thống [[Virchow
2]{.underline}](https://huggingface.co/paige-ai/Virchow2) và Virchow 2G
từ Paige và MSKCC thiết lập các chuẩn mực mới về quy mô với lần lượt 632
triệu và 1.85 tỷ tham số^38^. Để xử lý đặc thù của dữ liệu mô học,
Virchow 2 đưa ra hai cải tiến quan trọng trên thuật toán DINOv2: thay
thế bộ điều chuẩn khoảng cách KoLeo bằng ước lượng mật độ nhân (Kernel
Density Estimator - KDE) nhằm tránh hiện tượng sụp đổ gradient do sự dư
thừa mô học quá cao, đồng thời áp dụng phép dịch chuyển ngữ cảnh mở rộng
(Extended Context Translation - ECT) thay cho phép cắt-phóng to
(crop-and-resize) để ngăn chặn hiện tượng méo mó tỷ lệ tế bào^38^.

Phòng thí nghiệm Mahmood Lab tại Harvard và Brigham and Women\'s
Hospital đóng góp một hệ sinh thái mô hình nền tảng phong phú, khởi đầu
với [UNI](https://github.com/mahmoodlab/UNI) và phiên bản
kế nhiệm
[UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) xây
dựng trên ViT-H/14 với lớp kích hoạt SwiGLU, huấn luyện trên 350,000
tiêu bản đa dạng phương pháp nhuộm^12^. Bên cạnh đó, mô hình thị giác -
ngôn ngữ [CONCH](https://github.com/mahmoodlab/CONCH) mang
lại khả năng phân đoạn zero-shot có điều kiện văn bản nhờ căn chỉnh biểu
diễn giữa 1.17 triệu cặp ảnh và chú thích bệnh lý^40^. Gần đây nhất, mô
hình [TITAN](https://github.com/mahmoodlab/TITAN) đã kết
hợp tự giám sát WSI với hơn 600,000 báo cáo y khoa thực tế và văn bản
tổng hợp từ tác nhân AI PathChat, tạo nên bộ biểu diễn toàn diện phục vụ
chẩn đoán bệnh lý phức tạp^41^. Cùng với đó, mô hình
[CHIEF](https://github.com/hms-dbmi/CHIEF) của Harvard
DBMI tích hợp mã hóa vị trí giải phẫu bằng mô hình ngôn ngữ - hình ảnh,
tối ưu hóa việc phân tầng nguồn gốc ung thư và tiên lượng đột biến
gen^13^, trong khi
[H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0)
của Bioptimus cung cấp không gian nhúng 1536 chiều mạnh mẽ từ kiến trúc
ViT-G 1.1 tỷ tham số^39^.

Bảng dưới đây tổng hợp các đặc tính kỹ thuật, quy mô huấn luyện và bản
quyền của các mô hình nền tảng giải phẫu bệnh tiêu biểu:

Để thích ứng các bộ trích xuất đặc trưng đóng băng này cho các bài toán
phân đoạn dày đặc mà không làm phát sinh chi phí tính toán khổng lồ, các
phương pháp hiện đại sử dụng các bộ giải mã chuyên dụng kết hợp thích
ứng tham số hiệu quả (Parameter-Efficient Fine-Tuning - PEFT)^9^. Điển
hình như kiến trúc ViT-Adapter đưa các mô-đun không gian đa thang đo bên
ngoài vào thân ViT đông cứng, tạo ra các bản đồ đặc trưng ở các tỷ lệ
/4, 1/8, 1/16${width="1.7855938320209974in"
height="0.2580741469816273in"} tương thích hoàn hảo với bộ giải mã
Mask2Former hoặc UPerNet^12^. Tương tự, bộ giải mã SegFormer MLP cho
phép tổng hợp các đặc trưng đa tầng với chi phí tính toán tối thiểu^5^.
Khi cần tinh chỉnh trực tiếp các trọng số tự chú ý của mô hình nền tảng,
các bộ điều hợp LoRA (Low-Rank Adaptation) phân rã ma trận cập nhật
trọng số $\Delta W = B \cdot A${width="1.184896106736658in"
height="0.270239501312336in"} thành các thành phần có số chiều ẩn rất
thấp, cho phép mô hình học các biến thể hình thái vi mô mới mà chỉ cần
cập nhật dưới $< 1\%${width="0.2855905511811024in"
height="0.26417104111986in"} tổng số tham số^9^.

## Hệ sinh thái mã nguồn mở, công cụ kỹ thuật và tập dữ liệu chuẩn

Hệ sinh thái phần mềm mã nguồn mở dành cho WSI đã đạt đến độ hoàn thiện
cao, hỗ trợ toàn diện các quy trình từ nạp dữ liệu tiêu bản, trích xuất
đặc trưng cho đến phân đoạn ngữ nghĩa và hiển thị kết quả^4^.

Bộ công cụ
[TIAToolbox](https://github.com/TissueImageAnalytics/tiatoolbox)
do TIA Centre tại Đại học Warwick phát triển cung cấp hệ thống giao diện
mạnh mẽ hỗ trợ đọc truy cập ngẫu nhiên (random-access read) đa định dạng
như SVS, NDPI, MRXS cũng như định dạng lưu trữ hiện đại
OME-NGFF/Zarr^59^. TIAToolbox tích hợp sẵn các động cơ suy luận phân
đoạn như SemanticSegmentor và NucleusInstanceSegmentor, đồng thời hỗ trợ
nạp trực tiếp các mô hình nền tảng thông qua API TimmBackbone^59^.

Đối với nhu cầu xử lý dữ liệu tốc độ cao, thư viện
[Slideflow](https://github.com/slideflow/slideflow) cung
cấp giải pháp tối ưu hóa thông lượng dữ liệu bằng cách sử dụng các luồng
xử lý thời gian thực từ phần cứng NVIDIA GPU (qua cuCIM) và Libvips, hỗ
trợ chuẩn hóa màu sắc tức thời và tích hợp sâu rộng các mô hình MIL như
CLAM cùng các Foundation Model bao gồm UNI, GigaPath và PLIP^60^.

Gần đây hơn, thư viện
[Trident](https://github.com/mahmoodlab/trident) từ
Mahmood Lab chuẩn hóa quy trình phân tích WSI thành một pipeline khép
kín bao gồm bốn bước: tự động phân đoạn mô/nền, tạo lưới tile hình thái,
trích xuất đặc trưng patch bằng UNI/UNI2-h/CONCH và tổng hợp biểu diễn
toàn slide^4^. Bên cạnh đó, nền tảng
[CLAM](https://github.com/mahmoodlab/CLAM) vẫn giữ vững
vai trò là công cụ chuẩn mực cho việc đào tạo các mô hình MIL có diễn
giải trực quan trên bản đồ nhiệt WSI^10^.

Bảng dưới đây tóm tắt các tính năng cốt lõi và kiến trúc kỹ thuật của
các bộ công cụ mã nguồn mở phổ biến:

Hệ thống dữ liệu chuẩn và các cuộc thi quốc tế đóng vai trò then chốt
trong việc kiểm chuẩn các giải pháp giám sát yếu^6^. Chuẩn kiểm thử
[BCSS-WSSS](https://github.com/ChuHan89/WSSS-Tissue) và
[LUAD-HistoSeg](https://github.com/ChuHan89/WSSS-Tissue)
cung cấp các tập dữ liệu có nhãn yếu patch-level cho quá trình huấn
luyện kèm mặt nạ điểm ảnh chính xác đã qua thẩm định của chuyên gia để
đánh giá khả năng phân tách bốn thành phần mô cơ bản: u, mô đệm, lympho
bào và hoại tử^6^.

Trên quy mô lâm sàng, chuỗi thử thách [[PAIP (Pathology AI
Platform)]{.underline}](https://paip2019.grand-challenge.org/) tại các
hội nghị MICCAI và ISBI đã định hướng nghiên cứu qua nhiều năm: PAIP
2019 tập trung vào việc phân đoạn ung thư gan và ước tính gánh nặng khối
u sống (viable tumor burden)^66^; PAIP 2020 kết hợp phân đoạn vùng khối
u đại trực tràng với dự đoán tình trạng mất ổn định vi vệ tinh
(MSI)^69^; PAIP 2021 đánh giá khả năng phát hiện xâm lấn quanh dây thần
kinh (Perineural Invasion) trên đa cơ quan^65^; và PAIP 2023 thử thách
việc phân đoạn nhân tế bào và ước tính mật độ tế bào khối u trong điều
kiện học chuyển giao với dữ liệu mẫu cực kỳ khan hiếm^73^. Tương tự, thử
thách
[DigestPath](https://digestpath2019.grand-challenge.org/)
cung cấp nền tảng chuẩn hóa cho việc phân đoạn tổn thương biểu mô tuyến
đại tràng và nhận diện tế bào nhẫn trên tiêu bản gigapixel^71^.

## Thách thức mở và chân trời nghiên cứu tiếp theo

Dịch chuyển miền (Domain shift) tiếp tục là một trong những trở ngại lớn
nhất cản trở việc triển khai rộng rãi các mô hình phân đoạn mô học vào
thực tế^18^. Sự sai khác hình thái học xuất hiện do nhiều yếu tố khó
kiểm soát trong quy trình phòng xét nghiệm, bao gồm độ dày lát cắt mô,
nhiệt độ cố định mẫu, nồng độ các hóa chất nhuộm Hematoxylin và Eosin,
cũng như đặc tính quang học của các dòng máy quét WSI khác nhau^1^.

Các thực nghiệm trên khung mô hình giáo viên - học sinh cho thấy độ
chính xác có thể duy trì tốt trên các tập dữ liệu có cùng nguồn gốc
chuẩn bị (như TCGA-COAD và TCGA-READ) nhưng lại suy giảm nghiêm trọng
khi đối mặt với các miền dữ liệu độc lập bên ngoài như tập SPIDER^18^.
Các thuật toán chuẩn hóa màu nhuộm truyền thống như Macenko, Vahadane
hay Reinhard thường tỏ ra không đủ mạnh để xử lý các phân phối kết cấu
phức tạp và thậm chí có thể làm biến dạng các chi tiết hạt sắc tố của
nhân tế bào^17^. Do đó, các nghiên cứu tương lai đang hướng tới việc
tích hợp các cơ chế học bất biến miền trực tiếp (Domain-Invariant CAM -
D-CAM) hoặc tận dụng các biểu diễn đặc trưng có tính khái quát cao từ
các mô hình nền tảng được huấn luyện trên hàng triệu tiêu bản^46^.

Bên cạnh độ trôi dạt dữ liệu, rào cản về chi phí tính toán và bộ nhớ khi
suy luận trên các ma trận gigapixel đặt ra bài toán kinh tế lớn cho các
cơ sở y tế^3^. Mặc dù các PFM quy mô hàng tỷ tham số như Virchow 2G hay
H-optimus-0 mang lại độ chính xác biểu diễn vượt trội, việc xử lý toàn
bộ các tile của một slide duy nhất đòi hỏi tài nguyên tính toán cấp
trung tâm dữ liệu với các GPU chuyên dụng như NVIDIA A100^3^. Xu hướng
nén và chưng cất tri thức sang các mô hình siêu nhẹ phục vụ triển khai
tại chỗ (on-premise)--chẳng hạn như Prov-GigaPath-Flash, Virchow2G Mini
hay H0-mini--đang trở thành trọng tâm phát triển nhằm duy trì hiệu năng
chẩn đoán trong khi giảm mạnh độ trễ và chi phí hạ tầng phần cứng^3^.
Đồng thời, các giải pháp lưu trữ hình ảnh thông minh kết hợp kỹ thuật
nén độ phân giải thấp với mạng tái tạo siêu phân giải (Super-Resolution)
hướng tác vụ đang mở ra khả năng tiết kiệm không gian lưu trữ mà không
đánh đổi độ chính xác của các thuật toán phân đoạn giám sát yếu^6^.

Một chân trời công nghệ quan trọng khác là sự hội tụ đa phương thức giữa
hình thái học không gian và thông tin phân tử^3^. Các giải pháp phân
đoạn thế hệ mới không chỉ dừng lại ở việc gán nhãn mô học thuần túy mà
còn liên kết hình ảnh H&E với dữ liệu phiên mã không gian (Spatial
Transcriptomics) và protein học không gian (Spatial Proteomics)^3^. Các
mô hình tiên tiến như GigaTIME đã chứng minh khả năng dự đoán trực tiếp
các phân bố protein và thành phần tế bào miễn dịch ngay trên các lát cắt
mô thông thường^3^. Việc kết hợp các bản đồ phân đoạn cấu trúc với dữ
liệu hệ gen cho phép các thuật toán dự đoán chính xác các đột biến gen
đích (như EGFR, KRAS, BRAF) cũng như xác định các phân nhóm phân tử khối
u, trực tiếp hỗ trợ định hướng các phác đồ điều trị đích và liệu pháp
miễn dịch^13^.

Cuối cùng, việc chuyển giao các giải pháp phân đoạn giám sát yếu vào môi
trường lâm sàng thực tế đòi hỏi phải vượt qua các tiêu chuẩn phê duyệt
nghiêm ngặt từ các cơ quan quản lý y tế như US FDA hoặc CE-IVD^1^. Phần
lớn các thuật toán AI giải phẫu bệnh được phê chuẩn hiện nay, tiêu biểu
như *Paige Prostate Detect*, chủ yếu thực hiện chức năng phân loại hoặc
định vị vùng nghi ngờ ở mức tiêu bản để hỗ trợ chuyên gia đọc slide^80^.
Các hệ thống phân đoạn ngữ nghĩa dày đặc giám sát yếu vẫn cần chứng minh
độ tin cậy tuyệt đối tại các ranh giới tổn thương, loại bỏ hoàn toàn các
hiện tượng tạo nhãn giả sai lệch (pseudo-label hallucination) và đảm bảo
tính minh bạch của các vùng kích hoạt không gian^2^. Việc xây dựng các
quy trình kiểm định đa trung tâm độc lập và tích hợp các độ đo định
lượng độ bất định (Uncertainty Quantification) vào từng điểm ảnh phân
đoạn sẽ là bước chuyển cốt lõi để các thuật toán giám sát yếu chính thức
trở thành công cụ hỗ trợ tin cậy trong thực hành bệnh học hàng ngày^18^.

## Kết luận

Lĩnh vực phân tích và phân đoạn ngữ nghĩa ảnh giải phẫu bệnh gigapixel
đang trải qua giai đoạn chuyển mình sâu sắc nhờ sự kết hợp giữa các
nguyên lý học giám sát yếu, mô hình không gian trạng thái tuyến tính và
các mô hình nền tảng giải phẫu bệnh quy mô lớn^3^. Các sáng kiến vượt
qua giới hạn của Class Activation Maps thông qua nguyên mẫu hình ảnh,
trộn ảnh cấu trúc nhân tạo và chưng cất tri thức đa mô hình đã thu hẹp
khoảng cách hiệu năng với các phương pháp giám sát đầy đủ xuống mức tối
thiểu^5^.

Đồng thời, sự xuất hiện của các kiến trúc nền tảng chuyên biệt như
Prov-GigaPath, Virchow 2 và UNI2-h đã cung cấp nền tảng biểu diễn mô
bệnh học toàn diện, giải phóng các nhà nghiên cứu khỏi sự phụ thuộc vào
các nhãn thủ công tốn kém^3^. Khi các giải pháp kiểm soát dịch chuyển
miền, chưng cất mô hình gọn nhẹ và kiểm định lâm sàng tiếp tục hoàn
thiện, phân đoạn ngữ nghĩa giám sát yếu sẽ trở thành hạ tầng công nghệ
nền tảng, đưa giải phẫu bệnh số hóa tiến tới kỷ nguyên chẩn đoán định
lượng và y học cá thể hóa ở quy mô toàn cầu^1^.

#### Nguồn trích dẫn

1.  Survey on Whole Slide Image in Pathology: Deep learning and,
    [https://www.researchgate.net/publication/397264019_Survey_on_Whole_Slide_Image_in_Pathology_Deep_learning_and_Machine_learning_approaches](https://www.researchgate.net/publication/397264019_Survey_on_Whole_Slide_Image_in_Pathology_Deep_learning_and_Machine_learning_approaches)

2.  Geometric multi-instance learning for weakly supervised gastric,
    [https://pmc.ncbi.nlm.nih.gov/articles/PMC12860791/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12860791/)

3.  prov-gigapath/prov-gigapath-flash - Hugging Face,
    [https://huggingface.co/prov-gigapath/prov-gigapath-flash](https://huggingface.co/prov-gigapath/prov-gigapath-flash)

4.  GitHub - mahmoodlab/TRIDENT: Toolkit for large-scale whole-slide,
    [https://github.com/mahmoodlab/trident/](https://github.com/mahmoodlab/trident/)

5.  Prototype-Based Image Prompting for Weakly Supervised,
    [https://openaccess.thecvf.com/content/CVPR2025/papers/Tang_Prototype-Based_Image_Prompting_for_Weakly_Supervised_Histopathological_Image_Segmentation_CVPR_2025_paper.pdf](https://openaccess.thecvf.com/content/CVPR2025/papers/Tang_Prototype-Based_Image_Prompting_for_Weakly_Supervised_Histopathological_Image_Segmentation_CVPR_2025_paper.pdf)

6.  Toward Efficient Weakly Supervised Semantic Segmentation Using,
    [https://arxiv.org/html/2607.10783v1](https://arxiv.org/html/2607.10783v1)

7.  GitHub - ChuHan89/WSSS-Tissue,
    [https://github.com/ChuHan89/WSSS-Tissue](https://github.com/ChuHan89/WSSS-Tissue)

8.  Physics-Grounded Weakly Supervised Histopathological Tissue,
    [https://papers.miccai.org/miccai-2026/0778-Paper1322.html](https://papers.miccai.org/miccai-2026/0778-Paper1322.html)

9.  ConStruct: Structural Distillation of Foundation Models for
    Prototype,
    [https://arxiv.org/html/2512.10316v1](https://arxiv.org/html/2512.10316v1)

10. isyangshu/MambaMIL: \[MICCAI 2024\] Official Code for \... - GitHub,
    [https://github.com/isyangshu/MambaMIL](https://github.com/isyangshu/MambaMIL)

11. PathMamba: Weakly Supervised State Space Model for Multi-class,
    [https://papers.miccai.org/miccai-2024/paper/1354_paper.pdf](https://papers.miccai.org/miccai-2024/paper/1354_paper.pdf)

12. MahmoodLab/UNI2-h - Hugging Face,
    [https://huggingface.co/MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h)

13. Revolutionizing Cancer Diagnosis with the CHIEF Foundation Model,
    [https://blogs.torus.ai/revolutionizing-cancer-diagnosis-with-the-chief-foundation-model/](https://blogs.torus.ai/revolutionizing-cancer-diagnosis-with-the-chief-foundation-model/)

14. Weakly-Supervised Semantic Segmentation for Histopathology,
    [https://mmlab-sigs.github.io/groups/zwang/publications/6106.FangZ.pdf](https://mmlab-sigs.github.io/groups/zwang/publications/6106.FangZ.pdf)

15. HisynSeg: Weakly-Supervised Histopathological Image \... - arXiv,
    [https://arxiv.org/html/2412.20924v1](https://arxiv.org/html/2412.20924v1)

16. Weakly Supervised Histological Tissue Segmentation with Activation,
    [https://papers.miccai.org/miccai-2024/paper/1156_paper.pdf](https://papers.miccai.org/miccai-2024/paper/1156_paper.pdf)

17. Enhanced gastric cancer classification and quantification \... -
    PMC,
    [https://pmc.ncbi.nlm.nih.gov/articles/PMC11439054/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11439054/)

18. Weakly Supervised Teacher-Student Framework with Progressive,
    [https://www.xiahepublishing.com/2771-165X/JCTP-2025-00055](https://www.xiahepublishing.com/2771-165X/JCTP-2025-00055)

19. UniSemAlign: Text-Prototype Alignment with a Foundation Encoder,
    [https://openaccess.thecvf.com/content/CVPR2026W/CV4Clinic2026/papers/Van_Thai_UniSemAlign_Text-Prototype_Alignment_with_a_Foundation_Encoder_for_Semi-Supervised_Histopathology_CVPRW_2026_paper.pdf](https://openaccess.thecvf.com/content/CVPR2026W/CV4Clinic2026/papers/Van_Thai_UniSemAlign_Text-Prototype_Alignment_with_a_Foundation_Encoder_for_Semi-Supervised_Histopathology_CVPRW_2026_paper.pdf)

20. a survey on shape-constraint deep learning for - arXiv,
    [https://arxiv.org/pdf/2101.07721](https://arxiv.org/pdf/2101.07721)

21. Deep Learning in Breast Cancer Imaging: A Decade of Progress,
    [https://arxiv.org/html/2304.06662v4](https://arxiv.org/html/2304.06662v4)

22. Point-supervised Segmentation of Microscopy Images and Volumes,
    [https://www.researchgate.net/publication/349943905_Point-supervised_Segmentation_of_Microscopy_Images_and_Volumes_via_Objectness_Regularization](https://www.researchgate.net/publication/349943905_Point-supervised_Segmentation_of_Microscopy_Images_and_Volumes_via_Objectness_Regularization)

23. GitHub - hvcl/scribble2label: Official PyTorch implementation of,
    [https://github.com/hvcl/scribble2label](https://github.com/hvcl/scribble2label)

24. Publication - HVCL,
    [https://hvcl.korea.ac.kr/?page_id=5770](https://hvcl.korea.ac.kr/?page_id=5770)

25. 醫學影像分割的弱監督持續學習Weakly-supervised Continual,
    [https://tdr.lib.ntu.edu.tw/retrieve/6af20510-a65e-44c4-a4d5-534ec2761cfe/ntu-112-2.pdf](https://tdr.lib.ntu.edu.tw/retrieve/6af20510-a65e-44c4-a4d5-534ec2761cfe/ntu-112-2.pdf)

26. Jingwei Zhang - alphaXiv,
    [https://www.alphaxiv.org/@jingwei-zhang-2](https://www.alphaxiv.org/@jingwei-zhang-2)

27. Segment Anything Model (SAM) for Digital Pathology: Assess Zero,
    [https://pmc.ncbi.nlm.nih.gov/articles/PMC11971729/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11971729/)

28. Synchronous Inhibition and Activation for Weakly Supervised,
    [https://papers.miccai.org/miccai-2025/paper/5096_paper.pdf](https://papers.miccai.org/miccai-2025/paper/5096_paper.pdf)

29. 2DMamba: Efficient State Space Model for Image Representation,
    [https://arxiv.org/html/2412.00678v1](https://arxiv.org/html/2412.00678v1)

30. \[2403.06800\] MambaMIL: Enhancing Long Sequence Modeling with,
    [https://arxiv.org/abs/2403.06800](https://arxiv.org/abs/2403.06800)

31. PathMamba: Weakly Supervised State Space Model for Multi-class,
    [https://papers.miccai.org/miccai-2024/594-Paper1354.html](https://papers.miccai.org/miccai-2024/594-Paper1354.html)

32. MambaMIL: Enhancing Long Sequence Modeling with Sequence,
    [https://papers.miccai.org/miccai-2024/487-Paper3255.html](https://papers.miccai.org/miccai-2024/487-Paper3255.html)

33. MambaMIL: Enhancing Long Sequence Modeling with Sequence,
    [https://www.alphaxiv.org/abs/2403.06800](https://www.alphaxiv.org/abs/2403.06800)

34. GitHub - SZU-AdvTech-2023/382-Weakly-Supervised-Semantic,
    [https://github.com/SZU-AdvTech-2023/382-Weakly-Supervised-Semantic-Segmentation-for-Histopathology-Images-Based-on-Dataset-Synthesis-and](https://github.com/SZU-AdvTech-2023/382-Weakly-Supervised-Semantic-Segmentation-for-Histopathology-Images-Based-on-Dataset-Synthesis-and)

35. Weakly-Supervised Semantic Segmentation for Histopathology,
    [https://ojs.aaai.org/index.php/AAAI/article/view/25136](https://ojs.aaai.org/index.php/AAAI/article/view/25136)

36. A whole-slide foundation model for digital pathology from
    real-world,
    [https://www.researchgate.net/publication/380788897_A_whole-slide_foundation_model_for_digital_pathology_from_real-world_data](https://www.researchgate.net/publication/380788897_A_whole-slide_foundation_model_for_digital_pathology_from_real-world_data)

37. UNI/README_old.md at main · mahmoodlab/UNI - GitHub,
    [https://github.com/mahmoodlab/UNI/blob/main/README_old.md](https://github.com/mahmoodlab/UNI/blob/main/README_old.md)

38. Virchow2 \| Model Catalog \| Microsoft Foundry,
    [https://ai.azure.com/catalog/models/Virchow2](https://ai.azure.com/catalog/models/Virchow2)

39. Hugging Face Digital Pathology \| (Para) Pathology Notes,
    [https://www.parapathology.com/computational-digital-and-mathematical-pathology/hugging-face-digital-pathology](https://www.parapathology.com/computational-digital-and-mathematical-pathology/hugging-face-digital-pathology)

40. mahmoodlab/CONCH: Vision-Language Pathology Foundation Model,
    [https://github.com/mahmoodlab/CONCH](https://github.com/mahmoodlab/CONCH)

41. TITAN: A Multimodal AI Model for Digital Pathology Slides,
    [https://www.helmholtz-munich.de/en/newsroom/news-all/artikel/titan-a-multimodal-ai-model-for-digital-pathology-slides](https://www.helmholtz-munich.de/en/newsroom/news-all/artikel/titan-a-multimodal-ai-model-for-digital-pathology-slides)

42. Assessing Foundation Models for Computational Pathology in,
    [https://www.medrxiv.org/content/10.64898/2026.05.22.26353897v1.full.pdf](https://www.medrxiv.org/content/10.64898/2026.05.22.26353897v1.full.pdf)

43. MahmoodLab/UNI2-h-features · Datasets at Hugging Face,
    [https://huggingface.co/datasets/MahmoodLab/UNI2-h-features](https://huggingface.co/datasets/MahmoodLab/UNI2-h-features)

44. prov-gigapath/prov-gigapath - Hugging Face,
    [https://huggingface.co/prov-gigapath/prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath)

45. Prov-GigaPath - GitHub,
    [https://github.com/prov-gigapath/prov-gigapath](https://github.com/prov-gigapath/prov-gigapath)

46. Scaling Self-Supervised Mixed Magnification Models in Pathology,
    [https://arxiv.org/html/2408.00738v3](https://arxiv.org/html/2408.00738v3)

47. (PDF) Virchow 2: Scaling Self-Supervised Mixed Magnification,
    [https://www.researchgate.net/publication/382797319_Virchow_2_Scaling_Self-Supervised_Mixed_Magnification_Models_in_Pathology](https://www.researchgate.net/publication/382797319_Virchow_2_Scaling_Self-Supervised_Mixed_Magnification_Models_in_Pathology)

48. Virchow2: Scaling Self-Supervised Mixed Magnification Models in,
    [https://arxiv.org/pdf/2408.00738](https://arxiv.org/pdf/2408.00738)

49. Foundation models in computational pathology - PMC - NIH,
    [https://pmc.ncbi.nlm.nih.gov/articles/PMC13157769/](https://pmc.ncbi.nlm.nih.gov/articles/PMC13157769/)

50. Multi-Modal Foundation Models for Computational Pathology - PMC,
    [https://pmc.ncbi.nlm.nih.gov/articles/PMC13081565/](https://pmc.ncbi.nlm.nih.gov/articles/PMC13081565/)

51. wenhaozhang0066/Awesome-Pathology-VLMs: A curated \... - GitHub,
    [https://github.com/wenhaozhang0066/Awesome-Pathology-VLMs](https://github.com/wenhaozhang0066/Awesome-Pathology-VLMs)

52. MahmoodLab/TITAN - Hugging Face,
    [https://huggingface.co/MahmoodLab/TITAN](https://huggingface.co/MahmoodLab/TITAN)

53. A multimodal whole-slide foundation model for pathology - PubMed,
    [https://pubmed.ncbi.nlm.nih.gov/41193692/](https://pubmed.ncbi.nlm.nih.gov/41193692/)

54. A pathology foundation model for cancer diagnosis and prognosis,
    [https://pure.psu.edu/en/publications/a-pathology-foundation-model-for-cancer-diagnosis-and-prognosis-p/](https://pure.psu.edu/en/publications/a-pathology-foundation-model-for-cancer-diagnosis-and-prognosis-p/)

55. hms-dbmi/CHIEF: Clinical Histopathology Imaging Evaluation,
    [https://github.com/hms-dbmi/CHIEF](https://github.com/hms-dbmi/CHIEF)

56. Pathology Feature Extractors and Foundation Models - GitHub,
    [https://github.com/georg-wolflein/pathology-foundation-models](https://github.com/georg-wolflein/pathology-foundation-models)

57. GitHub - dibalokechanda/PFMs: A curated catalog of pathology,
    [https://github.com/dibalokechanda/PFMs](https://github.com/dibalokechanda/PFMs)

58. mahmoodlab/TITAN: Multimodal Whole Slide Foundation \... - GitHub,
    [https://github.com/mahmoodlab/TITAN](https://github.com/mahmoodlab/TITAN)

59. tiatoolbox - PyPI,
    [https://pypi.org/project/tiatoolbox/](https://pypi.org/project/tiatoolbox/)

60. slideflow · PyPI,
    [https://pypi.org/project/slideflow/](https://pypi.org/project/slideflow/)

61. TIAToolbox as an end-to-end library for advanced tissue image,
    [https://pmc.ncbi.nlm.nih.gov/articles/PMC9509319/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9509319/)

62. Computational Pathology Toolbox developed by TIA Centre \... -
    GitHub,
    [https://github.com/tissueimageanalytics/tiatoolbox](https://github.com/tissueimageanalytics/tiatoolbox)

63. semantic_segmentor - TIA Toolbox 2.1.3 Documentation,
    [https://tia-toolbox.readthedocs.io/en/latest/\_autosummary/tiatoolbox.models.engine.semantic_segmentor.html](https://tia-toolbox.readthedocs.io/en/latest/_autosummary/tiatoolbox.models.engine.semantic_segmentor.html)

64. nucleus_instance_segmentor.py - GitHub,
    [https://github.com/TissueImageAnalytics/tiatoolbox/blob/develop/tiatoolbox/models/engine/nucleus_instance_segmentor.py](https://github.com/TissueImageAnalytics/tiatoolbox/blob/develop/tiatoolbox/models/engine/nucleus_instance_segmentor.py)

65. pathology image segmentation challenges,
    [https://dipot.ulb.ac.be/dspace/bitstream/2013/352391/3/FOUCART22-Challenges-preprint.pdf](https://dipot.ulb.ac.be/dspace/bitstream/2013/352391/3/FOUCART22-Challenges-preprint.pdf)

66. Artificial Intelligence-Based Opportunities in Liver Pathology--A,
    [https://pmc.ncbi.nlm.nih.gov/articles/PMC10217251/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10217251/)

67. Home - PAIP 2019 - Grand Challenge,
    [https://paip2019.grand-challenge.org/](https://paip2019.grand-challenge.org/)

68. PAIP 2019: Liver cancer segmentation challenge - CUHK Research,
    [https://research.cuhk.edu.hk/files/256067802/PAIP_2019_Liver_cancer_segmentation_challenge.pdf](https://research.cuhk.edu.hk/files/256067802/PAIP_2019_Liver_cancer_segmentation_challenge.pdf)

69. Home - PAIP2020 - Grand Challenge,
    [https://paip2020.grand-challenge.org/](https://paip2020.grand-challenge.org/)

70. Deep learning for the detection of microsatellite instability from,
    [https://eprints.whiterose.ac.uk/id/eprint/181621/8/1-s2.0-S2667119021000082-main.pdf](https://eprints.whiterose.ac.uk/id/eprint/181621/8/1-s2.0-S2667119021000082-main.pdf)

71. A list of digital pathology image analysis challenges - Adrien
    Foucart,
    [https://research.adfoucart.be/challenges](https://research.adfoucart.be/challenges)

72. Automated Hybrid Model for Detecting Perineural Invasion in the,
    [https://www.mdpi.com/2076-3417/12/18/9159](https://www.mdpi.com/2076-3417/12/18/9159)

73. Home - PAIP 2023: TC prediction in pancreatic and colon cancer,
    [https://2023paip.grand-challenge.org/](https://2023paip.grand-challenge.org/)

74. Application of Artificial Intelligence in Pathology: Trends and
    \... - MDPI,
    [https://www.mdpi.com/2075-4418/12/11/2794](https://www.mdpi.com/2075-4418/12/11/2794)

75. A Deep Learning Approach for Colonoscopy Pathology WSI Analysis,
    [https://www.researchgate.net/publication/347257459_A_Deep_Learning_Approach_for_Colonoscopy_Pathology_WSI_Analysis_Accurate_Segmentation_and_Classification](https://www.researchgate.net/publication/347257459_A_Deep_Learning_Approach_for_Colonoscopy_Pathology_WSI_Analysis_Accurate_Segmentation_and_Classification)

76. MICCAI 2025 - Open Access,
    [https://papers.miccai.org/miccai-2025/](https://papers.miccai.org/miccai-2025/)

77. bioptimus/H0-mini - Hugging Face,
    [https://huggingface.co/bioptimus/H0-mini](https://huggingface.co/bioptimus/H0-mini)

78. mahmoodlab/UNI: Pathology Foundation Model - Nature Medicine,
    [https://github.com/mahmoodlab/UNI](https://github.com/mahmoodlab/UNI)

79. Foundational models for histology: H-Optimus - Bioptimus,
    [https://www.bioptimus.com/h-optimus](https://www.bioptimus.com/h-optimus)

80. Diagnostic AI - Paige.ai,
    [https://www.paige.ai/diagnostic-ai](https://www.paige.ai/diagnostic-ai)

81. An Interactive Semi-supervised Learning Framework for Gigapixel,
    [https://escholarship.org/content/qt4sk4g2d0/qt4sk4g2d0_noSplash_5956692b1fc53be99d0e7be93f731d89.pdf?t=sdc19t](https://escholarship.org/content/qt4sk4g2d0/qt4sk4g2d0_noSplash_5956692b1fc53be99d0e7be93f731d89.pdf?t=sdc19t)
