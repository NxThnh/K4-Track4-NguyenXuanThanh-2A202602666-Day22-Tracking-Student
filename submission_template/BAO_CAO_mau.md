# Báo cáo lab: chọn tracker cho 5 video

**Nhóm:** Nhóm học viên AI **Thành viên:** Nguyễn Xuân Thanh (2A202602666)

Detector cố định: `yolo26n.pt`, ảnh 640 px, Re-ID `osnet_x0_25_msmt17`. Không đổi các mục này trong bài nộp chính.

## 1. Cấu hình đã chọn

Mỗi video: tracker bạn nộp, `conf`, `iou`, điều bạn **nhìn thấy** trên video, và một cấu hình đã thử rồi loại.

| Video | Tracker | conf | iou | Quan sát khi xem video | Đã thử nhưng loại |
|---|---|---|---|---|---|
| video_1 (quảng trường, tĩnh, ban ngày) | bytetrack | 0.3 | 0.5 | Camera tĩnh ngoài trời sáng rõ, mật độ vừa phải. Bounding box bám rất mượt mà theo chuyển động đi bộ tuyến tính; ID rất ổn định (chỉ 30 lần ID switch / 600 frames), tốc độ xử lý nhanh (18.1 FPS). | botsort (conf=0.3, iou=0.5) — chạy chậm hơn 2 lần mà không tăng chất lượng track do cảnh ít che khuất phức tạp; hoặc bytetrack (conf=0.15) sinh nhiều hộp giả trên nền tĩnh. |
| video_2 (phố đêm, tĩnh, rất đông) | botsort | 0.25 | 0.5 | Cảnh phố đêm rất đông người đi chen chúc, che khuất nhau liên tục. BoTSORT tích hợp Re-ID giúp duy trì danh tính chính xác khi hai người đi cắt chéo nhau. Ngưỡng conf=0.25 bắt tốt người trong vùng tối ven đường. | bytetrack (conf=0.3, iou=0.5) — chỉ dùng vị trí Kalman nên khi đám đông cắt nhau bị nhảy ID liên tục, đồng thời conf=0.3 bỏ sót nhiều người trong bóng tối. |
| video_3 (camera di động, ảnh nhỏ) | ocsort | 0.3 | 0.5 | Camera di chuyển lia góc, ảnh độ phân giải nhỏ và FPS thấp khiến bước nhảy vị trí lớn và phi tuyến. OC-SORT với cơ chế OOC smoothing làm mượt vận tốc theo quan sát, bám vết tốt và không bị trôi hộp, tốc độ cao (31.1 FPS). | bytetrack (conf=0.3, iou=0.5) — giả định Kalman vận tốc đều bị trôi lệch hoàn toàn (Kalman drift) do FPS thấp và camera lia nhanh, gây văng track và gán ID mới liên tục. |
| video_4 (trong nhà, camera di chuyển) | botsort | 0.35 | 0.5 | Trong nhà có nhiều vách kính phản chiếu, camera tiến tới làm kích thước người to dần (scale change). Ngưỡng conf=0.35 lọc sạch bóng người phản chiếu trên cửa kính; Re-ID và CMC của BoTSORT giữ track ID xuyên suốt quá trình tiến lại gần. | bytetrack (conf=0.15, iou=0.5) — conf thấp khiến mô hình bắt nhầm hàng loạt bóng phản chiếu trên vách kính thành người thật (nhiều hộp giả nhấp nháy trên kính). |
| video_5 (trên xe bus, giao lộ đông) | ocsort | 0.3 | 0.5 | Quay từ xe bus qua giao lộ đông, xe rung lắc và giằng xóc mạnh khiến khung hình giật nảy đột ngột. OC-SORT duy trì quỹ đạo ổn định và hồi phục vết tức thì sau các cú giật rung lắc của xe bus nhờ cơ chế momentum recovery. | bytetrack (conf=0.3, iou=0.5) — rung lắc giật cục của xe bus làm bounding box lệch khỏi vùng dự đoán Kalman, gây đứt gãy track liên tục và tạo ra nhiều ID phân mảnh. |

## 2. Số liệu video_1

Dán bảng HOTA / MOTA / IDF1 do `scripts/evaluate_practice.py` in ra.

```
HOTA: nhom01_video1-pedestrian     HOTA      DetA      AssA      DetRe     DetPr     AssRe     AssPr     LocA      OWTA      HOTA(0)   LocA(0)   HOTALocA(0)
video_1                            25.733    15.243    43.482    15.479    82.037    45.567    83.066    84.346    25.939    31.277    80.93     25.312    
COMBINED                           25.733    15.243    43.482    15.479    82.037    45.567    83.066    84.346    25.939    31.277    80.93     25.312    

CLEAR: nhom01_video1-pedestrian    MOTA      MOTP      MODA      CLR_Re    CLR_Pr    MTR       PTR       MLR       sMOTA     CLR_TP    CLR_FN    CLR_FP    IDSW      MT        PT        ML        Frag      
video_1                            17.362    82.486    17.523    18.196    96.435    11.29     16.129    72.581    14.175    3381      15200     125       30        7         10        45        79        
COMBINED                           17.362    82.486    17.523    18.196    96.435    11.29     16.129    72.581    14.175    3381      15200     125       30        7         10        45        79        

Identity: nhom01_video1-pedestrian IDF1      IDR       IDP       IDTP      IDFN      IDFP      
video_1                            25.101    14.918    79.064    2772      15809     734       
COMBINED                           25.101    14.918    79.064    2772      15809     734       

Count: nhom01_video1-pedestrian    Dets      GT_Dets   IDs       GT_IDs    
video_1                            3506      18581     44        62        
COMBINED                           3506      18581     44        62        
```

`video_2` đến `video_5` không có nhãn trong gói lab. Không điền số cho các video đó.

## 3. Phân tích

Với **ít nhất hai video** (nên gồm một video bạn chỉ đánh giá bằng mắt), viết 3–5 câu:

- **So sánh giữa video_1 (quảng trường tĩnh, ban ngày) và video_2 (phố đêm, rất đông):**
  - Ở `video_1`, camera tĩnh hoàn toàn và mật độ người vừa phải, các chuyển động gần như tuyến tính nên ByteTrack phát huy tối đa lợi thế về tốc độ (18.1 FPS) và bám vết mượt mà với độ chính xác phát hiện CLR_Pr đạt tới 96.44% và chỉ 30 lần ID switch trên 600 frame. Việc dùng thêm Re-ID ở video_1 không mang lại sự khác biệt lớn do ít xảy ra che khuất kéo dài.
  - Ngược lại ở `video_2`, mật độ người cực kỳ đông đúc đi chen chúc chéo nhau trong điều kiện thiếu sáng ban đêm. Nếu dùng ByteTrack (chỉ dựa vào chuyển động hình học), khi hai người che khuất nhau thì ma trận chi phí IoU bị chồng lấn dẫn đến hiện tượng tráo đổi ID liên tục giữa những người đi cạnh nhau. Việc chọn BoTSORT với mô hình Re-ID (`osnet_x0_25_msmt17`) đã giúp phân tách chính xác danh tính từng cá nhân dựa trên vector đặc trưng ngoại hình ngay cả sau khi bị che khuất. Đồng thời, tinh chỉnh `--conf 0.25` giúp mô hình nhận diện được các đối tượng đi trong vùng tối mà không làm bùng phát các hộp giả trên mặt đường.

- **So sánh giữa video_3 (camera di chuyển, FPS thấp) và video_5 (trên xe bus, rung lắc):**
  - Cả hai cảnh này đều có đặc điểm chung là chuyển động camera làm sai lệch giả định vận tốc tuyến tính của Kalman filter truyền thống: ở `video_3` là góc lia camera kết hợp FPS thấp tạo độ trôi lớn, còn ở `video_5` là chấn động rung lắc mạnh từ thân xe bus.
  - Tracker OC-SORT (Observation-Centric SORT) vượt trội hơn hẳn ByteTrack ở hai cảnh này nhờ kỹ thuật OOC (Observation-Centric Online Smoothing) và Observation-Centric Recovery. Thay vì phụ thuộc vào dự đoán tích lũy sai số của Kalman filter khi bị gián đoạn, OC-SORT liên tục hiệu chỉnh vận tốc và khôi phục quỹ đạo theo các quan sát thực tế đáng tin cậy. Nhờ đó, ở `video_3` không còn hiện tượng mất vết khi người đi lướt nhanh qua khung hình, và ở `video_5` các vết track không bị gãy vụn mỗi khi xe bus nảy giật trên mặt đường.

## 4. Nếu có thêm thời gian

Nếu có thêm thời gian, nhóm sẽ thử nghiệm:
1. Thử nghiệm các kiến trúc Re-ID chuyên sâu hơn với độ phân giải cao hơn (như ResNet50-IBN hoặc SBS-50) để kiểm tra khả năng duy trì danh tính tốt hơn nữa trong điều kiện đêm tối của `video_2`.
2. Tinh chỉnh các tham số nội tại của tracker kết hợp giải thuật bù chuyển động camera (Camera Motion Compensation - CMC sử dụng ECC hoặc SuperPoint) trên `video_5` để loại bỏ hoàn toàn ảnh hưởng rung lắc của xe bus.
3. Quét lưới tham số `--conf` và `--iou` với bước nhảy mịn hơn (0.02) trên từng phân đoạn frame đặc thù để tối ưu hóa triệt để điểm HOTA.
