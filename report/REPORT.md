# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: 
`class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.51087 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Record này nói về cả bức ảnh vì có task là image_classification, model phân loại toàn bộ ảnh thành một lớp đại diện. sample_id = traffic chỉ là tên định danh của ảnh, không có nghĩa model đang phân loại riêng một vật thể.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Danh sách tên lớp (class list) đi kèm taxonomy/checkpoint không phải do model tự nghĩ ra
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
Cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp vì để xác định chính xác lớp được dự đoán, tránh nhầm lẫn và đảm bảo tính nhất quán của dữ liệu, cùng một tên lớp có thể xuất hiện trong nhiều taxonomy với mã ID khác nhau nên cần lưu cả ba.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Guideline cần quy định rõ tiêu chí lựa chọn một lớp đại diện cho ảnh.

Ví dụ: Ưu tiên vật thể hoặc chủ đề chính, nổi bật nhất trong ảnh, dựa trên kích thước, vị trí và nội dung của ảnh. Nếu có nhiều vật thể, cần có quy tắc ưu tiên thống nhất để tránh mỗi người chọn một lớp khác nhau.
6. Vì sao không được lấy `score` của model làm ground truth?
Vì `score` chỉ thể hiện mức độ tin cậy dự đoán của model, không đảm bảo dự đoán đó chính xác.
Ground truth phải được xác định dựa trên dữ liệu đã kiểm chứng hoặc nhãn do con người gán theo guideline, không thể lấy trực tiếp từ kết quả dự đoán của model.
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: 
`class_name`=person ; `score`=0.912647 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.59 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
- Góc trên-trái (385.33, 69.24), cách mép trái 385.33 pixel và mép trên 69.24 pixel.
- Góc dưới-phải (498.92, 348.92), cách mép trái 498.92 pixel và mép trên 348.92 pixel.
- Chiều rộng: 113.59 pixel.
- Chiều cao: 279.68 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
- Chọn ngưỡng 0.35 và 0.60;
- Ngưỡng 0.35 số prediction: 11
- Ngưỡng 0.60 số prediction: 6

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Khi hạ ngưỡng, model giữ lại nhiều bounding box hơn. Reviewer phải kiểm tra thêm các box có độ tin cậy thấp, loại bỏ box sai, box trùng lặp, sửa nhãn và điều chỉnh vị trí box nếu cần.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Khi gán nhãn, bounding box phải bao sát phần nhìn thấy của vật thể, không chứa quá nhiều nền xung quanh và không bao gồm các vật thể khác không liên quan.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
- Chọn vật thể oven, có bounding box [0.11, 187.54, 195.96, 292.81].
- Cách xử lý: 
+ Đề xuất đưa vào guideline quy tắc: Nếu vật thể bị cắt bởi mép ảnh nhưng vẫn nhận diện được, chỉ vẽ bounding box bao quanh phần vật thể nằm trong ảnh, không tự suy đoán phần nằm ngoài ảnh.

+ Trường hợp không xác định được vật thể hoặc không rõ cách vẽ box thì cần hỏi Lab Coach (escalation).
Box gần sát mép trái ảnh (x = 0.11), cho thấy vật thể có khả năng bị cắt bởi mép ảnh.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: 
`instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899166 ; số điểm polygon=`polygon_point_count`=350 ; 3–5 điểm đầu của `polygon_xy`=
[
  [446.0, 70.0],
  [445.0, 71.0],
  [444.0, 71.0],
  [443.0, 72.0],
  [442.0, 72.0]
]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Bounding box chỉ tạo một hình chữ nhật bao quanh vật thể.

Polygon sử dụng nhiều điểm nối với nhau để mô tả đường viền thực tế của vật thể, giúp thể hiện hình dạng chính xác hơn và hạn chế lấy cả phần nền xung quanh.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
instance_id dùng để định danh từng đối tượng cụ thể được phát hiện trong ảnh.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Quy tắc: Khi vẽ mask, phải bám sát đường viền phần vật thể nhìn thấy được, không lấy phần nền hoặc vật thể khác vào mask và không tự suy đoán phần bị che khuất.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Chọn vật thể potted plant (kitchen-004), có bounding box [0.05, 1.4, 63.44, 148.88].

Vật thể nằm sát mép trái và mép trên ảnh, nên có khả năng bị cắt bởi mép ảnh.

Cách xử lý: Sử dụng guideline với quy tắc chỉ vẽ mask cho phần vật thể nhìn thấy trong ảnh, không tự vẽ thêm phần nằm ngoài ảnh.

Lý do: Đây là trường hợp có thể quy định thống nhất cho nhiều ảnh, không nhất thiết phải escalation. Chỉ hỏi Lab Coach khi không xác định được ranh giới hoặc không có quy tắc xử lý phù hợp.
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

Dựa trên kết quả JSON của ba tác vụ và chỉ xét ảnh sample_id = kitchen

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp cho toàn bộ ảnh | Model dự đoán gong với score 0.42036 cho ảnh kitchen, có thể không phản ánh đúng nội dung chính | Xem toàn bộ ảnh, chọn nhãn đúng theo guideline | Kiểm tra nhãn có phù hợp nội dung ảnh không |
| Phát hiện vật thể | Mỗi vật thể có một nhãn lớp và một bounding box | Box oven có tọa độ x1 = 0.11, sát mép trái ảnh, có khả năng vật thể bị cắt mép | Gán nhãn và vẽ box bao sát phần vật thể nhìn thấy | Kiểm tra nhãn, vị trí box, vật thể bị bỏ sót hoặc box thừa |
| Instance segmentation | Mỗi vật thể có nhãn lớp và một mask riêng | potted plant (kitchen-004) có box bắt đầu tại (0.05, 1.4), gần mép ảnh, gây khó khăn khi xác định biên mask | Vẽ mask theo đường viền phần vật thể nhìn thấy, tách riêng từng đối tượng | Kiểm tra độ chính xác đường viền, vùng mask thừa/thiếu và các vật thể bị che khuất |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ sử dụng ảnh và dữ liệu được cấp cho mục đích bài lab, không tự ý chia sẻ, sao chép hoặc tải dữ liệu lên nền tảng bên ngoài khi chưa được cho phép.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Tôi sẽ dừng xử lý dữ liệu đó và báo ngay cho Lab Coach hoặc giảng viên hướng dẫn để được xác nhận và hướng dẫn xử lý.
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
