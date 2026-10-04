# Báo cáo Lab Day 1 - MSSV 2A202602364

## 1. Thiết lập

- Môi trường: Google Colab, Tesla T4, PyTorch `2.11.0+cu130`.
- Dữ liệu: Forest CoverType; `464.809` mẫu train và `116.203` mẫu eval theo `split_metadata.csv`. Validation chiếm 20% train, phân tầng với seed 42, tạo `371.847` mẫu train và `92.962` mẫu validation.
- Chuẩn hóa: mean và std chỉ được fit trên 10 cột số của train; 44 cột one-hot được giữ nguyên. Eval không được dùng để fit hoặc chọn cấu hình.
- Baseline: M-base `54 -> 256 -> 128 -> 7`, 47.879 tham số, ReLU, He initialization, cross-entropy, SGD momentum 0,9, learning rate 0,1, batch 512, 20 epoch, FP32.
- Mốc đoán lớp đa số trên validation: accuracy `0,4876`.
- Các chủ đề đã thử: loss, optimizer, hyper-parameter, dropout, gradient clipping, mixed precision và initialization.

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả |
|---|---:|
| Số tham số M-base | 47.879 |
| Shape logits | `(B, 7)` |
| Step-0 CE loss của base-s42 | 2,3776 |
| `ln(7)` | 1,9459 |
| Overfit 20 mẫu | loss `0,000001`, accuracy `100%` |
| Gradient tới mọi tham số | Có, norm đều khác 0 |

Step-0 loss cao hơn `ln(7)` khoảng `0,4317`. Đây không phải lỗi shape hay softmax: He initialization tạo logits có độ phân tán ban đầu đáng kể, với activation std sau các Linear là `[0,6962; 0,6614; 0,6522]`. Phép thử 20 mẫu xác nhận model và vòng cập nhật phối hợp đúng.

Baseline được chạy với ba seed:

| exp_id | Best epoch | Val accuracy | Val macro-F1 | Val loss |
|---|---:|---:|---:|---:|
| `base-s1` | 18 | 0,9054 | 0,8390 | 0,2379 |
| `base-s42` | 20 | 0,9059 | 0,8564 | 0,2380 |
| `base-s2026` | 20 | 0,9086 | 0,8496 | 0,2302 |

Validation accuracy trung bình là `0,9066 +/- 0,0017`; validation macro-F1 là `0,8484 +/- 0,0087`. Tôi dùng `2 sigma = 0,0175` làm ngưỡng tham khảo. Chênh lệch nhỏ hơn ngưỡng này không được xem là bằng chứng chắc chắn rằng một cấu hình tốt hơn.

## 3. Kết quả thí nghiệm

### 3.1 Hàm mất mát

Tôi dự đoán CE phù hợp hơn MSE cho phân loại đa lớp. `loss-mse` đạt validation accuracy `0,8686` và macro-F1 `0,7326`, thấp hơn `base-s42` lần lượt `0,0373` và `0,1238`. Không thể so trực tiếp MSE loss `0,0296` với CE loss `0,2380` vì hai loss khác thang đo. CE tối ưu xác suất lớp đúng trực tiếp và tạo gradient hữu ích hơn khi model dự đoán sai; MSE trên logits one-hot học chậm, nhất là với lớp hiếm. Xem `figures/compare_loss_f1.png`.

### 3.2 Bộ tối ưu

Mỗi optimizer được thử với ít nhất hai learning rate. SGD momentum dùng `0,03` và `0,1`; Adam dùng `0,0003` và `0,001`.

| Optimizer tốt nhất | exp_id | LR | Val macro-F1 |
|---|---|---:|---:|
| SGD momentum | `base-s42` | 0,1 | 0,8564 |
| Adam | `opt-adam-lr1e-3` | 0,001 | 0,8495 |

Adam tốt nhất thấp hơn SGD momentum `0,0069`, nhỏ hơn `2 sigma`, nên chưa thể kết luận optimizer nào tổng quát hóa tốt hơn. Learning rate có ảnh hưởng lớn: SGD `0,03` chỉ đạt `0,8166`; Adam `0,0003` chỉ đạt `0,7958`. Kết luận về optimizer sẽ sai nếu mỗi optimizer không được chỉnh learning rate riêng. Xem `figures/compare_optimizer_f1.png`.

### 3.3 Hyper-parameter: kiến trúc

`hparam-wide` dùng `54 -> 512 -> 256 -> 7`, 161.287 tham số và đạt validation macro-F1 `0,8721`. `hparam-deep` dùng `54 -> 256 -> 128 -> 64 -> 7`, 55.687 tham số và đạt `0,8704`. Hai mức tăng so với baseline là `0,0158` và `0,0140`, đều chưa vượt `2 sigma`. M-wide được chọn làm cấu hình cuối vì có macro-F1 lớn nhất tại checkpoint được chọn bằng validation loss, nhưng thí nghiệm kiến trúc mới chỉ có một seed nên kết luận còn hạn chế. Xem `figures/compare_hparam_f1.png`.

### 3.4 Dropout

Baseline có final validation-train loss gap `0,0208`, chưa biểu hiện quá khớp mạnh. Dropout `0,2` giảm gap xuống `0,0085` nhưng macro-F1 giảm còn `0,8147`; dropout `0,5` giảm gap xuống `0,0043` nhưng macro-F1 chỉ còn `0,6647`. Dropout làm train và validation gần nhau hơn bằng cách regularize model, nhưng trong trường hợp này nó gây thiếu khớp. Gap nhỏ không tự động đồng nghĩa với mô hình tốt. Xem `figures/compare_dropout_f1.png`.

### 3.5 Gradient clipping

Gradient norm baseline ổn định khoảng `0,51-0,58`, nên tôi chọn `c=0,6` và stress-test ở learning rate `0,3`. Không clip đạt macro-F1 `0,8508`; clip đạt `0,8589`; cả hai không phân kỳ. Chênh lệch `0,0081` nhỏ hơn nhiễu. Gradient norm trung bình theo epoch của hai run nằm khoảng `0,35-0,40`, nhưng log không lưu cực đại từng batch nên không thể khẳng định clipping chưa bao giờ kích hoạt. Kết quả chỉ cho thấy không có bùng nổ gradient rõ trong stress test này. Xem `figures/compare_clipping_f1.png`.

### 3.6 Mixed precision

FP16 không cho kết quả như dự đoán. `amp-fp16` mất `1,826 s/epoch`, chậm hơn FP32 `1,442 s/epoch`, bộ nhớ đo được cùng khoảng `190,9 MB`, và phân kỳ tại epoch 9. Checkpoint hợp lệ trước phân kỳ đạt macro-F1 `0,7791`. Mạng nhỏ làm chi phí autocast và GradScaler lấn át lợi ích Tensor Core; FP16 cũng có dải biểu diễn hẹp hơn. Xem `figures/compare_amp_f1.png`.

### 3.7 Khởi tạo

He đạt macro-F1 `0,8564`, Xavier đạt `0,8538`; chênh lệch `0,0026` nằm trong nhiễu. Activation std của He là `[0,6916; 0,6563; 0,6535]`, còn Xavier là `[0,2887; 0,2237; 0,2169]`. He giữ phương sai tốt hơn qua ReLU. Với zeros, activation std bằng 0 ở mọi lớp; model mắc tại accuracy lớp đa số `0,4876` và macro-F1 `0,0936`. Các nơ-ron khởi tạo giống nhau nhận cập nhật đối xứng và không học được đặc trưng khác nhau. Xem `figures/compare_init_f1.png`.

## 4. Đánh giá cuối trên eval

Cấu hình cuối được khóa trước eval là `hparam-wide`, dựa trên validation macro-F1 `0,8721` tại checkpoint có validation loss thấp nhất ở epoch 18.

| Cấu hình | Seed | Val macro-F1 | Eval macro-F1 | Eval accuracy |
|---|---:|---:|---:|---:|
| Baseline `base-s42` | 42 | 0,8564 | 0,8596 | 0,9053 |
| Final `hparam-wide` | 42 | 0,8721 | 0,8734 | 0,9169 |

Final cải thiện eval macro-F1 `0,0138`. Mức tăng đã được đo nhưng vẫn nhỏ hơn ngưỡng nhiễu validation `0,0175`, nên chưa thể khẳng định M-wide luôn tốt hơn với seed khác. Val và eval của final chỉ lệch `0,0013`, cho thấy hai tập khá nhất quán.

### 4.1 Phân tích lỗi

| Lớp | Support | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 0 | 42.368 | 0,9193 | 0,9095 | 0,9144 |
| 1 | 56.661 | 0,9286 | 0,9314 | 0,9300 |
| 2 | 7.151 | 0,8978 | 0,9241 | 0,9108 |
| 3 | 549 | 0,7903 | 0,8579 | 0,8227 |
| 4 | 1.899 | 0,8231 | 0,7546 | 0,7874 |
| 5 | 3.473 | 0,8137 | 0,8390 | 0,8262 |
| 6 | 4.102 | 0,9151 | 0,9300 | 0,9225 |

Lớp 4 khó nhất với F1 `0,7874` và thường bị nhầm thành lớp 1. Lớp 4 chỉ có 1.899 mẫu eval, trong khi lớp 1 có 56.661 mẫu. Mất cân bằng và đặc trưng chồng lấn có thể làm biên quyết định thiên về lớp 1. Một hướng thử tiếp là class-weighted CE hoặc sampling cân bằng, nhưng tôi không chỉnh mô hình sau khi đã xem eval. Ma trận nhầm lẫn nằm ở `figures/final_confusion_matrix.png`.

## 5. Trả lời câu hỏi dẫn dắt

1. **Optimizer nào thắng?** SGD momentum `lr=0,1` cao hơn Adam `lr=0,001` đúng `0,0069` macro-F1, nhưng chưa vượt nhiễu. Nếu dùng learning rate chưa chỉnh, kết luận thay đổi mạnh vì cả SGD `0,03` và Adam `0,0003` đều học chậm.
2. **Dropout có giúp không?** Không trong 20 epoch vì baseline chưa quá khớp rõ. Dropout phù hợp hơn khi train loss tiếp tục giảm nhưng validation loss tăng, hoặc khi model có dư năng lực.
3. **Clipping giải quyết gì?** Nó giới hạn gradient quá lớn để tránh bước cập nhật mất ổn định. Trong log này hai stress run đều ổn định và khác biệt dưới nhiễu, nên chưa có bằng chứng clipping là cần thiết.
4. **Mixed precision có nhanh hơn không?** Không. FP16 chậm hơn và phân kỳ; nguyên nhân phù hợp với overhead trên MLP nhỏ và độ ổn định số thấp hơn.
5. **Vì sao zeros hỏng, He khác Xavier thế nào?** Zeros giữ các nơ-ron đối xứng. He dùng phương sai lớn hơn phù hợp ReLU; Xavier nhắm giữ phương sai cho activation đối xứng hơn và activation std giảm rõ trong mạng này.
6. **Ba kiểm tra đầu tiên khi loss không giảm:** (i) kiểm tra dữ liệu, nhãn `0..6`, shape, dtype và chuẩn hóa; (ii) kiểm tra logits đi thẳng vào loss, gradient có tới mọi tham số và thử overfit 20 mẫu; (iii) kiểm tra optimizer thật sự chứa tham số model, thứ tự `zero_grad/backward/step` và learning rate bằng một sweep ngắn.

## 6. Hạn chế và điều bất ngờ

- Các thí nghiệm ngoài baseline chỉ chạy một seed; `2 sigma` từ ba baseline seed chỉ là ước lượng thô.
- Cùng 20 epoch không bảo đảm cùng chi phí tính toán khi kiến trúc thay đổi.
- Clipping chỉ log gradient norm trung bình theo epoch, chưa log tỷ lệ batch thực sự bị clip.
- Thời gian và bộ nhớ chỉ đo trên một Tesla T4; kết luận FP16 không nên khái quát sang model lớn hoặc GPU khác.
- FP16 phân kỳ là kết quả khác dự đoán và được giữ nguyên thay vì loại khỏi bảng.

## 7. File nộp

- `code/` và `code/lab.ipynb`
- `experiments.xlsx`
- `predictions_eval.csv`
- `eval_result.json`
- `figures/` gồm ảnh từng `exp_id`, ảnh so sánh nhóm và confusion matrix
- `REPORT.md`
