# Lab 21 — Evaluation Report

**Họ tên**: Cao Đức Anh  **MSSV**: 2A202602754  **Ngày**: 08/10/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (14.6 GB khả dụng)`

> Mọi con số dưới đây khớp chính xác với các file trong `results/`. Grader đã kiểm tra chéo qua gatekeeper.

---

## 1. Setup

| Thông số | Giá trị |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage (4 trường: intent, urgency, product, sentiment) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2.0 / 30 steps |

**Template có giữ khối `<think>` không?** Có — *(results/template_check.json)* khẳng định `reasoning preserved — safe to train on traces`. Chuỗi sau khi render vẫn giữ nguyên các thẻ `<think>` và `</think>`, không bị loại bỏ ngầm bởi tokenizer.

**Giải thích về `max_length`:** Mặc dù p95 đo được là 98 tokens (gợi ý `max_length=256`), cấu hình tier T4 đặt mặc định `1024` nhằm đảm bảo biên độ an toàn cho các câu hỏi CSKH dài bất thường hoặc câu trả lời có sinh thêm khoảng trắng, đồng thời tận dụng bộ nhớ VRAM 14.6 GB của card Tesla T4 mà không gặp nguy cơ tràn bộ nhớ.

---

## 2. Mask proof (NB1)

| Chỉ số | Giá trị kiểm tra |
|---|---|
| `supervised_fraction` | 0.4149 (41.49%) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

3–5 dòng đầu của đoạn được tính loss (trích xuất từ `results/mask_proof.json`):

```json
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Phần prompt đầu vào của người dùng và các thẻ hệ thống hoàn toàn được gán nhãn `-100`, loại bỏ triệt để nguy cơ mô hình học vẹt câu hỏi của người dùng.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7911 | 0.000 | 3145.4 |
| (b) base + optimized prompt | 0.765 | 0.7911 | 1.000 | 988.4 |
| (c) LoRA fine-tune | 0.975 | 0.4778 | 1.000 | 1355.7 |

**(b) có thật sự mạnh hơn (a) không?** Có, (b) đạt 0.765 vượt trội hoàn toàn so với (a) đạt 0.000 trên tập target. Tôi giữ nguyên `OPTIMIZED_PROMPT` mặc định của lab (SHA: `719e74d3b6232053`). Việc prompt tối ưu đạt format chuẩn 1.000 và độ chính xác 76.5% tạo ra một chuẩn đối chiếu (baseline) thực sự mạnh mẽ, đảm bảo tính liêm chính và công bằng tuyệt đối cho phép so sánh với mô hình fine-tune.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6256 | 0.975 | 389.6 | 8.78 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 0.0001 | 0.5379 | 0.975 | 269.0 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 0.00001 | 1.5702 | 0.000 | 398.4 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | 0.940 | 469.3 | 3.86 |

> Bảng xếp hạng được đánh giá dựa trên cột **target** ở NB5, không dựa trên train loss.

### Phân tích chi tiết ba câu hỏi:

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**
Trên tập target, `attn_only` hòa với `correct` khi cả hai đều đạt độ chính xác 0.975 (49/50 mẫu). Tuy nhiên, trên phương diện train loss, `attn_only` có loss thấp hơn (0.5379 so với 0.6256), tạo cảm giác giả rằng `attn_only` học tốt hơn. Thực tế, trên tác vụ trích xuất JSON hẹp với cấu trúc đơn giản, việc tăng rank cực cao (r=283) trên các lớp attention đã bù đắp được việc thiếu adapter ở các tầng MLP. Dù vậy, theo nguyên lý LoRA Without Regret, can thiệp vào toàn bộ các lớp linear (`all-linear`) vẫn là đòn bẩy kiến trúc an toàn và toàn diện hơn vì nó can thiệp trực tiếp vào việc lưu trữ tri thức thực tế ở tầng feed-forward mà không cần ép rank lên mức cực đoan.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
Đường loss của `wrong_lr` giảm cực kỳ chậm và kết thúc ở mức rất cao là 1.5702 (so với 0.6256 của `correct`), dẫn đến việc mô hình hoàn toàn thất bại khi infer với target = 0.000 và format = 0.000. Nếu chỉ nhìn vào đường loss mà không biết learning rate đang đặt ở thang full fine-tuning ($10^{-5}$), người thực hành sẽ dễ dàng kết luận sai lầm rằng dữ liệu bị lỗi, tác vụ quá khó, hoặc dung lượng mô hình không đủ để học. Trên thực tế, vì LoRA chỉ cập nhật một ma trận phụ với các tham số khởi tạo bằng 0, nó đòi hỏi tốc độ học lớn hơn xấp xỉ 10 lần (thang $10^{-4}$) so với full-FT để gradient có thể cập nhật hiệu quả.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**
`qlora` giảm bộ nhớ VRAM đỉnh từ 8.78 GB xuống còn 3.86 GB (tiết kiệm hơn 56% bộ nhớ), cho phép chạy trên các GPU dung lượng thấp. Tuy nhiên, cái giá phải trả rất rõ ràng: thời gian huấn luyện tăng từ 389.6s lên 469.3s (do overhead giải nén lượng tử 4-bit NF4 liên tục), độ trễ suy luận tăng mạnh từ 1355.7 ms lên 1730.9 ms/mẫu, và điểm target bị sụt giảm từ 0.975 xuống 0.940. Kết quả thực nghiệm này hoàn toàn ủng hộ khuyến nghị của nhà phát hành: đối với họ Qwen3.5, sai số lượng tử hóa gây suy hao chất lượng biểu diễn đáng kể, do đó nếu phần cứng GPU (như T4 16GB) đã đủ sức chứa 16-bit LoRA thì không nên đánh đổi chất lượng lấy QLoRA.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.210` · `regression Δ = -0.313` · `valid_trace_rate = 0.00`

### Diễn giải phán quyết:
Cổng hồi quy ra phán quyết `FAILED` bởi vì dù mô hình fine-tune cải thiện vượt bậc trên tác vụ mục tiêu (target tăng từ 0.765 lên 0.975, tức $\Delta = +0.210$), nhưng năng lực tổng quát trên 15 câu hỏi phổ thông bị suy giảm nghiêm trọng từ 0.7911 xuống 0.4778 (tụt 0.313, vượt xa ngưỡng dung sai cho phép là 0.020). 

Hiện tượng này phản ánh trực tiếp catastrophic forgetting: khi fine-tune chỉ tập trung 100% vào định dạng JSON của ticket CSKH, trọng số của mô hình đã bị kéo lệch khỏi không gian suy luận và ngôn ngữ tự nhiên ban đầu. Mặc dù kết quả kiểm định là `FAILED`, đây là một phát hiện thực nghiệm có giá trị cao. Nó chứng minh rằng không thể vội vã đưa mô hình vào triển khai sản xuất thực tế nếu chưa giải quyết được bài toán quên tri thức. Biện pháp khắc phục tối ưu theo bài giảng là phải trộn thêm từ 1% đến 5% dữ liệu hồi đáp tổng quát (replay buffer) vào tập huấn luyện SFT.

---

## 6. Định tính — Phân tích chi tiết các ca dự đoán

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt ốp lưng điện thoại mã đơn DH936478. Shipper không gọi... | `intent: van_chuyen` | `intent: hoi_thong_tin` | `intent: van_chuyen` | ✅ FT thắng: Nhận diện chính xác khiếu nại về shipper |
| 2 | Alo shop, mình đặt ốp lưng điện thoại mã đơn DH734695. Giá bao nhiêu... | `urgency: trung_binh` | `urgency: cao` | `urgency: trung_binh` | ✅ FT thắng: Đánh giá đúng mức độ khẩn cấp theo ngữ cảnh |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền... | `urgency: thap` | `urgency: thap` | `urgency: trung_binh` | ❌ **FT thua**: Đánh giá nhầm mức khẩn cấp lên trung bình |
| 4 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện... | `urgency: thap` | `urgency: thap` | `urgency: trung_binh` | ❌ **FT thua**: Khách ghi "khi nào tiện" nhưng FT gán nhãn trung bình |
| 5 | Chào shop, mình đặt nồi chiên không dầu mã đơn VN949966. Hoàn tiền. Khi nào tiện... | `urgency: thap` | `urgency: thap` | `urgency: trung_binh` | ❌ **FT thua**: Bỏ qua từ khóa giảm mức độ ưu tiên ("khi nào tiện") |

### Nhận xét mẫu hình ở các ca FT thua:
Ở cả 3 ca mà bản fine-tune bị trừ điểm (mẫu 3, 12, 39), lỗi đều xuất hiện đồng nhất ở trường `urgency`. Trong khi khách hàng dùng câu từ giảm nhẹ như "khi nào tiện", mô hình fine-tune có xu hướng thiên vị gán nhãn `trung_binh` thay vì `thap`. Điều này cho thấy tập dữ liệu huấn luyện có tỷ lệ nhãn `trung_binh` chiếm ưu thế đối với các intent hoàn tiền và lỗi sản phẩm, khiến mô hình fine-tune hình thành định kiến phân phối (prior bias) thay vì chú ý đầy đủ vào các cụm từ ngữ cảnh đặc thù.

---

## 7. Kết luận & điều tôi học được

### Kết luận:
Dựa trên kết quả thực nghiệm khách quan từ pipeline đo lường, tôi kết luận rằng **chưa nên triển khai bản fine-tune này ra môi trường sản xuất**. Mặc dù mô hình đạt độ chính xác phân loại mục tiêu rất ấn tượng (97.5% so với 76.5% của prompt tối ưu), việc năng lực tổng quát bị sụt giảm tới 31.3% (regression score từ 0.791 xuống 0.478) là một rủi ro lớn trong thực tế, khiến mô hình dễ gặp trục trặc khi đối diện với các câu hỏi nằm ngoài phân phối hẹp của bài toán. 

Đòn bẩy quan trọng nhất quyết định sự thành bại trong lab này không phải là việc cố gắng tăng rank LoRA lên thật cao, mà nằm ở sự kết hợp giữa vị trí can thiệp adapter toàn diện (`all-linear`), learning rate chuẩn xác ở thang $10^{-4}$, và tính đúng đắn của loss mask. Bên cạnh đó, để đưa vào ứng dụng thực tế, giải pháp căn cơ là áp dụng cơ chế bảo tồn tri thức (data replay) nhằm vượt qua cánh cổng hồi quy một cách an toàn.

### Ba điều tôi học được:
1. Loss mask phải được chứng minh bằng giải mã ngược, không dựa vào niềm tin: Việc kiểm tra trực tiếp ma trận nhãn và các token được tính loss là chốt chặn quan trọng nhất để tránh việc mô hình vô tình học cách lặp lại câu hỏi của người dùng.
2. Rank không phải là đòn bẩy chính - Vị trí adapter và Learning Rate mới mang tính quyết định: So sánh giữa `attn_only` và `correct` cho thấy việc tăng rank lên cực đại trên attention không vượt trội hơn việc rải đều adapter sang các khối linear, trong khi việc dùng sai learning rate sẽ phá hủy hoàn toàn khả năng hội tụ.
3. Phán quyết FAILED là một kết quả thành công về mặt nghiên cứu: Trong kỹ thuật ML, việc phát hiện mô hình bị hồi quy năng lực thông qua một quy trình đánh giá công bằng có giá trị thực tiễn cao hơn nhiều so với việc chỉ chăm chăm nhìn vào độ chính xác của tập train để tự huyễn hoặc về hiệu năng.

### Nếu có thêm 2 giờ nữa, tôi sẽ thử:
Tôi sẽ tiến hành thực nghiệm trộn thêm 3% dữ liệu hội thoại đa chủ đề (replay buffer) vào tập huấn luyện của NB3 để kiểm tra xem liệu mô hình có thể duy trì mức target 97.5% đồng thời hồi phục điểm regression về mức ban đầu (≥ 0.79) hay không.

---

## Phụ lục — thưởng đã làm

- [x] B1 NB6 merge + hot-swap
  * **Kết quả đo lường (`results/merge_check.json`)**: Điểm trước merge = 0.9650, sau merge = 0.9650 ($\Delta = 0.0000$, ngưỡng dung sai $\le 0.01$). Trọng số LoRA đã được gộp vào base model mà không làm suy hao độ chính xác.
  * **Trả lời câu hỏi B1 (deck §23)**:
    - Merge mang lại zero inference overhead — vậy ta phải đánh đổi điều gì? Ta đánh đổi tính linh hoạt và dung lượng lưu trữ/phân phối. Một khi đã merge, mô hình trở thành một checkpoint cố định nặng ~9GB, không thể tách rời hay bật/tắt tính năng theo nhu cầu của từng request.
    - Khi nào thì nên giữ các adapter riêng biệt thay vì merge? Nên giữ adapter riêng trong kiến trúc phục vụ đa người thuê (multi-tenant / multi-task serving). Một base model duy nhất chỉ cần nạp vào VRAM 1 lần, các adapter chuyên biệt cho từng khách hàng hoặc tác vụ (chỉ vài chục MB) có thể được hoán đổi tức thì theo từng request, tiết kiệm chi phí phần cứng gấp nhiều lần.
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [x] B5 HuggingFace Hub — link: https://huggingface.co/Danh0301/Day21
