# S1 — Fix Phase-1 Parser (V1 = 0 articles)

> **ROI cao nhất, rủi ro thấp nhất.** "Free win" ~+9 pp recall.
> Thuộc [Metrics Improvement Plan](../metrics_improvement_plan.md).

---

## 1. Vấn đề

Hai cặp ghi `V1=0 articles` trong Phase-1 → Phase 2/3 không có gì để so → recall 0%:

| Cặp | V1 articles | GT changes | Recall |
|---|---:|---:|---:|
| `33-hd.signed` | **0** | 241 | 0% |
| `29-cd.signed` | **0** | 20 | 0% |
| **Tổng** | — | **261** | **0%** |

→ **261 / 589 = 44% toàn bộ GT change bị "bỏ qua hoàn toàn"** chỉ vì parser không trích article.
Nếu sửa được, recall toàn cục nhảy từ **11.7% → ~21%** ngay lập tức (cùng matcher cũ).

## 2. Nguyên nhân gốc (cần confirm)

File `data_test/33-hd.signed_v1.docx` (49 KB) và `29-cd.signed_v1.docx` (39 KB) **không lớn** →
không phải vấn đề kích thước. Các khả năng trong `src/ingestion/docling_parser.py`:

- **(a) Article regex không match** — pattern nhận diện `Điều ...` không khớp định dạng heading
  của 2 văn bản này (có thể là "Hiến pháp"/"Nghị định" dùng style khác, hoặc heading là list-number).
- **(b) Confidence filter quá khắc nghiệt** — `min_confidence_to_include: 0.4` (config) có thể loại
  **toàn bộ** text block khi docling tự tin thấp → mất sạch.
- **(c) DOM builder không nhận heading** — docling trả text nhưng không gắn `label=section_header`,
  nên builder không tạo ArticleNode.

> Cần chạy `docling_parser` trực tiếp lên 2 file để log intermediate (raw text, confidence, heading labels)
> rồi xác định (a)/(b)/(c).

## 3. Giải pháp đề xuất

1. **Diagnose** — thêm debug dump cho 2 cặp: lưu raw docling output + confidence map + heading labels
   vào `tests/output/` để kiểm tra bằng mắt.
2. **Mở rộng nhận diện article** — nếu (a): bổ sung pattern phụ (Roman numeral, numbered list, hoặc
   fallback "dòng ngắn kết thúc bằng dấu chấm + có số/ký hiệu điều").
3. **Không drop thô bạo** — nếu (b): khi toàn văn confidence thấp, **không loại cả doc**; thay vào đó
   gom text còn lại vào **một "uncategorized article"** (giữ node_id) để Phase 2/3 vẫn xử lý được.
   Ưu tiên recall của parser hơn precision.
4. **Table-aware heading** — nếu doc có bảng biểu ở header, đảm bảo table extraction (`TableData`)
   không nuốt heading.
5. **Regression test** — thêm 2 cặp này vào test bộ, assert `len(articles) > 0` cho V1.

## 4. Các bước triển khai

```
1. Viết script diagnose:  python -m src.ingestion.docling_parser --debug <file>
   → dump raw_text / confidence / heading_labels
2. Xác định (a)/(b)/(c), chọn nhánh fix
3. Patch docling_parser.py (hoặc config min_confidence_to_include)
4. Thêm unit test cho 33-hd, 29-cd  →  assert V1 articles > 0
5. Re-run eval (run_full_eval.py) → so recall trước/sau
```

## 5. Impact kỳ vọng

- **Recall toàn cục: 11.7% → ~21%** (giả định 2 cặp đạt recall trung bình của phần còn lại).
- Đặc biệt giảm "category D miss" vì `33-hd` có nhiều D-changes.
- Gate: **G1 — số cặp V1=0 = 0**.

## 6. Rủi ro & trade-off

- Fallback "uncategorized article" có thể sinh **false-positive article** → ảnh hưởng Phase 2 alignment.
  → Giảm thiểu: chỉ dùng fallback khi `articles == 0`, log warning rõ.
- Nếu fix bằng cách nới `min_confidence_to_include` → có thể mang noise vào các cặp khác.
  → Re-eval **toàn bộ 23 cặp** để bắt regression.

## 7. Cách đo lường

| Chỉ số | Trước | Mục tiêu |
|---|---|---|
| Cặp V1=0 articles | 2 | **0** |
| `33-hd.signed` recall | 0% | > 0 |
| Recall toàn cục | 11.7% | ~21% |
| Cặp khác regression | — | không (re-eval 23 cặp) |
