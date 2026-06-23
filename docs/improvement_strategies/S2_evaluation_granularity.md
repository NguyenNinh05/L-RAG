# S2 — Sửa Granularity Evaluation (Matcher ngữ nghĩa + Article-level scoring)

> **Mở khóa "trần 30%".** Trả lời câu hỏi *"pipeline thực sự tốt đến đâu?"* trước khi tối ưu prompt.
> Thuộc [Metrics Improvement Plan](../metrics_improvement_plan.md).

---

## 1. Vấn đề

Hàm `match_gt_to_acus` (`evaluation/run_full_eval.py:266`) liên kết GT change ↔ ACU bằng:
- **exact substring** (`gt_orig in acu_evidence`) sau normalize, hoặc
- **word-overlap ratio > 0.6**.

Và **mỗi ACU chỉ được credit cho 1 GT** (`pipeline_matched.add(i); break` tại line ~320).

Hệ quả (bằng chứng `03-tand`): **30 ACU, 17 GT, chỉ 3 match** → có đủ ACU nhưng matcher không nối được.
Nguyên nhân: LLM `verbatim_evidence` thường **diễn đạt lại / cắt ngắn**, nên exact-substring hiếm khi khớp
chuỗi GT gốc → recall bị đánh giá thấp (under-count) mà không phản ánh chất lượng pipeline thật.

## 2. Nguyên nhân gốc

- GT ở granularity **câu/ký tự** ("quản lý"→"quãn lý").
- ACU ở granularity **"điều đã thay đổi"** (tóm tắt ngữ nghĩa + 1-2 câu evidence).
- Matcher **chuỗi-tuyến-tính 1:1** không thể bridge hai mức granularity này.

## 3. Giải pháp đề xuất

### 3A. Semantic matcher (nhiều-nhiều) — dùng chính BGE-M3 đã có

```
Mỗi GT change  →  vector v_gt  = embed(original_text + " → " + modified_text + " @ " + location)
Mỗi ACU        →  vector v_ac  = embed(change_summary + evidence_v1 + evidence_v2 + location)
Khớp nếu cosine(v_gt, v_ac) ≥ τ   (calibrate τ trên golden, khởi đầu τ = 0.55)
Một ACU được credit cho MỌI GT thỏa τ  →  phá giả định 1:1
```

- **De-dup pipeline-side**: tránh 1 ACU match 2 GT trùng lặp ngữ nghĩa (giữ precision công bằng).
- Dùng batch embedding đã tối ưu (`BGEM3Manager`, fp16) → chi phí thấp.

### 3B. Article-level complementary metric (recall thô, công bằng)

Định nghĩa **Article-level Recall** song song, không phụ thuộc matcher chuỗi:
- GT change được "cover" nếu **article chứa nó được matched** ở Phase 2 **và** Phase 3 sinh ≥1 ACU trong article đó có `change_type` tương thích category.
- Metric này cho thấy *"pipeline có chạm đúng vùng text không"* — tách biệt khỏi vấn đề evidence-string.

### 3C. Báo cáo song song

| Metric | Ý nghĩa |
|---|---|
| `recall_substring` (cũ) | khớp chuỗi nghiêm ngặt — benchmark cũ |
| `recall_semantic` (mới) | khớp ngữ nghĩa, nhiều-nhiều — phản ánh thật |
| `recall_article` | article-level coverage |

→ Theo dõi cả 3 để **không ảo giác** khi optimize S3/S4.

## 4. Các bước triển khai

```
1. Viết hàm semantic_match_gt_to_acus() bên cạnh hàm cũ (không xóa hàm cũ).
2. Dùng BGEM3Manager: embed GT changes + ACU evidence theo batch.
3. Calibrate τ: quét τ ∈ {0.4,0.45,...,0.7} trên golden, chọn τ cho recall_semantic ≈ recall_article
   (chỗ hai metric đồng thuận → τ "công bằng").
4. Thêm cột recall_semantic / recall_article vào aggregate_metrics.json + report.
5. Chạy lại 23 cặp → so sánh 3 recall.  Nếu recall_semantic >> recall_substring  →
   khẳng định "matcher cũ under-count" và S3/S4 đáng làm.
```

## 5. Impact kỳ vọng

- Không thay đổi pipeline, chỉ thay đổi **cách đo**.
- `recall_semantic` kỳ vọng **+8–12 pp** so với substring (phần lớn là under-count được khôi phục).
- Quan trọng hơn: cho **baseline trung thực** để đo S3/S4 — tránh tối ưu nhầm.

## 6. Rủi ro & trade-off

- **τ quá lỏng → over-credit** (recall ảo). Giảm thiểu: calibrate τ sao cho recall_semantic ≈ recall_article
  + spot-check 20 cặp GT–ACU bằng mắt.
- **Nhiều-nhiều có thể thổi phồng precision**. Giảm thiểu: de-dup pipeline-side + giữ precision ở mức ACU.
- Embedding thêm ~chi phí (nhỏ, vì cùng model BGE-M3 đã load).

## 7. Cách đo lường

| Chỉ số | Mục tiêu |
|---|---|
| `recall_semantic` − `recall_substring` | dương, ổn định qua các cặp |
| τ được calibrate trên golden | có giá trị cố định trong config |
| Spot-check 20 cặp GT↔ACU | ≥ 80% ghép đúng theo ý người |
| Gate G2 (chênh semantic vs article ≤ 5 pp) | đạt |
