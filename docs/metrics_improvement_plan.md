# L-RAG — Kế hoạch Cải thiện Metrics Evaluation

> **Mục tiêu:** Đưa Change Recall / F1 của pipeline LegalDiff lên mức chấp nhận được,
> giữ Zero-Hallucination. Đây là tài liệu **root** — dẫn đến các proposal chi tiết trong
> [`improvement_strategies/`](./improvement_strategies).
>
> **Baseline tham chiếu:** `eval_results/` chạy 2026-06-23, 23 cặp, 0 lỗi.

---

## 0. TL;DR cho Decision-maker

| Metric | Baseline (23/06) | Bottleneck chính | Đích 8 tuần |
|---|---:|---|---:|
| Change **Recall** | **11.71%** | Trần 30% do ACU thô + parser fail 2 cặp | **≥ 45%** |
| Change **Precision** | 39.20% | Hallucination cluster `bgddt` | **≥ 60%** |
| Change **F1** | 18.04% | Recall kéo xuống | **≥ 40%** |
| **Hallucination** | 3.83% | `20/21/22-bgddt` (11–50%) | **≤ 2%** |

**3 đòn bẩy lớn nhất (xem §4):**
1. **Fix Phase-1 parser** cho `33-hd.signed` / `29-cd.signed` (V1 = 0 articles) → +~9 pp recall ngay lập tức.
2. **Sửa granularity evaluation**: matcher hiện dựa trên substring/overlap >0.6 — không thể liên kết ACU thô với GT mức câu. Cần matcher ngữ nghĩa (embedding) + article-level scoring.
3. **Sinh ACU tinh hơn + target Category D** (logic pháp lý = 55% GT, recall 2.2%) — nâng chính trần 30%.

> ⚠️ **Insight quan trọng nhất — "trần 30%":** Pipeline sinh **176 ACU** nhưng GT có **589 change**
> (tỷ lệ 1:3.3). Ngay cả khi matcher *hoàn hảo 1:1*, recall không thể vượt `176/589 ≈ 30%`.
> Có nghĩa: **số 11.7% hiện tại bị kìm bởi cả hai thứ** — (a) matcher yếu + (b) pipeline sinh
> quá ít đơn vị thay đổi so với độ mịn của GT. Phải xử lý cả hai.

---

## 1. Bức tranh hiện tại (Baseline)

```
23 cặp · 0 error · 2279s (99.1s/cặp) · BGE-M3 + Qwen2.5-7B (ollama)

Change Recall    : 11.71%   (69 / 589 GT change được phát hiện)
Change Precision : 39.20%   (69 / 176 ACU khớp GT)
Change F1        : 18.04%
Hallucination    :  3.83%   (7 / 183 ACU bị reject)
ACU sinh / pass  : 183 / 176
```

### Phân rã theo category (xếp theo độ nặng)

| Cat | Mô tả | GT | Detected | Recall | Ghi chú |
|---|---|---:|---:|---:|---|
| **D** | Thay đổi logic pháp lý | 325 | 7 | **2.2%** | ⚠️ 55% toàn bộ GT — kìm recall |
| A | Thay đổi con số | 74 | 20 | 27.0% | |
| B | Diễn đạt lại ngữ nghĩa | 60 | 17 | 28.3% | |
| C | Thay đổi cấu trúc | 81 | 11 | 13.6% | |
| E | Xử lý bảng biểu | 19 | 2 | 10.5% | |
| F | Nhiễu định dạng | 30 | 12 | 40.0% | Largely noise (date/typo) |

### Các cặp "đỏ" (recall 0% hoặc hallucination cao)

| Cặp | V1 art | GT chg | Recall | Halluc% | Nguyên nhân sơ bộ |
|---|---:|---:|---:|---:|---|
| `33-hd.signed` | **0** | 241 | 0% | 0% | Phase-1 parser → 0 article |
| `29-cd.signed` | **0** | 20 | 0% | 0% | Phase-1 parser → 0 article |
| `20-bgddt` | 6 | 13 | 0% | **50%** | Table/structure misread |
| `21-bgddt` | 13 | 16 | 6% | **40%** | Table/structure misread |
| `22-bgddt` | 17 | 21 | 5% | 11% | Table/structure misread |
| `05/06-bvhttdl` | 1–2 | 11–14 | 0% | 0% | Quá ít article được align |

---

## 2. Chẩn đoán nguyên nhân gốc (ranked theo impact)

| # | Nguyên nhân | Bằng chứng | Tác động recall |
|---|---|---|---|
| R1 | **Trần cấu trúc 30%** — pipeline sinh 176 ACU vs 589 GT | `aggregate_metrics.json` | Capping |
| R2 | **2 cặp không ingest** (V1=0) = 44% GT = 0 contribution | 33-hd:241, 29-cd:20 | **−9 pp** |
| R3 | **Matcher substring/overlap** không liên kết ACU thô ↔ GT mịn | `run_full_eval.py:266` `match_gt_to_acus`; `03-tand`: 30 ACU, chỉ 3 match | Lớn nhất trên cặp "kha" |
| R4 | **ACU tiêu hao 1-đối-1** (`break` + `pipeline_matched.add`) | `run_full_eval.py:320` | Nhân hiệu ứng R1 |
| R5 | **Category D (55% GT) cực khó** — semantic inversion | 7/325 = 2.2% | −~23 pp so với nếu D≈30% |
| R6 | **GT chứa nhiều noise** (F+A = 104 change) pipeline đúng khi bỏ qua | F 40%, A 27% | Phạt recall "công bằng" |
| R7 | **Alignment gap** — Phase 3 chỉ chạy trên matched article | added 32, deleted 12 | Bỏ sót GT trong article chưa match |
| R8 | **LLM "báo không đổi"** → emit 0 ACU cho cả article | log "Tier 1 extracted 0/0" | Bỏ sót hàng loạt |

---

## 3. Trần kỹ thuật — giải thích "30% ceiling"

```
GT change (mịn, mức câu/ký tự):    589 đơn vị
        │
        ▼  [matcher 1:1, tối đa]
Pipeline ACU (thô, mức "điều đã thay đổi"): 176 đơn vị
        │
        ▼
recall_max = 176 / 589 = 29.9%   ← KHÔNG THỂ VƯỢT nếu giữ nguyên cả 2 đầu
```

Hai hướng phá trần (cần làm **cả hai**):
- **Tăng tử số** → pipeline sinh ACU mịn hơn (Strategy S3).
- **Thư giãn giả định 1:1** → một ACU được credit cho nhiều GT (Strategy S2, đổi cách chấm).

---

## 4. Lộ trình ưu tiên (Roadmap theo ROI)

| ID | Strategy | Recall lift | Effort | Risk | Detail |
|---|---|---:|---|---|---|
| **S1** | Fix parser V1=0 (33-hd, 29-cd) | **+9 pp** | Thấp | Thấp | [`S1_fix_parser_ingestion.md`](./improvement_strategies/S1_fix_parser_ingestion.md) |
| **S2** | Matcher ngữ nghĩa + article-level scoring | **+8–12 pp** (mở khóa trần) | Trung bình | Thấp | [`S2_evaluation_granularity.md`](./improvement_strategies/S2_evaluation_granularity.md) |
| **S3** | Tier-1 prompt sinh ACU mịn hơn (mỗi edit 1 ACU) | **Tăng chính trần** | Trung bình | TB | [`S3_finer_acu_extraction.md`](./improvement_strategies/S3_finer_acu_extraction.md) |
| **S4** | Target Category D (legal-logic) | **+5–8 pp** | Trung bình–Cao | TB | [`S4_category_d_legal_logic.md`](./improvement_strategies/S4_category_d_legal_logic.md) |
| **S5** | Giảm hallucination cluster `bgddt` (precision) | Precision +10–15 pp | Trung bình | Thấp | [`S5_reduce_hallucination.md`](./improvement_strategies/S5_reduce_hallucination.md) |

> Lưu ý: các lift là **ước lượng độc lập**, có trùng lặp (recall có ceiling tự nhiên ~ khi
> ACU đủ mịn). Tổng hợp thực tế kỳ vọng đưa Recall vào **45–55%**.

---

## 5. Trình tự triển khai (Sequencing)

### Phase A — Quick wins (Tuần 1–2) 🟢
- **S1**: sửa parser → +9 pp recall "free".
- **S5 (một phần)**: filter table-misread trong `bgddt` → precision ↑, halluc ↓.
- *Gate:* Recall ≥ 20%, Hallucination ≤ 3% sau khi re-eval.

### Phase B — Cấu trúc lại evaluation (Tuần 3–4) 🟡
- **S2**: matcher ngữ nghĩa (dùng chính BGE-M3 đã có) + article-level complementary metric.
- *Lý do làm trước S3:* S2 cho biết "pipeline thực sự tốt đến đâu" → tránh tối ưu prompt
  dựa trên matcher sai.
- *Gate:* recall "thật" (semantic) được ước lượng, so sánh với substring recall.

### Phase C — Tối ưu pipeline (Tuần 5–8) 🟠
- **S3**: prompt Tier-1 tinh hơn → phá trần.
- **S4**: chuyên biệt Category D (few-shot, negative-example, nhánh logic-shift).
- *Gate:* Recall ≥ 45%, F1 ≥ 40%, Hallucination ≤ 2%.

---

## 6. KPI & Definition of Done

| Gate | Metric | Mục tiêu |
|---|---|---|
| G1 — Parser | Số cặp V1=0 articles | **0** |
| G2 — Eval fairness | Matcher semantic vs substring (chênh ≤ 5 pp) | Đạt |
| G3 — Coverage | Category D recall | **≥ 20%** |
| G4 — Quality | Hallucination toàn cục | **≤ 2%** |
| G5 — Final | F1 toàn cục | **≥ 40%** |

**Definition of Done:** tái chạy `evaluation/run_full_eval.py` trên cùng 23 cặp → mọi gate G1–G5 đạt,
`num_pairs_with_errors == 0`, kết quả được commit vào `eval_results/` kèm ngày.

---

## 7. Liên kết & phụ thuộc

- Source-of-truth metrics: `eval_results/aggregate_metrics.json`, `eval_results/evaluation_report.md`
- Eval logic hiện tại: `evaluation/run_full_eval.py` (hàm `match_gt_to_acus` line ~266)
- ACU schema & prompt: `src/comparison/models.py` (`ACUOutput`), `src/comparison/acu_prompter.py`
- Parser: `src/ingestion/docling_parser.py`; config `configs/model_config.yaml` (`confidence_threshold=0.75`)
- GT format: `data_test/<pair>.json` (mỗi change: `category`, `original_text`, `modified_text`, `change_type`, `location`)

---

_Cập nhật lần cuối: 2026-06-23. Mọi ước lượng lift dựa trên baseline 23 cặp._
