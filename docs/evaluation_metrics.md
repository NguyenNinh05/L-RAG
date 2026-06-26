# L-RAG — Evaluation Metrics Reference

> **Tài liệu này giải thích chi tiết mọi metric dùng để đánh giá pipeline LegalDiff / L-RAG** —
> định nghĩa, công thức, ngưỡng (threshold), nơi sinh ra trong code, và giới hạn (caveat) của từng metric.
>
> Mục tiêu: một nguồn tham chiếu duy nhất để hiểu *"chúng ta đang đo gì, đo ở đâu, và tại sao"*.
>
- **Source-of-truth code:** [`evaluation/run_full_eval.py`](../evaluation/run_full_eval.py), [`evaluation/metrics.py`](../evaluation/metrics.py)
- **Kết quả mới nhất:** [`evaluation/outputs/eval_results_v3/`](../evaluation/outputs/eval_results_v3/) (23 cặp, 2026-06-24)
- **Bản đồ metric → code:** xem [§8 — Quick reference table](#8-quick-reference-table)

---

## 0. TL;DR — Pipeline đang bị đánh giá như thế nào?

LegalDiff là pipeline **so sánh văn bản pháp lý** (V1 ↔ V2) gồm 3 phase. Đánh giá đo lường mỗi phase
theo góc nhìn khác nhau, rồi cộng dồn thành **một chỉ số trung tâm: khả năng phát hiện đúng sự thay đổi**

```
                         ┌─────────────────────────────────────────────────────┐
  V1.docx, V2.docx ───►  │  Phase 1: Ingestion   (parse + chunk + store)        │
                         │  Phase 2: Alignment   (ghép cặp Điều V1 ↔ Điều V2)   │
                         │  Phase 3: Comparison  (LLM sinh ACU + verify)        │
                         └─────────────────────────────────────────────────────┘
                                              │
                                              ▼
            ┌───────────────────────────────────────────────────────────┐
            │  So sánh với Ground Truth (GT): data_test/<pair>.json     │
            │  → Recall / Precision / F1  +  Hallucination Rate         │
            └───────────────────────────────────────────────────────────┘
```

**5 nhóm metric chính** (chi tiết ở các mục sau):

| Nhóm | Metric chính | Phase | § |
|---|---|---|---|
| **Change Detection** | Recall / Precision / F1 (semantic) | 3 | [§4](#4-change-detection--recall--precision--f1-primary) |
| **Alignment** | matched / added / deleted / split / merged | 2 | [§3](#3-phase-2--alignment-metrics) |
| **Ingestion** | số Điều/Khoản/table, avg_confidence, chunk stats | 1 | [§2](#2-phase-1--ingestion-metrics) |
| **Hallucination** | ACU rejected / generated | 3 | [§5](#5-hallucination--acu-quality-metrics) |
| **Per-category** | Recall theo nhóm A–F | 3 | [§6](#6-per-category-breakdown-a--f) |

---

## 1. Khái niệm nền tảng

Trước khi đi vào công thức, cần nắm 3 khái niệm — vì mọi metric đều xoay quanh chúng.

### 1.1 Ground Truth (GT) — "đáp án"

Nằm ở `data_test/<pair>.json`. Mỗi file là một **list các change**, mỗi change có:

```json
{
  "change_id": 1,
  "category": "F",                       // nhóm A–F (xem §6)
  "technique": "Thay đổi định dạng ngày tháng",
  "location": "Phần tiêu đề (Header)",   // "Điều X, Khoản Y" — dùng cho article-coverage
  "original_text": "Hà Nội, ngày 13 tháng 01 năm 2026",
  "modified_text": "Hà Nội, ngày 13/01/2026",
  "change_type": "formatting_noise",     // addition / deletion / paraphrasing / ...
  "severity": "low"
}
```

> GT là **độ mịn mức câu/ký tự** — một "change" có thể chỉ là đổi dấu phẩy. Tổng 23 cặp hiện có **589 GT changes**.

### 1.2 ACU (Atomic Change Unit) — "đáp án của pipeline"

Output của Phase 3. Một ACU = một đơn vị thay đổi do LLM sinh ra, có cấu trúc
(định nghĩa tại [`src/comparison/models.py:63`](../src/comparison/models.py)):

| Trường | Ý nghĩa |
|---|---|
| `change_type` | `numerical` / `terminology` / `structural` / `addition` / `deletion` / `reorder` |
| `original_value` | Giá trị ở V1 |
| `new_value` | Giá trị ở V2 |
| `verbatim_evidence_v1` | **Trích dẫn nguyên văn** từ V1 (chống ảo giác) |
| `verbatim_evidence_v2` | Trích dẫn nguyên văn từ V2 |
| `location_v1` / `location_v2` | "Điều X, Khoản Y" |
| `confidence` | `[0.0, 1.0]` — tự tin của LLM; `< 0.5` → cần review thủ công |
| `reasoning` | Chain-of-thought ngắn (dùng cho self-verification với DeepSeek) |

> ACU là **độ thô mức "điều đã thay đổi"** — một ACU thường gộp nhiều micro-edit của GT.
> Đây chính là nguồn gốc "trần 30%" (xem [§4.4](#44--trần-kỹ-thuật--tại-sao-recall-có-ceiling)).

### 1.3 Verification — ACU "đạt" hay "bị drop"

Trước khi vào report, mỗi ACU đi qua **2 tầng xác minh zero-hallucination**
([`src/comparison/verifier.py`](../src/comparison/verifier.py)):

| Tầng | Kiểm tra gì | Phương pháp | Ngưỡng |
|---|---|---|---|
| **Tier 2 — Evidence** | `verbatim_evidence_v1/v2` có **thực sự nằm** trong raw_text gốc? | exact `in` → whitespace-normalized → fuzzy `SequenceMatcher` | fuzzy ≥ **0.85** |
| **Tier 3 — Numerical** | Mọi con số trong `original_value`/`new_value` có khớp raw_text? | regex trích xuất số (date/%/currency/int/float) → đối chiếu | **100%** match (strict mode) |

Kết quả: `status ∈ {PASSED, FAILED_EVIDENCE, FAILED_NUMERICAL}`.
Chỉ ACU `PASSED` mới vào report. ACU bị drop = **hallucination** (xem [§5](#5-hallucination--acu-quality-metrics)).

---

## 2. Phase 1 — Ingestion Metrics

**File:** [`evaluation/metrics.py`](../evaluation/metrics.py) (Module 1/2/3)

Phase 1 không so với GT. Nó đo **chất lượng parse & cấu trúc DOM** — trả lời câu hỏi:
*"Parser có hiểu đúng cấu trúc văn bản không?"*

### 2.1 Module 1 — Parser quality (`run_module1`)

| Metric | Ý nghĩa | Đo bằng |
|---|---|---|
| `articles`, `clauses`, `tables`, `sections` | Số lượng node parse được | đếm `doc.iter_all_articles()` |
| `engine_used` | Engine OCR/layout (docling) | `doc.quality_metrics.engine_used` |
| `avg_confidence` | Độ tự tin trung bình của layout detection | `doc.quality_metrics.avg_confidence` |
| `ocr_triggered` | Có phải gọi OCR fallback không? | bool |
| `parse_time` | Thời gian parse (giây) | `time.perf_counter()` |
| `doc_title`, `doc_number`, `signing_date` | Metadata có trích được không? | không rỗng = pass |

> ⚠️ **Metric quan trọng nhất: `articles`.** Nếu `articles = 0` → toàn bộ pipeline phía sau chết
> (ví dụ `33-hd.signed`, `29-cd.signed` cho V1 = 0 article → recall 0%). Đây là **gate G1**
> trong [improvement plan](./metrics_improvement_plan.md).

### 2.2 Module 2 — Chunk quality (`run_module2`)

| Metric | Ý nghĩa |
|---|---|
| `total_chunks` | Tổng số LSU chunk sinh ra |
| `article_level` vs `clause_level` | Chunk ở mức Điều hay Khoản |
| `with_tables` | Số chunk chứa table |
| `avg_chars`, `min_chars`, `max_chars` | Phân bố kích thước chunk |
| `content_types` | Phân loại nội dung (text/table/...) |

**Validation check (4 rule):** mỗi chunk phải có
(a) `breadcrumb`, (b) `source_node_id`, (c) `content_with_prefix`,
và (d) `content_with_prefix` **bắt đầu bằng** `breadcrumb`.

### 2.3 Module 3 — Knowledge Store linkage (`run_module3`)

Đo **sự gắn kết giữa Graph DB (Kuzu) và Vector DB (Qdrant/Chroma)**:

| Metric | Ý nghĩa |
|---|---|
| `nodes` | Số node ghi vào Kuzu graph |
| `ref_edges` | Số cạnh `REFERENCES` (trích dẫn chéo giữa các Điều) |
| `vectors` | Số vector lưu vào ChromaDB |
| **node_id linkage** | Metadata của vector có chứa `node_id` để join ngược về graph không? (cầu nối RAG) |

---

## 3. Phase 2 — Alignment Metrics

**File:** [`evaluation/run_full_eval.py`](../evaluation/run_full_eval.py) (hàm `evaluate_pair`) +
[`src/alignment/diff_catalog.py`](../src/alignment/diff_catalog.py) (enum `MatchType`)

Phase 2 ghép cặp **Điều X (V1) ↔ Điều Y (V2)**. Kết quả phân thành 5 loại:

| Match Type | Ý nghĩa | Ví dụ |
|---|---|---|
| `matched` | Cặp 1-1 rõ ràng, similarity ≥ threshold | Điều 4 V1 ↔ Điều 4 V2 |
| `added` | Node **chỉ có ở V2** (V2 thêm mới) | Điều mới không có ở V1 |
| `deleted` | Node **chỉ có ở V1** (V2 xoá) | Điều bị huỷ |
| `split` | 1 Điều V1 → 2+ Điều V2 | Điều 5 tách thành Điều 5, 5a |
| `merged` | 2+ Điều V1 → 1 Điều V2 | Điều 7, 8 gộp thành Điều 7 |

### 3.1 Similarity score — công thức ghép cặp

Mỗi cặp (i, j) nhận một điểm tổng hợp ([`src/alignment/similarity_matrix.py:65`](../src/alignment/similarity_matrix.py)):

```
S[i][j] = w_semantic  · Cosine(BGE-M3_dense_i, BGE-M3_dense_j)     # ngữ nghĩa
        + w_jaro      · JaroWinkler(title_i, title_j)              # khớp tiêu đề
        + w_ordinal   · OrdinalProximity(article_number_i, j)      # gần nhau về thứ tự
        + w_sparse    · SparseOverlap(lexical, BGE-M3_sparse)      # khớp từ (mặc định tắt)
```

| Trọng số | Default | Ghi chú |
|---|---|---|
| `w_semantic` | **0.60** | BGE-M3 dense (1024-dim) — thành phần chính |
| `w_jaro_winkler` | **0.30** | Jaro-Winkler trên tiêu đề Điều |
| `w_ordinal` | **0.10** | Gần nhau về số thứ tự → ưu tiên ghép Điều 4↔4 hơn Điều 4↔5 |
| `w_sparse` | 0.00 | Tắt mặc định; bật khi `NodeRecord` có sparse vector |
| **Ràng buộc** | — | `w_semantic + w_jaro + w_ordinal + w_sparse = 1.0` (validate ở `__post_init__`) |

### 3.2 Ngưỡng phân loại

| Tham số | Default | Tác dụng |
|---|---|---|
| `match_threshold` | **0.65** | `S ≥ 0.65` → `matched` |
| `split_merge_threshold` | **0.80** | `S ≥ 0.80` nhưng cấu trúc n-1/1-n → `split`/`merged` |

Ghép cặp tối ưu dùng **Hungarian algorithm** (tối đa tổng similarity toàn cục). Kết quả 23 cặp mới nhất:
`matched=136, added=29, deleted=21, split=0, merged=1`.

> 📌 **Lưu ý đánh giá:** các số `matched/added/deleted/split/merged` là **thống kê mô tả**, không phải
> recall/precision. Chúng cho biết alignment hoạt động ra sao, không trực tiếp so với GT.

---

## 4. Change Detection — Recall / Precision / F1 (PRIMARY)

**File:** [`evaluation/run_full_eval.py`](../evaluation/run_full_eval.py)

Đây là **nhóm metric trung tâm** — đo xem pipeline có phát hiện đúng các thay đổi trong GT không.
Được tính theo **3 matcher khác nhau**, từ khắt khe nhất đến "thật" nhất.

### 4.1 Các thành phần của phân tử / mẫu số

```
TP (true positive)   = số GT change được pipeline "chạm tới" (detected)
FN (false negative)  = GT change nhưng pipeline bỏ sót
FP (false positive)  = ACU pipeline sinh ra nhưng KHÔNG khớp GT nào

recall    = TP / (TP + FN) = detected / GT_total
precision = TP / (TP + FP) = matched / ACU_total
F1        = 2 · P · R / (P + R)
```

### 4.2 Matcher #1 — Substring / overlap (STRICT, legacy)

**Hàm:** `match_gt_to_acus` + inline loop ([`run_full_eval.py:279`](../evaluation/run_full_eval.py))

Cách khớp 1 GT với 1 ACU:
- **Chuẩn hoá** text: lowercase, collapse whitespace, bỏ dấu câu.
- **Addition / deletion:** kiểm `modified_text`/`original_text` ∈ ACU field (exact substring).
- **Change thường:** `gt_orig` khớp `acu_orig` **HOẶC** `gt_mod` khớp `acu_new` (substring hoặc
  word-overlap > 0.5–0.6).
- **Greedy 1:1:** mỗi GT và mỗi ACU chỉ khớp 1 lần duy nhất (dùng `break` + `matched_acu_indices.add`).

> ⚠️ Đây là **giới hạn dưới** — vì ACU thô khó khớp chính xác chuỗi với GT mịn, matcher này **under-count**.
> Kết quả 23 cặp: `recall = 28.18%`, `precision = 31.38%`, `F1 = 29.70%`.
> Không dùng làm headline; giữ để so sánh với matcher ngữ nghĩa ([gate G2](./metrics_improvement_plan.md)).

### 4.3 Matcher #2 — Semantic BGE-M3 (PRIMARY ⭐)

**Hàm:** `semantic_match_gt_to_acus` ([`run_full_eval.py:389`](../evaluation/run_full_eval.py))

Cách khớp:
1. Mỗi GT change và mỗi ACU được biểu diễn thành text (gộp `original + modified + location`).
2. Embed bằng **BGE-M3** (chính model đã dùng ở Phase 2) → 1024-dim, L2-normalized.
3. Tính **cosine similarity** = tích vô hướng (dot product) ma trận `(G × A)`.
4. Một GT được tính là **detected** nếu **≥ 1 ACU** có cosine ≥ **ngưỡng 0.60**.

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `SEMANTIC_MATCH_THRESHOLD` | **0.60** | middle-ground: τ=0.55 over-credit (74%), τ=0.65 under-credit (33%) |

**Điểm khác cốt lõi so với matcher #1:** đây là **many-to-many** — một ACU có thể thỏa nhiều GT
(vì ACU thô gộp nhiều micro-edit). Điều này phản ánh đúng thực tế và **phá trần 1:1** của matcher cũ.

> **Kết quả 23 cặp (PRIMARY):**
> - `recall_semantic = 77.93%` (459 / 589 GT detected)
> - `precision_semantic = 38.94%` (206 / 529 ACU matched)
> - `F1_semantic = 51.93%`

### 4.4 ▶ Trần kỹ thuật — tại sao recall có ceiling?

```
GT change (mịn, mức câu):       589 đơn vị
        │  ▼ [matcher 1:1 tối đa]
Pipeline ACU (thô, mức "điều"): ~529 đơn vị (sau verify)
        │
recall_max ≈ ACU / GT  ← KHÔNG THỂ VƯỢT nếu giữ nguyên cả 2 đầu
```

→ Để nâng recall phải làm **cả hai**: (a) sinh ACU mịn hơn [Strategy S3],
(b) thư giãn giả định 1:1 [Strategy S2 — chính matcher #2 này]. Xem [§9](#9-liên-kết).

### 4.5 Matcher #3 — Article-level coverage (bổ sung)

**Hàm:** `article_coverage` ([`run_full_eval.py:438`](../evaluation/run_full_eval.py))

Không phụ thuộc matcher chuỗi hay ngữ nghĩa. Chỉ hỏi:
> *"Pipeline có **chạm đúng Điều** chứa GT change không?"*

- Trích `Điều X` từ `location` của GT → tập GT-articles.
- Trích `Điều X` từ tất cả ACU + tất cả matched pair → tập `reached`.
- `recall_article = gt_article_covered / gt_with_article`

Kết quả 23 cặp: `recall_article = 30.0%` (72 / 240 GT có article number).
> Lưu ý: thấp hơn semantic vì nhiều GT có `location` mơ hồ hoặc pipeline chưa align tới Điều đó.

---

## 5. Hallucination — ACU Quality Metrics

**File:** [`src/comparison/verifier.py`](../src/comparison/verifier.py) +
`hallucination_rate` trong [`run_full_eval.py`](../evaluation/run_full_eval.py)

**Hallucination** = ACU bị LLM **bịa ra** (evidence không nằm trong text gốc, hoặc số liệu sai).

### 5.1 Công thức

```
hallucination_rate = ACU_rejected / ACU_generated
                   = (FAILED_EVIDENCE + FAILED_NUMERICAL) / (passed + rejected)
```

| Trạng thái | Ý nghĩa |
|---|---|
| `PASSED` | Vượt qua cả Tier 2 (evidence) và Tier 3 (numerical) → vào report |
| `FAILED_EVIDENCE` | Tier 2 fail: `verbatim_evidence` không tìm thấy trong raw_text |
| `FAILED_NUMERICAL` | Tier 3 fail: con số không khớp raw_text |

### 5.2 Cờ cảnh báo (flagging)

- **`halluc_flagged_pairs`:** mọi cặp có `hallucination_rate > 15%` → cần review
  (ví dụ `02-tand.signed`: 17.65%).
- **Gate G4:** hallucination toàn cục **≤ 2%**.

> **Kết quả 23 cặp:** `hallucination = 3.29%` (18 / 547 ACU rejected).
> Headline (loại outlier `33-hd.signed`): **2.38%**.

### 5.3 ACU confidence — chỉ số phụ trợ

Trường `confidence ∈ [0, 1]` (LLM tự đánh giá, bị clamp về `[0,1]`):

| Khoảng | Ý nghĩa |
|---|---|
| 0.85 – 1.0 | Chắc chắn cao (dành cho logic-shift) |
| 0.7 – 0.9 | Khá chắc chắn |
| **< 0.5** | Không chắc → **gợi ý review thủ công** |
| default 0.5 | Khi LLM quên điền (tránh validation reject) |

> `confidence` **không phải metric đánh giá** — nó là tín hiệu nội bộ để route review.
> Metric chất lượng chính vẫn là `hallucination_rate` (deterministic, regex/sequence-based).

---

## 6. Per-Category Breakdown (A–F)

**File:** `CATEGORY_MAP` ([`run_full_eval.py:69`](../evaluation/run_full_eval.py))

Mỗi GT change thuộc 1 trong 6 nhóm. Recall được tính **riêng từng nhóm** → biết pipeline mạnh/yếu ở loại thay đổi nào.

| Cat | Mô tả | GT count (23 cặp) | Recall (semantic) |
|---|---|---:|---:|
| **A** | Thay đổi con số (numerical) | 74 | **56.8%** |
| **B** | Diễn đạt lại ngữ nghĩa (paraphrasing) | 60 | **81.7%** |
| **C** | Thay đổi cấu trúc (structural) | 81 | **85.2%** |
| **D** | Thay đổi logic pháp lý (legal logic) | 325 | **84.9%** |
| **E** | Xử lý bảng biểu (tables) | 19 | **68.4%** |
| **F** | Nhiễu định dạng (formatting noise) | 30 | **33.3%** |

> 📌 **Insight:** Category **D chiếm 55% toàn bộ GT** (325/589) — đây là nhóm nặng nhất và đáng quan tâm nhất.
> (Lưu ý: số semantic recall này của bản v3; bản baseline `metrics_improvement_plan.md` có D chỉ 2.2% —
> matcher ngữ nghĩa đã nâng D từ 2.2% → 84.9%.)

---

## 7. Aggregation & Reporting

**File:** `compute_aggregate_metrics` + `_write_markdown_report` ([`run_full_eval.py`](../evaluation/run_full_eval.py))

### 7.1 Các mức tổng hợp

Pipeline tính metrics ở **3 mức**:

1. **Per-pair** (`per_pair_results.json`) — mỗi cặp (V1,V2) có riêng recall/precision/F1/halluc.
2. **Aggregate (ALL)** (`aggregate_metrics.json`) — cộng dồn TP/FN/FP trên toàn bộ cặp rồi chia.
3. **Headline aggregate** — loại outlier (`--exclude 33-hd.signed`) để số headline không bị méo.

### 7.2 Headline aggregate — tại sao loại outlier?

Một số cặp có đặc thù (VD `33-hd.signed` parser V1=0 → recall 0% nhưng có 241 GT) kéo méo trung bình.
`--exclude` vẫn **đánh giá & lưu** cặp đó, chỉ tính thêm một aggregate thứ hai bỏ cặp đó ra.

| Metric (semantic) | ALL (23 cặp) | Headline (22, excl. outlier) |
|---|---:|---:|
| **F1** | 51.93% | **46.72%** |
| Recall | 77.93% | 71.26% |
| Precision | 38.94% | 34.76% |
| Hallucination | 3.29% | **2.38%** |

> Đảo ngược kỳ vọng: headline thấp hơn ALL vì outlier `33-hd.signed` có recall_semantic rất cao (87.5%)
> dù recall strict = 0 — bỏ nó ra làm headline trung thực hơn về "cặp điển hình".

### 7.3 Output files

Mỗi lần chạy `python evaluation/run_full_eval.py` sinh ra trong `--output-dir`:

| File | Nội dung |
|---|---|
| `aggregate_metrics.json` | Toàn bộ số tổng hợp + per-pair + headline |
| `per_pair_results.json` | Chi tiết từng cặp (kèm list ACU đã verify) |
| `evaluation_report.md` | Báo cáo Markdown tự sinh |

---

## 8. Quick reference table

| Metric | Biểu thức / nguồn | Ngưỡng | File:line |
|---|---|---|---|
| Change Recall (strict) | `detected / GT_total` | — | `run_full_eval.py:818` |
| Change Precision (strict) | `matched / ACU_total` | — | `run_full_eval.py:819` |
| Change F1 (strict) | `2PR/(P+R)` | — | `run_full_eval.py:820` |
| **Change Recall (semantic)** ⭐ | cosine ≥ τ | **τ = 0.60** | `run_full_eval.py:350, 825` |
| **Change F1 (semantic)** ⭐ | `2PR/(P+R)` | — | `run_full_eval.py:829` |
| Article coverage | `covered / with_article` | — | `run_full_eval.py:835` |
| Hallucination rate | `rejected / generated` | flag > **15%** | `run_full_eval.py:839, 877` |
| Alignment similarity `S` | tổng có trọng số | match **0.65** / split-merge **0.80** | `similarity_matrix.py:65,82` |
| Evidence verify | exact → ws-norm → fuzzy | fuzzy ≥ **0.85** | `verifier.py:58, 396` |
| Numerical verify | regex trích số → match | **100%** strict | `verifier.py:85, 480` |
| ACU confidence | LLM `[0,1]` | review < **0.5** | `models.py:137` |

---

## 9. Liên kết & phụ thuộc

- **Source-of-truth số liệu:** [`evaluation/outputs/eval_results_v3/aggregate_metrics.json`](../evaluation/outputs/eval_results_v3/aggregate_metrics.json)
- **Logic đánh giá chính:** [`evaluation/run_full_eval.py`](../evaluation/run_full_eval.py)
- **Đánh giá Phase 1 (cấu trúc):** [`evaluation/metrics.py`](../evaluation/metrics.py)
- **GT schema:** `data_test/<pair>.json` (mỗi change: `category`, `location`, `original_text`, `modified_text`, `change_type`)
- **ACU schema & verification:** [`src/comparison/models.py`](../src/comparison/models.py), [`src/comparison/verifier.py`](../src/comparison/verifier.py)
- **Alignment similarity:** [`src/alignment/similarity_matrix.py`](../src/alignment/similarity_matrix.py)
- **Kế hoạch cải thiện metrics (context):** [`docs/metrics_improvement_plan.md`](./metrics_improvement_plan.md)

> ⚠️ **Chưa triển khai (stub):** [`evaluation/golden_dataset_gen.py`](../evaluation/golden_dataset_gen.py)
> (sinh synthetic V2 từ V1) và [`evaluation/ragas_eval.py`](../evaluation/ragas_eval.py) (RAGAS faithfulness)
> hiện là placeholder — chưa được implement. Hiện pipeline dùng GT do người tạo (`data_test/`) thay vì
> golden tự sinh.

---

_Cập nhật lần cuối: 2026-06-26. Số liệu tham chiếu: bản eval_results_v3 (23 cặp, 2026-06-24)._
