# S4 — Target Category D (Thay đổi logic pháp lý)

> **Category D = 55% toàn bộ GT nhưng recall chỉ 2.2%.** Đây là đòn bẩy lớn nhất theo category.
> Thuộc [Metrics Improvement Plan](../metrics_improvement_plan.md).

---

## 1. Vấn đề

| Cat | GT | Detected | Recall |
|---|---:|---:|---:|
| **D** (logic pháp lý) | **325** | **7** | **2.2%** |

D chiếm **55% pool GT** (325/589). Nếu đưa D lên recall 20% (tăng ~18 pp trên 325 change)
→ đóng góp **+~10 pp recall toàn cục** — lớn nhất trong mọi category.

Đặc trưng Category D: **inversion ngữ nghĩa tinh tế** — đổi điều kiện, thêm/bớt ngoại lệ, đảo nghĩa.

```
V1: "...không sử dụng cho mục đích khác."
V2: "...có thể sử dụng cho mục đích khác nếu được Chánh án TANDTC phê duyệt."
→ logic shift (cấm  ↔  cho phép có điều kiện)
```

## 2. Nguyên nhân gốc

- LLM **bỏ sót** logic-shift vì thay đổi tinh tế, không phải thêm/xóa đoạn rõ ràng.
- Prompt Tier-1 **không có signal** ưu tiên logic/modal/negative; không có few-shot cho D.
- Không có cơ chế phát hiện **thay đổi token modal/negative/conditional** (`không` ↔ `có thể`,
  `phải` ↔ `được`, thêm `nếu...`, `trừ khi...`).

## 3. Giải pháp đề xuất

### 3A. Pre-filter tín hiệu logic (deterministic, rẻ)

Quét diff token-level giữa V1/V2 clause, flag khi xuất hiện thay đổi nhóm:
- **Modal/negative**: `không` ↔ `có thể/được/phải`, `bắt buộc` ↔ `không bắt buộc`.
- **Conditional**: thêm/bớt `nếu`, `khi`, `trừ khi`, `ngoại trừ`, `trong trường hợp`.
- **Scope/exception**: thêm/bớt `chỉ`, `duy nhất`, `tất cả`, `một phần`.

→ Sinh **candidate D-list** để Tier-1 **bắt buộc** xét các clause này (không được skip).

### 3B. Few-shot chuyên biệt Category D

Thêm 3–5 ví dụ D vào prompt, kèm **negative example** (đổi chữ nhưng **không** đổi logic → không báo D)
để giảm false positive.

### 3C. NLI-light check (xác nhận logic thực sự đổi)

Với mỗi candidate D, chạy kiểm tra nhẹ (hoặc 1 LLM call riêng):
- **Premise** = clause V1, **Hypothesis** = clause V2.
- Nếu quan hệ **contradiction / neutral** (không entailment) → xác nhận logic-shift → giữ ACU.
- Nếu **entailment** (chỉ đổi chữ, nghĩa giữ) → **loại** (tránh halluc).

### 3D. Priority routing

Clause bị flag D → ưu tiên sinh ACU sớm, gán `severity=high` (phù hợp GT: D thường severity cao).

## 4. Các bước triển khai

```
1. Xây danh sách modal/negative/conditional tokens VN (file data/lexicon_logic.json).
2. Viết detector token-level trong src/comparison (pre-filter) → output candidate clause list.
3. Thêm few-shot D (+ negative) vào acu_prompter.py.
4. Thêm NLI-light check (LLM 1-call hoặc heuristic entailment) làm gate cho ACU type=logic_shift.
5. Đo trên các cặp giàu D: 33-hd (sau S1), 01-tand_signed, 29-ttlt29.
6. Re-eval → theo dõi recall category D + hallucination.
```

## 5. Impact kỳ vọng

- Category D recall: **2.2% → ≥ 20%**.
- Recall toàn cục: **+5–8 pp** (sau khi đã có S1+S2+S3 để ACU đủ mịn).
- Giá trị phụ: tín hiệu logic là **đáng tin nhất** cho người dùng pháp lý (high-severity).

## 6. Rủi ro & trade-off

- **False positive** (báo logic-shift khi chỉ đổi chữ) → halluc. Mitigate: NLI-light gate + negative few-shot.
- **Token-diff noise** với văn bản dài. Mitigate: chỉ xét clause đã matched ở Phase 2.
- **Chi phí** thêm 1 LLM call/candidate. Mitigate: candidate list thường nhỏ (vài chục/pair).

## 7. Cách đo lường

| Chỉ số | Trước | Mục tiêu |
|---|---|---|
| Category D recall | 2.2% | **≥ 20%** |
| Category D precision | thấp | ≥ 50% (không báo oan) |
| Hallucination (logic_shift ACU) | — | ≤ 5% |
| Recall toàn cục | — | +5–8 pp |
| Gate G3 | — | đạt (D ≥ 20%) |
