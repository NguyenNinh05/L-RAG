# S3 — Tier-1 Prompt sinh ACU tinh hơn (phá trần 30%)

> **Tăng chính tử số của recall.** Hiện 176 ACU vs 589 GT → trần 30%. Cần ACU mịn hơn.
> Thuộc [Metrics Improvement Plan](../metrics_improvement_plan.md).

---

## 1. Vấn đề

Pipeline sinh **176 ACU** trên 23 cặp; GT có **589 change** (tỷ lệ 1 : 3.3).
Ngay cả matcher hoàn hảo 1:1 cũng cap recall ở `176/589 ≈ 30%`.

Tier-1 hiện gộp **nhiều micro-edit thành 1 ACU** (mức "điều đã thay đổi"), do đó:

```
1 article có 8 edit nhỏ  →  1 ACU (thô)  →  tối đa credit 1/8 GT  →  recall thấp
```

Kèm theo: bug validation `confidence` bị thiếu cho `change_type=reorder`
(xem warning trong log eval) → vài ACU bị reject oan.

## 2. Nguyên nhân gốc

- `src/comparison/acu_prompter.py` yêu cầu **1 ACU / semantic-change** thay vì **1 ACU / atomic-edit**.
- Không có hướng dẫn tách number-change / paraphrase / logic-shift thành các ACU riêng.
- Trường `location` thô ở mức "Điều", không đủ mịn để match GT `location` (khoản/điểm).

## 3. Giải pháp đề xuất

### 3A. Đổi contract prompt: "atomic ACU"

> **Quy tắc:** mỗi thay đổi **có thể kiểm chứng độc lập** = **1 ACU riêng**.
> Một điều có 3 con số đổi + 1 câu diễn đạt lại → **4 ACU**, không gộp.

### 3B. Two-pass extraction (giảm halluc, tăng precision evidence)

```
Pass 1 (liệt kê):  LLM output danh sách candidate edits  — [{location, type, snippet_v1, snippet_v2}]
Pass 2 (đóng gói): mỗi candidate → ACU với verbatim_evidence CHÍNH XÁC (copy từ raw text, không diễn đạt lại)
```

→ Evidence verbatim giúp cả verifier (S5) và matcher (S2) hoạt động tốt.

### 3C. Mịn hóa `location`

Bắt buộc location xuống **khoản/điểm** khi có thể (`Điều 3, Khoản 2, Điểm a`) để match GT location.

### 3D. Fix bug `confidence`

- Đảm bảo prompt + schema luôn yêu cầu `confidence` cho mọi `change_type` (kể cả `reorder`).
- Đã có validator clamp `[0,1]` (`models.py:156`) — chỉ thiếu ở output, fix ở prompt instruction.

### 3E. Giới hạn để tránh nổ chi phí

- **Cap ACU/pair** ở mức hợp lý (vd 40) để tránh cặp lớn (`29-ttlt29` đã 1061s).
- Ưu tiên severity cao → sinh ACU cho high/medium trước.

## 4. Các bước triển khai

```
1. Sửa acu_prompter.py: đổi instruction sang atomic-ACU + few-shot (1 điều → nhiều ACU).
2. Thêm Pass-1 list schema (structured output) trước Pass-2 ACU schema.
3. Yêu cầu location khoản/điểm; yêu cầu confidence cho mọi change_type.
4. Chạy trên 3 cặp đại diện (01-tand_signed, 03-tand, 29-ttlt29) — đo:
     - #ACU/pair tăng bao nhiêu
     - recall_semantic (S2) tăng bao nhiêu
     - hallucination_rate (verifier) có tăng không
5. Tune cap ACU/pair để cân recall vs latency.
6. Re-eval toàn 23 cặp.
```

## 5. Impact kỳ vọng

- #ACU/pair tăng ~2–3× → trần recall từ 30% đẩy lên **50–60%**.
- Recall_semantic kỳ vọng **+10–15 pp** (kết hợp S2).
- Side-effect tốt: evidence verbatim → hallucination **giảm** (S5 hưởng lợi).

## 6. Rủi ro & trade-off

- **Latency / cost tăng**: nhiều ACU hơn = nhiều LLM call. Cặp lớn có thể chậm (`29-ttlt29`).
  → Mitigate: cap ACU/pair, concurrency đã =4, batch Pass-1.
- **Hallucination có thể tăng** nếu prompt ép sinh quá nhiều. → Mitigate: verifier mạnh (S5),
  Pass-2 bắt buộc verbatim evidence.
- **Over-fragmentation**: tách quá mịn thành noise. → Mitigate: chỉ tách edit kiểm chứng độc lập được.

## 7. Cách đo lường

| Chỉ số | Trước | Mục tiêu |
|---|---|---|
| ACU / cặp (trung bình) | ~7.7 (176/23) | 18–25 |
| Trần recall lý thuyết | 30% | ≥ 55% |
| Recall_semantic (sau S2) | — | +10–15 pp |
| `confidence` missing warning | > 0 | **0** |
| Hallucination toàn cục | 3.83% | ≤ 3% (không tăng) |
