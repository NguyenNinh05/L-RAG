# S5 — Giảm Hallucination (precision) — cluster `bgddt`

> **Đẩy Precision lên + F1 theo.** Hiện hallucination 3.83%, nhưng `bgddt` lên tới 11–50%.
> Thuộc [Metrics Improvement Plan](../metrics_improvement_plan.md).

---

## 1. Vấn đề

| Cặp | Hallucination | ACU pass |
|---|---:|---:|
| `20-bgddt` | **50.0%** | 1 |
| `21-bgddt` | **40.0%** | 3 |
| `22-bgddt` | 11.1% | 8 |
| `41-btc` | 12.5% | 7 |
| Toàn cục | 3.83% | 176/183 |

`bgddt` (Bộ Giáo dục) — văn bản nhiều **bảng biểu + cấu trúc phức tạp** → LLM bịa change
không có thật trong raw text. Verifier hiện không bắt hết.

## 2. Nguyên nhân gốc

- **Table misread**: docling flatten bảng → text rời rạc → LLM hiểu sai cấu trúc, bịa change.
- **Verifier chưa enforce đủ**: schema có `verbatim_found` (`models.py:210` — "True nếu
  verbatim_evidence_v1 tìm thấy trong raw_text_v1") nhưng ACU với evidence **gần đúng** vẫn pass.
- **Numerical/format** đổi trong bảng → LLM hallucinate con số.

## 3. Giải pháp đề xuất

### 3A. Enforce verbatim gate (cứng)

- ACU chỉ pass nếu `verbatim_evidence_v1` **thực sự substring** của `raw_text_v1` (và tương tự V2),
  sau normalize. Nếu không → **reject** (đếm vào hallucination, không vào report).
- Nới `fuzzy_threshold` (hiện 0.85) **lên 0.9** cho cặp giàu bảng.

### 3B. Cải table extraction → feed structured context

- Giữ `TableData` JSON (không flatten) từ Phase-1.
- Trong Tier-1 prompt, **inject table cells** dưới dạng markdown table làm context → LLM thấy đúng
  cấu trúc, giảm bịa.
- Nếu cell đổi → ACU tham chiếu `(table_id, row, col)` thay vì text mờ.

### 3C. Per-pair hallucination gate

- Nếu `hallucination_rate(pair) > 15%` → **flag pair** cần review, **không** góp vào aggregate tốt.
- Thêm cột "flagged" vào report để không ảo giác precision.

### 3D. Numerical regex verification (đã có, tăng cường)

- `strict_numerical=True` đã bật. Thêm: mọi con số trong `new_value` phải xuất hiện trong `raw_text_v2`
  (exact), nếu không → reject.

## 4. Các bước triển khai

```
1. Trong verifier.py: implement verbatim gate (substring sau normalize) — reject nếu fail.
2. Bump fuzzy_threshold 0.85 → 0.90 (config), đo precision/recall trade-off.
3. Patch acu_prompter: inject TableData markdown context cho article có bảng.
4. Thêm numerical exact-check trong verifier.
5. Thêm per-pair hallucination flag vào aggregate_metrics.json.
6. Re-eval → đặc biệt theo dõi bgddt + các cặp khác không regression.
```

## 5. Impact kỳ vọng

- Hallucination toàn cục: **3.83% → ≤ 2%**.
- `bgddt` cluster: 11–50% → **≤ 10%**.
- Precision toàn cục: **+10–15 pp** (39% → ~50–55%); F1 tăng tương ứng.
- Side-effect tốt cho S3: evidence verbatim → cả S2 matcher hoạt động chính xác hơn.

## 6. Rủi ro & trade-off

- **Over-reject** → giảm recall (đặc biệt nếu verbatim gate quá cứng với văn bản có OCR noise).
  Mitigate: normalize aggressive trước khi substring; giữ fuzzy gate thứ cấp ở 0.9.
- **Table context làm prompt dài** → tăng latency. Mitigate: chỉ inject khi article có TableData.
- Cần tách bạch: reject do halluc thật vs reject do evidence cắt ngắn (xem log `reason`).

## 7. Cách đo lường

| Chỉ số | Trước | Mục tiêu |
|---|---|---|
| Hallucination toàn cục | 3.83% | **≤ 2%** |
| `bgddt` hallucination | 11–50% | ≤ 10% |
| Precision toàn cục | 39.2% | ≥ 50% |
| Cặp bị flag (>15%) | 2 | 0 |
| Recall (không regression) | — | giữ / tăng |
| Gate G4 | — | đạt (halluc ≤ 2%) |
