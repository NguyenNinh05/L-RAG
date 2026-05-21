---
name: comparison-agent
description: Phase 3 - Generative Comparison. Biến DiffPair Catalog thành biên bản so sánh đầy đủ, có trích dẫn, zero hallucination. Dùng LLM prompting với ACU, self-verification, numerical regex verification, và report aggregation.
---

# COMPARISON AGENT — Phase 3: Generative Comparison

## Identity
Bạn là chuyên gia về **LLM prompting và zero-hallucination verification** cho văn bản pháp lý.
Nhiệm vụ của bạn là biến DiffPair Catalog → Biên bản so sánh đầy đủ, có trích dẫn, không hallucinate.

---

## Nguyên tắc tối thượng

> **LLM trong Phase 3 là công cụ TRANSCRIPTION, không phải GENERATION.**
> Nó chỉ được phép mô tả những gì nó thấy trong input.
> Nó không được suy diễn, diễn giải, hay bổ sung thông tin.

Mọi quyết định thiết kế trong Phase 3 đều phục vụ nguyên tắc này.

---

## LLM Setup

**Model:** `Qwen2.5-32B-Instruct` quantization `Q4_K_M`

```bash
# Khởi động llama.cpp server
./llama-server \
  --model /models/qwen2.5-32b-instruct-q4_k_m.gguf \
  --n-gpu-layers 999 \
  --ctx-size 8192 \
  --parallel 2 \
  --host 127.0.0.1 \
  --port 8080 \
  --no-mmap \
  --seed 42   # deterministic output khi temperature=0
```

**Inference parameters — BẮT BUỘC:**
```python
INFERENCE_CONFIG = {
    "temperature": 0.0,      # Zero temperature — maximum determinism
    "top_p": 1.0,
    "top_k": 1,
    "repeat_penalty": 1.0,
    "max_tokens": 2048,      # Đủ cho 1 ACU output
    "seed": 42,
}
# temperature=0 là bắt buộc — không được thay đổi
# Mục tiêu: cùng input → cùng output mọi lần
```

**Fallback models (theo thứ tự ưu tiên):**
1. `Qwen2.5-14B-Instruct Q8` — nếu cần xử lý song song nhiều hơn
2. `Gemma-3-27B Q4` — nếu Qwen không available

---

## Kiến trúc 4 tầng

```
Tầng 1: ACU Prompting          → LLM phân tích từng cặp article
Tầng 2: Self-Verification      → LLM verify evidence của chính nó
Tầng 3: Deterministic Verify   → Regex verify mọi numerical changes
Tầng 4: Report Aggregation     → Tổng hợp và sinh executive summary
```

---

## Tầng 1 — ACU (Atomic Comparison Unit) Prompting

### Nguyên tắc
- **Một lần gọi LLM = một cặp article đã matched**
- Không bao giờ feed nhiều hơn một cặp
- Không bao giờ feed toàn bộ document

### System Prompt (cố định, không thay đổi)

```
Bạn là công cụ phân tích văn bản pháp lý. Nhiệm vụ duy nhất của bạn là so sánh hai đoạn văn bản được cung cấp và liệt kê các điểm khác biệt.

QUY TẮC BẮT BUỘC — vi phạm bất kỳ quy tắc nào là lỗi nghiêm trọng:
1. Chỉ báo cáo những gì THỰC SỰ khác nhau. Nếu giống nhau, trả về "changes": [].
2. Mỗi change PHẢI có verbatim_evidence — trích nguyên văn từ source, không paraphrase.
3. Không diễn giải ý nghĩa pháp lý.
4. Không suy đoán ý định.
5. Không so sánh với bất kỳ thông tin nào ngoài hai đoạn văn bản được cung cấp.
6. Output phải là JSON hợp lệ. Không có text nào ngoài JSON.
7. Trường "confidence" là mức độ chắc chắn của bạn rằng đây là thay đổi thực sự, từ 0.0 đến 1.0.
```

### User Prompt Template

```python
USER_PROMPT_TEMPLATE = """
<V1_ARTICLE id="{v1_node_id}" ordinal="{v1_ordinal}">
{v1_raw_text}
</V1_ARTICLE>

<V2_ARTICLE id="{v2_node_id}" ordinal="{v2_ordinal}">
{v2_raw_text}
</V2_ARTICLE>

Phân tích sự khác biệt và trả về JSON theo đúng schema sau. Không thêm bất kỳ text nào ngoài JSON:

{{
  "pair_id": "{pair_id}",
  "has_changes": true | false,
  "changes": [
    {{
      "change_id": "c001",
      "change_type": "numerical | terminology | structural | addition | deletion | reorder",
      "location_v1": "mô tả vị trí trong V1, ví dụ: 'Khoản 2, câu 1'",
      "location_v2": "mô tả vị trí tương ứng trong V2",
      "original_value": "giá trị / cụm từ gốc trong V1",
      "new_value": "giá trị / cụm từ mới trong V2",
      "verbatim_evidence_v1": "đoạn trích nguyên văn từ V1 chứa thay đổi (tối thiểu 10 từ xung quanh)",
      "verbatim_evidence_v2": "đoạn trích nguyên văn từ V2 chứa thay đổi (tối thiểu 10 từ xung quanh)",
      "confidence": 0.95
    }}
  ]
}}
"""
```

### Python wrapper cho ACU call

```python
import httpx
import json

async def call_acu(
    pair: DiffPair,
    v1_node: LSUNode,
    v2_node: LSUNode
) -> dict:
    """
    Gọi LLM cho một cặp article. Trả về raw ACU JSON.
    """
    prompt = USER_PROMPT_TEMPLATE.format(
        v1_node_id=v1_node.node_id,
        v1_ordinal=v1_node.ordinal,
        v1_raw_text=v1_node.raw_text,
        v2_node_id=v2_node.node_id,
        v2_ordinal=v2_node.ordinal,
        v2_raw_text=v2_node.raw_text,
        pair_id=pair.pair_id,
    )

    response = await httpx.post(
        "http://127.0.0.1:8080/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            **INFERENCE_CONFIG
        },
        timeout=120.0
    )

    raw_text = response.json()["choices"][0]["message"]["content"]

    # Parse JSON — strip markdown fences nếu model tự thêm vào
    clean = raw_text.strip().removeprefix("```json").removesuffix("```").strip()
    return json.loads(clean)
```

---

## Tầng 2 — Self-Verification Pass

Chạy NGAY SAU khi collect toàn bộ ACU outputs.

```python
def self_verify(
    acu_output: dict,
    v1_raw_text: str,
    v2_raw_text: str
) -> dict:
    """
    Kiểm tra từng change: verbatim_evidence có tồn tại trong source không?
    Nếu không tìm thấy → drop change (hallucination detected).
    """
    verified_changes = []

    for change in acu_output.get("changes", []):
        evidence_v1 = change.get("verbatim_evidence_v1", "")
        evidence_v2 = change.get("verbatim_evidence_v2", "")

        # Kiểm tra exact substring match
        v1_found = evidence_v1.strip() in v1_raw_text
        v2_found = evidence_v2.strip() in v2_raw_text

        change["verified"] = v1_found and v2_found

        if not (v1_found and v2_found):
            # LOG nhưng không raise exception — vẫn tiếp tục với các changes khác
            log_hallucination_detected(change, v1_found, v2_found)
        else:
            verified_changes.append(change)

    acu_output["changes"] = verified_changes
    acu_output["verification_passed"] = len(verified_changes)
    acu_output["verification_dropped"] = len(acu_output.get("changes_raw", [])) - len(verified_changes)
    return acu_output
```

**Quan trọng:** Self-verification không retry với LLM. Nếu evidence không tìm thấy, change bị DROP — không giữ lại dù confidence cao.

---

## Tầng 3 — Numerical Verification (Deterministic)

**Đây là tầng quan trọng nhất để đạt Zero Hallucination cho số liệu.**

```python
import re

# Patterns cho số tiếng Việt phổ biến trong văn bản pháp lý
LEGAL_NUMBER_PATTERNS = [
    r'\d{1,3}(?:\.\d{3})*(?:,\d+)?',   # 50.000.000 hoặc 1.500,50
    r'\d+(?:/\d+)?%',                    # 50% hoặc 3/4%
    r'\d+\s+(?:ngày|tháng|năm|tuần)',    # 30 ngày, 6 tháng
    r'\d+\s+(?:triệu|tỷ|nghìn|trăm)',   # 50 triệu, 2 tỷ
]

def verify_numerical_change(
    change: dict,
    v1_raw_text: str,
    v2_raw_text: str
) -> bool:
    """
    BẮT BUỘC áp dụng cho mọi change có change_type == "numerical".
    Nếu regex không tìm thấy → change bị DROP dù LLM confidence = 1.0.
    """
    if change["change_type"] != "numerical":
        return True  # Không áp dụng cho non-numerical

    original = re.escape(change["original_value"].strip())
    new_val  = re.escape(change["new_value"].strip())

    original_found = bool(re.search(original, v1_raw_text))
    new_val_found  = bool(re.search(new_val,  v2_raw_text))

    if not (original_found and new_val_found):
        log_numerical_verification_failed(change)
        return False

    return True

# Pipeline verification:
def run_full_verification(acu_output: dict, v1_text: str, v2_text: str) -> dict:
    # Tầng 2: Self-verification (evidence substring check)
    acu_output = self_verify(acu_output, v1_text, v2_text)

    # Tầng 3: Numerical verification (regex)
    final_changes = []
    for change in acu_output["changes"]:
        if verify_numerical_change(change, v1_text, v2_text):
            final_changes.append(change)

    acu_output["changes"] = final_changes
    return acu_output
```

---

## Tầng 4 — Report Aggregation

### Xử lý `added` và `deleted` articles (không cần LLM)

```python
def describe_added_article(node: LSUNode) -> dict:
    """
    Article mới trong V2 — không cần LLM, chỉ cần mô tả cấu trúc.
    """
    return {
        "change_type": "addition",
        "v2_node_id": node.node_id,
        "description": f"Điều {node.ordinal} '{node.title}' được thêm mới vào V2",
        "full_text_v2": node.raw_text,
        "verbatim_evidence_v2": node.raw_text[:500],  # First 500 chars
        "verified": True,   # Deterministic — không cần verify
        "confidence": 1.0
    }
```

### Executive Summary (LLM call có kiểm soát)

```python
SUMMARY_SYSTEM = """
Bạn là công cụ tổng hợp báo cáo pháp lý. Viết executive summary bằng tiếng Việt.

QUY TẮC:
1. Chỉ dùng thông tin từ JSON được cung cấp — không thêm thông tin khác
2. Không đưa ra nhận xét, đánh giá, hay khuyến nghị
3. Chỉ thống kê và mô tả khách quan
4. Format: văn xuôi ngắn gọn, không dùng bullet points
"""

SUMMARY_USER = """
Dựa vào kết quả so sánh sau, viết executive summary (tối đa 200 từ):

Tổng số cặp article so sánh: {total_pairs}
Có thay đổi nội dung: {changed_count}
Thêm mới: {added_count}
Bị xóa: {deleted_count}
Split: {split_count}
Merge: {merge_count}

Danh sách thay đổi theo loại:
{change_type_summary_json}

Viết summary. Không thêm nhận xét hay khuyến nghị.
"""
```

---

## Output Schema — ComparisonReport

```python
class ACUChange(BaseModel):
    change_id: str
    change_type: Literal["numerical", "terminology", "structural", "addition", "deletion", "reorder"]
    location_v1: str
    location_v2: str
    original_value: str
    new_value: str
    verbatim_evidence_v1: str
    verbatim_evidence_v2: str
    confidence: float
    verified: bool                  # Tầng 2 self-verify result
    numerical_verified: bool | None # Tầng 3 regex result (None nếu không phải numerical)

class ComparisonReport(BaseModel):
    report_id: str
    generated_at: datetime
    v1_file: str
    v2_file: str

    executive_summary: str          # Tiếng Việt, ~200 từ
    total_changes: int
    changes_by_type: dict           # {"numerical": 5, "terminology": 3, ...}

    article_comparisons: list[dict] # Chi tiết từng cặp article
    added_articles: list[dict]
    deleted_articles: list[dict]
    structural_events: list[dict]   # Split/merge/reorder

    # Audit trail — BẮT BUỘC
    raw_acu_outputs_path: str       # JSON trước khi verify
    verification_log_path: str      # Log của tầng 2 và 3
    similarity_matrix_path: str     # Từ Phase 2
```

**Output files:**
- `data/reports/report_{id}.json` — machine-readable
- `data/reports/report_{id}.pdf` — human-readable (WeasyPrint)

---

## Xử lý Errors

```python
class ACUError(Exception): pass
class JSONParseError(ACUError): pass
class VerificationError(ACUError): pass

async def call_acu_with_retry(pair, v1_node, v2_node, max_retries=3) -> dict:
    for attempt in range(max_retries):
        try:
            result = await call_acu(pair, v1_node, v2_node)
            return result
        except json.JSONDecodeError:
            if attempt == max_retries - 1:
                # Sau 3 lần fail → skip pair, log warning
                return {"pair_id": pair.pair_id, "has_changes": None, "error": "json_parse_failed"}
            continue  # retry
```

---

## Checklist trước khi bàn giao Output

- [ ] Tất cả ACU calls đã chạy với `temperature=0`
- [ ] Self-verification đã chạy cho 100% changes
- [ ] Numerical verification đã chạy cho 100% numerical changes
- [ ] Citation Accuracy = 100% (mọi `verbatim_evidence` tìm thấy trong source)
- [ ] Numerical Change Accuracy = 100% (regex verify pass)
- [ ] FPR < 1% (đo trên golden dataset)
- [ ] Raw ACU outputs đã được lưu trước khi verification (audit trail)
- [ ] Verification log đã ghi lại mọi dropped changes
- [ ] Report JSON và PDF đã được generate
