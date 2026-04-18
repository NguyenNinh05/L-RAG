"""
src/comparison/acu_prompter.py
================================
Prompt templates và builders cho Phase 3 — Generative Comparison.

Được trích xuất từ generative_comparison/comparison_pipeline.py để
tách biệt prompt engineering khỏi orchestration logic.

Nội dung:
    - ACU_SYSTEM_PROMPT     : System prompt cho Tầng 1 (ACU Extraction)
    - build_acu_user_prompt : Builder cho user prompt Tầng 1
    - SUMMARY_SYSTEM_PROMPT : System prompt cho Tầng 4 (Executive Summary)
    - build_summary_user_prompt : Builder cho user prompt Tầng 4
"""

from __future__ import annotations

import json
import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ACUOutput

ACU_SYSTEM_PROMPT = textwrap.dedent("""\
    Bạn là một chuyên gia phân tích pháp lý AI với nhiệm vụ DUY NHẤT là so sánh \
hai đoạn văn bản pháp lý và xác định CÁC THAY ĐỔI NGUYÊN TỬ (Atomic Comparison Units - ACUs).

## QUY TẮC BẮT BUỘC — VI PHẠM BẤT KỲ QUY TẮC NÀO SẼ KHIẾN OUTPUT BỊ BÁC BỎ HOÀN TOÀN:

### RULE 1: CHỈ TRÍCH DẪN NGUYÊN VĂN
- `verbatim_evidence_v1` PHẢI là copy-paste NGUYÊN VĂN từ <v1_text>.
- `verbatim_evidence_v2` PHẢI là copy-paste NGUYÊN VĂN từ <v2_text>.
- NGHIÊM CẤM paraphrase, tóm tắt, hoặc tạo ra bất kỳ chuỗi văn bản nào không có trong input.

### RULE 2: MỖI ACU CHỈ MÔ TẢ 1 THAY ĐỔI DUY NHẤT
- Không gộp nhiều thay đổi vào 1 ACU.
- Nếu có 3 thay đổi, tạo 3 ACU riêng biệt.

### RULE 3: PHÂN LOẠI CHÍNH XÁC change_type
- "numerical"   → Thay đổi con số, ngày tháng, phần trăm, tiền tệ, thời hạn.
- "terminology" → Thay đổi thuật ngữ/từ ngữ pháp lý quan trọng (không phải số).
- "structural"  → Thêm/bỏ mệnh đề, thay đổi cấu trúc câu mà không thêm/xoá hoàn toàn.
- "addition"    → Đoạn/câu/khoản hoàn toàn MỚI chỉ có trong V2, không có trong V1.
- "deletion"    → Đoạn/câu/khoản trong V1 bị XOÁ HOÀN TOÀN khỏi V2.
- "reorder"     → Thứ tự nội dung bị đổi chỗ, nội dung không thay đổi.

### RULE 4: EVIDENCE CHO TỪNG change_type
- "addition"   → verbatim_evidence_v1 = "" (chuỗi rỗng), verbatim_evidence_v2 = đoạn mới.
- "deletion"   → verbatim_evidence_v1 = đoạn bị xoá, verbatim_evidence_v2 = "" (chuỗi rỗng).
- Các type khác → CẢ HAI evidence phải có nội dung.

### RULE 5: KHÔNG SUY DIỄN
- KHÔNG suy diễn ý nghĩa pháp lý, hậu quả, hay rủi ro.
- CHỈ mô tả thực tế thay đổi trong original_value và new_value.

### RULE 6: CONFIDENCE
- confidence = 1.0 → Chắc chắn 100%, evidence rõ ràng.
- confidence = 0.7–0.9 → Khá chắc chắn.
- confidence < 0.5 → Không chắc, có thể là cách diễn đạt khác của cùng nội dung.

## FORMAT OUTPUT:
Trả về DUY NHẤT một JSON object hợp lệ với schema sau. KHÔNG thêm bất kỳ text nào ngoài JSON:

```json
{
  "acus": [
    {
      "change_type": "<numerical|terminology|structural|addition|deletion|reorder>",
      "location_v1": "<Điều X, Khoản Y, Điểm Z hoặc rỗng>",
      "location_v2": "<Điều X, Khoản Y, Điểm Z hoặc rỗng>",
      "original_value": "<giá trị/cụm từ GỐC ngắn gọn>",
      "new_value": "<giá trị/cụm từ MỚI ngắn gọn>",
      "verbatim_evidence_v1": "<copy-paste NGUYÊN VĂN từ V1 hoặc chuỗi rỗng>",
      "verbatim_evidence_v2": "<copy-paste NGUYÊN VĂN từ V2 hoặc chuỗi rỗng>",
      "confidence": <0.0 đến 1.0>
    }
  ]
}
```

Nếu không phát hiện được thay đổi nào, trả về: {"acus": []}
""")


def build_acu_user_prompt(
    raw_text_v1: str,
    raw_text_v2: str,
    breadcrumb_v1: str = "",
    breadcrumb_v2: str = "",
    match_type: str = "matched",
) -> str:
    """
    Xây dựng user prompt cho Tầng 1 (ACU extraction).

    Text V1 và V2 được đóng khung trong thẻ XML để LLM phân biệt rõ ràng.
    Thêm hướng dẫn đặc biệt tuỳ theo match_type.
    """
    # Hướng dẫn bổ sung theo match_type
    guidance_map = {
        "matched": (
            "Hai đoạn này được xác định là TƯƠNG ĐƯƠNG nhau (matched pair). "
            "Hãy tập trung tìm các thay đổi tinh tế trong nội dung."
        ),
        "added": (
            "Đoạn V2 là NỘI DUNG HOÀN TOÀN MỚI không có trong V1. "
            "Tạo một ACU duy nhất với change_type='addition'."
        ),
        "deleted": (
            "Đoạn V1 đã bị XOÁ HOÀN TOÀN khỏi V2. "
            "Tạo một ACU duy nhất với change_type='deletion'."
        ),
        "split": (
            "Đoạn V1 đã được TÁCH THÀNH NHIỀU đoạn trong V2. "
            "Hãy xác định nội dung nào được giữ nguyên và nội dung nào thay đổi/bổ sung."
        ),
        "merged": (
            "Nhiều đoạn V1 đã được GỘP LẠI thành một đoạn V2. "
            "Hãy xác định nội dung nào được giữ nguyên và nội dung nào bị xoá/thay đổi."
        ),
    }
    guidance = guidance_map.get(match_type, "So sánh hai đoạn văn bản.")

    # Build breadcrumb context
    context_lines: list[str] = []
    if breadcrumb_v1:
        context_lines.append(f"Vị trí V1: {breadcrumb_v1}")
    if breadcrumb_v2:
        context_lines.append(f"Vị trí V2: {breadcrumb_v2}")
    context_block = "\n".join(context_lines)

    return textwrap.dedent(f"""\
        {context_block}

        Hướng dẫn đặc biệt: {guidance}

        Hãy xác định tất cả thay đổi nguyên tử (ACU) giữa V1 và V2 dưới đây:

        <v1_text>
        {raw_text_v1}
        </v1_text>

        <v2_text>
        {raw_text_v2}
        </v2_text>

        Nhớ: Chỉ trả về JSON. Mọi chuỗi trong "verbatim_evidence_v1/v2" PHẢI xuất hiện \
nguyên văn trong <v1_text> hoặc <v2_text> tương ứng ở trên.
    """).strip()


# ============================================================
# TẦNG 4: EXECUTIVE SUMMARY PROMPTS
# ============================================================

SUMMARY_SYSTEM_PROMPT = textwrap.dedent("""\
    Bạn là chuyên gia pháp lý soạn thảo báo cáo so sánh văn bản bằng tiếng Việt.

## QUY TẮC BẮT BUỘC:

1. CHỈ dựa vào danh sách ACU (Atomic Comparison Units) được cung cấp.
2. KHÔNG suy diễn, KHÔNG thêm thông tin ngoài những gì có trong ACU.
3. Viết tiếng Việt, văn phong pháp lý trang trọng, ngắn gọn, súc tích.
4. Nếu không có ACU nào → nêu rõ "Không phát hiện thay đổi đáng kể".
5. Trả về DUY NHẤT một JSON object theo schema dưới đây. KHÔNG thêm text nào khác.

## OUTPUT SCHEMA:
```json
{
  "overall_assessment": "<Đánh giá tổng thể: mức độ và tính chất thay đổi>",
  "critical_changes": [
    "<Thay đổi quan trọng nhất — 1 câu ngắn gọn>",
    "<Thay đổi quan trọng thứ 2>",
    "..."
  ],
  "numerical_changes_summary": "<Tóm tắt tất cả thay đổi số liệu hoặc chuỗi rỗng>",
  "risk_flags": [
    "<Rủi ro pháp lý cụ thể nếu có bằng chứng rõ ràng>",
    "..."
  ],
  "recommendation": "<Khuyến nghị xem xét thêm hoặc chuỗi rỗng nếu không cần>"
}
```
""")


def build_summary_user_prompt(
    verified_acus: list[ACUOutput],
    v1_doc_id: str = "",
    v2_doc_id: str = "",
    location_context: str = "",
) -> str:
    """
    Xây dựng user prompt cho Tầng 4 (Executive Summary).

    Truyền danh sách ACU đã verified dưới dạng JSON compact.
    """
    acu_list_json = json.dumps(
        [
            {
                "change_type": acu.change_type.value,
                "location_v1": acu.location_v1,
                "location_v2": acu.location_v2,
                "original_value": acu.original_value,
                "new_value": acu.new_value,
                "confidence": round(acu.confidence, 2),
            }
            for acu in verified_acus
        ],
        ensure_ascii=False,
        indent=2,
    )

    context_parts = []
    if v1_doc_id:
        context_parts.append(f"Tài liệu V1: {v1_doc_id}")
    if v2_doc_id:
        context_parts.append(f"Tài liệu V2: {v2_doc_id}")
    if location_context:
        context_parts.append(f"Vị trí: {location_context}")
    context_str = " | ".join(context_parts) if context_parts else "Không có thông tin"

    return textwrap.dedent(f"""\
        Ngữ cảnh: {context_str}
        Tổng số thay đổi đã xác minh: {len(verified_acus)} ACU

        Danh sách các thay đổi đã được xác minh (ACUs):
        {acu_list_json}

        Hãy soạn báo cáo tóm tắt theo JSON schema đã quy định.
    """).strip()