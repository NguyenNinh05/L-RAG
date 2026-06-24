"""
src/comparison/report_generator.py
=============================================
GenerativeComparisonPipeline — Orchestrator của Phase 3.

Kiến trúc 4 tầng:

  Tầng 1 — ACU Extraction (LLM):
    Nhận cặp (raw_text_v1, raw_text_v2).
    Gửi system + user prompt cực kỳ chặt chẽ.
    LLM trả về: {"acus": [ {...ACUOutput JSON...}, ... ]}
    Parse và validate qua Pydantic ACUOutput.

  Tầng 2 — Evidence Self-Verification:
    Mỗi ACU được kiểm tra: verbatim evidence CÓ trong raw_text không?
    → Drop ACU hallucination

  Tầng 3 — Numerical Verification:
    ACU với change_type=numerical: extra strict regex check.
    → Drop ACU có số liệu sai

  Tầng 4 — Report Aggregation (LLM):
    Gọi LLM một lần cuối với toàn bộ ACU đã PASS.
    Sinh ExecutiveSummary tiếng Việt, không suy diễn.
    Render Markdown report hoàn chỉnh.

Usage:
    import asyncio
    from generative_comparison import (
        GenerativeComparisonPipeline,
        PipelineConfig,
        ComparisonRequest,
    )

    config = PipelineConfig(llm_base_url="http://localhost:8000/v1")
    pipeline = GenerativeComparisonPipeline(config=config)

    requests = [
        ComparisonRequest(
            pair_id="pair_abc123",
            match_type="matched",
            raw_text_v1="Điều 5. Bên A phải thanh toán 500.000.000 đồng trong vòng 30 ngày...",
            raw_text_v2="Điều 5. Bên A phải thanh toán 600.000.000 đồng trong vòng 45 ngày...",
            breadcrumb_v1="[Chương II > Điều 5]",
            breadcrumb_v2="[Chương II > Điều 5]",
        )
    ]

    reports = asyncio.run(pipeline.run_batch(requests, max_concurrency=4))
    for report in reports:
        print(report.to_human_readable())
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import textwrap
from datetime import datetime, timezone
from math import ceil
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .models import (
    ACUOutput,
    ChangeType,
    ComparisonReport,
    ComparisonRequest,
    ExecutiveSummary,
    VerificationResult,
    VerificationStatus,
)
from .llm_client import LocalLLMClient, LLMConfig
from .verifier import (
    VerificationConfig,
    VerificationEngine,
    extract_numbers,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers cho windowing & dedupe (chiến lược B/C)
# ---------------------------------------------------------------------------

# Sentence boundary cho tiếng Việt (giữ nguyên chữ hoa có dấu).
_RE_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?;])\s+(?=[A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ0-9])",
    re.UNICODE,
)


def _split_text_windows(text: str, max_chars: int, overlap: int) -> list[str]:
    """
    Chia text thành các cửa sổ ~max_chars ký tự, chồng nhau `overlap` ký tự.
    Cắt theo ranh giới câu khi có thể để không vỡ câu.
    Trả về [text] nếu text ngắn hơn max_chars.
    """
    if not text or len(text) <= max_chars:
        return [text] if text else []

    # Cắt theo câu trước, rồi gộp câu vào cửa sổ tới khi tới max_chars.
    sentences = _RE_SENTENCE_BOUNDARY.split(text)
    # Ghép lại giữ whitespace: split ở boundary nên cần re-add; đơn giản hoá bằng
    # cách duyệt cumulative.
    windows: list[str] = []
    cur = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        candidate = (cur + " " + sent) if cur else sent
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            if cur:
                windows.append(cur)
            # Nếu 1 câu dài hơn max_chars → cắt cứng theo max_chars.
            if len(sent) > max_chars:
                for i in range(0, len(sent), max_chars - overlap):
                    windows.append(sent[i : i + max_chars])
                cur = ""
            else:
                cur = sent
    if cur:
        windows.append(cur)

    # Thêm overlap: nối lại phần đuôi cửa sổ trước vào đầu cửa sổ sau (nếu muốn).
    # Ở đây overlap được dùng khi cắt cứng; giữ windows theo câu đã an toàn.
    return windows or [text]


def _acu_dedupe_key(acu: ACUOutput) -> tuple:
    """ khoá chuẩn hoá để phát hiện ACU trùng (cross-window / cross-pass)."""
    def _norm(s: str) -> str:
        s = (s or "").lower().strip()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[,\.;:\(\)\[\]\"'«»/\\]", "", s)
        return s
    return (
        acu.change_type.value,
        _norm(acu.original_value),
        _norm(acu.new_value),
    )


# ---------------------------------------------------------------------------
# Pipeline Configuration
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    """Cấu hình cho GenerativeComparisonPipeline."""

    # LLM settings
    llm_base_url: str = Field(
        default="http://localhost:8000/v1",
        description="URL của local LLM server (OpenAI-compatible)",
    )
    llm_model_name: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct",
        description="Tên model",
    )
    llm_api_key: str = Field(
        default="not-needed",
        description="API key (cho cloud provider như DeepSeek). Local server thường không cần.",
    )
    acu_temperature: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description=(
            "Temperature cho Tầng 1 (ACU extraction). "
            "CỰC THẤP để đảm bảo output deterministic và bám sát văn bản."
        ),
    )
    summary_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Temperature cho Tầng 4 (Executive Summary).",
    )
    max_tokens_acu: int = Field(
        default=4096,
        gt=0,
        description="Max tokens cho ACU extraction call",
    )
    max_tokens_summary: int = Field(
        default=1024,
        gt=0,
        description="Max tokens cho Executive Summary call",
    )
    timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        description="Timeout mỗi LLM request",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Số lần retry mỗi LLM request",
    )

    # Verification settings
    fuzzy_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Ngưỡng fuzzy match cho Tầng 2",
    )
    strict_numerical: bool = Field(
        default=True,
        description="100% strict mode cho Tầng 3",
    )

    # Concurrency
    max_concurrency: int = Field(
        default=4,
        gt=0,
        description="Số cặp xử lý song song (asyncio Semaphore)",
    )

    # Confidence filter
    min_confidence_to_include: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description=(
            "ACU có confidence < threshold này sẽ bị drop TRƯỚC KHI verification "
            "(pre-filter để tránh noise). Verification (Tầng 2&3) vẫn là guardrail "
            "chống hallucination nên có thể đặt thấp để tăng recall."
        ),
    )

    # LLM Provider — "local" (Qwen) hoặc "deepseek"
    provider: str = Field(
        default="local",
        description=(
            "LLM provider: 'local' (Qwen/Ollama/vLLM) hoặc 'deepseek' (DeepSeek V4 API). "
            "Ảnh hưởng đến: system prompt, windowing threshold, self-verification strategy."
        ),
    )

    # DeepSeek self-verification (chiến lược chống hallucination)
    enable_self_verification: bool = Field(
        default=False,
        description=(
            "DeepSeek-only: sau khi trích ACU, gửi lại danh sách ACU cho model "
            "tự kiểm tra — flag các ACU hallucination TRƯỚC khi vào Verification Engine. "
            "Giảm hallucination rate 3-5% trên DeepSeek."
        ),
    )

    # Long-article windowing (chiến lược B) — tránh truncation output token
    windowing_threshold_chars: int = Field(
        default=5000,
        ge=0,
        description=(
            "Nếu max(len_v1, len_v2) > threshold → chia text thành cửa sổ để trích "
            "ACU riêng (tránh LLM bị cắt ngắn khi quá nhiều thay đổi)."
        ),
    )
    window_max_chars: int = Field(default=2500, gt=0)
    window_overlap_chars: int = Field(default=400, ge=0)

    # Two-pass extraction + number pre-enumeration (chiến lược C)
    enable_number_enumeration: bool = Field(
        default=True,
        description="Đưa danh sách số liệu/ngày tháng của V1&V2 vào prompt (buộc LLM kiểm tra từng số).",
    )
    enable_second_pass: bool = Field(
        default=True,
        description="Lượt 2: hỏi LLM các thay đổi BỔ SUNG chưa có trong danh sách (tăng recall).",
    )


# ---------------------------------------------------------------------------
# System & User Prompts
# ---------------------------------------------------------------------------

# ============================================================
# TẦNG 1: ACU EXTRACTION PROMPTS
# ============================================================

ACU_SYSTEM_PROMPT = textwrap.dedent("""\
    Bạn là một chuyên gia phân tích pháp lý AI với nhiệm vụ DUY NHẤT là so sánh \
hai đoạn văn bản pháp lý và xác định CÁC THAY ĐỔI NGUYÊN TỬ (Atomic Comparison Units - ACUs).

## QUY TẮC BẮT BUỘC — VI PHẠM BẤT KỲ QUY TẮC NÀO SẼ KHIẾN OUTPUT BỊ BÁC BỎ HOÀN TOÀN:

### RULE 1: CHỈ TRÍCH DẪN NGUYÊN VĂN
- `verbatim_evidence_v1` PHẢI là copy-paste NGUYÊN VĂN từ <v1_text>.
- `verbatim_evidence_v2` PHẢI là copy-paste NGUYÊN VĂN từ <v2_text>.
- NGHIÊM CẤM paraphrase, tóm tắt, hoặc tạo ra bất kỳ chuỗi văn bản nào không có trong input.

### RULE 2: MỖI ACU CHỈ MÔ TẢ 1 THAY ĐỔI DUY NHẤT — TÁCH MỊN TỐI ĐA
- Không gộp nhiều thay đổi vào 1 ACU.
- Nếu có 3 thay đổi, tạo 3 ACU riêng biệt.
- Tách từng con số, từng ngày, từng cụm từ đổi thành 1 ACU riêng.
  VD: đổi "30 ngày → 45 ngày" VÀ "500.000 đồng → 600.000 đồng" là 2 ACU, không phải 1.

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
- BẮT BUỘC điền confidence cho MỌI ACU (kể cả reorder/addition/deletion).

### RULE 7: KHÔNG BỎ QUA THAY ĐỔI NHỎ
- Mọi khác biệt đều phải được báo cáo: đổi ngày tháng, lỗi chính tả, đổi dấu câu,
  đổi thứ tự từ, thêm/bớt một chữ. Mỗi cái = 1 ACU riêng.
- KHÔNG tự quyết "thay đổi này quá nhỏ nên bỏ qua". Nhiệm vụ của bạn là LIỆT KÊ toàn bộ.

### RULE 8: THAY ĐỔI LOGIC / NGHĨA PHÁP LÝ (RẤT QUAN TRỌNG)
- Khi V2 ĐẢO NGHĨA so với V1 (đổi "không" ↔ "có thể/được", "bắt buộc" ↔ "không bắt buộc",
  thêm/bớt "nếu…", "trừ khi…", "chỉ khi…", đổi điều kiện/phạm vi đối tượng) → đây là
  thay đổi quan trọng, dễ bị bỏ sót.
- Phân loại: đổi từ đòn bẩy nghĩa → "terminology"; đổi mệnh đề điều kiện → "structural".
- Đặt confidence CAO (0.85–1.0) cho logic-shift. KHÔNG gộp với thay đổi nhỏ khác.

## VÍ DỤ (few-shot — cách tách ACU)
<v1_text>Bên A phải thanh toán 500.000 đồng trong vòng 30 ngày. Bên A không được chuyển nhượng.</v1_text>
<v2_text>Bên A phải thanh toán 600.000 đồng trong vòng 30 ngày. Bên A có thể chuyển nhượng nếu được bên B đồng ý.</v2_text>
→ Trả về 2 ACU:
  1. {"change_type":"numerical","original_value":"500.000 đồng","new_value":"600.000 đồng", ...}
  2. {"change_type":"terminology","original_value":"không được chuyển nhượng","new_value":"có thể chuyển nhượng nếu được bên B đồng ý","confidence":0.95, ...}
→ "30 ngày" KHÔNG đổi → KHÔNG tạo ACU.

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


# ============================================================
# TẦNG 1: ACU EXTRACTION PROMPT — DEEPSEEK V4 PRO (OPTIMIZED)
# ============================================================
#
# Khác biệt so với prompt gốc (cho Qwen):
#   1. Thêm "Think step-by-step" — tận dụng khả năng Chain-of-Thought của DeepSeek
#   2. Thêm trường "reasoning" vào JSON schema — model tự giải thích từng ACU
#   3. Hướng dẫn logic-shift mạnh hơn — DeepSeek mạnh về reasoning cho Category D
#   4. Tận dụng 128K context — khuyến khích model đọc kỹ toàn bộ text không bỏ sót

ACU_SYSTEM_PROMPT_DEEPSEEK = textwrap.dedent("""\
    Bạn là một chuyên gia phân tích pháp lý AI với khả năng suy luận sâu (deep reasoning). \
Nhiệm vụ DUY NHẤT của bạn là so sánh hai đoạn văn bản pháp lý và xác định \
CÁC THAY ĐỔI NGUYÊN TỬ (Atomic Comparison Units - ACUs).

## HƯỚNG DẪN TƯ DUY (THINK STEP-BY-STEP)

Trước khi xuất JSON, hãy thực hiện các bước suy luận sau (TRONG ĐẦU, không ghi ra output):
1. ĐỌC LẠI toàn bộ <v1_text> và <v2_text> ít nhất 2 lần.
2. XÁC ĐỊNH từng vị trí khác biệt giữa hai văn bản — đánh dấu TẤT CẢ, không bỏ sót.
3. Với MỖI khác biệt, tự hỏi:
   a) Đây là thay đổi gì? (số? từ ngữ? cấu trúc? logic?)
   b) Bằng chứng nguyên văn từ V1 và V2 là gì?
   c) Mức độ chắc chắn của tôi về thay đổi này? (0.0-1.0)
4. PHÂN LOẠI từng thay đổi theo change_type.
5. ĐIỀN trường "reasoning" — giải thích NGẮN GỌN (1-2 câu) tại sao đây là một thay đổi.

## QUY TẮC BẮT BUỘC — VI PHẠM BẤT KỲ QUY TẮC NÀO SẼ KHIẾN OUTPUT BỊ BÁC BỎ HOÀN TOÀN:

### RULE 1: CHỈ TRÍCH DẪN NGUYÊN VĂN
- `verbatim_evidence_v1` PHẢI là copy-paste NGUYÊN VĂN từ <v1_text>.
- `verbatim_evidence_v2` PHẢI là copy-paste NGUYÊN VĂN từ <v2_text>.
- NGHIÊM CẤM paraphrase, tóm tắt, hoặc tạo ra bất kỳ chuỗi văn bản nào không có trong input.

### RULE 2: MỖI ACU CHỈ MÔ TẢ 1 THAY ĐỔI DUY NHẤT — TÁCH MỊN TỐI ĐA
- Không gộp nhiều thay đổi vào 1 ACU.
- Nếu có 3 thay đổi, tạo 3 ACU riêng biệt.
- Tách từng con số, từng ngày, từng cụm từ đổi thành 1 ACU riêng.
  VD: đổi "30 ngày → 45 ngày" VÀ "500.000 đồng → 600.000 đồng" là 2 ACU, không phải 1.

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
- Trường "reasoning" chỉ mô tả CƠ CHẾ của thay đổi (vd: "từ phủ định chuyển thành cho phép"),
  không diễn giải hậu quả pháp lý.

### RULE 6: CONFIDENCE
- confidence = 1.0 → Chắc chắn 100%, evidence rõ ràng trong văn bản.
- confidence = 0.7-0.9 → Khá chắc chắn.
- confidence < 0.5 → Không chắc, có thể là cách diễn đạt khác của cùng nội dung.
- BẮT BUỘC điền confidence cho MỌI ACU.

### RULE 7: KHÔNG BỎ QUA THAY ĐỔI NHỎ
- Mọi khác biệt đều phải được báo cáo: đổi ngày tháng, lỗi chính tả, đổi dấu câu,
  đổi thứ tự từ, thêm/bớt một chữ. Mỗi cái = 1 ACU riêng.
- KHÔNG tự quyết "thay đổi này quá nhỏ nên bỏ qua".

### RULE 8: THAY ĐỔI LOGIC / NGHĨA PHÁP LÝ (ĐẶC BIỆT QUAN TRỌNG — TẬN DỤNG KHẢ NĂNG REASONING)
- Khi V2 ĐẢO NGHĨA so với V1 → ĐÂY LÀ THAY ĐỔI CRITICAL, phải phát hiện bằng được.
  Ví dụ cụ thể:
  * "không được" → "có thể", "được phép"
  * "phải" → "có trách nhiệm", "nên"
  * "cấm" → "được phép", "cho phép"
  * Thêm "nếu...", "trừ khi...", "chỉ khi..." → thay đổi điều kiện
  * Bỏ "trong mọi trường hợp" → mở rộng phạm vi
- Phân loại: đổi từ đòn bẩy nghĩa → "terminology"; đổi mệnh đề điều kiện → "structural".
- Đặt confidence CAO (0.85-1.0) cho logic-shift. GHI RÕ trong reasoning cơ chế thay đổi.

### RULE 9: ĐIỀN REASONING CHO MỌI ACU
- Trường "reasoning" là BẮT BUỘC với DeepSeek.
- Viết 1-2 câu tiếng Việt ngắn gọn, giải thích:
  * Thay đổi là gì? (vd: "từ 'không được' thành 'có thể'")
  * Tại sao đây là một thay đổi? (vd: "đảo nghĩa từ cấm sang cho phép")
- Reasoning giúp model TỰ KIỂM TRA ACU của chính nó — viết CÀNG RÕ RÀNG,
  tỉ lệ hallucination CÀNG THẤP.

## VÍ DỤ (few-shot — cách tách ACU + reasoning)
<v1_text>Bên A phải thanh toán 500.000 đồng trong vòng 30 ngày. Bên A không được chuyển nhượng.</v1_text>
<v2_text>Bên A phải thanh toán 600.000 đồng trong vòng 30 ngày. Bên A có thể chuyển nhượng nếu được bên B đồng ý.</v2_text>
→ Trả về 2 ACU:
  1. {{"change_type":"numerical","original_value":"500.000 đồng","new_value":"600.000 đồng","reasoning":"Số tiền thay đổi từ 500.000 lên 600.000 đồng","confidence":1.0, ...}}
  2. {{"change_type":"terminology","original_value":"không được chuyển nhượng","new_value":"có thể chuyển nhượng nếu được bên B đồng ý","reasoning":"Đảo nghĩa từ cấm sang cho phép, thêm điều kiện 'nếu được bên B đồng ý'","confidence":0.95, ...}}
→ "30 ngày" KHÔNG đổi → KHÔNG tạo ACU.

## FORMAT OUTPUT:
Trả về DUY NHẤT một JSON object hợp lệ. KHÔNG thêm bất kỳ text nào ngoài JSON:

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
      "reasoning": "<1-2 câu giải thích tại sao đây là thay đổi>",
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


# ---------------------------------------------------------------------------
# Markdown Report Renderer
# ---------------------------------------------------------------------------


def render_markdown_report(
    report: ComparisonReport,
    exec_summary: ExecutiveSummary | None = None,
) -> str:
    """
    Render ComparisonReport thành Markdown string hoàn chỉnh.

    Layout:
        # Báo cáo So sánh Văn bản Pháp lý
        ## Tóm tắt Điều hành
        ## Các Thay đổi Chi tiết (ACU Table)
        ## Thay đổi bị Bác bỏ (Hallucination Log)
        ## Thống kê
    """
    lines: list[str] = []

    # ── Header ──────────────────────────────────────────────
    now = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    lines.append(f"# 📋 Báo cáo So sánh Văn bản Pháp lý")
    lines.append("")
    lines.append(f"**Report ID:** `{report.report_id}`  ")
    lines.append(f"**Tài liệu V1:** `{report.v1_doc_id or 'N/A'}`  ")
    lines.append(f"**Tài liệu V2:** `{report.v2_doc_id or 'N/A'}`  ")
    lines.append(f"**Vị trí:** {report.location_context or 'Toàn bộ văn bản'}  ")
    lines.append(f"**Thời điểm:** {now}  ")
    lines.append("")

    # ── Thống kê nhanh ───────────────────────────────────────
    total = report.total_acus_raw
    passed = report.total_acus_passed
    rejected = report.total_acus_rejected
    hal_rate = report.hallucination_rate * 100

    lines.append("---")
    lines.append("")
    lines.append("## 📊 Thống kê Tổng quan")
    lines.append("")
    lines.append(f"| Chỉ số | Giá trị |")
    lines.append(f"|--------|---------|")
    lines.append(f"| Tổng ACU phát hiện | **{total}** |")
    lines.append(f"| ACU đã xác minh (PASSED) | **{passed}** |")
    lines.append(f"| ACU bị bác bỏ | **{rejected}** |")
    lines.append(f"| Tỷ lệ ảo giác | **{hal_rate:.1f}%** |")
    lines.append("")

    # ── Executive Summary ────────────────────────────────────
    if exec_summary:
        lines.append("---")
        lines.append("")
        lines.append("## 🔍 Tóm tắt Điều hành")
        lines.append("")
        lines.append(f"**Đánh giá tổng thể:**  ")
        lines.append(f"{exec_summary.overall_assessment}")
        lines.append("")

        if exec_summary.critical_changes:
            lines.append("**Các thay đổi quan trọng:**")
            for change in exec_summary.critical_changes:
                lines.append(f"- {change}")
            lines.append("")

        if exec_summary.numerical_changes_summary:
            lines.append(f"**Thay đổi số liệu:**  ")
            lines.append(exec_summary.numerical_changes_summary)
            lines.append("")

        if exec_summary.risk_flags:
            lines.append("**⚠️ Rủi ro pháp lý cần lưu ý:**")
            for flag in exec_summary.risk_flags:
                lines.append(f"- ⚠️ {flag}")
            lines.append("")

        if exec_summary.recommendation:
            lines.append(f"**💡 Khuyến nghị:**  ")
            lines.append(exec_summary.recommendation)
            lines.append("")

    # ── Chi tiết ACU đã Pass ─────────────────────────────────
    if report.verified_acus:
        lines.append("---")
        lines.append("")
        lines.append(f"## ✅ Các Thay đổi Đã Xác minh ({passed} ACU)")
        lines.append("")

        # Group by change_type
        from collections import defaultdict
        grouped: dict[str, list[ACUOutput]] = defaultdict(list)
        for acu in report.verified_acus:
            grouped[acu.change_type.value].append(acu)

        type_icons = {
            "numerical": "🔢",
            "terminology": "📝",
            "structural": "🏗️",
            "addition": "➕",
            "deletion": "➖",
            "reorder": "🔄",
        }
        type_labels = {
            "numerical": "Thay đổi Số liệu",
            "terminology": "Thay đổi Thuật ngữ",
            "structural": "Thay đổi Cấu trúc",
            "addition": "Nội dung Mới thêm",
            "deletion": "Nội dung Bị xoá",
            "reorder": "Sắp xếp lại",
        }

        for ct_value, acus_in_group in grouped.items():
            icon = type_icons.get(ct_value, "•")
            label = type_labels.get(ct_value, ct_value)
            lines.append(f"### {icon} {label} ({len(acus_in_group)})")
            lines.append("")

            for i, acu in enumerate(acus_in_group, 1):
                loc = acu.location_v2 or acu.location_v1 or "—"
                lines.append(f"**{i}. {loc}** _(confidence: {acu.confidence:.0%})_")

                if acu.original_value:
                    lines.append(f"- **Trước (V1):** {acu.original_value}")
                if acu.new_value:
                    lines.append(f"- **Sau (V2):** {acu.new_value}")

                if acu.verbatim_evidence_v1:
                    ev1_preview = acu.verbatim_evidence_v1[:200]
                    if len(acu.verbatim_evidence_v1) > 200:
                        ev1_preview += "..."
                    lines.append(f'- **Bằng chứng V1:** > *"{ev1_preview}"*')

                if acu.verbatim_evidence_v2:
                    ev2_preview = acu.verbatim_evidence_v2[:200]
                    if len(acu.verbatim_evidence_v2) > 200:
                        ev2_preview += "..."
                    lines.append(f'- **Bằng chứng V2:** > *"{ev2_preview}"*')

                lines.append("")

    else:
        lines.append("---")
        lines.append("")
        lines.append("## ✅ Các Thay đổi Đã Xác minh")
        lines.append("")
        lines.append("> _Không phát hiện thay đổi nào qua quá trình xác minh._")
        lines.append("")

    # ── Hallucination Log ────────────────────────────────────
    if report.rejected_acus:
        lines.append("---")
        lines.append("")
        lines.append(f"## ❌ ACU Bị Bác bỏ — Nhật ký Ảo giác ({rejected})")
        lines.append("")
        lines.append(
            "> _Các ACU này đã bị hệ thống xác minh tự động phát hiện là không chính xác._"
        )
        lines.append("")

        for i, vres in enumerate(report.rejected_acus, 1):
            acu = vres.acu
            status_label = {
                VerificationStatus.FAILED_EVIDENCE: "❌ Evidence không tìm thấy",
                VerificationStatus.FAILED_NUMERICAL: "🔢 Số liệu không khớp",
            }.get(vres.status, "❌ Bị bác bỏ")

            lines.append(
                f"**{i}. [{status_label}]** change_type=`{acu.change_type.value}` | "
                f"Lý do: {vres.rejection_reason}"
            )
        lines.append("")

    # ── Footer ──────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append(
        "_Báo cáo được tạo tự động bởi L-RAG Phase 3 — Generative Comparison Pipeline. "
        "Mọi ACU đã được xác minh qua 3 tầng kiểm tra Zero-Hallucination._"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GenerativeComparisonPipeline
# ---------------------------------------------------------------------------


class GenerativeComparisonPipeline:
    """
    Orchestrator chính của Phase 3 — Generative Comparison.

    Xử lý một hoặc nhiều ComparisonRequest song song (asyncio).

    Example:
        pipeline = GenerativeComparisonPipeline(
            config=PipelineConfig(llm_base_url="http://localhost:8000/v1")
        )
        reports = await pipeline.run_batch(requests, max_concurrency=4)
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._cfg = config or PipelineConfig()

        # ── Provider auto-config ────────────────────────────
        if self._cfg.provider == "deepseek":
            # Windowing: 128K context → nâng threshold lên 20K ký tự
            if self._cfg.windowing_threshold_chars <= 5000:
                self._cfg.windowing_threshold_chars = 20000
            if self._cfg.window_max_chars <= 2500:
                self._cfg.window_max_chars = 8000
            # Self-verification mặc định ON cho DeepSeek
            # Để tắt: PipelineConfig(provider="deepseek", enable_self_verification=False)
            #   rồi truyền config đã chỉnh sửa vào GenerativeComparisonPipeline
            self._cfg.enable_self_verification = True
            # Chọn DeepSeek-optimized prompt
            self._acu_system_prompt = ACU_SYSTEM_PROMPT_DEEPSEEK
        else:
            self._acu_system_prompt = ACU_SYSTEM_PROMPT

        # Khởi tạo LLM Client cho ACU extraction (low temperature)
        self._acu_llm = LocalLLMClient(
            config=LLMConfig(
                base_url=self._cfg.llm_base_url,
                api_key=self._cfg.llm_api_key,
                model_name=self._cfg.llm_model_name,
                temperature=self._cfg.acu_temperature,
                max_tokens=self._cfg.max_tokens_acu,
                timeout_seconds=self._cfg.timeout_seconds,
                max_retries=self._cfg.max_retries,
            )
        )

        # LLM Client riêng cho Summary (slightly higher temperature)
        self._summary_llm = LocalLLMClient(
            config=LLMConfig(
                base_url=self._cfg.llm_base_url,
                api_key=self._cfg.llm_api_key,
                model_name=self._cfg.llm_model_name,
                temperature=self._cfg.summary_temperature,
                max_tokens=self._cfg.max_tokens_summary,
                timeout_seconds=self._cfg.timeout_seconds,
                max_retries=self._cfg.max_retries,
            )
        )

        # Verification Engine
        self._verifier = VerificationEngine(
            config=VerificationConfig(
                fuzzy_match_threshold=self._cfg.fuzzy_threshold,
                strict_numerical=self._cfg.strict_numerical,
            )
        )
        logger.info(
            "GenerativeComparisonPipeline khởi tạo: model=%s, max_concurrency=%d",
            self._cfg.llm_model_name,
            self._cfg.max_concurrency,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_single(self, request: ComparisonRequest) -> ComparisonReport:
        """
        Xử lý một ComparisonRequest qua toàn bộ 4 tầng.

        Returns:
            ComparisonReport với đầy đủ machine_readable và human_readable.
        """
        logger.info(
            "Processing pair_id=%s (match_type=%s)",
            request.pair_id,
            request.match_type,
        )

        # ── Tầng 1: ACU Extraction ─────────────────────────────
        raw_acus = await self._tier1_extract_acus(request)

        # ── DeepSeek Self-Verification (pre-Verification Engine) ──
        if self._cfg.provider == "deepseek" and self._cfg.enable_self_verification:
            raw_acus = await self._deepseek_self_verify(
                acus=raw_acus,
                raw_text_v1=request.raw_text_v1,
                raw_text_v2=request.raw_text_v2,
                pair_id=request.pair_id,
            )

        # ── Pre-filter: Bỏ ACU confidence quá thấp ────────────
        filtered_acus = [
            acu for acu in raw_acus
            if acu.confidence >= self._cfg.min_confidence_to_include
        ]
        if len(raw_acus) != len(filtered_acus):
            logger.info(
                "Pre-filter: dropped %d/%d ACUs với confidence < %.2f",
                len(raw_acus) - len(filtered_acus),
                len(raw_acus),
                self._cfg.min_confidence_to_include,
            )

        # Inject pair_id vào ACUs
        for acu in filtered_acus:
            acu.pair_id = request.pair_id

        # ── Tầng 2 & 3: Verification ───────────────────────────
        verification_results = self._verifier.verify_batch(
            acus=filtered_acus,
            raw_text_v1=request.raw_text_v1,
            raw_text_v2=request.raw_text_v2,
        )

        # Phân loại kết quả
        passed_results = [r for r in verification_results if r.is_passed]
        rejected_results = [r for r in verification_results if not r.is_passed]
        verified_acus = [r.acu for r in passed_results]

        logger.info(
            "Verification: %d passed, %d rejected (pair_id=%s)",
            len(verified_acus),
            len(rejected_results),
            request.pair_id,
        )

        # ── Tầng 4: Executive Summary & Report ─────────────────
        exec_summary = await self._tier4_generate_summary(
            verified_acus=verified_acus,
            request=request,
        )

        # Tạo report sơ bộ để render Markdown
        report = ComparisonReport(
            pair_id=request.pair_id,
            v1_doc_id=request.v1_doc_id,
            v2_doc_id=request.v2_doc_id,
            location_context=request.breadcrumb_v1 or request.breadcrumb_v2,
            verified_acus=verified_acus,
            rejected_acus=rejected_results,
            executive_summary=exec_summary,
        )

        # Render Markdown
        report.markdown_report = render_markdown_report(report, exec_summary)

        logger.info(
            "Report hoàn thành: pair_id=%s, passed=%d, rejected=%d, "
            "hallucination_rate=%.1f%%",
            request.pair_id,
            report.total_acus_passed,
            report.total_acus_rejected,
            report.hallucination_rate * 100,
        )

        return report

    async def run_batch(
        self,
        requests: list[ComparisonRequest],
        max_concurrency: int | None = None,
    ) -> list[ComparisonReport]:
        """
        Xử lý nhiều ComparisonRequest song song với asyncio Semaphore.

        Args:
            requests:        Danh sách ComparisonRequest cần xử lý.
            max_concurrency: Override max_concurrency từ config.

        Returns:
            List[ComparisonReport] tương ứng với từng request (giữ nguyên thứ tự).
        """
        concurrency = max_concurrency or self._cfg.max_concurrency
        semaphore = asyncio.Semaphore(concurrency)

        async def _bounded_run(req: ComparisonRequest) -> ComparisonReport:
            async with semaphore:
                try:
                    return await self.run_single(req)
                except Exception as exc:
                    logger.exception(
                        "Lỗi nghiêm trọng khi xử lý pair_id=%s: %s",
                        req.pair_id,
                        exc,
                    )
                    # Trả về report rỗng thay vì crash toàn batch
                    return ComparisonReport(
                        pair_id=req.pair_id,
                        v1_doc_id=req.v1_doc_id,
                        v2_doc_id=req.v2_doc_id,
                        markdown_report=(
                            f"# ❌ Lỗi xử lý\n\n"
                            f"Không thể xử lý pair `{req.pair_id}`.\n\n"
                            f"**Error:** {exc}"
                        ),
                    )

        logger.info(
            "Bắt đầu xử lý batch: %d requests, concurrency=%d",
            len(requests),
            concurrency,
        )
        tasks = [_bounded_run(req) for req in requests]
        reports = await asyncio.gather(*tasks)
        logger.info("Batch hoàn thành: %d reports.", len(reports))
        return list(reports)

    async def health_check(self) -> dict[str, Any]:
        """Kiểm tra kết nối LLM server."""
        ok = await self._acu_llm.health_check()
        return {
            "llm_server": "ok" if ok else "unreachable",
            "base_url": self._cfg.llm_base_url,
            "model": self._cfg.llm_model_name,
        }

    # ------------------------------------------------------------------
    # Tầng 1: ACU Extraction
    # ------------------------------------------------------------------

    async def _tier1_extract_acus(
        self,
        request: ComparisonRequest,
    ) -> list[ACUOutput]:
        """
        Tầng 1: Trích xuất ACU qua LLM với 3 chiến lược tăng recall:

          B. Windowing — nếu raw text dài (> threshold), chia thành cửa sổ và
             trích ACU riêng từng cửa sổ để tránh truncation output token.
          C1. Number enumeration — đưa danh sách số/ngày của V1&V2 vào prompt.
          C2. Two-pass — lượt 2 hỏi các thay đổi BỔ SUNG chưa có.

        Mọi ACU (cả 2 lượt, mọi cửa sổ) được dedupe trước khi trả về.
        Verification (Tầng 2&3) vẫn chạy sau ở run_single với raw text đầy đủ.
        """
        v1, v2 = request.raw_text_v1, request.raw_text_v2
        mt = request.match_type
        bc1, bc2 = request.breadcrumb_v1, request.breadcrumb_v2
        pid = request.pair_id

        # C1 — number enumeration hint
        num_hint = ""
        if self._cfg.enable_number_enumeration:
            num_hint = self._build_number_hint(v1, v2)

        # Lượt 1 (có thể windowed)
        first = await self._extract_windowed(v1, v2, mt, bc1, bc2, num_hint, pid)
        first = self._dedupe_acus(first)

        # C2 — lượt 2: thay đổi bổ sung.
        # Bỏ qua cho added/deleted: theo guidance_map chúng chỉ sinh 1 ACU (toàn bài
        # thêm/xoá) → lượt 2 không thêm gì được, chỉ tốn thời gian.
        do_second_pass = (
            self._cfg.enable_second_pass
            and (v1 or v2)
            and mt not in ("added", "deleted")
        )
        if do_second_pass:
            extra = await self._second_pass(v1, v2, mt, bc1, bc2, first, pid)
            merged = self._merge_dedupe(first, extra)
        else:
            merged = first

        logger.info(
            "Tier 1 extracted %d ACUs (pair_id=%s) [first=%d, second_pass=%s]",
            len(merged),
            pid,
            len(first),
            self._cfg.enable_second_pass,
        )
        return merged

    async def _extract_windowed(
        self,
        v1: str,
        v2: str,
        match_type: str,
        bc1: str,
        bc2: str,
        extra_instructions: str,
        pair_id: str,
    ) -> list[ACUOutput]:
        """Lượt 1: single-call nếu text ngắn, windowing nếu dài."""
        max_len = max(len(v1 or ""), len(v2 or ""))
        if max_len <= self._cfg.windowing_threshold_chars:
            return await self._llm_extract_acus(
                v1, v2, match_type, bc1, bc2, extra_instructions, pair_id
            )

        # Windowing: chia từng phía theo cửa sổ, ghép cặp theo vị trí.
        w1 = _split_text_windows(v1, self._cfg.window_max_chars, self._cfg.window_overlap_chars)
        w2 = _split_text_windows(v2, self._cfg.window_max_chars, self._cfg.window_overlap_chars)
        if not w1 and not w2:
            return []

        all_acus: list[ACUOutput] = []
        n = max(len(w1), len(w2))
        for i in range(n):
            a = w1[min(i, len(w1) - 1)] if w1 else ""
            b = w2[min(i, len(w2) - 1)] if w2 else ""
            if not a and not b:
                continue
            try:
                win_acus = await self._llm_extract_acus(
                    a, b, match_type, bc1, bc2, extra_instructions, f"{pair_id}#w{i}"
                )
                all_acus.extend(win_acus)
            except Exception as exc:
                logger.warning("Window %d extract failed (pair_id=%s): %s", i, pair_id, exc)

        logger.info(
            "Windowing: %d windows (v1=%d, v2=%d) → %d raw ACUs (pair_id=%s)",
            n, len(w1), len(w2), len(all_acus), pair_id,
        )
        return all_acus

    async def _second_pass(
        self,
        v1: str,
        v2: str,
        match_type: str,
        bc1: str,
        bc2: str,
        found_acus: list[ACUOutput],
        pair_id: str,
    ) -> list[ACUOutput]:
        """C2: lượt 2 — hỏi LLM các thay đổi BỔ SUNG chưa có trong found_acus."""
        found_compact = json.dumps(
            [
                {
                    "change_type": a.change_type.value,
                    "original_value": a.original_value,
                    "new_value": a.new_value,
                }
                for a in found_acus
            ],
            ensure_ascii=False,
        )
        extra = (
            f"\n\n🔄 LƯỢT 2 — Kiểm tra bổ sung (recall boost):\n"
            f"Đã tìm thấy {len(found_acus)} ACU ở lượt 1: {found_compact}\n"
            f"Hãy xem LẠI <v1_text> và <v2_text>, liệt kê CHỈ CÁC THAY ĐỔI BỔ SUNG "
            f"chưa có trong danh sách trên. Ưu tiên: số/ngày/thời hạn, từ phủ định "
            f"hoặc cho phép (không ↔ được/có thể, phải, chỉ, cấm), phạm vi & điều "
            f"kiện (nếu / trừ khi / chỉ khi / trong trường hợp). KHÔNG lặp lại ACU "
            f"đã có. Trả về JSON theo đúng schema cũ (trả về danh sách rỗng nếu không còn gì)."
        )
        try:
            return await self._llm_extract_acus(v1, v2, match_type, bc1, bc2, extra, f"{pair_id}#p2")
        except Exception as exc:
            logger.warning("Second pass failed (pair_id=%s): %s", pair_id, exc)
            return []

    def _build_number_hint(self, v1: str, v2: str) -> str:
        """C1: liệt kê số/ngày/tiền đã trích xuất tự động để buộc LLM kiểm tra từng mục."""
        try:
            n1 = sorted({x.raw_str for x in extract_numbers(v1 or "")})
            n2 = sorted({x.raw_str for x in extract_numbers(v2 or "")})
        except Exception:
            return ""
        if not n1 and not n2:
            return ""
        lines = []
        if n1:
            lines.append(f"- V1: {', '.join(n1[:80])}")
        if n2:
            lines.append(f"- V2: {', '.join(n2[:80])}")
        return (
            "\n\nℹ️ DANH SÁCH SỐ LIỆU/NGÀY ĐÃ TRÍCH XUẤT TỰ ĐỘNG (bắt buộc đối chiếu từng mục):\n"
            + "\n".join(lines)
            + "\n→ Mỗi sự KHÁC NHAU giữa hai bên PHẢI thành 1 ACU numerical riêng. "
              "Một số chỉ có ở 1 bên có thể là addition/deletion."
        )

    def _dedupe_acus(self, acus: list[ACUOutput]) -> list[ACUOutput]:
        """Bỏ ACU trùng lặp (giữ lần xuất hiện đầu)."""
        seen: set[tuple] = set()
        out: list[ACUOutput] = []
        for acu in acus:
            key = _acu_dedupe_key(acu)
            # Bỏ qua dedup cho ACU không có giá trị (addition/deletion thuần).
            if not (acu.original_value or acu.new_value):
                out.append(acu)
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(acu)
        if len(out) != len(acus):
            logger.info("Dedup: %d → %d ACUs", len(acus), len(out))
        return out

    def _merge_dedupe(self, base: list[ACUOutput], extra: list[ACUOutput]) -> list[ACUOutput]:
        """Gộp extra vào base, bỏ các ACU trùng với base."""
        seen = {_acu_dedupe_key(a) for a in base if (a.original_value or a.new_value)}
        merged = list(base)
        added = 0
        for acu in extra:
            key = _acu_dedupe_key(acu)
            if not (acu.original_value or acu.new_value):
                merged.append(acu)
                added += 1
                continue
            if key in seen:
                continue
            seen.add(key)
            merged.append(acu)
            added += 1
        if added:
            logger.info("Second pass added %d new ACUs", added)
        return merged

    # ------------------------------------------------------------------
    # DeepSeek Self-Verification (Anti-Hallucination Pass)
    # ------------------------------------------------------------------

    async def _deepseek_self_verify(
        self,
        acus: list[ACUOutput],
        raw_text_v1: str,
        raw_text_v2: str,
        pair_id: str,
    ) -> list[ACUOutput]:
        """
        DeepSeek-only: Self-Verification Pass.

        Sau khi trích xuất ACU (Tier 1), gửi lại danh sách ACU cho DeepSeek
        tự kiểm tra. Model được yêu cầu:
          1. So sánh verbatim_evidence với raw text gốc
          2. Flag các ACU mà evidence KHÔNG khớp với source (hallucination risk)
          3. Xác nhận hoặc sửa confidence score

        Điều này chạy TRƯỚC Verification Engine (Tier 2/3) và chỉ áp dụng
        cho DeepSeek — tận dụng khả năng reasoning + self-reflection của model.

        Args:
            acus: Danh sách ACU đã trích từ Tier 1
            raw_text_v1, raw_text_v2: Văn bản gốc để đối chiếu
            pair_id: ID của cặp đang xử lý

        Returns:
            Danh sách ACU đã được self-verify (có thể bị loại bỏ bớt).
        """
        if not acus:
            return acus

        # Build compact ACU list for verification
        acus_compact: list[dict] = []
        for a in acus:
            acus_compact.append({
                "acu_id": a.acu_id[-8:],  # short ID for readability
                "change_type": a.change_type.value,
                "original_value": a.original_value,
                "new_value": a.new_value,
                "verbatim_evidence_v1": a.verbatim_evidence_v1[:200] if a.verbatim_evidence_v1 else "",
                "verbatim_evidence_v2": a.verbatim_evidence_v2[:200] if a.verbatim_evidence_v2 else "",
                "reasoning": a.reasoning,
                "confidence": a.confidence,
            })

        verify_prompt = (
            f"## Nhiệm vụ: TỰ KIỂM TRA (Self-Verification)\n\n"
            f"Bạn vừa tạo ra {len(acus)} ACU khi so sánh hai văn bản. "
            f"Hãy kiểm tra lại CHÍNH CÁC ACU CỦA BẠN.\n\n"
            f"### Văn bản gốc V1:\n<v1_text>\n{raw_text_v1[:8000]}\n</v1_text>\n\n"
            f"### Văn bản gốc V2:\n<v2_text>\n{raw_text_v2[:8000]}\n</v2_text>\n\n"
            f"### Danh sách ACU cần kiểm tra:\n```json\n"
            f"{json.dumps(acus_compact, ensure_ascii=False, indent=2)}\n"
            f"```\n\n"
            f"### Yêu cầu:\n"
            f"1. Với MỖI ACU, kiểm tra xem verbatim_evidence có THỰC SỰ "
            f"xuất hiện trong văn bản gốc không.\n"
            f"2. Nếu evidence KHÔNG khớp → flag là hallucination, giải thích lý do.\n"
            f"3. Nếu confidence không phù hợp → điều chỉnh.\n"
            f"4. Nếu ACU trùng lặp → merge, giữ ACU có confidence cao hơn.\n\n"
            f"### OUTPUT FORMAT:\n"
            f"Trả về JSON với format:\n"
            f'{{"verified": [{{"acu_id": "...", "confidence": x.x}}], '
            f'"flagged": [{{"acu_id": "...", "reason": "..."}}]}}\n\n'
            f"verified = ACU hợp lệ (evidence khớp). flagged = ACU cần LOẠI BỎ."
        )

        try:
            result = await self._acu_llm.chat_json(
                system_prompt=(
                    "Bạn là chuyên gia kiểm tra chất lượng (Quality Assurance) cho "
                    "hệ thống so sánh văn bản pháp lý. Nhiệm vụ duy nhất: xác minh "
                    "từng ACU dựa trên bằng chứng nguyên văn. Trả lời bằng JSON."
                ),
                user_prompt=verify_prompt,
            )
        except Exception as exc:
            logger.warning(
                "Self-verification LLM call failed (pair_id=%s): %s — keeping all ACUs",
                pair_id, exc,
            )
            return acus

        # Parse kết quả self-verification
        flagged_ids = set()
        verified_ids = set()

        for item in result.get("verified", []):
            vid = item.get("acu_id", "")
            verified_ids.add(vid)
            # Update confidence if adjusted
            if "confidence" in item:
                try:
                    new_conf = float(item["confidence"])
                    for a in acus:
                        if a.acu_id.endswith(vid):
                            a.confidence = max(0.0, min(1.0, new_conf))
                            break
                except (ValueError, TypeError):
                    pass

        for item in result.get("flagged", []):
            fid = item.get("acu_id", "")
            flagged_ids.add(fid)
            reason = item.get("reason", "unknown")
            logger.debug("Self-verify flagged ACU %s: %s", fid, reason)

        # Drop flagged ACUs (hallucination)
        if flagged_ids:
            original_count = len(acus)
            acus = [
                a for a in acus
                if not any(a.acu_id.endswith(fid) for fid in flagged_ids)
            ]
            logger.info(
                "Self-verification: dropped %d/%d flagged ACUs (pair_id=%s)",
                original_count - len(acus), original_count, pair_id,
            )

        return acus
        return merged

    async def _llm_extract_acus(
        self,
        raw_text_v1: str,
        raw_text_v2: str,
        match_type: str,
        breadcrumb_v1: str,
        breadcrumb_v2: str,
        extra_instructions: str,
        pair_id: str,
    ) -> list[ACUOutput]:
        """
        Core: build prompt → gọi LLM → parse & validate ACU.
        Trả về list[ACUOutput] (rỗng nếu LLM lỗi). Dùng bởi cả windowing & two-pass.
        """
        user_prompt = build_acu_user_prompt(
            raw_text_v1=raw_text_v1,
            raw_text_v2=raw_text_v2,
            breadcrumb_v1=breadcrumb_v1,
            breadcrumb_v2=breadcrumb_v2,
            match_type=match_type,
        )

        # S4 — logic-signal hint (Category D awareness), best-effort
        try:
            from .logic_detector import build_logic_hint
            hint = build_logic_hint(raw_text_v1, raw_text_v2)
            if hint:
                user_prompt += hint
        except Exception:
            pass

        # C1/C2 — extra instructions (number enum / second pass)
        if extra_instructions:
            user_prompt += extra_instructions

        try:
            raw_json = await self._acu_llm.chat_json(
                system_prompt=self._acu_system_prompt,
                user_prompt=user_prompt,
            )
        except (ValueError, RuntimeError) as exc:
            logger.error("Tier 1 LLM call failed (pair_id=%s): %s", pair_id, exc)
            return []

        acus_raw = raw_json.get("acus", [])
        if not isinstance(acus_raw, list):
            logger.warning(
                "Tier 1: 'acus' không phải list (pair_id=%s). Type: %s",
                pair_id,
                type(acus_raw).__name__,
            )
            return []

        validated_acus: list[ACUOutput] = []
        for i, acu_dict in enumerate(acus_raw):
            if not isinstance(acu_dict, dict):
                continue
            try:
                validated_acus.append(ACUOutput.model_validate(acu_dict))
            except ValidationError as exc:
                logger.warning(
                    "Tier 1: ACU #%d validation failed (pair_id=%s): %s. Raw: %s",
                    i, pair_id, exc, str(acu_dict)[:200],
                )
                recovered = self._attempt_acu_recovery(acu_dict, exc)
                if recovered:
                    validated_acus.append(recovered)

        return validated_acus

    @staticmethod
    def _attempt_acu_recovery(
        acu_dict: dict[str, Any],
        original_error: ValidationError,
    ) -> ACUOutput | None:
        """
        Cố gắng phục hồi ACU bị lỗi validation bằng cách điền giá trị mặc định.

        Chiến lược: Log field lỗi, set mặc định an toàn, retry.
        """
        try:
            repaired = dict(acu_dict)

            # Fix change_type nếu giá trị không hợp lệ
            valid_types = {ct.value for ct in ChangeType}
            if repaired.get("change_type") not in valid_types:
                logger.warning(
                    "ACU recovery: change_type='%s' không hợp lệ → fallback 'structural'",
                    repaired.get("change_type"),
                )
                repaired["change_type"] = "structural"

            # Fix confidence nếu None hoặc ngoài range
            conf = repaired.get("confidence")
            if conf is None or not (0.0 <= float(conf) <= 1.0):
                repaired["confidence"] = 0.5

            return ACUOutput.model_validate(repaired)
        except Exception:
            return None  # Recovery thất bại → drop ACU

    # ------------------------------------------------------------------
    # Tầng 4: Executive Summary Generation
    # ------------------------------------------------------------------

    async def _tier4_generate_summary(
        self,
        verified_acus: list[ACUOutput],
        request: ComparisonRequest,
    ) -> ExecutiveSummary | None:
        """
        Tầng 4: Gọi LLM để sinh ExecutiveSummary từ các ACU đã pass verification.

        Returns:
            ExecutiveSummary hoặc None nếu có lỗi.
        """
        if not verified_acus:
            logger.info(
                "Tier 4: Không có ACU nào để tóm tắt (pair_id=%s).",
                request.pair_id,
            )
            return ExecutiveSummary(
                overall_assessment=(
                    "Không phát hiện thay đổi nào được xác minh giữa hai phiên bản văn bản."
                ),
                critical_changes=[],
                risk_flags=[],
            )

        user_prompt = build_summary_user_prompt(
            verified_acus=verified_acus,
            v1_doc_id=request.v1_doc_id,
            v2_doc_id=request.v2_doc_id,
            location_context=request.breadcrumb_v1 or request.breadcrumb_v2,
        )

        try:
            raw_json = await self._summary_llm.chat_json(
                system_prompt=SUMMARY_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            summary = ExecutiveSummary.model_validate(raw_json)
            logger.info(
                "Tier 4 summary generated (pair_id=%s): %d critical changes, %d risks.",
                request.pair_id,
                len(summary.critical_changes),
                len(summary.risk_flags),
            )
            return summary

        except ValidationError as exc:
            logger.error(
                "Tier 4 summary validation failed (pair_id=%s): %s",
                request.pair_id,
                exc,
            )
            # Fallback: tạo summary tối giản từ ACU list
            return self._fallback_summary(verified_acus)

        except (ValueError, RuntimeError) as exc:
            logger.error(
                "Tier 4 LLM call failed (pair_id=%s): %s",
                request.pair_id,
                exc,
            )
            return self._fallback_summary(verified_acus)

    @staticmethod
    def _fallback_summary(verified_acus: list[ACUOutput]) -> ExecutiveSummary:
        """
        Tạo summary tối giản (không cần LLM) khi Tầng 4 thất bại.

        Dựa hoàn toàn vào Python logic — không suy diễn.
        """
        from collections import Counter
        type_counts = Counter(acu.change_type.value for acu in verified_acus)

        desc_parts = []
        label_map = {
            "numerical": "thay đổi số liệu",
            "terminology": "thay đổi thuật ngữ",
            "structural": "thay đổi cấu trúc",
            "addition": "nội dung bổ sung",
            "deletion": "nội dung xoá bỏ",
            "reorder": "sắp xếp lại",
        }
        for ct, count in type_counts.most_common():
            desc_parts.append(f"{count} {label_map.get(ct, ct)}")

        overall = (
            f"Phát hiện {len(verified_acus)} thay đổi đã xác minh, bao gồm: "
            + ", ".join(desc_parts) + "."
        )

        critical = [
            f"{acu.change_type.value.upper()}: {acu.original_value} → {acu.new_value}"
            for acu in sorted(verified_acus, key=lambda a: a.confidence, reverse=True)[:5]
            if acu.original_value or acu.new_value
        ]

        numerical_acus = [a for a in verified_acus if a.change_type == ChangeType.NUMERICAL]
        num_summary = ""
        if numerical_acus:
            num_items = [
                f"'{a.original_value}' → '{a.new_value}' ({a.location_v2 or a.location_v1})"
                for a in numerical_acus
            ]
            num_summary = "Các thay đổi số liệu: " + "; ".join(num_items)

        return ExecutiveSummary(
            overall_assessment=overall,
            critical_changes=critical,
            numerical_changes_summary=num_summary,
            risk_flags=[],
            recommendation=(
                "Fallback summary — Khuyến nghị xem xét lại kết quả tóm tắt "
                "do LLM tier 4 gặp lỗi."
            ),
        )
