"""
generative_comparison/comparison_pipeline.py
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
import textwrap
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from generative_comparison.models import (
    ACUOutput,
    ChangeType,
    ComparisonReport,
    ComparisonRequest,
    ExecutiveSummary,
    VerificationResult,
    VerificationStatus,
)
from generative_comparison.llm_manager import LocalLLMClient, LLMConfig
from generative_comparison.verification_engine import (
    VerificationConfig,
    VerificationEngine,
)

logger = logging.getLogger(__name__)


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
        default=0.4,
        ge=0.0,
        le=1.0,
        description=(
            "ACU có confidence < threshold này sẽ bị drop TRƯỚC KHI verification "
            "(pre-filter để tránh noise)"
        ),
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

        # Khởi tạo LLM Client cho ACU extraction (low temperature)
        self._acu_llm = LocalLLMClient(
            config=LLMConfig(
                base_url=self._cfg.llm_base_url,
                model_name=self._cfg.llm_model_name,
                temperature=self._cfg.acu_temperature,
                max_tokens=self._cfg.max_tokens_acu,
            )
        )

        # LLM Client riêng cho Summary (slightly higher temperature)
        self._summary_llm = LocalLLMClient(
            config=LLMConfig(
                base_url=self._cfg.llm_base_url,
                model_name=self._cfg.llm_model_name,
                temperature=self._cfg.summary_temperature,
                max_tokens=self._cfg.max_tokens_summary,
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
        Tầng 1: Gọi LLM với JSON mode để trích xuất danh sách ACU.

        Returns:
            List[ACUOutput] đã được validate qua Pydantic.
            Trả về list rỗng nếu có lỗi.
        """
        user_prompt = build_acu_user_prompt(
            raw_text_v1=request.raw_text_v1,
            raw_text_v2=request.raw_text_v2,
            breadcrumb_v1=request.breadcrumb_v1,
            breadcrumb_v2=request.breadcrumb_v2,
            match_type=request.match_type,
        )

        try:
            raw_json = await self._acu_llm.chat_json(
                system_prompt=ACU_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except (ValueError, RuntimeError) as exc:
            logger.error(
                "Tier 1 LLM call failed cho pair_id=%s: %s",
                request.pair_id,
                exc,
            )
            return []

        # Parse và validate từng ACU
        acus_raw = raw_json.get("acus", [])
        if not isinstance(acus_raw, list):
            logger.warning(
                "Tier 1: LLM trả về 'acus' không phải list cho pair_id=%s. "
                "Actual type: %s",
                request.pair_id,
                type(acus_raw).__name__,
            )
            return []

        validated_acus: list[ACUOutput] = []
        for i, acu_dict in enumerate(acus_raw):
            if not isinstance(acu_dict, dict):
                logger.warning(
                    "Tier 1: ACU #%d không phải dict, bỏ qua. pair_id=%s",
                    i,
                    request.pair_id,
                )
                continue
            try:
                acu = ACUOutput.model_validate(acu_dict)
                validated_acus.append(acu)
            except ValidationError as exc:
                logger.warning(
                    "Tier 1: ACU #%d validation failed cho pair_id=%s: %s. "
                    "Raw dict: %s",
                    i,
                    request.pair_id,
                    exc,
                    str(acu_dict)[:200],
                )
                # Cố gắng recovery: bỏ field lỗi và thử lại
                recovered = self._attempt_acu_recovery(acu_dict, exc)
                if recovered:
                    validated_acus.append(recovered)

        logger.info(
            "Tier 1 extracted %d/%d ACUs (pair_id=%s)",
            len(validated_acus),
            len(acus_raw),
            request.pair_id,
        )

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
