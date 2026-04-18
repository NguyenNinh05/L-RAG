"""
generative_comparison/models.py
================================
Pydantic data models cho Phase 3 — Generative Comparison.

Hierarchy:
    ComparisonRequest        — Đầu vào pipeline (DiffPair + raw texts)
    ACUOutput                — Atomic Comparison Unit (1 thay đổi nguyên tử)
    VerificationResult       — Kết quả sau Tầng 2 & 3 verification
    ExecutiveSummary         — Tóm tắt tiếng Việt từ LLM (Tầng 4)
    ComparisonReport         — Đầu ra cuối: machine_readable + human_readable

Không import vòng — chỉ import từ comparison.models (Phase 2).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums Phase 3
# ---------------------------------------------------------------------------


class ChangeType(str, Enum):
    """
    Loại thay đổi nguyên tử được phát hiện bởi LLM.

    - numerical    : Thay đổi con số / giá trị định lượng (ngày, %, tiền tệ...)
    - terminology  : Thay đổi thuật ngữ / từ ngữ pháp lý quan trọng
    - structural   : Thay đổi cấu trúc câu / điều khoản (thêm/xoá mệnh đề phụ)
    - addition     : Nội dung hoàn toàn mới được thêm vào V2
    - deletion     : Nội dung trong V1 bị xoá khỏi V2
    - reorder      : Thứ tự nội dung bị sắp xếp lại (không đổi nội dung)
    """
    NUMERICAL = "numerical"
    TERMINOLOGY = "terminology"
    STRUCTURAL = "structural"
    ADDITION = "addition"
    DELETION = "deletion"
    REORDER = "reorder"


class VerificationStatus(str, Enum):
    """Trạng thái sau khi qua Verification Engine."""
    PASSED = "passed"               # Vượt qua tất cả các tầng xác minh
    FAILED_EVIDENCE = "failed_evidence"    # Hallucination: evidence không tìm thấy trong text gốc
    FAILED_NUMERICAL = "failed_numerical"  # Số liệu không khớp với văn bản gốc
    SKIPPED = "skipped"             # Bỏ qua (vd: change_type=addition không cần v1 evidence)


# ---------------------------------------------------------------------------
# Core: ACUOutput — Atomic Comparison Unit
# ---------------------------------------------------------------------------


class ACUOutput(BaseModel):
    """
    Atomic Comparison Unit — đơn vị thay đổi nguyên tử nhỏ nhất.

    Mỗi ACU đại diện cho MỘT thay đổi cụ thể, có thể xác minh độc lập.
    LLM phải trả về danh sách các ACU cho mỗi cặp điều khoản được so sánh.

    ⚠️  CRITICAL: verbatim_evidence_v1 và verbatim_evidence_v2 phải là
        trích dẫn NGUYÊN VĂN từ raw_text — không paraphrase, không tóm tắt.
        Đây là nền tảng của Zero-Hallucination approach.
    """

    acu_id: str = Field(
        default_factory=lambda: f"acu_{uuid.uuid4().hex[:12]}",
        description="ID duy nhất của ACU này",
    )

    # --- Phân loại ---
    change_type: ChangeType = Field(
        ...,
        description=(
            "Loại thay đổi: numerical/terminology/structural/addition/deletion/reorder"
        ),
    )

    # --- Vị trí trong văn bản ---
    location_v1: str = Field(
        default="",
        description=(
            "Vị trí trong V1: 'Điều X, Khoản Y, Điểm Z' hoặc rỗng nếu là addition"
        ),
    )
    location_v2: str = Field(
        default="",
        description=(
            "Vị trí trong V2: 'Điều X, Khoản Y, Điểm Z' hoặc rỗng nếu là deletion"
        ),
    )

    # --- Giá trị thay đổi ---
    original_value: str = Field(
        default="",
        description=(
            "Giá trị / cụm từ GỐC trong V1 (ngắn gọn, chính xác). "
            "Rỗng nếu change_type=addition."
        ),
    )
    new_value: str = Field(
        default="",
        description=(
            "Giá trị / cụm từ MỚI trong V2 (ngắn gọn, chính xác). "
            "Rỗng nếu change_type=deletion."
        ),
    )

    # --- CRITICAL: Evidence — Bằng chứng nguyên văn (Zero-Hallucination) ---
    verbatim_evidence_v1: str = Field(
        default="",
        description=(
            "[CRITICAL] Trích dẫn NGUYÊN VĂN từ V1 chứa original_value. "
            "PHẢI copy-paste chính xác từ <v1_text>. "
            "Rỗng CHÍNH XÁC KHI VÀ CHỈ KHI change_type=addition."
        ),
    )
    verbatim_evidence_v2: str = Field(
        default="",
        description=(
            "[CRITICAL] Trích dẫn NGUYÊN VĂN từ V2 chứa new_value. "
            "PHẢI copy-paste chính xác từ <v2_text>. "
            "Rỗng CHÍNH XÁC KHI VÀ CHỈ KHI change_type=deletion."
        ),
    )

    # --- Confidence ---
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Mức độ tự tin của LLM về ACU này [0.0, 1.0]. "
            "Dưới 0.5 → nên xem xét thủ công."
        ),
    )

    # --- Metadata (được điền bởi pipeline, không phải LLM) ---
    pair_id: str = Field(
        default="",
        description="ID của DiffPair (Phase 2) mà ACU này thuộc về",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: Any) -> float:
        """Clamp confidence về [0.0, 1.0] nếu LLM trả về ngoài range."""
        try:
            return max(0.0, min(1.0, float(v)))
        except (ValueError, TypeError):
            return 0.5

    @model_validator(mode="after")
    def _validate_evidence_consistency(self) -> "ACUOutput":
        """
        Xác nhận evidence nhất quán với change_type.

        Rule:
          - addition  → verbatim_evidence_v1 phải rỗng
          - deletion  → verbatim_evidence_v2 phải rỗng
          - Các type khác → cả 2 evidence phải có giá trị
        """
        ct = self.change_type
        if ct == ChangeType.ADDITION and self.verbatim_evidence_v1:
            # LLM đôi khi vẫn thêm evidence_v1 cho addition — cho phép nhưng log warning
            pass  # Sẽ được xử lý bởi VerificationEngine
        if ct == ChangeType.DELETION and self.verbatim_evidence_v2:
            pass  # Tương tự
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize sang dict cho JSON export."""
        d = self.model_dump(mode="json")
        d["change_type"] = self.change_type.value
        return d


# ---------------------------------------------------------------------------
# VerificationResult — Kết quả sau Verification Engine
# ---------------------------------------------------------------------------


class VerificationResult(BaseModel):
    """
    Wrapper gói ACUOutput cùng với kết quả verification.

    Đây là output của VerificationEngine (Module 2).
    Chỉ các ACU có status=PASSED mới được đưa vào report cuối.
    """

    acu: ACUOutput = Field(..., description="ACU gốc được xác minh")
    status: VerificationStatus = Field(
        default=VerificationStatus.PASSED,
        description="Kết quả verification",
    )
    evidence_v1_found: bool | None = Field(
        default=None,
        description="True nếu verbatim_evidence_v1 tìm thấy trong raw_text_v1",
    )
    evidence_v2_found: bool | None = Field(
        default=None,
        description="True nếu verbatim_evidence_v2 tìm thấy trong raw_text_v2",
    )
    numerical_verified: bool | None = Field(
        default=None,
        description="True nếu số liệu extracted đúng 100% với raw_text (chỉ cho numerical)",
    )
    rejection_reason: str = Field(
        default="",
        description="Lý do reject nếu status != PASSED",
    )
    fuzzy_match_score_v1: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Điểm fuzzy match cho evidence_v1 (nếu exact match thất bại)",
    )
    fuzzy_match_score_v2: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Điểm fuzzy match cho evidence_v2 (nếu exact match thất bại)",
    )

    @property
    def is_passed(self) -> bool:
        return self.status == VerificationStatus.PASSED


# ---------------------------------------------------------------------------
# ExecutiveSummary — Tóm tắt điều hành (Tầng 4)
# ---------------------------------------------------------------------------


class ExecutiveSummary(BaseModel):
    """
    Tóm tắt điều hành bằng tiếng Việt do LLM sinh ra ở Tầng 4.

    Quy tắc bắt buộc khi LLM tạo summary này:
      - CHỈ dựa vào các ACU đã PASSED verification
      - KHÔNG suy diễn, KHÔNG thêm thông tin ngoài văn bản
      - Viết bằng tiếng Việt, văn phong pháp lý trang trọng
    """

    overall_assessment: str = Field(
        ...,
        description=(
            "Đánh giá tổng thể: mức độ thay đổi (đáng kể/nhỏ/rất lớn) "
            "và tính chất chung của các thay đổi"
        ),
    )
    critical_changes: list[str] = Field(
        default_factory=list,
        description=(
            "Danh sách các thay đổi quan trọng nhất (ảnh hưởng lớn đến "
            "quyền/nghĩa vụ các bên). Mỗi phần tử là 1 câu ngắn gọn."
        ),
    )
    numerical_changes_summary: str = Field(
        default="",
        description=(
            "Tóm tắt riêng các thay đổi số liệu (ngày, %, tiền, thời hạn...)"
        ),
    )
    risk_flags: list[str] = Field(
        default_factory=list,
        description=(
            "Các điểm rủi ro pháp lý cần chú ý đặc biệt "
            "(chỉ nêu nếu có bằng chứng rõ ràng trong ACU)"
        ),
    )
    recommendation: str = Field(
        default="",
        description=(
            "Khuyến nghị xem xét thủ công (nếu có ACU confidence thấp hoặc "
            "thay đổi cấu trúc phức tạp)"
        ),
    )

    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# ComparisonReport — Đầu ra cuối cùng của Phase 3
# ---------------------------------------------------------------------------


class ComparisonReport(BaseModel):
    """
    Báo cáo so sánh hoàn chỉnh — đầu ra cuối cùng của Phase 3.

    Gồm 2 lớp:
      - machine_readable : List[ACUOutput] đã verified → dùng cho downstream systems
      - human_readable   : Markdown string → render trực tiếp cho người dùng
    """

    report_id: str = Field(
        default_factory=lambda: f"report_{uuid.uuid4().hex[:12]}",
        description="ID duy nhất của báo cáo",
    )

    # --- Input metadata ---
    pair_id: str = Field(default="", description="DiffPair ID từ Phase 2")
    v1_doc_id: str = Field(default="", description="doc_id của tài liệu V1")
    v2_doc_id: str = Field(default="", description="doc_id của tài liệu V2")
    location_context: str = Field(
        default="",
        description="Ngữ cảnh vị trí (vd: 'Điều 5, Khoản 2')",
    )

    # --- Machine-readable output ---
    verified_acus: list[ACUOutput] = Field(
        default_factory=list,
        description="[machine_readable] Danh sách ACU đã qua verification (status=PASSED)",
    )
    rejected_acus: list[VerificationResult] = Field(
        default_factory=list,
        description="[machine_readable] Các ACU bị reject (hallucination / số liệu sai)",
    )

    # --- Human-readable output ---
    executive_summary: ExecutiveSummary | None = Field(
        default=None,
        description="[human_readable] Tóm tắt điều hành tiếng Việt từ Tầng 4",
    )
    markdown_report: str = Field(
        default="",
        description="[human_readable] Báo cáo Markdown đầy đủ sẵn sàng render",
    )

    # --- Stats ---
    total_acus_raw: int = Field(default=0, ge=0, description="Tổng số ACU LLM sinh ra")
    total_acus_passed: int = Field(default=0, ge=0, description="Số ACU qua verification")
    total_acus_rejected: int = Field(default=0, ge=0, description="Số ACU bị reject")
    hallucination_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Tỷ lệ ACU bị reject do hallucination [0, 1]",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @model_validator(mode="after")
    def _compute_stats(self) -> "ComparisonReport":
        """Tính toán thống kê từ danh sách ACU."""
        self.total_acus_passed = len(self.verified_acus)
        self.total_acus_rejected = len(self.rejected_acus)
        self.total_acus_raw = self.total_acus_passed + self.total_acus_rejected
        if self.total_acus_raw > 0:
            self.hallucination_rate = self.total_acus_rejected / self.total_acus_raw
        return self

    def to_machine_readable(self) -> dict[str, Any]:
        """Export machine-readable layer (JSON-serializable dict)."""
        return {
            "report_id": self.report_id,
            "pair_id": self.pair_id,
            "v1_doc_id": self.v1_doc_id,
            "v2_doc_id": self.v2_doc_id,
            "location_context": self.location_context,
            "stats": {
                "total_raw": self.total_acus_raw,
                "passed": self.total_acus_passed,
                "rejected": self.total_acus_rejected,
                "hallucination_rate": round(self.hallucination_rate, 4),
            },
            "verified_acus": [acu.to_dict() for acu in self.verified_acus],
            "rejected_acus": [
                {
                    "acu": r.acu.to_dict(),
                    "status": r.status.value,
                    "reason": r.rejection_reason,
                }
                for r in self.rejected_acus
            ],
            "created_at": self.created_at.isoformat(),
        }

    def to_human_readable(self) -> str:
        """Export human-readable Markdown report."""
        return self.markdown_report


# ---------------------------------------------------------------------------
# ComparisonRequest — Đầu vào của pipeline (internal use)
# ---------------------------------------------------------------------------


class ComparisonRequest(BaseModel):
    """
    Đầu vào cho GenerativeComparisonPipeline.

    Được tạo từ một DiffPair (Phase 2) + raw_text của từng phiên bản.
    """

    pair_id: str = Field(..., description="DiffPair.pair_id từ Phase 2")
    match_type: str = Field(..., description="MatchType value: matched/added/deleted/split/merged")

    # Raw text của từng phiên bản (để LLM so sánh và để Verify)
    raw_text_v1: str = Field(default="", description="Nội dung text thuần của V1 (rỗng nếu ADDED)")
    raw_text_v2: str = Field(default="", description="Nội dung text thuần của V2 (rỗng nếu DELETED)")

    # Metadata vị trí
    breadcrumb_v1: str = Field(default="", description="Breadcrumb V1: '[Chương II > Điều 5]'")
    breadcrumb_v2: str = Field(default="", description="Breadcrumb V2: '[Chương II > Điều 5]'")

    # Doc IDs
    v1_doc_id: str = Field(default="")
    v2_doc_id: str = Field(default="")
