"""
src/comparison/verifier.py
==============================================
VerificationEngine — Tầng 2 & Tầng 3 trong Zero-Hallucination Pipeline.

Kiến trúc 2 tầng:

  Tầng 2 — Self-Verification (Evidence Check):
    Kiểm tra verbatim_evidence_v1 và verbatim_evidence_v2 có THỰC SỰ
    xuất hiện trong raw_text gốc hay không.
    → Dùng: Python `in` operator (exact) + SequenceMatcher (fuzzy fallback)
    → Ngưỡng fuzzy: configurable (mặc định 0.85)
    → Kết quả: PASSED / FAILED_EVIDENCE

  Tầng 3 — Numerical Verification:
    Chỉ áp dụng với ACU có change_type=ChangeType.NUMERICAL.
    → Dùng: Regex chuyên dụng trích xuất tất cả số liệu từ original_value
      và new_value, sau đó đối chiếu ngược với raw_text.
    → Yêu cầu: Phải tìm thấy ĐÚNG 100% số liệu trong raw_text.
    → Kết quả: PASSED / FAILED_NUMERICAL

Quy trình:
    ACU  →  [Tầng 2: Evidence Check]  →  [Tầng 3: Numerical if applicable]
          ↓ PASS                      ↓ PASS
         VerificationResult(PASSED)  VerificationResult(PASSED)
          ↓ FAIL                      ↓ FAIL
         VerificationResult(FAILED_*)  (bị drop khỏi final report)
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import NamedTuple

from pydantic import BaseModel, Field

from .models import (
    ACUOutput,
    ChangeType,
    VerificationResult,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class VerificationConfig(BaseModel):
    """Cấu hình cho VerificationEngine."""

    # Tầng 2 — Evidence check
    fuzzy_match_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=(
            "Ngưỡng tối thiểu của SequenceMatcher ratio để chấp nhận fuzzy match. "
            "0.85 = 85% tương đồng ký tự. "
            "Giảm xuống 0.80 nếu văn bản có lỗi OCR."
        ),
    )
    min_evidence_length: int = Field(
        default=5,
        gt=0,
        description=(
            "Độ dài tối thiểu (ký tự) của evidence để thực hiện verification. "
            "Evidence ngắn hơn → SKIPPED (tránh false positives)."
        ),
    )
    normalize_whitespace: bool = Field(
        default=True,
        description=(
            "Chuẩn hoá khoảng trắng trước khi so sánh "
            "(xử lý lỗi OCR / PDF extraction)."
        ),
    )

    # Tầng 3 — Numerical verification
    strict_numerical: bool = Field(
        default=True,
        description=(
            "True = yêu cầu 100%% số liệu phải tìm thấy trong raw_text. "
            "False = dùng heuristic linh hoạt hơn (không khuyến nghị)."
        ),
    )
    numerical_context_window: int = Field(
        default=50,
        gt=0,
        description=(
            "Số ký tự xung quanh mỗi số liệu để tìm trong raw_text "
            "(giúp xác định đúng ngữ cảnh số)."
        ),
    )


# ---------------------------------------------------------------------------
# Internal data types
# ---------------------------------------------------------------------------


class ExtractedNumber(NamedTuple):
    """Số liệu được trích xuất bởi Regex."""
    raw_str: str          # Chuỗi số gốc (vd: "30%", "500.000 đồng", "01/06/2025")
    normalized: str       # Dạng chuẩn hoá (vd: "30", "500000", "01062025")
    num_type: str         # Loại: "percentage", "currency", "date", "integer", "float"
    value: float | None   # Giá trị số học nếu convert được


# ---------------------------------------------------------------------------
# Regex patterns cho Numerical Extraction
# ---------------------------------------------------------------------------

# Pattern phức hợp — thứ tự quan trọng: ưu tiên pattern dài hơn
_NUMERICAL_PATTERNS: list[tuple[str, str]] = [
    # Ngày tháng: DD/MM/YYYY, DD-MM-YYYY, YYYY/MM/DD
    (
        r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b",
        "date",
    ),
    # Phần trăm: 10%, 10,5%, 0.05%
    (
        r"\b(\d+(?:[,\.]\d+)?)\s*%",
        "percentage",
    ),
    # Tiền tệ VN: 500.000 đồng, 1.000.000 VNĐ, 500,000 VND
    (
        r"\b(\d{1,3}(?:[\.]\d{3})+(?:[,]\d+)?)\s*(?:đồng|VNĐ|VND|đ\b)?",
        "currency_vn",
    ),
    # Tiền tệ quốc tế: USD 1,000 / $500.00
    (
        r"(?:USD|EUR|GBP|\$|€|£)\s*(\d{1,3}(?:[,\.]\d{3})+(?:\.\d+)?)",
        "currency_intl",
    ),
    # Số thập phân (đặt sau currency để không overlap)
    (
        r"\b(\d+[,\.]\d+)\b",
        "float",
    ),
    # Số nguyên thuần túy (>= 2 chữ số để tránh false positive với 'a', 'b', etc.)
    (
        r"\b(\d{2,})\b",
        "integer",
    ),
    # Số nguyên 1 chữ số standalone (ít quan trọng hơn)
    (
        r"\b([1-9])\b",
        "integer_single",
    ),
]

_COMPILED_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(pat, re.UNICODE), num_type)
    for pat, num_type in _NUMERICAL_PATTERNS
]


# ---------------------------------------------------------------------------
# VerificationEngine
# ---------------------------------------------------------------------------


class VerificationEngine:
    """
    Engine xác minh chống ảo giác (Anti-Hallucination) cho các ACU.

    Workflow:
        result = engine.verify(acu, raw_text_v1, raw_text_v2)
        if result.is_passed:
            # đưa vào final report
        else:
            # drop, log rejection_reason
    """

    def __init__(self, config: VerificationConfig | None = None) -> None:
        self._cfg = config or VerificationConfig()
        logger.info(
            "VerificationEngine khởi tạo: fuzzy_threshold=%.2f, strict_numerical=%s",
            self._cfg.fuzzy_match_threshold,
            self._cfg.strict_numerical,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        acu: ACUOutput,
        raw_text_v1: str,
        raw_text_v2: str,
    ) -> VerificationResult:
        """
        Xác minh một ACU qua Tầng 2 (evidence) và Tầng 3 (numerical).

        Args:
            acu:          ACU cần kiểm tra.
            raw_text_v1:  Nội dung text thuần của phiên bản V1.
            raw_text_v2:  Nội dung text thuần của phiên bản V2.

        Returns:
            VerificationResult với trường status cho biết kết quả.
        """
        result = VerificationResult(acu=acu)

        # ---- Tầng 2: Evidence Verification ----
        result = self._verify_evidence(result, raw_text_v1, raw_text_v2)
        if not result.is_passed:
            return result

        # ---- Tầng 3: Numerical Verification (chỉ nếu applicable) ----
        if acu.change_type == ChangeType.NUMERICAL:
            result = self._verify_numerical(result, raw_text_v1, raw_text_v2)

        return result

    def verify_batch(
        self,
        acus: list[ACUOutput],
        raw_text_v1: str,
        raw_text_v2: str,
    ) -> list[VerificationResult]:
        """
        Xác minh danh sách ACU (synchronous batch).

        Returns:
            List[VerificationResult] tương ứng với mỗi ACU đầu vào.
        """
        results: list[VerificationResult] = []
        for acu in acus:
            try:
                r = self.verify(acu, raw_text_v1, raw_text_v2)
                results.append(r)
            except Exception as exc:
                logger.exception(
                    "Lỗi khi verify ACU %s: %s. Marking as FAILED_EVIDENCE.",
                    acu.acu_id,
                    exc,
                )
                results.append(
                    VerificationResult(
                        acu=acu,
                        status=VerificationStatus.FAILED_EVIDENCE,
                        rejection_reason=f"Exception during verification: {exc}",
                    )
                )
        return results

    # ------------------------------------------------------------------
    # Tầng 2: Evidence Verification (Self-Check)
    # ------------------------------------------------------------------

    def _verify_evidence(
        self,
        result: VerificationResult,
        raw_text_v1: str,
        raw_text_v2: str,
    ) -> VerificationResult:
        """
        Tầng 2: Kiểm tra verbatim evidence CÓ NẰM trong raw_text hay không.

        Rules per change_type:
          - ADDITION : chỉ verify evidence_v2 (evidence_v1 bỏ qua)
          - DELETION : chỉ verify evidence_v1 (evidence_v2 bỏ qua)
          - Others   : verify cả 2 evidence
        """
        acu = result.acu
        ct = acu.change_type

        # ---- Xác định cần verify evidence nào ----
        check_v1 = ct not in (ChangeType.ADDITION,)
        check_v2 = ct not in (ChangeType.DELETION,)

        # ---- Verify V1 evidence ----
        v1_ok: bool | None = None
        fuzzy_v1: float | None = None
        if check_v1:
            if not acu.verbatim_evidence_v1 or len(acu.verbatim_evidence_v1) < self._cfg.min_evidence_length:
                # Evidence rỗng hoặc quá ngắn → SKIP (tránh false positive)
                v1_ok = True  # Treat as SKIPPED = PASS
                logger.debug(
                    "ACU %s: evidence_v1 quá ngắn (%d chars), bỏ qua verify.",
                    acu.acu_id,
                    len(acu.verbatim_evidence_v1),
                )
            else:
                v1_ok, fuzzy_v1 = self._evidence_in_text(
                    evidence=acu.verbatim_evidence_v1,
                    text=raw_text_v1,
                )
                if not v1_ok:
                    logger.warning(
                        "ACU %s: HALLUCINATION DETECTED — evidence_v1 không tìm thấy "
                        "trong raw_text_v1. Evidence: %r (first 100 chars). "
                        "Fuzzy score: %.3f",
                        acu.acu_id,
                        acu.verbatim_evidence_v1[:100],
                        fuzzy_v1 or 0.0,
                    )
        else:
            v1_ok = None  # Không applicable

        # ---- Verify V2 evidence ----
        v2_ok: bool | None = None
        fuzzy_v2: float | None = None
        if check_v2:
            if not acu.verbatim_evidence_v2 or len(acu.verbatim_evidence_v2) < self._cfg.min_evidence_length:
                v2_ok = True  # SKIP = PASS
                logger.debug(
                    "ACU %s: evidence_v2 quá ngắn (%d chars), bỏ qua verify.",
                    acu.acu_id,
                    len(acu.verbatim_evidence_v2),
                )
            else:
                v2_ok, fuzzy_v2 = self._evidence_in_text(
                    evidence=acu.verbatim_evidence_v2,
                    text=raw_text_v2,
                )
                if not v2_ok:
                    logger.warning(
                        "ACU %s: HALLUCINATION DETECTED — evidence_v2 không tìm thấy "
                        "trong raw_text_v2. Evidence: %r (first 100 chars). "
                        "Fuzzy score: %.3f",
                        acu.acu_id,
                        acu.verbatim_evidence_v2[:100],
                        fuzzy_v2 or 0.0,
                    )
        else:
            v2_ok = None  # Không applicable

        # ---- Tổng hợp kết quả ----
        result.evidence_v1_found = v1_ok
        result.evidence_v2_found = v2_ok
        result.fuzzy_match_score_v1 = fuzzy_v1
        result.fuzzy_match_score_v2 = fuzzy_v2

        # Fail nếu bất kỳ evidence nào bị reject
        if (v1_ok is False) or (v2_ok is False):
            failed_sides = []
            if v1_ok is False:
                failed_sides.append(f"V1 (score={fuzzy_v1:.3f})")
            if v2_ok is False:
                failed_sides.append(f"V2 (score={fuzzy_v2:.3f})")

            result.status = VerificationStatus.FAILED_EVIDENCE
            result.rejection_reason = (
                f"Evidence không tìm thấy trong raw_text: {', '.join(failed_sides)}. "
                f"ACU bị đánh dấu là hallucination."
            )

        return result

    def _evidence_in_text(
        self,
        evidence: str,
        text: str,
    ) -> tuple[bool, float | None]:
        """
        Kiểm tra evidence có trong text không.

        Bước 1: Exact match (nhanh, O(n)).
        Bước 2: Normalised whitespace match.
        Bước 3: Fuzzy sliding-window match (chậm hơn, dùng làm tiebreak).

        Returns:
            (is_found: bool, fuzzy_score: float | None)
            fuzzy_score chỉ có giá trị nếu exact match thất bại.
        """
        # Bước 1: Exact match
        if evidence in text:
            return True, None

        # Bước 2: Whitespace-normalised match
        if self._cfg.normalize_whitespace:
            normed_evidence = self._normalize_ws(evidence)
            normed_text = self._normalize_ws(text)
            if normed_evidence in normed_text:
                return True, None

        # Bước 3: Fuzzy sliding window match
        fuzzy_score = self._fuzzy_evidence_match(evidence, text)
        if fuzzy_score >= self._cfg.fuzzy_match_threshold:
            logger.debug(
                "Evidence khớp qua fuzzy match: score=%.3f (ngưỡng=%.3f)",
                fuzzy_score,
                self._cfg.fuzzy_match_threshold,
            )
            return True, fuzzy_score

        return False, fuzzy_score

    @staticmethod
    def _normalize_ws(text: str) -> str:
        """Chuẩn hoá khoảng trắng: nhiều space → 1 space, strip."""
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _fuzzy_evidence_match(evidence: str, text: str) -> float:
        """
        Tìm max fuzzy similarity giữa evidence và mọi cửa sổ độ dài tương đương trong text.

        SequenceMatcher.find_longest_match tiếp cận tốt hơn full-string comparison
        vì evidence thường là sub-string của text.

        Optimisation: Chỉ dùng SequenceMatcher trên window có độ dài ±50% so với evidence.
        """
        ev_len = len(evidence)
        if ev_len == 0:
            return 0.0

        # Giới hạn text length để tránh O(n²) quá lớn
        text_to_search = text[:10_000] if len(text) > 10_000 else text

        # Window size: từ 80% đến 150% độ dài evidence
        min_win = max(1, int(ev_len * 0.8))
        max_win = min(len(text_to_search), int(ev_len * 1.5))

        best_ratio = 0.0
        step = max(1, ev_len // 4)  # Stride để tránh quá chậm

        for start in range(0, len(text_to_search) - min_win + 1, step):
            for win_len in range(min_win, min(max_win, len(text_to_search) - start) + 1):
                window = text_to_search[start: start + win_len]
                ratio = SequenceMatcher(None, evidence, window, autojunk=False).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                if best_ratio >= 1.0:
                    return 1.0  # Perfect match → early exit

        return best_ratio

    # ------------------------------------------------------------------
    # Tầng 3: Numerical Verification
    # ------------------------------------------------------------------

    def _verify_numerical(
        self,
        result: VerificationResult,
        raw_text_v1: str,
        raw_text_v2: str,
    ) -> VerificationResult:
        """
        Tầng 3: Xác minh số liệu trong ACU có khớp với raw_text không.

        Quy trình:
          1. Trích xuất tất cả số liệu từ original_value → numbers_v1
          2. Trích xuất tất cả số liệu từ new_value → numbers_v2
          3. Mỗi số trong numbers_v1 phải xuất hiện trong raw_text_v1
          4. Mỗi số trong numbers_v2 phải xuất hiện trong raw_text_v2
          5. Nếu bất kỳ số nào không tìm thấy → FAILED_NUMERICAL

        Trong strict mode: 100% số phải match.
        """
        acu = result.acu

        # --- Trích xuất số từ original_value và new_value ---
        numbers_from_original = extract_numbers(acu.original_value or "")
        numbers_from_new = extract_numbers(acu.new_value or "")

        failed_checks: list[str] = []

        # --- Verify V1 numbers trong raw_text_v1 ---
        for num in numbers_from_original:
            found = self._number_in_raw_text(num, raw_text_v1)
            if not found:
                if self._cfg.strict_numerical:
                    failed_checks.append(
                        f"Số '{num.raw_str}' (original_value) không tìm thấy trong V1"
                    )
                else:
                    logger.warning(
                        "ACU %s: Số '%s' không tìm thấy trong V1 raw_text (non-strict mode).",
                        acu.acu_id,
                        num.raw_str,
                    )

        # --- Verify V2 numbers trong raw_text_v2 ---
        for num in numbers_from_new:
            found = self._number_in_raw_text(num, raw_text_v2)
            if not found:
                if self._cfg.strict_numerical:
                    failed_checks.append(
                        f"Số '{num.raw_str}' (new_value) không tìm thấy trong V2"
                    )
                else:
                    logger.warning(
                        "ACU %s: Số '%s' không tìm thấy trong V2 raw_text (non-strict mode).",
                        acu.acu_id,
                        num.raw_str,
                    )

        # --- Tổng hợp ---
        if failed_checks:
            result.status = VerificationStatus.FAILED_NUMERICAL
            result.numerical_verified = False
            result.rejection_reason = (
                f"Numerical verification thất bại: "
                + "; ".join(failed_checks)
            )
            logger.error(
                "ACU %s FAILED numerical verification: %s",
                acu.acu_id,
                result.rejection_reason,
            )
        else:
            result.numerical_verified = True
            logger.debug(
                "ACU %s: Numerical verification PASSED (%d V1 nums, %d V2 nums).",
                acu.acu_id,
                len(numbers_from_original),
                len(numbers_from_new),
            )

        return result

    def _number_in_raw_text(self, num: ExtractedNumber, text: str) -> bool:
        """
        Kiểm tra một số liệu có xuất hiện trong raw_text không.

        Chiến lược (theo thứ tự ưu tiên):
          1. Exact raw_str match
          2. Normalised raw_str match (bỏ khoảng trắng, chuẩn hoá dấu chấm/phẩy)
          3. Pattern match: tìm số có cùng giá trị số học (nếu value không None)
        """
        # Bước 1: Exact match
        if num.raw_str in text:
            return True

        # Bước 2: Normalised match (bỏ separator)
        normed = re.sub(r"[\s\.]", "", num.raw_str).replace(",", ".")
        normed_text = re.sub(r"[\s]", "", text)
        if normed in normed_text:
            return True

        # Bước 3: Value-based match (nếu có thể parse thành số)
        if num.value is not None and num.num_type not in ("date", "integer_single"):
            # Tạo regex tìm số có cùng magnitude trong text
            value_str = str(int(num.value)) if num.value == int(num.value) else str(num.value)
            pattern = re.escape(value_str)
            if re.search(pattern, re.sub(r"[\.,]", "", text)):
                return True

        return False


# ---------------------------------------------------------------------------
# Public utility: extract_numbers
# ---------------------------------------------------------------------------


def extract_numbers(text: str) -> list[ExtractedNumber]:
    """
    Trích xuất tất cả số liệu từ một chuỗi văn bản.

    Hỗ trợ: ngày tháng, phần trăm, tiền tệ, số nguyên, số thập phân.
    Sử dụng multi-pattern với ưu tiên từ phức tạp đến đơn giản.

    Args:
        text: Chuỗi cần trích xuất (thường là original_value / new_value)

    Returns:
        List[ExtractedNumber] — có thể rỗng nếu không có số nào.

    Example:
        >>> extract_numbers("Tăng từ 30% lên 45% trong vòng 90 ngày")
        [
            ExtractedNumber(raw_str="30%", num_type="percentage", value=30.0, ...),
            ExtractedNumber(raw_str="45%", num_type="percentage", value=45.0, ...),
            ExtractedNumber(raw_str="90", num_type="integer", value=90.0, ...),
        ]
    """
    if not text:
        return []

    extracted: list[ExtractedNumber] = []
    covered_spans: set[tuple[int, int]] = set()  # Tránh duplicate từ overlapping patterns

    for pattern, num_type in _COMPILED_PATTERNS:
        for match in pattern.finditer(text):
            span = match.span()

            # Kiểm tra overlap với spans đã captured
            if any(
                not (span[1] <= covered_start or span[0] >= covered_end)
                for covered_start, covered_end in covered_spans
            ):
                continue

            raw_str = match.group(0).strip()
            # Lấy captured group nếu có, fallback sang toàn bộ match
            captured = match.group(1) if match.lastindex else raw_str

            normalized = _normalize_number(captured, num_type)
            value = _parse_numeric_value(captured, num_type)

            extracted.append(
                ExtractedNumber(
                    raw_str=raw_str,
                    normalized=normalized,
                    num_type=num_type,
                    value=value,
                )
            )
            covered_spans.add(span)

    return extracted


def _normalize_number(raw: str, num_type: str) -> str:
    """Chuẩn hoá số về dạng đơn giản để so sánh."""
    # Bỏ tất cả dấu phân cách (., ,) trừ dấu thập phân cuối cùng
    cleaned = re.sub(r"[^\d,\.]", "", raw)
    if num_type == "date":
        return re.sub(r"[^0-9]", "", raw)
    return cleaned


def _parse_numeric_value(raw: str, num_type: str) -> float | None:
    """Parse raw_str thành giá trị float nếu có thể."""
    if num_type == "date":
        return None  # Ngày không parse thành float

    try:
        # Chuẩn hoá: Việt Nam dùng dấu . làm thousands sep, , làm decimal
        # Quốc tế thì ngược lại → phải xử lý cả 2 trường hợp
        cleaned = raw.strip()

        # Bỏ currency symbol và unit
        cleaned = re.sub(r"[^\d,\.]", "", cleaned)

        if not cleaned:
            return None

        # Nếu có cả , và . → xác định decimal separator
        if "," in cleaned and "." in cleaned:
            # VN format: 1.000.000,50 → dot là thousands, comma là decimal
            last_comma = cleaned.rfind(",")
            last_dot = cleaned.rfind(".")
            if last_comma > last_dot:
                # VN format
                cleaned = cleaned.replace(".", "").replace(",", ".")
            else:
                # International format: 1,000,000.50
                cleaned = cleaned.replace(",", "")
        elif "," in cleaned:
            # Có thể là thousands (1,000) hoặc decimal (1,5)
            parts = cleaned.split(",")
            if len(parts) == 2 and len(parts[1]) <= 2:
                # Decimal: 1,5 → 1.5
                cleaned = cleaned.replace(",", ".")
            else:
                # Thousands: 1,000,000
                cleaned = cleaned.replace(",", "")

        return float(cleaned)
    except (ValueError, AttributeError):
        return None
