from __future__ import annotations

import asyncio
import difflib
import json
import logging
import re
from datetime import datetime
from typing import Any

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from comparison import build_comparison_result
from comparison.models import ChangeRecord, ClauseResult, ComparisonResult
from config import (
    LLM_MAX_TOKENS as MAX_TOKENS,
    LLM_NUM_CTX,
    LLM_PRESENCE_PENALTY,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_MAX_CONCURRENT,
    OLLAMA_CHAT_URL as OLLAMA_API_URL,
    OLLAMA_LLM_MODEL as OLLAMA_MODEL,
)

logger = logging.getLogger(__name__)

SUPPORT_WINDOW_ANOMALY_RE = re.compile(r"\b(?:24/7|8/5|24x7|8x5|7/24)\b", re.IGNORECASE)
SLA_RESPONSE_HOURS_ANOMALY_RE = re.compile(
    r"\bsla[^\n\r]{0,40}?(\d+(?:[.,]\d+)?)\s*gi(?:ờ|o)\b",
    re.IGNORECASE,
)
RESPONSE_HOURS_ANOMALY_RE = re.compile(
    r"thời\s*gian\s*phản\s*hồi[^\n\r]{0,40}?(\d+(?:[.,]\d+)?)\s*gi(?:ờ|o)\b",
    re.IGNORECASE,
)
RESOLUTION_HOURS_ANOMALY_RE = re.compile(
    r"thời\s*gian\s*khắc\s*phục[^\n\r]{0,40}?(\d+(?:[.,]\d+)?)\s*gi(?:ờ|o)\b",
    re.IGNORECASE,
)
UPTIME_PERCENT_ANOMALY_RE = re.compile(
    r"\buptime[^\n\r]{0,20}?(\d+(?:[.,]\d+)?)\s*%",
    re.IGNORECASE,
)

CLAUSE_COMPARE_SYSTEM_PROMPT = """Bạn là trợ lý so sánh văn bản pháp lý chạy local.
Bạn sẽ nhận một điều khoản/phụ lục đã được so sánh bằng deterministic pipeline và phải tạo so sánh có cấu trúc dựa hoàn toàn trên evidence đã cho.
Bạn không được đổi change_kind, impact_level, confidence, tags, counts hoặc citation đã khóa sẵn.
Chỉ trả về JSON hợp lệ, không markdown, không giải thích thêm.
Schema:
{
  "clause_id": "string",
  "comparison_summary": "1-2 câu ngắn, bám sát thay đổi quan trọng nhất",
  "important_changes": ["tối đa 3 bullet ngắn, mỗi bullet bám vào evidence cụ thể"],
  "review_points": ["tối đa 2 điểm cần rà soát tiếp, nếu không có thì mảng rỗng"]
}"""

EXECUTIVE_SUMMARY_SYSTEM_PROMPT = """Bạn là trợ lý lập executive summary cho báo cáo so sánh văn bản pháp lý.
Bạn chỉ được dùng số liệu và change summaries đã khóa sẵn.
Không được bịa thêm thay đổi, không được đổi counts.
Không nêu lại bất kỳ số thống kê nào trong overview/key_points/review_alerts; phần thống kê đã được deterministic layer hiển thị riêng.
Hãy tập trung vào bản chất thay đổi, nhóm thay đổi đáng chú ý và điểm cần rà soát.
Chỉ trả về JSON hợp lệ, không markdown.
Schema:
{
  "overview": "1-2 câu ngắn tổng hợp thay đổi quan trọng nhất",
  "key_points": ["tối đa 4 ý quan trọng nhất"],
  "review_alerts": ["tối đa 3 cảnh báo cần kiểm tra thủ công, nếu không có thì mảng rỗng"]
}"""

SEMANTIC_ADJUDICATION_SYSTEM_PROMPT = """Bạn là bộ phân xử ngữ nghĩa cho thay đổi điều khoản pháp lý.
Bạn chỉ được đánh giá sự thay đổi ngữ nghĩa của MỘT change record đã có citation và diff rõ ràng.
Bạn không được thêm change mới, không được sửa citation, không được đổi tags gợi ý nếu evidence không ủng hộ.
Nếu thay đổi chỉ là cách diễn đạt khác đi nhưng nghĩa của mệnh đề không đổi, phải trả về LEXICAL_EQUIVALENT.
Không được đưa khuyến nghị, không được đánh giá rủi ro, không dùng các từ cần/nên/xác minh.
Chỉ trả về JSON hợp lệ.
Schema:
{
  "semantic_effect": "LEXICAL_EQUIVALENT hoặc SUBSTANTIVE",
  "review_status": "AUTO hoặc REVIEW_NEEDED",
  "factual_summary": "1 câu factual, không dùng các từ cần/nên/rủi ro/xác minh",
  "reason": "1 câu ngắn giải thích vì sao"
}

Ví dụ 1:
old_text: "Bên A có quyền chấm dứt hợp đồng nếu bên kia vi phạm nghiêm trọng."
new_text: "Bên A có quyền kết thúc hợp đồng nếu bên kia vi phạm nghiêm trọng."
Kết quả đúng:
{"semantic_effect":"LEXICAL_EQUIVALENT","review_status":"AUTO","factual_summary":"Cụm từ diễn đạt được thay đổi từ 'chấm dứt hợp đồng' sang 'kết thúc hợp đồng' nhưng nghĩa của mệnh đề không thay đổi.","reason":"Khác biệt nằm ở cách diễn đạt, không làm đổi điều kiện hoặc hậu quả pháp lý của câu."}

Ví dụ 2:
old_text: "Bên A có quyền đơn phương chấm dứt hợp đồng sau 30 ngày thông báo."
new_text: "Bên A có quyền đơn phương chấm dứt hợp đồng sau 15 ngày thông báo."
Kết quả đúng:
{"semantic_effect":"SUBSTANTIVE","review_status":"AUTO","factual_summary":"Thời hạn thông báo được thay đổi từ 30 ngày xuống 15 ngày.","reason":"Thay đổi này làm đổi nội dung nghĩa vụ và thời hạn áp dụng."}"""


def get_llm_model() -> str:
    return OLLAMA_MODEL


def _format_source(source) -> str:
    if source is None:
        return "-"
    parts = [source.file]
    if source.page is not None:
        if source.page_end is not None and source.page_end != source.page:
            parts.append(f"p.{source.page}-{source.page_end}")
        else:
            parts.append(f"p.{source.page}")
    if source.line_start is not None:
        if source.line_end is not None and source.line_end != source.line_start:
            parts.append(f"l.{source.line_start}-{source.line_end}")
        else:
            parts.append(f"l.{source.line_start}")
    if source.char_start is not None:
        if source.char_end is not None and source.char_end != source.char_start:
            parts.append(f"c.{source.char_start}-{source.char_end}")
        else:
            parts.append(f"c.{source.char_start}")
    if source.chunk_id:
        parts.append(source.chunk_id)
    return " · ".join(parts)


def _format_diff_block(record: ChangeRecord) -> str:
    old_text = record.diff_snippet.old or "(không có)"
    new_text = record.diff_snippet.new or "(không có)"
    return "\n".join(
        [
            "```diff",
            f"- {old_text}",
            f"+ {new_text}",
            "```",
        ]
    )


def _impact_badge(record: ChangeRecord) -> str:
    return _semantic_effect_badges(record)[0]


def _change_badge(clause: ClauseResult) -> str:
    mapping = {
        "MODIFIED": "[Sửa đổi]",
        "REPLACED": "[Thay thế]",
        "ADDED": "[Bổ sung]",
        "DELETED": "[Loại bỏ]",
    }
    return mapping.get(clause.clause_change_kind, "[Sửa đổi]")


def _sanitize_text(value: Any, max_length: int = 320) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(value.replace("\r", " ").replace("\n", " ").split())
    text = re.sub(r"^[\-\*\d\.\)\s]+", "", text)
    return text[:max_length].strip()


def _sanitize_list(value: Any, limit: int, max_length: int = 220) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = _sanitize_text(item, max_length=max_length)
        if text and text not in cleaned:
            cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def _contains_numeric_claim(text: str) -> bool:
    return bool(re.search(r"\b\d+(?:[.,]\d+)?\b", text))


def _normalize_for_dedupe(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _text_similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(
        None,
        _normalize_for_dedupe(left),
        _normalize_for_dedupe(right),
        autojunk=False,
    ).ratio()


def _is_generic_llm_summary(summary: str, clause_id: str) -> bool:
    normalized = _normalize_for_dedupe(summary)
    clause_token = _normalize_for_dedupe(clause_id)
    generic_patterns = [
        r"có \d+ thay đổi thực chất và \d+ thay đổi hình thức",
        r"được thay nội dung theo cùng cấu trúc điều khoản",
        r"được thêm mới",
        r"bị loại bỏ",
        r"điều khoản thay đổi nội dung",
        r"điều khoản thay đổi hình thức trình bày",
    ]
    if clause_token and clause_token in normalized:
        for pattern in generic_patterns:
            if re.search(pattern, normalized):
                return True
    return any(re.fullmatch(r".*" + pattern + r".*", normalized) for pattern in generic_patterns)


def _is_redundant_text(text: str, references: list[str], threshold: float = 0.82) -> bool:
    for reference in references:
        if not reference:
            continue
        if _normalize_for_dedupe(text) == _normalize_for_dedupe(reference):
            return True
        if _text_similarity(text, reference) >= threshold:
            return True
    return False


def _dedupe_texts(items: list[str], references: list[str] | None = None, limit: int | None = None) -> list[str]:
    deduped: list[str] = []
    known = list(references or [])
    for item in items:
        if not item:
            continue
        if _is_redundant_text(item, known):
            continue
        deduped.append(item)
        known.append(item)
        if limit is not None and len(deduped) >= limit:
            break
    return deduped


def _ground_key_points(
    result: ComparisonResult,
    key_points: list[str],
    threshold: float = 0.38,
) -> tuple[list[str], list[dict[str, Any]]]:
    if not key_points:
        return [], []

    candidates = _top_substantive_changes(result, limit=12)
    if not candidates:
        return [], []

    grounded: list[str] = []
    evidence_map: list[dict[str, Any]] = []
    seen_clause_sets: set[tuple[str, ...]] = set()

    for point in key_points:
        scored: list[tuple[float, ChangeRecord]] = []
        for change in candidates:
            haystack = " | ".join(
                filter(
                    None,
                    [
                        change.clause_id,
                        change.summary,
                        " ".join(change.tags),
                        change.diff_snippet.old or "",
                        change.diff_snippet.new or "",
                    ],
                )
            )
            score = max(
                _text_similarity(point, change.summary),
                _text_similarity(point, change.clause_id),
                _text_similarity(point, " ".join(change.tags)),
                _text_similarity(point, haystack),
            )
            if score >= threshold:
                scored.append((score, change))

        if not scored:
            continue

        scored.sort(key=lambda item: item[0], reverse=True)
        matched_changes = [change for _, change in scored[:3]]
        clause_ids = sorted({change.clause_id for change in matched_changes})
        clause_key = tuple(clause_ids)
        if clause_key in seen_clause_sets:
            continue

        seen_clause_sets.add(clause_key)
        grounded.append(point)
        evidence_map.append(
            {
                "text": point,
                "evidence_clause_ids": clause_ids,
            }
        )

    return grounded, evidence_map


def _extract_metric_values(text: str | None) -> dict[str, set[str]]:
    if not text:
        return {
            "support_window": set(),
            "sla_response_hours": set(),
            "response_hours": set(),
            "resolution_hours": set(),
            "uptime_percent": set(),
        }

    normalized = " ".join(text.split())
    return {
        "support_window": {match.group(0).lower() for match in SUPPORT_WINDOW_ANOMALY_RE.finditer(normalized)},
        "sla_response_hours": {
            match.group(1).replace(",", ".") for match in SLA_RESPONSE_HOURS_ANOMALY_RE.finditer(normalized)
        },
        "response_hours": {
            match.group(1).replace(",", ".") for match in RESPONSE_HOURS_ANOMALY_RE.finditer(normalized)
        },
        "resolution_hours": {
            match.group(1).replace(",", ".") for match in RESOLUTION_HOURS_ANOMALY_RE.finditer(normalized)
        },
        "uptime_percent": {
            match.group(1).replace(",", ".") for match in UPTIME_PERCENT_ANOMALY_RE.finditer(normalized)
        },
    }


def _detect_metric_consistency_anomalies(result: ComparisonResult) -> list[str]:
    metric_to_values: dict[str, dict[str, set[str]]] = {
        "support_window": {},
        "sla_response_hours": {},
        "response_hours": {},
        "resolution_hours": {},
        "uptime_percent": {},
    }

    for clause in result.clauses:
        if clause.clause_change_kind == "DELETED":
            continue
        values = _extract_metric_values(clause.text_b)
        for metric_name, metric_values in values.items():
            for metric_value in metric_values:
                metric_to_values[metric_name].setdefault(metric_value, set()).add(clause.clause_id)

    label_map = {
        "support_window": "khung hỗ trợ",
        "sla_response_hours": "SLA phản hồi (giờ)",
        "response_hours": "thời gian phản hồi (giờ)",
        "resolution_hours": "thời gian khắc phục (giờ)",
        "uptime_percent": "uptime (%)",
    }
    alerts: list[str] = []
    for metric_name, value_map in metric_to_values.items():
        if len(value_map) < 2:
            continue

        distinct_clauses = set()
        for clause_ids in value_map.values():
            distinct_clauses.update(clause_ids)
        if len(distinct_clauses) < 2:
            continue

        sorted_entries = sorted(value_map.items(), key=lambda item: (len(item[1]), item[0]), reverse=True)
        conflict_pair: tuple[tuple[str, set[str]], tuple[str, set[str]]] | None = None
        for left_idx in range(len(sorted_entries)):
            for right_idx in range(left_idx + 1, len(sorted_entries)):
                left = sorted_entries[left_idx]
                right = sorted_entries[right_idx]
                if left[1].isdisjoint(right[1]):
                    conflict_pair = (left, right)
                    break
            if conflict_pair:
                break

        if not conflict_pair:
            continue

        (left_value, left_clauses), (right_value, right_clauses) = conflict_pair
        left_sample = ", ".join(sorted(left_clauses)[:2])
        right_sample = ", ".join(sorted(right_clauses)[:2])
        alerts.append(
            "[CRITICAL ANOMALY] "
            f"Chỉ số {label_map.get(metric_name, metric_name)} không nhất quán giữa các điều khoản: "
            f"{left_value} ở {left_sample} vs {right_value} ở {right_sample}."
        )

    return alerts


def _detect_cross_clause_anomalies(result: ComparisonResult) -> list[str]:
    alerts: list[str] = []

    # SLA consistency: nếu điều khoản SLA cũ bị thay/xóa nhưng nơi khác vẫn nhắc SLA ở bản mới.
    sla_anchor_clauses: list[ClauseResult] = []
    for clause in result.clauses:
        old_text = _normalize_for_dedupe(clause.text_a or "")
        new_text = _normalize_for_dedupe(clause.text_b or "")
        if clause.clause_change_kind in {"REPLACED", "DELETED"} and "sla" in old_text and "sla" not in new_text:
            sla_anchor_clauses.append(clause)

    if sla_anchor_clauses:
        reference_clauses: list[str] = []
        anchor_ids = {anchor.clause_id for anchor in sla_anchor_clauses}
        for clause in result.clauses:
            if clause.clause_id in anchor_ids:
                continue
            text_b = _normalize_for_dedupe(clause.text_b or "")
            if "sla" in text_b:
                reference_clauses.append(clause.clause_id)

        if reference_clauses:
            anchors = ", ".join(anchor.clause_id for anchor in sla_anchor_clauses[:2])
            refs = ", ".join(reference_clauses[:3])
            alerts.append(
                f"[CRITICAL ANOMALY] Tham chiếu SLA còn xuất hiện ở {refs} dù điều khoản SLA chính đã bị thay thế/loại bỏ ({anchors})."
            )

    # Orphaned legal references: điều khoản mới tham chiếu điều/phụ lục đã bị xóa.
    deleted_norm_map = {
        _normalize_for_dedupe(clause.clause_id): clause
        for clause in result.clauses
        if clause.clause_change_kind == "DELETED"
    }
    if deleted_norm_map:
        mention_re = re.compile(r"(điều\s+\d+|phụ\s*lục\s+[a-z])", re.IGNORECASE)
        for clause in result.clauses:
            text_b_norm = _normalize_for_dedupe(clause.text_b or "")
            if not text_b_norm:
                continue
            mentions = { _normalize_for_dedupe(m.group(1)) for m in mention_re.finditer(text_b_norm) }
            for mention in sorted(mentions):
                matched_deleted = next((d_clause for d_key, d_clause in deleted_norm_map.items() if mention in d_key), None)
                if not matched_deleted:
                    continue
                alerts.append(
                    f"[ANOMALY] {clause.clause_id} tham chiếu {mention} nhưng điều khoản tương ứng đã bị loại bỏ ({matched_deleted.clause_id})."
                )

    alerts.extend(_detect_metric_consistency_anomalies(result))

    return _dedupe_texts(alerts, limit=3)


def _semantic_effect_badges(record: ChangeRecord) -> list[str]:
    if record.semantic_effect == "FORMAL":
        return ["[HÌNH THỨC]"]
    if record.semantic_effect == "LEXICAL_EQUIVALENT":
        return ["[THAY ĐỔI TỪ NGỮ]", "[KHÔNG ĐỔI NGỮ NGHĨA]"]
    return ["[THỰC CHẤT]"]


def _semantic_status_line(record: ChangeRecord) -> str:
    parts = [record.semantic_effect]
    if record.review_status != "AUTO":
        parts.append(record.review_status)
    if record.semantic_source != "deterministic":
        parts.append(record.semantic_source.upper())
    return " · ".join(parts)


def _validate_semantic_adjudication(record: ChangeRecord, raw_result: dict[str, Any] | None) -> dict[str, str]:
    validated = {
        "semantic_effect": record.semantic_effect,
        "review_status": record.review_status,
        "semantic_source": record.semantic_source,
        "factual_summary": "",
    }
    if not raw_result:
        return validated

    obvious_substantive_tags = {"Số liệu", "Thời hạn", "Địa điểm", "Chủ thể", "Nhân sự"}
    proposed_effect = _sanitize_text(raw_result.get("semantic_effect"), max_length=40).upper()
    proposed_status = _sanitize_text(raw_result.get("review_status"), max_length=40).upper()
    factual_summary = _sanitize_text(raw_result.get("factual_summary"), max_length=320)

    if proposed_effect not in {"LEXICAL_EQUIVALENT", "SUBSTANTIVE"}:
        proposed_effect = record.semantic_effect
    if proposed_status not in {"AUTO", "REVIEW_NEEDED"}:
        proposed_status = record.review_status

    if (
        proposed_effect == "LEXICAL_EQUIVALENT"
        and (
            record.change_kind in {"ADDED", "DELETED"}
            or record.entity_changes
            or set(record.tags) & obvious_substantive_tags
        )
    ):
        proposed_effect = "SUBSTANTIVE"
        proposed_status = "REVIEW_NEEDED"

    if _is_redundant_text(factual_summary, [record.summary], threshold=0.8):
        factual_summary = ""

    validated["semantic_effect"] = proposed_effect
    validated["review_status"] = proposed_status
    validated["semantic_source"] = "llm"
    validated["factual_summary"] = factual_summary
    return validated


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    if not raw_text:
        return None
    candidate = raw_text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", candidate, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if match:
            candidate = match.group(0)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _record_payload(record: ChangeRecord) -> dict[str, Any]:
    return {
        "summary": record.summary,
        "change_kind": record.change_kind,
        "impact_level": record.impact_level,
        "semantic_effect": record.semantic_effect,
        "review_status": record.review_status,
        "tags": record.tags,
        "confidence_score": round(record.confidence_score, 4),
        "confidence_band": record.confidence_band,
        "source_a": _format_source(record.source_a),
        "source_b": _format_source(record.source_b),
        "diff_old": record.diff_snippet.old,
        "diff_new": record.diff_snippet.new,
        "entity_changes": [item.to_dict() for item in record.entity_changes],
    }


def _validate_clause_analysis(
    clause: ClauseResult,
    raw_analysis: dict[str, Any] | None,
) -> dict[str, Any]:
    deterministic_summary = clause.summary
    analysis: dict[str, Any] = {
        "summary": deterministic_summary,
        "important_changes": [],
        "review_points": [],
    }
    if not raw_analysis:
        return analysis

    clause_id = _sanitize_text(raw_analysis.get("clause_id"), max_length=180)
    if clause_id and clause_id != clause.clause_id:
        return analysis

    summary = _sanitize_text(raw_analysis.get("comparison_summary"), max_length=360)
    if summary and not _is_generic_llm_summary(summary, clause.clause_id) and not _is_redundant_text(summary, [deterministic_summary], threshold=0.8):
        analysis["summary"] = summary
    important_changes = _sanitize_list(
        raw_analysis.get("important_changes"),
        limit=6,
    )
    analysis["important_changes"] = _dedupe_texts(
        important_changes,
        references=[analysis["summary"], deterministic_summary],
        limit=min(3, max(len(clause.records), 1)),
    )
    review_points = _sanitize_list(raw_analysis.get("review_points"), limit=4)
    analysis["review_points"] = _dedupe_texts(
        review_points,
        references=[analysis["summary"], deterministic_summary] + analysis["important_changes"],
        limit=2,
    )
    return analysis


def _validate_executive_summary(
    result: ComparisonResult,
    raw_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    validated: dict[str, Any] = {
        "overview": "",
        "key_points": [],
        "review_alerts": [],
    }
    if raw_summary:
        overview = _sanitize_text(raw_summary.get("overview"), max_length=400)
        validated["overview"] = "" if _contains_numeric_claim(overview) else overview
        key_points = [
            item for item in _sanitize_list(raw_summary.get("key_points"), limit=4, max_length=220)
            if not _contains_numeric_claim(item)
        ]
        validated["key_points"] = _dedupe_texts(key_points, references=[validated["overview"]], limit=4)
        review_alerts = [
            item for item in _sanitize_list(raw_summary.get("review_alerts"), limit=3, max_length=220)
            if not _contains_numeric_claim(item)
        ]
        validated["review_alerts"] = _dedupe_texts(
            review_alerts,
            references=[validated["overview"]] + validated["key_points"],
            limit=3,
        )

    grounded_points, grounding = _ground_key_points(result, validated["key_points"])
    validated["key_points"] = grounded_points
    if grounding:
        validated["key_points_evidence"] = grounding

    for anomaly in _detect_cross_clause_anomalies(result):
        if anomaly not in validated["review_alerts"]:
            validated["review_alerts"].append(anomaly)

    low_confidence = [
        f"{change.clause_id}: {change.confidence_band} ({change.confidence_score:.2f})"
        for change in result.changes
        if change.confidence_band in {"LOW", "SUSPECT"}
    ]
    for alert in low_confidence[:3]:
        if alert not in validated["review_alerts"]:
            validated["review_alerts"].append(alert)
    return validated


def _render_record(record: ChangeRecord) -> list[str]:
    tags = " ".join(_semantic_effect_badges(record) + [f"[{tag}]" for tag in record.tags])
    if record.semantic_effect == "LEXICAL_EQUIVALENT":
        factual_summary = record.llm_notes or "Cách diễn đạt được thay đổi nhưng không làm đổi ngữ nghĩa của mệnh đề."
        return [
            tags,
            f"**Factual summary:** {factual_summary}",
            f"**Confidence:** `{record.confidence_band} ({record.confidence_score:.2f})` | **Nguồn:** `{_format_source(record.source_a)}` → `{_format_source(record.source_b)}`",
            _format_diff_block(record),
        ]
    lines = [
        f"{tags}",
        f"**Semantic status:** `{_semantic_status_line(record)}`",
        f"**Tóm tắt deterministic:** {record.summary}",
        f"**Confidence:** `{record.confidence_band} ({record.confidence_score:.2f})`",
        f"**Nguồn:** `{_format_source(record.source_a)}` → `{_format_source(record.source_b)}`",
        _format_diff_block(record),
    ]
    if record.llm_notes:
        lines.append(f"**Factual summary:** {record.llm_notes}")
    if record.entity_changes:
        entity_parts = [
            f"`{change.field}`: `{change.before or '-'} → {change.after or '-'}`"
            for change in record.entity_changes
        ]
        lines.append(f"**Entity diff:** {', '.join(entity_parts)}")
    return lines


def _top_substantive_changes(result: ComparisonResult, limit: int = 5) -> list[ChangeRecord]:
    substantive = [change for change in result.changes if change.impact_level == "substantive"]
    substantive.sort(key=lambda item: item.confidence_score, reverse=True)
    return substantive[:limit]


def _low_confidence_changes(result: ComparisonResult) -> list[ChangeRecord]:
    return [
        change
        for change in result.changes
        if change.confidence_band in {"LOW", "SUSPECT"}
    ]


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=4))
async def _call_json_llm(
    session: aiohttp.ClientSession,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any] | None:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": LLM_TEMPERATURE,
            "presence_penalty": LLM_PRESENCE_PENALTY,
            "num_predict": MAX_TOKENS,
            "top_p": LLM_TOP_P,
            "num_ctx": LLM_NUM_CTX,
        },
    }

    timeout = aiohttp.ClientTimeout(total=60)
    async with session.post(OLLAMA_API_URL, json=payload, timeout=timeout) as response:
        response.raise_for_status()
        result = await response.json()
        return _extract_json_object(result["message"]["content"].strip())


def _build_clause_prompt(clause: ClauseResult, file_a: str, file_b: str) -> str:
    deterministic_tags = sorted({tag for record in clause.records for tag in record.tags})
    evidence = [_record_payload(record) for record in clause.records]
    return (
        f"Điều khoản/phụ lục: {clause.clause_id}\n"
        f"Loại thay đổi đã khóa: {clause.clause_change_kind}\n"
        f"Tags đã khóa: {deterministic_tags}\n"
        f"Tóm tắt deterministic: {clause.summary}\n"
        f"Văn bản cũ: {file_a}\n"
        f"Văn bản mới: {file_b}\n"
        f"Evidence bundle: {json.dumps(evidence, ensure_ascii=False)}\n"
        "Hãy trả về so sánh có cấu trúc theo schema JSON."
    )


def _build_semantic_prompt(
    clause: ClauseResult,
    record: ChangeRecord,
    file_a: str,
    file_b: str,
) -> str:
    payload = {
        "clause_id": clause.clause_id,
        "file_a": file_a,
        "file_b": file_b,
        "change_kind": record.change_kind,
        "impact_level": record.impact_level,
        "suggested_tags": record.tags,
        "deterministic_summary": record.summary,
        "old_text": record.diff_snippet.old,
        "new_text": record.diff_snippet.new,
        "source_a": _format_source(record.source_a),
        "source_b": _format_source(record.source_b),
    }
    return f"Semantic adjudication input: {json.dumps(payload, ensure_ascii=False)}"


def _build_executive_prompt(result: ComparisonResult, file_a: str, file_b: str) -> str:
    top_changes = [
        {
            "clause_id": change.clause_id,
            "summary": change.summary,
            "impact_level": change.impact_level,
            "semantic_effect": change.semantic_effect,
            "tags": change.tags,
            "confidence": round(change.confidence_score, 4),
        }
        for change in _top_substantive_changes(result)
    ]
    low_confidence = [
        {
            "clause_id": change.clause_id,
            "confidence_band": change.confidence_band,
            "confidence_score": round(change.confidence_score, 4),
            "summary": change.summary,
        }
        for change in _low_confidence_changes(result)
    ]
    payload = {
        "file_a": file_a,
        "file_b": file_b,
        "stats": result.stats.to_dict(),
        "top_substantive_changes": top_changes,
        "low_confidence_items": low_confidence,
        "replaced_clauses": [clause.clause_id for clause in result.clauses if clause.clause_change_kind == "REPLACED"],
        "deterministic_anomalies": _detect_cross_clause_anomalies(result),
    }
    return f"Executive summary input: {json.dumps(payload, ensure_ascii=False)}"


def _semantic_candidates(result: ComparisonResult) -> list[tuple[ClauseResult, ChangeRecord]]:
    candidates: list[tuple[ClauseResult, ChangeRecord]] = []
    for clause in result.clauses:
        for record in clause.records:
            if record.impact_level != "substantive":
                continue
            if record.review_status != "REVIEW_NEEDED":
                continue
            candidates.append((clause, record))
    return candidates


def _should_use_clause_llm(clause: ClauseResult) -> bool:
    # Skip LLM cho clause chỉ có thay đổi hình thức — deterministic đã phân loại đúng
    if clause.clause_change_kind == "MODIFIED":
        if all(record.impact_level == "formal" for record in clause.records):
            return False

    if clause.clause_change_kind in {"REPLACED", "ADDED", "DELETED"}:
        return True
    substantive_records = [record for record in clause.records if record.impact_level == "substantive"]
    if len(substantive_records) >= 1:
        return True
    return False


async def _apply_semantic_adjudication(
    session: aiohttp.ClientSession,
    result: ComparisonResult,
    file_a: str,
    file_b: str,
) -> None:
    candidates = _semantic_candidates(result)
    if not candidates:
        return

    semaphore = asyncio.Semaphore(LLM_MAX_CONCURRENT)

    async def _bounded_call(clause, record):
        async with semaphore:
            return await _call_json_llm(
                session,
                SEMANTIC_ADJUDICATION_SYSTEM_PROMPT,
                _build_semantic_prompt(clause, record, file_a, file_b),
            )

    tasks = [_bounded_call(clause, record) for clause, record in candidates]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    for (clause, record), response in zip(candidates, responses):
        if isinstance(response, Exception):
            logger.warning("Skip semantic adjudication for %s: %s", clause.clause_id, response)
            continue
        validated = _validate_semantic_adjudication(record, response)
        record.semantic_effect = validated["semantic_effect"]
        record.review_status = validated["review_status"]
        record.semantic_source = validated["semantic_source"]
        if validated["factual_summary"]:
            record.llm_notes = validated["factual_summary"]


async def _apply_llm_structured_analysis(result: ComparisonResult, file_a: str, file_b: str) -> None:
    if not result.clauses:
        return

    try:
        async with aiohttp.ClientSession() as session:
            await _apply_semantic_adjudication(session, result, file_a, file_b)
            candidate_clauses = [clause for clause in result.clauses if _should_use_clause_llm(clause)]
            semaphore = asyncio.Semaphore(LLM_MAX_CONCURRENT)

            async def _bounded_clause_call(clause):
                async with semaphore:
                    return await _call_json_llm(
                        session,
                        CLAUSE_COMPARE_SYSTEM_PROMPT,
                        _build_clause_prompt(clause, file_a, file_b),
                    )

            clause_tasks = [_bounded_clause_call(clause) for clause in candidate_clauses]
            clause_responses = await asyncio.gather(*clause_tasks, return_exceptions=True)

            async with semaphore:
                executive_response = await _call_json_llm(
                    session,
                    EXECUTIVE_SUMMARY_SYSTEM_PROMPT,
                    _build_executive_prompt(result, file_a, file_b),
                )
    except Exception as exc:  # pragma: no cover - graceful fallback
        logger.warning("Skip structured LLM analysis due to connection failure: %s", exc)
        return

    for clause, response in zip(candidate_clauses, clause_responses):
        if isinstance(response, Exception):
            logger.warning("Skip clause structured analysis for %s: %s", clause.clause_id, response)
            clause.llm_analysis = _validate_clause_analysis(clause, None)
            continue
        clause.llm_analysis = _validate_clause_analysis(clause, response)
        clause.llm_notes = clause.llm_analysis.get("summary") or clause.summary

    # Set deterministic notes cho formal-only clauses (không gọi LLM)
    for clause in result.clauses:
        if not clause.llm_notes and clause.clause_change_kind == "MODIFIED":
            if all(r.impact_level == "formal" for r in clause.records):
                clause.llm_analysis = _validate_clause_analysis(clause, None)
                clause.llm_notes = "Các thay đổi chỉ mang tính hình thức trình bày, không làm thay đổi nội dung pháp lý."

    result.executive_summary = _validate_executive_summary(result, executive_response)


def _ensure_result(
    comparison_input,
    file_a: str,
    file_b: str,
) -> ComparisonResult:
    if isinstance(comparison_input, ComparisonResult):
        return comparison_input
    return build_comparison_result(comparison_input, file_a=file_a, file_b=file_b)


def _clause_status_line(clause: ClauseResult) -> str:
    substantive = sum(1 for record in clause.records if record.impact_level == "substantive")
    formal = sum(1 for record in clause.records if record.impact_level == "formal")
    if clause.clause_change_kind == "REPLACED":
        return f"`REPLACED` · {substantive} thực chất · {formal} hình thức"
    if clause.clause_change_kind == "ADDED":
        return "`ADDED`"
    if clause.clause_change_kind == "DELETED":
        return "`DELETED`"
    return f"{substantive} thực chất · {formal} hình thức"


def generate_comparison_report(
    comparison_input,
    file_a: str = "v1",
    file_b: str = "v2",
    enable_llm: bool = False,
) -> str:
    result = _ensure_result(comparison_input, file_a=file_a, file_b=file_b)
    if enable_llm:
        try:
            asyncio.run(_apply_llm_structured_analysis(result, file_a=file_a, file_b=file_b))
        except RuntimeError:
            logger.warning("Skip LLM structured analysis because no clean event loop is available.")

    if not result.executive_summary:
        result.executive_summary = _validate_executive_summary(result, None)

    stats = result.stats
    lines = [
        "# BÁO CÁO SO SÁNH CHI TIẾT VĂN BẢN PHÁP LÝ",
        "",
        f"**Ngày lập:** {datetime.now().strftime('%Y-%m-%d')}",
        f"**Văn bản gốc:** `{file_a}`",
        f"**Văn bản mới:** `{file_b}`",
        "",
        "## I. EXECUTIVE SUMMARY",
        f"- Atomic changes: **{stats.atomic_changes}**",
        f"- Clauses affected: **{stats.clauses_affected}**",
        f"- Substantive: **{stats.substantive}**",
        f"- Formal: **{stats.formal}**",
        f"- Replaced clauses: **{stats.replaced}**",
        f"- Added clauses: **{stats.added}**",
        f"- Deleted clauses: **{stats.deleted}**",
        f"- Modified clauses: **{stats.modified}**",
        f"- Unchanged clauses: **{stats.unchanged}**",
        "",
    ]

    overview = result.executive_summary.get("overview")
    if overview:
        lines.append(f"**LLM Executive Summary:** {overview}")
        lines.append("")

    key_points = result.executive_summary.get("key_points") or []
    if key_points:
        lines.append("**Key points:**")
        for point in key_points:
            lines.append(f"- {point}")
        lines.append("")

    review_alerts = result.executive_summary.get("review_alerts") or []
    if review_alerts:
        lines.append("**Review alerts:**")
        for alert in review_alerts:
            lines.append(f"- {alert}")
        lines.append("")

    low_confidence = _low_confidence_changes(result)
    if low_confidence:
        lines.append("**Deterministic low-confidence items:**")
        for record in low_confidence:
            lines.append(
                f"- {record.clause_id}: `{record.confidence_band} ({record.confidence_score:.2f})` - {record.summary}"
            )
        lines.append("")

    lines.extend(["## II. FULL DETAIL", ""])
    detailed_clauses = [
        clause
        for clause in result.clauses
        if clause.records and clause.clause_change_kind in {"MODIFIED", "REPLACED"}
    ]
    if not detailed_clauses:
        lines.append("Không có điều khoản sửa đổi hoặc thay thế.")
        lines.append("")
    else:
        for index, clause in enumerate(detailed_clauses, start=1):
            lines.append(f"### {index}. {clause.clause_id}")
            lines.append(f"{_change_badge(clause)}")
            lines.append(f"**Deterministic status:** {_clause_status_line(clause)}")
            clause_analysis = clause.llm_analysis or {}
            if clause_analysis.get("summary") and not _is_redundant_text(clause_analysis["summary"], [clause.summary], threshold=0.8):
                lines.append(f"**LLM comparison summary:** {clause_analysis['summary']}")
            important_changes = clause_analysis.get("important_changes") or []
            if important_changes:
                lines.append("**LLM key changes:**")
                for point in important_changes:
                    lines.append(f"- {point}")
            review_points = clause_analysis.get("review_points") or []
            if review_points:
                lines.append("**LLM review points:**")
                for point in review_points:
                    lines.append(f"- {point}")
            lines.append("")
            for record in clause.records:
                lines.extend(_render_record(record))
                lines.append("")

    added_clauses = [clause for clause in result.clauses if clause.clause_change_kind == "ADDED"]
    lines.extend(["## III. ĐIỀU KHOẢN/PHỤ LỤC THÊM MỚI", ""])
    if not added_clauses:
        lines.append("Không có điều khoản/phụ lục thêm mới.")
        lines.append("")
    else:
        for index, clause in enumerate(added_clauses, start=1):
            lines.append(f"### {index}. {clause.clause_id}")
            lines.append("[Bổ sung]")
            if clause.llm_analysis.get("summary") and not _is_redundant_text(clause.llm_analysis["summary"], [clause.summary], threshold=0.8):
                lines.append(f"**LLM comparison summary:** {clause.llm_analysis['summary']}")
            lines.append(f"**Nguồn:** `{_format_source(clause.source_b)}`")
            lines.append(_format_diff_block(clause.records[0]))
            lines.append("")

    deleted_clauses = [clause for clause in result.clauses if clause.clause_change_kind == "DELETED"]
    lines.extend(["## IV. ĐIỀU KHOẢN BỊ LOẠI BỎ", ""])
    if not deleted_clauses:
        lines.append("Không có điều khoản/phụ lục bị loại bỏ.")
        lines.append("")
    else:
        for index, clause in enumerate(deleted_clauses, start=1):
            lines.append(f"### {index}. {clause.clause_id}")
            lines.append("[Loại bỏ]")
            if clause.llm_analysis.get("summary") and not _is_redundant_text(clause.llm_analysis["summary"], [clause.summary], threshold=0.8):
                lines.append(f"**LLM comparison summary:** {clause.llm_analysis['summary']}")
            lines.append(f"**Nguồn:** `{_format_source(clause.source_a)}`")
            lines.append(_format_diff_block(clause.records[0]))
            lines.append("")

    lines.extend(
        [
            "**Ghi chú kiến trúc:**",
            "- LLM được dùng để so sánh theo từng clause và sinh executive summary ở dạng có cấu trúc.",
            "- Deterministic layer giữ quyền chốt classification, counts, confidence, diff và citation.",
        ]
    )

    markdown = "\n".join(lines)
    result.report_markdown = markdown
    return markdown
