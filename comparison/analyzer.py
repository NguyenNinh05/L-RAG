from __future__ import annotations

import difflib
import re
from collections import OrderedDict
from typing import Iterable

from ingestion.models import ArticleChunk
from retrieval.matcher import ComparedPair

from comparison.models import (
    ChangeRecord,
    ClauseResult,
    ComparisonResult,
    ComparisonStats,
    DiffSnippet,
    EntityChange,
    SourceRef,
)

FORMAL_TAGS = {
    "Định dạng",
    "Khoảng trắng",
    "Cấu trúc danh sách",
    "Chính tả",
}
ALL_TAGS = FORMAL_TAGS | {
    "Chủ thể",
    "Nhân sự",
    "Số liệu",
    "Thời hạn",
    "Địa điểm",
    "Ngôn ngữ chuyên môn",
}
META_TAGS = {"CASCADING_CHANGE", "LOW_CONFIDENCE", "CRITICAL_VALUE_DELTA"}
ROLE_HINTS = (
    "developer",
    "lead",
    "qa",
    "manager",
    "architect",
    "analyst",
    "tester",
    "engineer",
)
TIME_TOKEN_RE = re.compile(
    r"\b\d+\s*(?:ngày|tháng|năm|tuần|giờ|phút)\b",
    re.IGNORECASE,
)
NUMBER_TOKEN_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
EMAIL_RE = re.compile(r"[\w.\-+]+@[\w.\-]+\.\w+", re.IGNORECASE)
SUBJECT_PATTERNS = (
    re.compile(r'tên công ty:\s*([^|.\n]+)', re.IGNORECASE),
    re.compile(r'"bên [ab]"\s+có nghĩa là\s+([^|.\n]+)', re.IGNORECASE),
    re.compile(r'đại diện ký kết:\s*([^|.\n]+)', re.IGNORECASE),
    re.compile(r'mã số thuế:\s*([^|.\n]+)', re.IGNORECASE),
)
LOCATION_RE = re.compile(
    r"\b(?:tại|ở)\s+([A-ZĐ][^,;\n.]+)",
    re.IGNORECASE,
)
BOILERPLATE_PATTERNS = (
    re.compile(r"hợp đồng này được lập", re.IGNORECASE),
    re.compile(r"ngày\s+tháng\s+năm", re.IGNORECASE),
    re.compile(r"bản gốc", re.IGNORECASE),
    re.compile(r"lưu tại", re.IGNORECASE),
)
SUPPORT_WINDOW_RE = re.compile(r"\b(?:24/7|8/5|24x7|8x5|7/24)\b", re.IGNORECASE)
SLA_RESPONSE_HOURS_RE = re.compile(
    r"\bsla[^\n\r]{0,40}?(\d+(?:[.,]\d+)?)\s*gi(?:ờ|o)\b",
    re.IGNORECASE,
)
RESPONSE_TIME_HOURS_RE = re.compile(
    r"thời\s*gian\s*phản\s*hồi[^\n\r]{0,40}?(\d+(?:[.,]\d+)?)\s*gi(?:ờ|o)\b",
    re.IGNORECASE,
)
RESOLUTION_TIME_HOURS_RE = re.compile(
    r"thời\s*gian\s*khắc\s*phục[^\n\r]{0,40}?(\d+(?:[.,]\d+)?)\s*gi(?:ờ|o)\b",
    re.IGNORECASE,
)
UPTIME_PERCENT_RE = re.compile(
    r"\buptime[^\n\r]{0,20}?(\d+(?:[.,]\d+)?)\s*%",
    re.IGNORECASE,
)


def _normalize_space(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.split())


def _normalize_clause_id(value: str | None) -> str:
    return _normalize_space(value).lower()


def _normalize_text(value: str | None) -> str:
    normalized = _normalize_space(value)
    normalized = normalized.replace("–", "-")
    return normalized


def _normalize_for_semantics(value: str | None) -> str:
    if not value:
        return ""
    normalized = value.replace("–", "-").replace("<br>", " ")
    cleaned_lines: list[str] = []
    for raw_line in normalized.splitlines():
        line = " ".join(raw_line.split())
        line = re.sub(r"^\s*[-*•]+\s*", "", line)
        cleaned_lines.append(line)
    normalized = " ".join(cleaned_lines).lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[ \t]*([,.;:])", r"\1", normalized)
    normalized = re.sub(r"([,.;:])(?=\S)", r"\1 ", normalized)
    return normalized.strip()


def _line_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [line.strip() for line in value.splitlines() if line.strip()]


def _text_ratio(left: str | None, right: str | None) -> float:
    return difflib.SequenceMatcher(
        None,
        _normalize_for_semantics(left),
        _normalize_for_semantics(right),
        autojunk=False,
    ).ratio()


def _is_boilerplate_line(line: str) -> bool:
    stripped = _normalize_space(line)
    return any(pattern.search(stripped) for pattern in BOILERPLATE_PATTERNS)


def _only_boilerplate_delta(old_text: str, new_text: str) -> bool:
    lines_a = [line for line in _line_list(old_text) if not _is_boilerplate_line(line)]
    lines_b = [line for line in _line_list(new_text) if not _is_boilerplate_line(line)]
    return _normalize_for_semantics("\n".join(lines_a)) == _normalize_for_semantics("\n".join(lines_b))


def _detect_formal_tags(old_text: str, new_text: str) -> list[str]:
    tags: list[str] = []
    old_semantic = _normalize_for_semantics(old_text)
    new_semantic = _normalize_for_semantics(new_text)
    if old_semantic == new_semantic:
        tags.extend(["Định dạng", "Khoảng trắng"])
    if any(token in old_text + new_text for token in ("•", "-", "*")):
        if "Cấu trúc danh sách" not in tags:
            tags.append("Cấu trúc danh sách")
    punctuation_only = re.sub(r"[^\w\s]", "", old_text.lower()) == re.sub(r"[^\w\s]", "", new_text.lower())
    if punctuation_only and old_text != new_text and "Chính tả" not in tags:
        tags.append("Chính tả")
    if not tags:
        tags.extend(["Định dạng", "Khoảng trắng"])
    return [tag for tag in tags if tag in ALL_TAGS]


def _extract_tokens(pattern: re.Pattern[str], text: str | None) -> set[str]:
    if not text:
        return set()
    return {match.group(0).strip().lower() for match in pattern.finditer(text)}


def _extract_locations(text: str | None) -> set[str]:
    if not text:
        return set()
    results: set[str] = set()
    for match in LOCATION_RE.finditer(text):
        results.add(_normalize_space(match.group(1)).lower())
    for keyword in ("hà nội", "tp. hồ chí minh", "tp.hồ chí minh", "tp.hcm", "hồ chí minh"):
        if keyword in text.lower():
            results.add(keyword)
    return results


def _extract_subject_values(text: str | None) -> set[str]:
    if not text:
        return set()
    results: set[str] = set()
    for pattern in SUBJECT_PATTERNS:
        for match in pattern.finditer(text):
            results.add(_normalize_space(match.group(1)).lower())
    for match in EMAIL_RE.finditer(text):
        results.add(match.group(0).lower())
    return {value for value in results if value}


def _looks_like_personnel_clause(clause_id: str, old_text: str, new_text: str) -> bool:
    combined = f"{clause_id}\n{old_text}\n{new_text}".lower()
    return "phụ lục a" in combined and any(hint in combined for hint in ROLE_HINTS)


def _append_unique(target: list[str], *values: str) -> None:
    for value in values:
        if value and value not in target:
            target.append(value)


def _extract_critical_metrics(text: str | None) -> dict[str, set[str]]:
    if not text:
        return {
            "support_window": set(),
            "sla_response_hours": set(),
            "response_hours": set(),
            "resolution_hours": set(),
            "uptime_percent": set(),
        }

    normalized = _normalize_space(text)
    return {
        "support_window": {m.group(0).lower() for m in SUPPORT_WINDOW_RE.finditer(normalized)},
        "sla_response_hours": {m.group(1).replace(",", ".") for m in SLA_RESPONSE_HOURS_RE.finditer(normalized)},
        "response_hours": {m.group(1).replace(",", ".") for m in RESPONSE_TIME_HOURS_RE.finditer(normalized)},
        "resolution_hours": {m.group(1).replace(",", ".") for m in RESOLUTION_TIME_HOURS_RE.finditer(normalized)},
        "uptime_percent": {m.group(1).replace(",", ".") for m in UPTIME_PERCENT_RE.finditer(normalized)},
    }


def _critical_metric_deltas(old_text: str | None, new_text: str | None) -> list[tuple[str, str | None, str | None]]:
    before = _extract_critical_metrics(old_text)
    after = _extract_critical_metrics(new_text)
    deltas: list[tuple[str, str | None, str | None]] = []

    for metric_name in before:
        left = sorted(before.get(metric_name, set()))
        right = sorted(after.get(metric_name, set()))
        if left == right:
            continue
        max_len = max(len(left), len(right))
        for idx in range(max_len):
            delta_before = left[idx] if idx < len(left) else None
            delta_after = right[idx] if idx < len(right) else None
            if delta_before == delta_after:
                continue
            deltas.append((metric_name, delta_before, delta_after))
    return deltas


def _detect_substantive_tags(clause_id: str, old_text: str, new_text: str) -> list[str]:
    tags: list[str] = []
    clause_lower = clause_id.lower()
    critical_deltas = _critical_metric_deltas(old_text, new_text)
    if critical_deltas:
        _append_unique(tags, "Số liệu")
        if any(name in {"support_window", "sla_response_hours", "response_hours", "resolution_hours"} for name, _, _ in critical_deltas):
            _append_unique(tags, "Thời hạn")
    if _looks_like_personnel_clause(clause_id, old_text, new_text):
        _append_unique(tags, "Nhân sự")
    if _extract_locations(old_text) != _extract_locations(new_text):
        _append_unique(tags, "Địa điểm")
    if _extract_tokens(TIME_TOKEN_RE, old_text) != _extract_tokens(TIME_TOKEN_RE, new_text):
        _append_unique(tags, "Thời hạn")
    if _extract_tokens(NUMBER_TOKEN_RE, old_text) != _extract_tokens(NUMBER_TOKEN_RE, new_text):
        _append_unique(tags, "Số liệu")
    if _extract_subject_values(old_text) != _extract_subject_values(new_text):
        _append_unique(tags, "Chủ thể")
    if any(keyword in (old_text + "\n" + new_text).lower() for keyword in ("oracle", "java", "selenium", "istqb", "chứng chỉ")):
        _append_unique(tags, "Ngôn ngữ chuyên môn")
    if "phụ lục" in clause_lower and not tags:
        _append_unique(tags, "Ngôn ngữ chuyên môn")
    if not tags:
        _append_unique(tags, "Ngôn ngữ chuyên môn")
    return [tag for tag in tags if tag in ALL_TAGS]


def _classify_impact(clause_id: str, old_text: str, new_text: str) -> tuple[str, list[str]]:
    if (old_text or "") == (new_text or ""):
        return "none", []
    if _normalize_for_semantics(old_text) == _normalize_for_semantics(new_text):
        return "formal", _detect_formal_tags(old_text, new_text)
    if _only_boilerplate_delta(old_text, new_text):
        return "formal", _detect_formal_tags(old_text, new_text)
    if _critical_metric_deltas(old_text, new_text):
        return "substantive", _detect_substantive_tags(clause_id, old_text, new_text)
    return "substantive", _detect_substantive_tags(clause_id, old_text, new_text)


def _confidence_band(score: float) -> str:
    if score >= 0.85:
        return "HIGH"
    if score >= 0.60:
        return "MEDIUM"
    if score >= 0.30:
        return "LOW"
    return "SUSPECT"


def _confidence_score(
    structural_match: float,
    semantic_alignment: float,
    text_diff_quality: float,
    entity_match_quality: float,
) -> tuple[float, str]:
    score = (
        0.35 * structural_match
        + 0.25 * semantic_alignment
        + 0.25 * text_diff_quality
        + 0.15 * entity_match_quality
    )
    score = max(0.0, min(1.0, score))
    return score, _confidence_band(score)


def _initial_semantic_judgment(
    change_kind: str,
    impact_level: str,
    tags: list[str],
    old_text: str | None,
    new_text: str | None,
    entity_changes: list[EntityChange] | None,
) -> tuple[str, str, str]:
    if impact_level == "formal":
        return "FORMAL", "AUTO", "deterministic"

    obvious_substantive_tags = {"Số liệu", "Thời hạn", "Địa điểm", "Chủ thể", "Nhân sự"}
    if entity_changes:
        return "SUBSTANTIVE", "AUTO", "deterministic"
    if set(tags) & obvious_substantive_tags:
        return "SUBSTANTIVE", "AUTO", "deterministic"
    if change_kind in {"ADDED", "DELETED"}:
        return "SUBSTANTIVE", "AUTO", "deterministic"

    ratio = _text_ratio(old_text, new_text)
    if change_kind == "REPLACED" and ratio < 0.35:
        return "SUBSTANTIVE", "REVIEW_NEEDED", "deterministic"
    if ratio >= 0.55:
        return "SUBSTANTIVE", "REVIEW_NEEDED", "deterministic"
    return "SUBSTANTIVE", "AUTO", "deterministic"


def _source_ref(chunk: ArticleChunk | None, file_name: str) -> SourceRef | None:
    if chunk is None:
        return None

    metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
    page = chunk.page
    if page is None:
        metadata_page = metadata.get("page")
        page = metadata_page if metadata_page is not None else chunk.raw_index + 1

    return SourceRef(
        file=file_name,
        page=page,
        page_end=chunk.page_end if chunk.page_end is not None else metadata.get("page_end") or metadata.get("page_last"),
        line_start=chunk.line_start if chunk.line_start is not None else metadata.get("line_start"),
        line_end=chunk.line_end if chunk.line_end is not None else metadata.get("line_end"),
        char_start=chunk.char_start if chunk.char_start is not None else metadata.get("char_start"),
        char_end=chunk.char_end if chunk.char_end is not None else metadata.get("char_end"),
        chunk_id=chunk.chunk_id(),
        context_snippet=_normalize_space(chunk.content)[:160] or None,
    )


def _truncate(text: str | None, limit: int = 240) -> str:
    normalized = _normalize_space(text)
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _diff_snippet(old_text: str | None, new_text: str | None) -> DiffSnippet:
    return DiffSnippet(
        old=_truncate(old_text),
        new=_truncate(new_text),
        format="diff",
    )


def _validate_tags(tags: Iterable[str]) -> list[str]:
    valid: list[str] = []
    for tag in tags:
        if tag in ALL_TAGS and tag not in valid:
            valid.append(tag)
    return valid


def _validate_meta_tags(tags: Iterable[str]) -> list[str]:
    valid: list[str] = []
    for tag in tags:
        if tag in META_TAGS and tag not in valid:
            valid.append(tag)
    return valid


def _build_line_pairs(old_text: str, new_text: str) -> list[tuple[str, str]]:
    lines_a = _line_list(old_text)
    lines_b = _line_list(new_text)
    if not lines_a and not lines_b:
        return []

    matcher = difflib.SequenceMatcher(
        None,
        [_normalize_for_semantics(line) for line in lines_a],
        [_normalize_for_semantics(line) for line in lines_b],
        autojunk=False,
    )
    segments: list[tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        segments.append(("\n".join(lines_a[i1:i2]), "\n".join(lines_b[j1:j2])))
    if not segments:
        segments.append((old_text, new_text))
    return segments


def _pair_clause_id(pair: ComparedPair) -> str:
    return pair.chunk_a.article_number if pair.chunk_a else pair.chunk_b.article_number  # type: ignore[return-value]


def _collapse_replacements(pairs: list[ComparedPair]) -> list[ComparedPair | tuple[str, ComparedPair, ComparedPair]]:
    collapsed: list[ComparedPair | tuple[str, ComparedPair, ComparedPair]] = []
    index = 0
    while index < len(pairs):
        current = pairs[index]
        if index + 1 < len(pairs):
            following = pairs[index + 1]
            clause_current = _normalize_clause_id(_pair_clause_id(current))
            clause_following = _normalize_clause_id(_pair_clause_id(following))
            opposite = {current.match_type, following.match_type} == {"DELETED", "ADDED"}
            if clause_current and clause_current == clause_following and opposite:
                if current.match_type == "DELETED":
                    collapsed.append(("REPLACED", current, following))
                else:
                    collapsed.append(("REPLACED", following, current))
                index += 2
                continue
        collapsed.append(current)
        index += 1
    return collapsed


def _table_rows(text: str) -> tuple[list[str], OrderedDict[str, dict[str, str]]]:
    headers: list[str] = []
    rows: OrderedDict[str, dict[str, str]] = OrderedDict()
    raw_lines = _line_list(text)
    row_index = 0

    for line in raw_lines:
        if "|" not in line:
            if not rows:
                headers.append(line)
            continue
        cells = [cell.strip() for cell in line.split("|") if cell.strip()]
        if len(cells) < 3:
            if not rows:
                headers.append(line)
            continue
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        lowered_cells = [cell.lower() for cell in cells]
        if any(cell in {"họ tên", "vai trò", "kinh nghiệm", "chứng chỉ", "stt"} for cell in lowered_cells):
            continue
        role = next((cell for cell in cells if any(hint in cell.lower() for hint in ROLE_HINTS)), None)
        if role is None:
            role = f"row_{row_index}"
        non_role_cells = [cell for cell in cells if cell != role]
        name = ""
        experience = ""
        certification = ""
        for cell in non_role_cells:
            if NUMBER_TOKEN_RE.search(cell) and "năm" in cell.lower():
                experience = cell
            elif not name and not cell.isdigit():
                name = cell
            elif cell != name:
                certification = f"{certification}, {cell}".strip(", ")
        key = _normalize_space(role)
        rows[key] = {
            "name": name,
            "role": role,
            "experience": experience,
            "certification": certification,
            "raw": " | ".join(cells),
        }
        row_index += 1
    return headers, rows


def _record(
    clause_id: str,
    change_kind: str,
    impact_level: str,
    old_text: str | None,
    new_text: str | None,
    source_a: SourceRef | None,
    source_b: SourceRef | None,
    tags: list[str],
    summary: str,
    structural_match: float,
    semantic_alignment: float,
    text_diff_quality: float,
    entity_match_quality: float,
    entity_changes: list[EntityChange] | None = None,
    meta_tags: list[str] | None = None,
) -> ChangeRecord:
    score, band = _confidence_score(
        structural_match=structural_match,
        semantic_alignment=semantic_alignment,
        text_diff_quality=text_diff_quality,
        entity_match_quality=entity_match_quality,
    )
    semantic_effect, review_status, semantic_source = _initial_semantic_judgment(
        change_kind=change_kind,
        impact_level=impact_level,
        tags=tags,
        old_text=old_text,
        new_text=new_text,
        entity_changes=entity_changes,
    )
    merged_meta = list(meta_tags or [])
    if "CRITICAL_VALUE_DELTA" in merged_meta and impact_level == "substantive":
        review_status = "REVIEW_NEEDED"
    if band in {"LOW", "SUSPECT"}:
        _append_unique(merged_meta, "LOW_CONFIDENCE")
        review_status = "REVIEW_NEEDED"
    return ChangeRecord(
        clause_id=clause_id,
        change_kind=change_kind,
        impact_level=impact_level,
        semantic_effect=semantic_effect,
        review_status=review_status,
        semantic_source=semantic_source,
        tags=_validate_tags(tags),
        meta_tags=_validate_meta_tags(merged_meta),
        confidence_score=score,
        confidence_band=band,
        diff_snippet=_diff_snippet(old_text, new_text),
        source_a=source_a,
        source_b=source_b,
        entity_changes=entity_changes or [],
        summary=summary,
    )


def _summarize_clause(clause_id: str, records: list[ChangeRecord], clause_change_kind: str) -> str:
    if clause_change_kind == "ADDED":
        return f"{clause_id} được thêm mới."
    if clause_change_kind == "DELETED":
        return f"{clause_id} bị loại bỏ."
    if clause_change_kind == "REPLACED":
        return f"{clause_id} được thay nội dung theo cùng cấu trúc điều khoản."
    substantive = sum(1 for record in records if record.impact_level == "substantive")
    formal = sum(1 for record in records if record.impact_level == "formal")
    return f"{clause_id} có {substantive} thay đổi thực chất và {formal} thay đổi hình thức."


def _analyze_personnel_clause(
    clause_id: str,
    text_a: str,
    text_b: str,
    pair_similarity: float,
    source_a: SourceRef | None,
    source_b: SourceRef | None,
) -> ClauseResult:
    headers_a, rows_a = _table_rows(text_a)
    headers_b, rows_b = _table_rows(text_b)
    if not rows_a or not rows_b:
        return _analyze_modified_clause(
            clause_id=clause_id,
            text_a=text_a,
            text_b=text_b,
            pair_similarity=pair_similarity,
            source_a=source_a,
            source_b=source_b,
            allow_personnel=False,
        )

    records: list[ChangeRecord] = []
    if _normalize_for_semantics("\n".join(headers_a)) == _normalize_for_semantics("\n".join(headers_b)):
        if "\n".join(headers_a) != "\n".join(headers_b):
            records.append(
                _record(
                    clause_id=clause_id,
                    change_kind="MODIFIED",
                    impact_level="formal",
                    old_text="\n".join(headers_a),
                    new_text="\n".join(headers_b),
                    source_a=source_a,
                    source_b=source_b,
                    tags=["Định dạng", "Khoảng trắng"],
                    summary="Tiêu đề phụ lục thay đổi xuống dòng/định dạng nhưng không đổi nghĩa.",
                    structural_match=1.0,
                    semantic_alignment=max(pair_similarity, 0.95),
                    text_diff_quality=0.9,
                    entity_match_quality=0.7,
                )
            )

    all_roles = list(OrderedDict.fromkeys(list(rows_a.keys()) + list(rows_b.keys())))
    for role in all_roles:
        left = rows_a.get(role)
        right = rows_b.get(role)
        if left and right:
            if _normalize_for_semantics(left["raw"]) == _normalize_for_semantics(right["raw"]):
                continue
            entity_changes: list[EntityChange] = []
            tags = ["Nhân sự"]
            change_kind = "REPLACED" if left["name"] != right["name"] else "MODIFIED"
            summary = f"Vai trò {left['role']} thay đổi thông tin nhân sự."
            if left["name"] != right["name"]:
                entity_changes.append(EntityChange(role, "name", left["name"], right["name"]))
                summary = f"Vai trò {left['role']} được thay người phụ trách."
            if left["experience"] != right["experience"]:
                entity_changes.append(EntityChange(role, "experience", left["experience"], right["experience"]))
                _append_unique(tags, "Số liệu")
            if left["certification"] != right["certification"]:
                entity_changes.append(EntityChange(role, "certification", left["certification"], right["certification"]))
                _append_unique(tags, "Ngôn ngữ chuyên môn")
            meta_tags = ["CASCADING_CHANGE"] if len(entity_changes) > 1 else []
            records.append(
                _record(
                    clause_id=clause_id,
                    change_kind=change_kind,
                    impact_level="substantive",
                    old_text=left["raw"],
                    new_text=right["raw"],
                    source_a=source_a,
                    source_b=source_b,
                    tags=tags,
                    summary=summary,
                    structural_match=1.0,
                    semantic_alignment=max(pair_similarity, 0.8),
                    text_diff_quality=0.9,
                    entity_match_quality=1.0,
                    entity_changes=entity_changes,
                    meta_tags=meta_tags,
                )
            )
            continue
        if left:
            records.append(
                _record(
                    clause_id=clause_id,
                    change_kind="DELETED",
                    impact_level="substantive",
                    old_text=left["raw"],
                    new_text=None,
                    source_a=source_a,
                    source_b=source_b,
                    tags=["Nhân sự"],
                    summary=f"Vai trò {left['role']} bị loại khỏi phụ lục nhân sự.",
                    structural_match=0.95,
                    semantic_alignment=0.7,
                    text_diff_quality=0.85,
                    entity_match_quality=0.8,
                )
            )
        if right:
            records.append(
                _record(
                    clause_id=clause_id,
                    change_kind="ADDED",
                    impact_level="substantive",
                    old_text=None,
                    new_text=right["raw"],
                    source_a=source_a,
                    source_b=source_b,
                    tags=["Nhân sự"],
                    summary=f"Vai trò {right['role']} được bổ sung trong phụ lục nhân sự.",
                    structural_match=0.95,
                    semantic_alignment=0.7,
                    text_diff_quality=0.85,
                    entity_match_quality=0.8,
                )
            )

    clause_change_kind = "REPLACED" if any(record.change_kind == "REPLACED" for record in records) else "MODIFIED"
    return ClauseResult(
        clause_id=clause_id,
        clause_change_kind=clause_change_kind,
        citation_type=clause_change_kind,
        semantic_similarity=pair_similarity,
        source_a=source_a,
        source_b=source_b,
        text_a=text_a,
        text_b=text_b,
        records=records,
        summary=_summarize_clause(clause_id, records, clause_change_kind),
    )


def _analyze_modified_clause(
    clause_id: str,
    text_a: str,
    text_b: str,
    pair_similarity: float,
    source_a: SourceRef | None,
    source_b: SourceRef | None,
    forced_kind: str | None = None,
    allow_personnel: bool = True,
) -> ClauseResult:
    if allow_personnel and _looks_like_personnel_clause(clause_id, text_a, text_b):
        return _analyze_personnel_clause(
            clause_id=clause_id,
            text_a=text_a,
            text_b=text_b,
            pair_similarity=pair_similarity,
            source_a=source_a,
            source_b=source_b,
        )

    records: list[ChangeRecord] = []
    critical_deltas = _critical_metric_deltas(text_a, text_b)
    for old_segment, new_segment in _build_line_pairs(text_a, text_b):
        impact_level, tags = _classify_impact(clause_id, old_segment, new_segment)
        if impact_level == "none":
            continue
        segment_has_critical = bool(_critical_metric_deltas(old_segment, new_segment))
        deep_rewrite = forced_kind == "REPLACED" or _text_ratio(old_segment, new_segment) < 0.35
        change_kind = forced_kind or ("REPLACED" if deep_rewrite and impact_level == "substantive" else "MODIFIED")
        tags = tags or (
            _detect_formal_tags(old_segment, new_segment)
            if impact_level == "formal"
            else _detect_substantive_tags(clause_id, old_segment, new_segment)
        )
        summary = "Điều khoản thay đổi hình thức trình bày." if impact_level == "formal" else "Điều khoản thay đổi nội dung."
        semantic_alignment = max(min(pair_similarity, 1.0), 0.2)
        text_diff_quality = 0.9 if impact_level == "formal" else 0.85
        entity_match_quality = 0.65 if impact_level == "formal" else 0.75
        meta_tags = ["CRITICAL_VALUE_DELTA"] if segment_has_critical and impact_level == "substantive" else []
        records.append(
            _record(
                clause_id=clause_id,
                change_kind=change_kind,
                impact_level=impact_level,
                old_text=old_segment,
                new_text=new_segment,
                source_a=source_a,
                source_b=source_b,
                tags=tags,
                summary=summary,
                structural_match=1.0,
                semantic_alignment=semantic_alignment,
                text_diff_quality=text_diff_quality,
                entity_match_quality=entity_match_quality,
                meta_tags=meta_tags,
            )
        )

    if critical_deltas:
        for metric_name, before_value, after_value in critical_deltas:
            before_str = before_value if before_value is not None else "(không có)"
            after_str = after_value if after_value is not None else "(không có)"
            already_captured = any(
                before_str in (record.diff_snippet.old or "")
                and after_str in (record.diff_snippet.new or "")
                for record in records
            )
            if already_captured:
                continue

            metric_label = {
                "support_window": "khung hỗ trợ",
                "sla_response_hours": "SLA phản hồi",
                "response_hours": "thời gian phản hồi",
                "resolution_hours": "thời gian khắc phục",
                "uptime_percent": "uptime",
            }.get(metric_name, metric_name)

            metric_tags = ["Số liệu"]
            if metric_name in {"support_window", "sla_response_hours", "response_hours", "resolution_hours"}:
                metric_tags.append("Thời hạn")

            records.append(
                _record(
                    clause_id=clause_id,
                    change_kind=forced_kind or "MODIFIED",
                    impact_level="substantive",
                    old_text=f"{metric_label}: {before_str}",
                    new_text=f"{metric_label}: {after_str}",
                    source_a=source_a,
                    source_b=source_b,
                    tags=metric_tags,
                    summary=f"Chỉ số {metric_label} thay đổi từ {before_str} thành {after_str}.",
                    structural_match=1.0,
                    semantic_alignment=max(min(pair_similarity, 1.0), 0.3),
                    text_diff_quality=0.9,
                    entity_match_quality=0.95,
                    meta_tags=["CRITICAL_VALUE_DELTA"],
                )
            )

    if not records:
        records.append(
            _record(
                clause_id=clause_id,
                change_kind=forced_kind or "MODIFIED",
                impact_level="formal",
                old_text=text_a,
                new_text=text_b,
                source_a=source_a,
                source_b=source_b,
                tags=["Định dạng", "Khoảng trắng"],
                summary="Nội dung giữ nguyên, chỉ thay đổi cách trình bày.",
                structural_match=1.0,
                semantic_alignment=max(pair_similarity, 0.95),
                text_diff_quality=0.9,
                entity_match_quality=0.6,
            )
        )

    clause_change_kind = forced_kind or ("REPLACED" if any(record.change_kind == "REPLACED" for record in records) else "MODIFIED")
    return ClauseResult(
        clause_id=clause_id,
        clause_change_kind=clause_change_kind,
        citation_type=clause_change_kind,
        semantic_similarity=pair_similarity,
        source_a=source_a,
        source_b=source_b,
        text_a=text_a,
        text_b=text_b,
        records=records,
        summary=_summarize_clause(clause_id, records, clause_change_kind),
    )


def _analyze_single_clause(
    clause_id: str,
    clause_change_kind: str,
    text_a: str | None,
    text_b: str | None,
    similarity: float,
    source_a: SourceRef | None,
    source_b: SourceRef | None,
) -> ClauseResult:
    if clause_change_kind in {"MODIFIED", "REPLACED"}:
        return _analyze_modified_clause(
            clause_id=clause_id,
            text_a=text_a or "",
            text_b=text_b or "",
            pair_similarity=similarity,
            source_a=source_a,
            source_b=source_b,
            forced_kind="REPLACED" if clause_change_kind == "REPLACED" else None,
        )

    if clause_change_kind == "ADDED":
        record = _record(
            clause_id=clause_id,
            change_kind="ADDED",
            impact_level="substantive",
            old_text=None,
            new_text=text_b,
            source_a=source_a,
            source_b=source_b,
            tags=_detect_substantive_tags(clause_id, "", text_b or ""),
            summary="Điều khoản/phụ lục được thêm mới.",
            structural_match=0.95,
            semantic_alignment=0.75,
            text_diff_quality=0.85,
            entity_match_quality=0.6,
        )
        return ClauseResult(
            clause_id=clause_id,
            clause_change_kind="ADDED",
            citation_type="ADDED",
            semantic_similarity=similarity,
            source_a=source_a,
            source_b=source_b,
            text_a=text_a,
            text_b=text_b,
            records=[record],
            summary=_summarize_clause(clause_id, [record], "ADDED"),
        )

    record = _record(
        clause_id=clause_id,
        change_kind="DELETED",
        impact_level="substantive",
        old_text=text_a,
        new_text=None,
        source_a=source_a,
        source_b=source_b,
        tags=_detect_substantive_tags(clause_id, text_a or "", ""),
        summary="Điều khoản/phụ lục bị loại bỏ.",
        structural_match=0.95,
        semantic_alignment=0.75,
        text_diff_quality=0.85,
        entity_match_quality=0.6,
    )
    return ClauseResult(
        clause_id=clause_id,
        clause_change_kind="DELETED",
        citation_type="DELETED",
        semantic_similarity=similarity,
        source_a=source_a,
        source_b=source_b,
        text_a=text_a,
        text_b=text_b,
        records=[record],
        summary=_summarize_clause(clause_id, [record], "DELETED"),
    )


def build_comparison_result(
    pairs: list[ComparedPair],
    file_a: str = "v1",
    file_b: str = "v2",
) -> ComparisonResult:
    unchanged = sum(1 for pair in pairs if pair.match_type == "UNCHANGED")
    clauses: list[ClauseResult] = []
    changes: list[ChangeRecord] = []

    for item in _collapse_replacements(pairs):
        if isinstance(item, tuple):
            _, deleted_pair, added_pair = item
            clause_id = deleted_pair.chunk_a.article_number or added_pair.chunk_b.article_number or "N/A"
            clause = _analyze_single_clause(
                clause_id=clause_id,
                clause_change_kind="REPLACED",
                text_a=deleted_pair.chunk_a.content if deleted_pair.chunk_a else None,
                text_b=added_pair.chunk_b.content if added_pair.chunk_b else None,
                similarity=max(deleted_pair.similarity, added_pair.similarity),
                source_a=_source_ref(deleted_pair.chunk_a, file_a),
                source_b=_source_ref(added_pair.chunk_b, file_b),
            )
        else:
            pair = item
            if pair.match_type == "UNCHANGED":
                continue
            clause_id = _pair_clause_id(pair) or "N/A"
            clause = _analyze_single_clause(
                clause_id=clause_id,
                clause_change_kind=pair.match_type,
                text_a=pair.chunk_a.content if pair.chunk_a else None,
                text_b=pair.chunk_b.content if pair.chunk_b else None,
                similarity=pair.similarity,
                source_a=_source_ref(pair.chunk_a, file_a),
                source_b=_source_ref(pair.chunk_b, file_b),
            )
        clauses.append(clause)
        changes.extend(clause.records)

    stats = ComparisonStats(
        modified=sum(1 for clause in clauses if clause.clause_change_kind == "MODIFIED"),
        added=sum(1 for clause in clauses if clause.clause_change_kind == "ADDED"),
        deleted=sum(1 for clause in clauses if clause.clause_change_kind == "DELETED"),
        unchanged=unchanged,
        substantive=sum(1 for change in changes if change.impact_level == "substantive"),
        formal=sum(1 for change in changes if change.impact_level == "formal"),
        replaced=sum(1 for clause in clauses if clause.clause_change_kind == "REPLACED"),
        clauses_affected=len(clauses),
        atomic_changes=sum(1 for change in changes if change.impact_level != "none"),
    )
    return ComparisonResult(
        stats=stats,
        changes=changes,
        clauses=clauses,
    )
