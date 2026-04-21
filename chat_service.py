from __future__ import annotations

import asyncio
import difflib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from config import (
    LLM_MAX_TOKENS,
    LLM_NUM_CTX,
    LLM_PRESENCE_PENALTY,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    OLLAMA_CHAT_URL,
    OLLAMA_LLM_MODEL,
)
from session_store import SessionStore, is_quote_request

CHAT_SYSTEM_PROMPT = """Bạn là trợ lý hỏi đáp hậu so sánh văn bản pháp lý chạy local.
Bạn chỉ được trả lời dựa trên evidence bundle đã được grounded và có citation_id.
Không được tự tạo số liệu mới, không được tự tạo citation mới, không được tư vấn pháp lý.
Nếu bằng chứng không đủ thì phải nói rõ là không đủ bằng chứng trong các đoạn đã truy xuất.
Chỉ trả về JSON hợp lệ.
Schema:
{
  "answer_markdown": "câu trả lời factual ngắn gọn, có thể dùng bullet",
  "used_citation_ids": ["citation_id_1", "citation_id_2"],
  "answer_type": "diff_answer hoặc document_answer hoặc insufficient_evidence",
  "insufficient_evidence": false
}"""

TAG_KEYWORDS = {
    "Thời hạn": ("thời hạn", "gia hạn", "deadline", "hiệu lực", "tháng", "ngày hết hạn"),
    "Địa điểm": ("địa điểm", "tòa án", "thành phố", "hà nội", "hồ chí minh", "tp hcm", "tp.hcm", "nơi giải quyết"),
    "Nhân sự": ("nhân sự", "người phụ trách", "lead developer", "qa lead", "vai trò"),
    "Chủ thể": ("chủ thể", "bên a", "bên b", "công ty", "pháp nhân", "đại diện"),
    "Số liệu": ("số liệu", "giá trị", "số tiền", "chi phí", "phần trăm", "tỷ lệ", "sla", "uptime", "24/7", "8/5"),
    "Định dạng": ("định dạng", "xuống dòng", "khoảng trắng", "dấu chấm", "dấu gạch", "bullet"),
    "Ngôn ngữ chuyên môn": ("từ ngữ", "câu chữ", "thuật ngữ", "ngôn ngữ"),
}

CHANGE_KIND_KEYWORDS = {
    "ADDED": ("thêm", "bổ sung", "thêm mới"),
    "DELETED": ("xóa", "loại bỏ", "bị loại"),
    "REPLACED": ("thay thế", "thay toàn bộ", "thay hẳn"),
    "MODIFIED": ("sửa", "chỉnh sửa", "thay đổi", "khác"),
}

IMPACT_KEYWORDS = {
    "formal": ("hình thức", "định dạng", "trình bày", "spacing", "format"),
    "substantive": ("thực chất", "nội dung"),
}

SEMANTIC_KEYWORDS = {
    "LEXICAL_EQUIVALENT": ("không đổi nghĩa", "không thay đổi ngữ nghĩa", "thay đổi từ ngữ", "đồng nghĩa"),
    "SUBSTANTIVE": ("đổi nghĩa", "khác nghĩa", "thay đổi ý nghĩa"),
}

SUMMARY_HINTS = (
    "tóm tắt",
    "tổng quan",
    "khác với bản v1",
    "khác với bản cũ",
    "những thay đổi",
    "chi tiết đã được sửa",
    "điểm khác biệt",
    "có gì thay đổi",
)

DOCUMENT_HINTS = (
    "nói gì",
    "quy định gì",
    "nội dung",
    "chi tiết điều khoản",
    "chi tiết phụ lục",
)


DOMAIN_HINTS = (
    "so sanh",
    "thay doi",
    "sua",
    "chinh sua",
    "thay the",
    "them moi",
    "xoa",
    "bi xoa",
    "khac voi",
    "khac biet",
    "ban v1",
    "ban v2",
    "ban cu",
    "ban moi",
    "phien ban",
    "hop dong",
    "van ban",
    "tai lieu",
    "dieu khoan",
    "phu luc",
    "trich",
    "bang chung",
    "citation",
    "nguon",
)


@dataclass(slots=True)
class QueryPlan:
    intent: str
    answer_type: str
    primary_source: str
    needs_quote: bool = False
    clause_hint: str | None = None
    tag_filters: list[str] = field(default_factory=list)
    change_kind_filters: list[str] = field(default_factory=list)
    impact_filters: list[str] = field(default_factory=list)
    semantic_filters: list[str] = field(default_factory=list)
    structured_limit: int = 4
    chunk_limit: int = 3
    planner_source: str = "rule"

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "answer_type": self.answer_type,
            "primary_source": self.primary_source,
            "needs_quote": self.needs_quote,
            "clause_hint": self.clause_hint,
            "tag_filters": list(self.tag_filters),
            "change_kind_filters": list(self.change_kind_filters),
            "impact_filters": list(self.impact_filters),
            "semantic_filters": list(self.semantic_filters),
            "structured_limit": self.structured_limit,
            "chunk_limit": self.chunk_limit,
            "planner_source": self.planner_source,
        }


def _normalize(text: str) -> str:
    lowered = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", lowered).strip()


def _normalize_for_keyword_match(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    no_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    lowered = re.sub(r"[^\w\s]", " ", no_marks)
    return re.sub(r"\s+", " ", lowered).strip()


def _contains_keyword(haystack_normalized: str, haystack_tokens: set[str], keyword: str) -> bool:
    keyword_norm = _normalize_for_keyword_match(keyword)
    if not keyword_norm:
        return False
    if " " in keyword_norm:
        return f" {keyword_norm} " in f" {haystack_normalized} "
    return keyword_norm in haystack_tokens


def _tokens(text: str) -> set[str]:
    return {token for token in _normalize(text).split() if len(token) > 1}


def _score(query: str, candidate: str) -> float:
    left = _normalize(query)
    right = _normalize(candidate)
    if not left or not right:
        return 0.0
    ratio = difflib.SequenceMatcher(None, left, right, autojunk=False).ratio()
    query_tokens = _tokens(query)
    candidate_tokens = _tokens(candidate)
    overlap = len(query_tokens & candidate_tokens) / len(query_tokens) if query_tokens else 0.0
    clause_boost = 0.12 if re.search(r"điều\s+\d+|phụ lục\s+[a-z]", left) and any(token in right for token in query_tokens) else 0.0
    return min(1.0, 0.55 * overlap + 0.45 * ratio + clause_boost)


def _extract_clause_hint(question: str) -> str | None:
    match = re.search(r"(điều\s+\d+|phụ lục\s+[a-z]|mở đầu)", question, re.IGNORECASE)
    return match.group(0) if match else None


def _extract_filters(question: str, mapping: dict[str, tuple[str, ...]]) -> list[str]:
    normalized = _normalize_for_keyword_match(question)
    tokens = set(normalized.split())
    selected = []
    for canonical, keywords in mapping.items():
        if any(_contains_keyword(normalized, tokens, keyword) for keyword in keywords):
            selected.append(canonical)
    return selected


def _looks_like_domain_question(question: str, plan: QueryPlan) -> bool:
    normalized = _normalize_for_keyword_match(question)
    tokens = set(normalized.split())
    if plan.clause_hint or plan.needs_quote:
        return True
    if plan.tag_filters or plan.change_kind_filters or plan.impact_filters or plan.semantic_filters:
        return True
    return any(_contains_keyword(normalized, tokens, hint) for hint in DOMAIN_HINTS)


def _build_query_plan(question: str) -> QueryPlan:
    normalized = _normalize(question)
    needs_quote = is_quote_request(question)
    clause_hint = _extract_clause_hint(question)
    tag_filters = _extract_filters(question, TAG_KEYWORDS)
    change_kind_filters = _extract_filters(question, CHANGE_KIND_KEYWORDS)
    impact_filters = _extract_filters(question, IMPACT_KEYWORDS)
    semantic_filters = _extract_filters(question, SEMANTIC_KEYWORDS)

    if needs_quote:
        return QueryPlan(
            intent="quote_request",
            answer_type="document_answer",
            primary_source="chunk",
            needs_quote=True,
            clause_hint=clause_hint,
            tag_filters=tag_filters,
            structured_limit=2,
            chunk_limit=3,
        )

    if clause_hint and any(hint in normalized for hint in DOCUMENT_HINTS):
        return QueryPlan(
            intent="clause_lookup",
            answer_type="document_answer",
            primary_source="hybrid",
            clause_hint=clause_hint,
            tag_filters=tag_filters,
            change_kind_filters=change_kind_filters,
            impact_filters=impact_filters,
            semantic_filters=semantic_filters,
            structured_limit=3,
            chunk_limit=2,
        )

    if tag_filters or change_kind_filters or impact_filters or semantic_filters:
        return QueryPlan(
            intent="filtered_change_query",
            answer_type="diff_answer",
            primary_source="structured",
            clause_hint=clause_hint,
            tag_filters=tag_filters,
            change_kind_filters=change_kind_filters,
            impact_filters=impact_filters,
            semantic_filters=semantic_filters,
            structured_limit=5,
            chunk_limit=2,
        )

    if clause_hint:
        return QueryPlan(
            intent="clause_lookup",
            answer_type="diff_answer",
            primary_source="structured",
            clause_hint=clause_hint,
            structured_limit=4,
            chunk_limit=2,
        )

    if any(hint in normalized for hint in SUMMARY_HINTS):
        return QueryPlan(
            intent="diff_summary",
            answer_type="diff_answer",
            primary_source="structured",
            structured_limit=6,
            chunk_limit=0,
        )

    return QueryPlan(
        intent="general",
        answer_type="diff_answer",
        primary_source="hybrid",
        clause_hint=clause_hint,
        structured_limit=4,
        chunk_limit=2,
    )


def _matches_clause_hint(clause_hint: str | None, clause_id: str | None) -> bool:
    if not clause_hint:
        return True
    if not clause_id:
        return False
    left = _normalize(clause_hint)
    right = _normalize(clause_id)
    return left in right or _score(clause_hint, clause_id) >= 0.55


def _candidate_tags(item: dict[str, Any]) -> set[str]:
    payload = item.get("payload") or {}
    if item.get("candidate_type") == "change":
        return set(payload.get("tags") or [])
    tags: set[str] = set()
    for record in payload.get("records", []):
        tags.update(record.get("tags") or [])
    return tags


def _candidate_change_kinds(item: dict[str, Any]) -> set[str]:
    payload = item.get("payload") or {}
    if item.get("candidate_type") == "change":
        return {payload.get("change_kind") or item.get("change_kind") or ""}
    kind = payload.get("clause_change_kind") or payload.get("change_kind")
    return {kind} if kind else set()


def _candidate_impact_levels(item: dict[str, Any]) -> set[str]:
    payload = item.get("payload") or {}
    if item.get("candidate_type") == "change":
        return {payload.get("impact_level") or item.get("impact_level") or ""}
    levels = {record.get("impact_level") for record in payload.get("records", []) if record.get("impact_level")}
    if not levels and item.get("impact_level"):
        levels.add(item["impact_level"])
    return levels


def _candidate_semantic_effects(item: dict[str, Any]) -> set[str]:
    payload = item.get("payload") or {}
    if item.get("candidate_type") == "change":
        return {payload.get("semantic_effect") or item.get("semantic_effect") or ""}
    effects = {record.get("semantic_effect") for record in payload.get("records", []) if record.get("semantic_effect")}
    if not effects and item.get("semantic_effect"):
        effects.add(item["semantic_effect"])
    return effects


def _passes_plan_filters(item: dict[str, Any], plan: QueryPlan) -> bool:
    if plan.clause_hint and not _matches_clause_hint(plan.clause_hint, item.get("clause_id")):
        return False
    if plan.tag_filters and not (set(plan.tag_filters) & _candidate_tags(item)):
        return False
    if plan.change_kind_filters and not (set(plan.change_kind_filters) & _candidate_change_kinds(item)):
        return False
    if plan.impact_filters and not (set(plan.impact_filters) & _candidate_impact_levels(item)):
        return False
    if plan.semantic_filters and not (set(plan.semantic_filters) & _candidate_semantic_effects(item)):
        return False
    return True


def _score_structured_candidate(item: dict[str, Any], question: str, plan: QueryPlan) -> float:
    score = _score(question, item.get("search_text", ""))
    if item.get("candidate_type") == "change":
        score += 0.08
        score += min(float(item.get("confidence_score") or 0.0), 1.0) * 0.08
        if item.get("impact_level") == "substantive":
            score += 0.08
    elif item.get("candidate_type") == "clause":
        score += 0.04

    if plan.intent == "diff_summary" and item.get("candidate_type") == "change":
        score = max(score, 0.58)
    if plan.intent == "clause_lookup" and item.get("candidate_type") == "clause":
        score += 0.18
    if plan.clause_hint and _matches_clause_hint(plan.clause_hint, item.get("clause_id")):
        score += 0.25
    if plan.tag_filters and set(plan.tag_filters) & _candidate_tags(item):
        score += 0.22
    if plan.change_kind_filters and set(plan.change_kind_filters) & _candidate_change_kinds(item):
        score += 0.16
    if plan.impact_filters and set(plan.impact_filters) & _candidate_impact_levels(item):
        score += 0.14
    if plan.semantic_filters and set(plan.semantic_filters) & _candidate_semantic_effects(item):
        score += 0.14
    return score


def _pick_structured_hits(items: list[dict[str, Any]], question: str, plan: QueryPlan) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for item in items:
        if plan.intent in {"diff_summary", "filtered_change_query"} and item.get("candidate_type") == "clause" and not plan.clause_hint:
            continue
        if not _passes_plan_filters(item, plan):
            continue
        score = _score_structured_candidate(item, question, plan)
        if plan.intent != "diff_summary" and score <= 0.12:
            continue
        enriched = dict(item)
        enriched["score"] = min(score, 1.4)
        ranked.append(enriched)

    ranked.sort(key=lambda item: item["score"], reverse=True)
    if plan.intent != "diff_summary":
        return ranked[: plan.structured_limit]

    selected: list[dict[str, Any]] = []
    seen_clauses: set[str] = set()
    for item in ranked:
        clause_id = item.get("clause_id") or ""
        if clause_id in seen_clauses:
            continue
        selected.append(item)
        seen_clauses.add(clause_id)
        if len(selected) >= plan.structured_limit:
            break
    return selected


def _score_chunk_candidate(item: dict[str, Any], question: str, plan: QueryPlan) -> float:
    score = _score(question, item.get("search_text", ""))
    if plan.clause_hint and _matches_clause_hint(plan.clause_hint, item.get("clause_id")):
        score += 0.28
    if plan.needs_quote:
        score += 0.14
    return score


def _pick_chunk_hits(items: list[dict[str, Any]], question: str, plan: QueryPlan) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for item in items:
        if plan.clause_hint and not _matches_clause_hint(plan.clause_hint, item.get("clause_id")):
            continue
        score = _score_chunk_candidate(item, question, plan)
        if score <= 0.12 and not (plan.clause_hint and _matches_clause_hint(plan.clause_hint, item.get("clause_id"))):
            continue
        enriched = dict(item)
        enriched["score"] = min(score, 1.3)
        ranked.append(enriched)
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[: plan.chunk_limit]


def _should_fallback_to_chunks(question: str, plan: QueryPlan, structured_hits: list[dict[str, Any]]) -> bool:
    if plan.primary_source == "chunk" or plan.needs_quote:
        return True
    if not structured_hits:
        return True
    if plan.intent in {"diff_summary", "filtered_change_query"}:
        return False
    top_score = structured_hits[0]["score"]
    if top_score < 0.72:
        return True
    if is_quote_request(question):
        return True
    if all(hit.get("impact_level") == "formal" for hit in structured_hits[:2]) and "hình thức" not in _normalize(question):
        return True
    return False


def _build_evidence_bundle(structured_hits: list[dict[str, Any]], chunk_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for hit in structured_hits:
        evidence.append(
            {
                "citation_id": hit["citation_id"],
                "candidate_type": hit["candidate_type"],
                "clause_id": hit["clause_id"],
                "score": round(hit["score"], 4),
                "payload": hit["payload"],
            }
        )
    for hit in chunk_hits:
        evidence.append(
            {
                "citation_id": hit["citation_id"],
                "candidate_type": "chunk",
                "clause_id": hit["clause_id"],
                "score": round(hit["score"], 4),
                "payload": hit["payload"],
            }
        )
    return evidence


def _should_reject_as_out_of_scope(
    question: str,
    plan: QueryPlan,
    structured_hits: list[dict[str, Any]],
    chunk_hits: list[dict[str, Any]],
) -> bool:
    if _looks_like_domain_question(question, plan):
        return False
    top_hits = [*structured_hits[:1], *chunk_hits[:1]]
    if not top_hits:
        return True
    best_hit = max(top_hits, key=lambda item: item["score"])
    query_tokens = _tokens(question)
    candidate_tokens = _tokens(best_hit.get("search_text", ""))
    overlap_ratio = len(query_tokens & candidate_tokens) / len(query_tokens) if query_tokens else 0.0
    return best_hit["score"] < 0.72 or overlap_ratio < 0.34


async def _call_json_llm(user_prompt: str) -> dict[str, Any] | None:
    payload = {
        "model": OLLAMA_LLM_MODEL,
        "messages": [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": LLM_TEMPERATURE,
            "presence_penalty": LLM_PRESENCE_PENALTY,
            "num_predict": LLM_MAX_TOKENS,
            "top_p": LLM_TOP_P,
            "num_ctx": LLM_NUM_CTX,
        },
    }
    timeout = aiohttp.ClientTimeout(total=45)
    async with aiohttp.ClientSession() as session:
        async with session.post(OLLAMA_CHAT_URL, json=payload, timeout=timeout) as response:
            response.raise_for_status()
            result = await response.json()
    raw_content = (result.get("message") or {}).get("content", "").strip()
    match = re.search(r"\{.*\}", raw_content, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _deterministic_answer(question: str, evidence: list[dict[str, Any]], answer_type: str, plan: QueryPlan) -> dict[str, Any]:
    if not evidence:
        return {
            "answer_markdown": "Không đủ bằng chứng trong các đoạn đã truy xuất để trả lời câu hỏi này.",
            "used_citation_ids": [],
            "answer_type": "insufficient_evidence",
            "insufficient_evidence": True,
        }

    if plan.intent == "diff_summary":
        intro = "**Tóm tắt thay đổi:** Các điểm sửa đáng chú ý giữa bản v2 và bản v1 gồm:"
    elif plan.intent == "filtered_change_query":
        intro = f"**Kết quả lọc:** Các bằng chứng phù hợp nhất cho câu hỏi `{question}` gồm:"
    elif plan.needs_quote or answer_type == "document_answer":
        intro = f"**Nội dung truy xuất:** Các đoạn phù hợp nhất cho câu hỏi `{question}` gồm:"
    else:
        intro = f"**Trả lời:** Dưới đây là các bằng chứng phù hợp nhất cho câu hỏi `{question}`."
    lines = []
    for item in evidence[: max(3, min(5, len(evidence)))]:
        payload = item["payload"]
        if item["candidate_type"] == "chunk":
            excerpt = payload.get("content", "")[:220]
            lines.append(f"- {payload.get('clause_id')}: {excerpt}")
        elif item["candidate_type"] == "clause":
            summary = payload.get("summary") or payload.get("llm_notes") or "Có thay đổi ở điều khoản này."
            lines.append(f"- {payload.get('clause_id')}: {summary}")
        else:
            summary = payload.get("summary") or payload.get("llm_notes") or "Có thay đổi được phát hiện."
            lines.append(f"- {payload.get('clause_id')}: {summary}")

    return {
        "answer_markdown": "\n".join([intro, *lines]),
        "used_citation_ids": [item["citation_id"] for item in evidence[: max(3, min(5, len(evidence)))]],
        "answer_type": answer_type,
        "insufficient_evidence": False,
    }


def _out_of_scope_answer() -> dict[str, Any]:
    return {
        "answer_markdown": (
            "Câu hỏi này không thuộc phạm vi phiên so sánh hiện tại. "
            "Tôi chỉ trả lời về khác biệt giữa hai tài liệu hoặc nội dung có trong các đoạn đã truy xuất."
        ),
        "used_citation_ids": [],
        "answer_type": "insufficient_evidence",
        "insufficient_evidence": True,
    }


def _validate_answer(raw_answer: dict[str, Any] | None, allowed_citation_ids: set[str], fallback: dict[str, Any]) -> dict[str, Any]:
    if not raw_answer:
        return fallback
    used_citation_ids = [
        citation_id
        for citation_id in raw_answer.get("used_citation_ids", [])
        if citation_id in allowed_citation_ids
    ]
    answer_markdown = str(raw_answer.get("answer_markdown") or "").strip()
    answer_type = str(raw_answer.get("answer_type") or fallback["answer_type"]).strip() or fallback["answer_type"]
    insufficient = bool(raw_answer.get("insufficient_evidence"))
    if not answer_markdown:
        return fallback
    if not used_citation_ids and not insufficient:
        return fallback
    lowered = answer_markdown.lower()
    if "cần " in lowered or "nên " in lowered:
        return fallback
    return {
        "answer_markdown": answer_markdown,
        "used_citation_ids": used_citation_ids,
        "answer_type": answer_type,
        "insufficient_evidence": insufficient,
    }


async def answer_session_question(store: SessionStore, session_id: str, question: str) -> dict[str, Any]:
    plan = _build_query_plan(question)

    structured_candidates = store.get_structured_candidates(session_id)
    structured_hits = _pick_structured_hits(structured_candidates, question, plan)

    chunk_hits: list[dict[str, Any]] = []
    if _should_fallback_to_chunks(question, plan, structured_hits):
        chunk_hits = _pick_chunk_hits(store.get_chunk_candidates(session_id), question, plan)

    if _should_reject_as_out_of_scope(question, plan, structured_hits, chunk_hits):
        return _out_of_scope_answer()

    answer_type = plan.answer_type
    if chunk_hits and (plan.primary_source == "chunk" or not structured_hits):
        answer_type = "document_answer"

    if plan.primary_source == "chunk":
        evidence = _build_evidence_bundle([], chunk_hits[: plan.chunk_limit])
    elif plan.primary_source == "structured":
        fallback_chunks = chunk_hits[:1] if not structured_hits else []
        evidence = _build_evidence_bundle(structured_hits[: plan.structured_limit], fallback_chunks)
    else:
        evidence = _build_evidence_bundle(structured_hits[: plan.structured_limit], chunk_hits[: plan.chunk_limit])
    fallback = _deterministic_answer(question, evidence, answer_type, plan)
    if not evidence:
        return fallback

    prompt = json.dumps(
        {
            "question": question,
            "retrieval_plan": plan.to_dict(),
            "answer_type": answer_type,
            "evidence_bundle": evidence,
        },
        ensure_ascii=False,
    )
    try:
        raw_answer = await _call_json_llm(prompt)
    except Exception:
        raw_answer = None
    return _validate_answer(raw_answer, {item["citation_id"] for item in evidence}, fallback)


def answer_session_question_sync(store: SessionStore, session_id: str, question: str) -> dict[str, Any]:
    return asyncio.run(answer_session_question(store, session_id, question))
