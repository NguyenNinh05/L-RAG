from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

import chat_service


def make_change_candidate(
    citation_id: str,
    clause_id: str,
    summary: str,
    *,
    tags: list[str],
    search_text: str,
    impact_level: str = "substantive",
    semantic_effect: str = "SUBSTANTIVE",
    confidence_score: float = 0.9,
    change_kind: str = "MODIFIED",
) -> dict:
    return {
        "candidate_type": "change",
        "clause_id": clause_id,
        "citation_id": citation_id,
        "search_text": search_text,
        "impact_level": impact_level,
        "semantic_effect": semantic_effect,
        "confidence_score": confidence_score,
        "payload": {
            "clause_id": clause_id,
            "summary": summary,
            "tags": tags,
            "impact_level": impact_level,
            "semantic_effect": semantic_effect,
            "change_kind": change_kind,
        },
    }


def make_clause_candidate(citation_id: str, clause_id: str, summary: str, search_text: str) -> dict:
    return {
        "candidate_type": "clause",
        "clause_id": clause_id,
        "citation_id": citation_id,
        "search_text": search_text,
        "impact_level": None,
        "semantic_effect": None,
        "confidence_score": 0.0,
        "payload": {
            "clause_id": clause_id,
            "summary": summary,
            "records": [],
            "clause_change_kind": "MODIFIED",
        },
    }


def make_chunk_candidate(citation_id: str, clause_id: str, content: str, *, search_text: str | None = None) -> dict:
    return {
        "candidate_type": "chunk",
        "citation_id": citation_id,
        "doc_side": "B",
        "clause_id": clause_id,
        "page": 5,
        "chunk_id": f"{citation_id}_chunk",
        "search_text": search_text or f"{clause_id} {content}",
        "payload": {
            "clause_id": clause_id,
            "content": content,
        },
    }


class FakeStore:
    def __init__(self, structured: list[dict], chunks: list[dict]) -> None:
        self.structured = structured
        self.chunks = chunks

    def get_structured_candidates(self, session_id: str) -> list[dict]:
        return list(self.structured)

    def get_chunk_candidates(self, session_id: str) -> list[dict]:
        return list(self.chunks)


class ChatRouterTests(unittest.TestCase):
    def test_diff_summary_prefers_structured_changes(self) -> None:
        store = FakeStore(
            structured=[
                make_change_candidate(
                    "chg_time",
                    "Phần I > Điều 3",
                    "Thời hạn hợp đồng tăng từ 24 tháng lên 36 tháng.",
                    tags=["Thời hạn"],
                    search_text="Phần I Điều 3 thời hạn hợp đồng tăng 24 tháng lên 36 tháng",
                ),
                make_change_candidate(
                    "chg_location",
                    "Phần III > Điều 5",
                    "Địa điểm giải quyết tranh chấp đổi từ Hà Nội sang TP. Hồ Chí Minh.",
                    tags=["Địa điểm"],
                    search_text="Phần III Điều 5 địa điểm giải quyết tranh chấp Hà Nội TP Hồ Chí Minh",
                ),
            ],
            chunks=[
                make_chunk_candidate(
                    "chunk_doc",
                    "Phần III > Điều 5",
                    "Bước 3: tranh chấp được đưa ra Tòa án nhân dân có thẩm quyền tại TP. Hồ Chí Minh.",
                )
            ],
        )

        with patch("chat_service._call_json_llm", return_value=None):
            answer = asyncio.run(
                chat_service.answer_session_question(
                    store,
                    "session-1",
                    "Cho tôi thông tin về những chi tiết đã được sửa ở bản v2 mà khác với bản v1 nhé",
                )
            )

        self.assertEqual(answer["answer_type"], "diff_answer")
        self.assertEqual(answer["used_citation_ids"], ["chg_time", "chg_location"])
        self.assertIn("Thời hạn hợp đồng tăng", answer["answer_markdown"])
        self.assertIn("Địa điểm giải quyết tranh chấp", answer["answer_markdown"])

    def test_filtered_query_applies_tag_filter_before_answer(self) -> None:
        store = FakeStore(
            structured=[
                make_change_candidate(
                    "chg_time",
                    "Phần I > Điều 3",
                    "Thời hạn hợp đồng tăng từ 24 tháng lên 36 tháng.",
                    tags=["Thời hạn"],
                    search_text="Phần I Điều 3 thời hạn hợp đồng tăng 24 tháng lên 36 tháng",
                ),
                make_change_candidate(
                    "chg_location",
                    "Phần III > Điều 5",
                    "Địa điểm giải quyết tranh chấp đổi từ Hà Nội sang TP. Hồ Chí Minh.",
                    tags=["Địa điểm"],
                    search_text="Phần III Điều 5 địa điểm giải quyết tranh chấp Hà Nội TP Hồ Chí Minh",
                ),
            ],
            chunks=[],
        )

        with patch("chat_service._call_json_llm", return_value=None):
            answer = asyncio.run(
                chat_service.answer_session_question(
                    store,
                    "session-1",
                    "Có thay đổi nào về thời hạn không?",
                )
            )

        self.assertEqual(answer["answer_type"], "diff_answer")
        self.assertEqual(answer["used_citation_ids"], ["chg_time"])
        self.assertIn("Thời hạn hợp đồng tăng", answer["answer_markdown"])
        self.assertNotIn("Địa điểm giải quyết tranh chấp", answer["answer_markdown"])

    def test_quote_request_routes_to_chunk_retrieval(self) -> None:
        store = FakeStore(
            structured=[
                make_clause_candidate(
                    "clause_d7",
                    "Phần II > Điều 7",
                    "Điều 7 đã được thay thế bằng nội dung về bảo mật dữ liệu.",
                    "Phần II Điều 7 bảo mật dữ liệu thay thế SLA",
                )
            ],
            chunks=[
                make_chunk_candidate(
                    "chunk_d7",
                    "Phần II > Điều 7",
                    "Điều 7: Bảo mật và bảo vệ dữ liệu. Bên A cam kết tuân thủ Nghị định 13/2023/NĐ-CP.",
                    search_text="Phần II Điều 7 bản mới nói gì bảo mật và bảo vệ dữ liệu nghị định 13",
                )
            ],
        )

        with patch("chat_service._call_json_llm", return_value=None):
            answer = asyncio.run(
                chat_service.answer_session_question(
                    store,
                    "session-1",
                    "Điều 7 bản mới nói gì?",
                )
            )

        self.assertEqual(answer["answer_type"], "document_answer")
        self.assertEqual(answer["used_citation_ids"], ["chunk_d7"])
        self.assertIn("Bảo mật và bảo vệ dữ liệu", answer["answer_markdown"])

    def test_out_of_scope_question_returns_insufficient_evidence(self) -> None:
        store = FakeStore(
            structured=[
                make_change_candidate(
                    "chg_time",
                    "Phần I > Điều 3",
                    "Thời hạn hợp đồng tăng từ 24 tháng lên 36 tháng.",
                    tags=["Thời hạn"],
                    search_text="Phần I Điều 3 thời hạn hợp đồng tăng 24 tháng lên 36 tháng",
                ),
                make_change_candidate(
                    "chg_location",
                    "Phần III > Điều 5",
                    "Địa điểm giải quyết tranh chấp đổi từ Hà Nội sang TP. Hồ Chí Minh.",
                    tags=["Địa điểm"],
                    search_text="Phần III Điều 5 địa điểm giải quyết tranh chấp Hà Nội TP Hồ Chí Minh",
                ),
            ],
            chunks=[],
        )

        with patch("chat_service._call_json_llm", return_value=None):
            answer = asyncio.run(
                chat_service.answer_session_question(
                    store,
                    "session-1",
                    "Thời tiết hôm nay thế nào?",
                )
            )

        self.assertEqual(answer["answer_type"], "insufficient_evidence")
        self.assertEqual(answer["used_citation_ids"], [])
        self.assertTrue(answer["insufficient_evidence"])
        self.assertGreater(len(answer["answer_markdown"]), 0)

    def test_filtered_query_without_diacritics_still_matches_tag(self) -> None:
        store = FakeStore(
            structured=[
                make_change_candidate(
                    "chg_time",
                    "Phần I > Điều 3",
                    "Thời hạn hợp đồng tăng từ 24 tháng lên 36 tháng.",
                    tags=["Thời hạn"],
                    search_text="Phần I Điều 3 thời hạn hợp đồng tăng 24 tháng lên 36 tháng",
                ),
                make_change_candidate(
                    "chg_location",
                    "Phần III > Điều 5",
                    "Địa điểm giải quyết tranh chấp đổi từ Hà Nội sang TP. Hồ Chí Minh.",
                    tags=["Địa điểm"],
                    search_text="Phần III Điều 5 địa điểm giải quyết tranh chấp Hà Nội TP Hồ Chí Minh",
                ),
            ],
            chunks=[],
        )

        with patch("chat_service._call_json_llm", return_value=None):
            answer = asyncio.run(
                chat_service.answer_session_question(
                    store,
                    "session-1",
                    "Co thay doi nao ve thoi han khong?",
                )
            )

        self.assertEqual(answer["answer_type"], "diff_answer")
        self.assertEqual(answer["used_citation_ids"], ["chg_time"])
        self.assertIn("Thời hạn hợp đồng tăng", answer["answer_markdown"])

    def test_domain_hint_without_diacritics_not_out_of_scope(self) -> None:
        store = FakeStore(
            structured=[
                make_change_candidate(
                    "chg_location",
                    "Phần III > Điều 5",
                    "Địa điểm giải quyết tranh chấp đổi từ Hà Nội sang TP. Hồ Chí Minh.",
                    tags=["Địa điểm"],
                    search_text="Phần III Điều 5 địa điểm giải quyết tranh chấp Hà Nội TP Hồ Chí Minh",
                )
            ],
            chunks=[],
        )

        with patch("chat_service._call_json_llm", return_value=None):
            answer = asyncio.run(
                chat_service.answer_session_question(
                    store,
                    "session-1",
                    "So sanh dieu 5 ban moi voi ban cu",
                )
            )

        self.assertEqual(answer["answer_type"], "diff_answer")
        self.assertFalse(answer["insufficient_evidence"])
        self.assertIn("Điều", answer["answer_markdown"])


if __name__ == "__main__":
    unittest.main()
