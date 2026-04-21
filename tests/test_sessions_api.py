from __future__ import annotations

import asyncio
import json
import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import api
from comparison.models import (
    ChangeRecord,
    ClauseResult,
    ComparisonResult,
    ComparisonStats,
    DiffSnippet,
    SourceRef,
)
from session_store import SessionStore


def make_result() -> ComparisonResult:
    record = ChangeRecord(
        clause_id="Phần I > Điều 3",
        change_kind="MODIFIED",
        impact_level="substantive",
        semantic_effect="SUBSTANTIVE",
        tags=["Thời hạn"],
        confidence_score=0.91,
        confidence_band="HIGH",
        diff_snippet=DiffSnippet(old="24 tháng", new="36 tháng", format="diff"),
        source_a=SourceRef(file="v1.pdf", page=2, chunk_id="chunk_a"),
        source_b=SourceRef(file="v2.pdf", page=2, chunk_id="chunk_b"),
        summary="Thời hạn hợp đồng tăng từ 24 tháng lên 36 tháng.",
    )
    clause = ClauseResult(
        clause_id="Phần I > Điều 3",
        clause_change_kind="MODIFIED",
        citation_type="MODIFIED",
        semantic_similarity=0.91,
        source_a=SourceRef(file="v1.pdf", page=2, chunk_id="chunk_a"),
        source_b=SourceRef(file="v2.pdf", page=2, chunk_id="chunk_b"),
        text_a="Hợp đồng có thời hạn 24 tháng.",
        text_b="Hợp đồng có thời hạn 36 tháng.",
        records=[record],
        summary="Phần I > Điều 3 có 1 thay đổi thực chất.",
    )
    stats = ComparisonStats(
        modified=1,
        added=0,
        deleted=0,
        unchanged=0,
        substantive=1,
        formal=0,
        replaced=0,
        clauses_affected=1,
        atomic_changes=1,
    )
    return ComparisonResult(stats=stats, changes=[record], clauses=[clause], report_markdown="")


def make_chunk(doc_label: str, doc_id: str, article_number: str, content: str) -> SimpleNamespace:
    return SimpleNamespace(
        doc_label=doc_label,
        doc_id=doc_id,
        article_number=article_number,
        title=article_number,
        content=content,
        page=2,
        raw_index=0,
        sub_index=0,
        metadata={"top_level": "Phần I", "mid_level": article_number},
    )


class SessionStoreAndApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(__file__).resolve().parent / f"_tmp_sessions_api_{uuid.uuid4().hex}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.temp_dir / "sessions.db"
        self.store = SessionStore(self.db_path)
        self.store.initialize()
        self._old_store = api.session_store
        api.session_store = self.store
        self.client = TestClient(api.app)

    def tearDown(self) -> None:
        api.session_store = self._old_store
        self.client.close()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def seed_completed_session(self) -> tuple[str, dict]:
        session_id = self.store.create_session("v1.pdf", "v2.pdf", "hash-a", "hash-b")
        persisted = self.store.complete_session(
            session_id=session_id,
            comparison_result=make_result(),
            report_markdown="# demo report",
            chunks_a=[make_chunk("doc_A", "v1.pdf", "Phần I > Điều 3", "Hợp đồng có thời hạn 24 tháng.")],
            chunks_b=[make_chunk("doc_B", "v2.pdf", "Phần I > Điều 3", "Hợp đồng có thời hạn 36 tháng.")],
        )
        return session_id, persisted

    def test_recovery_marks_processing_sessions_failed_and_hidden(self) -> None:
        session_id = self.store.create_session("a.pdf", "b.pdf", "hash-a", "hash-b")
        recovered = self.store.recover_interrupted_sessions()

        self.assertEqual(recovered, 1)
        self.assertEqual(self.store.get_session_status(session_id), "failed")
        self.assertEqual(self.store.list_completed_sessions(), [])

    def test_list_sessions_returns_only_completed(self) -> None:
        self.store.create_session("draft_a.pdf", "draft_b.pdf", "hash-a", "hash-b")
        completed_id, _ = self.seed_completed_session()

        response = self.client.get("/api/sessions")
        self.assertEqual(response.status_code, 200)
        items = response.json()["items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["session_id"], completed_id)

    def test_chat_endpoint_returns_409_for_processing_session(self) -> None:
        session_id = self.store.create_session("v1.pdf", "v2.pdf", "hash-a", "hash-b")
        response = self.client.post(f"/api/sessions/{session_id}/chat", json={"question": "Có thay đổi nào không?"})

        self.assertEqual(response.status_code, 409)

    def test_citation_lookup_and_chat_persistence(self) -> None:
        session_id, persisted = self.seed_completed_session()
        citation_id = persisted["clause_citations"][0]["citation_id"]

        citation_response = self.client.get(f"/api/sessions/{session_id}/citations/{citation_id}")
        self.assertEqual(citation_response.status_code, 200)
        self.assertEqual(citation_response.json()["citation_id"], citation_id)

        async def fake_answer(*args, **kwargs):
            return {
                "answer_markdown": "Có 1 thay đổi về thời hạn.",
                "used_citation_ids": [citation_id],
                "answer_type": "diff_answer",
                "insufficient_evidence": False,
            }

        with patch("api.answer_session_question", new=fake_answer):
            response = self.client.post(
                f"/api/sessions/{session_id}/chat",
                json={"question": "Có thay đổi nào về thời hạn không?"},
            )

        self.assertEqual(response.status_code, 200)
        body = response.text
        self.assertIn("event: message", body)
        self.assertIn("Có 1 thay đổi về thời hạn.", body)

        session = self.store.get_session(session_id)
        assistant_messages = [item for item in session["messages"] if item["role"] == "assistant"]
        self.assertEqual(len(assistant_messages), 1)
        self.assertEqual(assistant_messages[0]["citation_ids"], [citation_id])

    def test_complete_session_merges_duplicate_clause_ids(self) -> None:
        base = make_result()
        duplicate_record = ChangeRecord(
            clause_id="Phần I > Điều 3",
            change_kind="MODIFIED",
            impact_level="formal",
            semantic_effect="LEXICAL_EQUIVALENT",
            tags=["Ngôn ngữ chuyên môn"],
            confidence_score=0.83,
            confidence_band="MEDIUM",
            diff_snippet=DiffSnippet(old="Bên A", new="Bên ký kết A", format="diff"),
            source_a=SourceRef(file="v1.pdf", page=2, chunk_id="chunk_a2"),
            source_b=SourceRef(file="v2.pdf", page=2, chunk_id="chunk_b2"),
            summary="Điều chỉnh câu chữ mô tả chủ thể.",
        )
        duplicate_clause = ClauseResult(
            clause_id="Phần I > Điều 3",
            clause_change_kind="MODIFIED",
            citation_type="MODIFIED",
            semantic_similarity=0.88,
            source_a=SourceRef(file="v1.pdf", page=2, chunk_id="chunk_a2"),
            source_b=SourceRef(file="v2.pdf", page=2, chunk_id="chunk_b2"),
            text_a="Bên A có trách nhiệm bảo mật.",
            text_b="Bên ký kết A có trách nhiệm bảo mật.",
            records=[duplicate_record],
            summary="Biến thể câu chữ cùng điều khoản.",
        )

        duplicated = ComparisonResult(
            stats=base.stats,
            changes=[*base.changes, duplicate_record],
            clauses=[*base.clauses, duplicate_clause],
            report_markdown=base.report_markdown,
        )

        session_id = self.store.create_session("v1.pdf", "v2.pdf", "hash-a", "hash-b")
        self.store.complete_session(
            session_id=session_id,
            comparison_result=duplicated,
            report_markdown="# duplicate clause test",
            chunks_a=[make_chunk("doc_A", "v1.pdf", "Phần I > Điều 3", "A")],
            chunks_b=[make_chunk("doc_B", "v2.pdf", "Phần I > Điều 3", "B")],
        )

        self.assertEqual(self.store.get_session_status(session_id), "completed")
        candidates = self.store.get_structured_candidates(session_id)
        clause_candidates = [item for item in candidates if item["candidate_type"] == "clause"]
        change_candidates = [item for item in candidates if item["candidate_type"] == "change"]

        self.assertEqual(len(clause_candidates), 1)
        self.assertEqual(clause_candidates[0]["clause_id"], "Phần I > Điều 3")
        self.assertGreaterEqual(len(change_candidates), 2)
