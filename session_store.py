from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from comparison.models import ChangeRecord, ClauseResult, ComparisonResult


logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _clip(text: str | None, limit: int = 1200) -> str:
    if not text:
        return ""
    normalized = " ".join(str(text).split())
    return normalized[:limit]


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


class SessionStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    file_a_name TEXT NOT NULL,
                    file_b_name TEXT NOT NULL,
                    file_a_hash TEXT NOT NULL,
                    file_b_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    last_step TEXT NOT NULL,
                    error_code TEXT,
                    error_message TEXT,
                    summary_counts_json TEXT,
                    report_markdown TEXT,
                    analysis_json TEXT
                );

                CREATE TABLE IF NOT EXISTS session_results (
                    result_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    clause_id TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    result_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_changes (
                    change_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    clause_id TEXT NOT NULL,
                    clause_result_id TEXT,
                    citation_id TEXT,
                    change_kind TEXT NOT NULL,
                    impact_level TEXT NOT NULL,
                    semantic_effect TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    search_text TEXT NOT NULL,
                    change_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_citations (
                    citation_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    change_id TEXT,
                    clause_id TEXT,
                    citation_type TEXT NOT NULL,
                    page_a INTEGER,
                    page_b INTEGER,
                    chunk_id_a TEXT,
                    chunk_id_b TEXT,
                    confidence_band TEXT,
                    citation_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    citation_ids_json TEXT,
                    answer_type TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_chunks (
                    chunk_row_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    citation_id TEXT NOT NULL,
                    doc_side TEXT NOT NULL,
                    clause_id TEXT NOT NULL,
                    page INTEGER,
                    chunk_id TEXT,
                    search_text TEXT NOT NULL,
                    chunk_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_status_completed
                    ON sessions(status, completed_at DESC, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_results_session
                    ON session_results(session_id);
                CREATE INDEX IF NOT EXISTS idx_changes_session
                    ON session_changes(session_id);
                CREATE INDEX IF NOT EXISTS idx_citations_session
                    ON session_citations(session_id);
                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON session_messages(session_id, message_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_session
                    ON session_chunks(session_id);
                """
            )

    def recover_interrupted_sessions(self) -> int:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id FROM sessions WHERE status = 'processing'"
            ).fetchall()
            session_ids = [row["session_id"] for row in rows]
            if not session_ids:
                return 0

            now = _utc_now()
            for session_id in session_ids:
                conn.execute(
                    """
                    UPDATE sessions
                    SET status = 'failed',
                        updated_at = ?,
                        last_step = 'recovery',
                        error_code = 'startup_recovery_interrupted',
                        error_message = 'Session was still processing when the server restarted.'
                    WHERE session_id = ?
                    """,
                    (now, session_id),
                )
                self._delete_children(conn, session_id)
            conn.commit()
            return len(session_ids)

    def create_session(
        self,
        file_a_name: str,
        file_b_name: str,
        file_a_hash: str,
        file_b_hash: str,
        last_step: str = "ingestion",
    ) -> str:
        session_id = uuid.uuid4().hex
        now = _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id, status, file_a_name, file_b_name,
                    file_a_hash, file_b_hash, created_at, updated_at, last_step
                )
                VALUES (?, 'processing', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    file_a_name,
                    file_b_name,
                    file_a_hash,
                    file_b_hash,
                    now,
                    now,
                    last_step,
                ),
            )
            conn.commit()
        return session_id

    def update_session_step(self, session_id: str, last_step: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET last_step = ?, updated_at = ?
                WHERE session_id = ? AND status = 'processing'
                """,
                (last_step, _utc_now(), session_id),
            )
            conn.commit()

    def fail_session(self, session_id: str, error_code: str, error_message: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET status = 'failed',
                    updated_at = ?,
                    error_code = ?,
                    error_message = ?
                WHERE session_id = ? AND status != 'completed'
                """,
                (_utc_now(), error_code, error_message[:1000], session_id),
            )
            conn.commit()

    def complete_session(
        self,
        session_id: str,
        comparison_result: ComparisonResult,
        report_markdown: str,
        chunks_a: list[Any],
        chunks_b: list[Any],
    ) -> dict[str, list[dict[str, Any]]]:
        stats_dict = comparison_result.stats.to_dict()
        analysis_dict = comparison_result.to_dict()
        clause_citations: list[dict[str, Any]] = []
        all_citations: list[dict[str, Any]] = []
        now = _utc_now()
        clauses = self._dedupe_clause_results(comparison_result.clauses)

        with self._lock, self._connect() as conn:
            self._delete_children(conn, session_id)
            clause_result_ids: dict[str, str] = {}

            for clause in clauses:
                result_id = _stable_id("result", session_id, clause.clause_id, clause.clause_change_kind)
                clause_result_ids[clause.clause_id] = result_id
                clause_search_text = " ".join(
                    filter(
                        None,
                        [
                            clause.clause_id,
                            clause.summary,
                            clause.llm_analysis.get("summary") if clause.llm_analysis else "",
                            clause.text_a or "",
                            clause.text_b or "",
                            " ".join(record.summary for record in clause.records),
                        ],
                    )
                )
                conn.execute(
                    """
                    INSERT INTO session_results (result_id, session_id, clause_id, search_text, result_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        result_id,
                        session_id,
                        clause.clause_id,
                        clause_search_text,
                        _safe_json(clause.to_dict()),
                    ),
                )

                clause_citation = self._build_clause_citation(session_id, clause)
                clause_citations.append(clause_citation)
                all_citations.append(clause_citation)
                self._insert_citation(conn, session_id, clause_citation)

                for index, record in enumerate(clause.records):
                    change_id = _stable_id(
                        "change",
                        session_id,
                        clause.clause_id,
                        str(index),
                        record.change_kind,
                        record.diff_snippet.old or "",
                        record.diff_snippet.new or "",
                    )
                    change_citation = self._build_change_citation(session_id, clause, record, change_id)
                    all_citations.append(change_citation)
                    self._insert_citation(conn, session_id, change_citation)

                    search_text = " ".join(
                        filter(
                            None,
                            [
                                clause.clause_id,
                                record.summary,
                                record.llm_notes or "",
                                " ".join(record.tags),
                                record.change_kind,
                                record.impact_level,
                                record.semantic_effect,
                                record.diff_snippet.old or "",
                                record.diff_snippet.new or "",
                            ],
                        )
                    )
                    conn.execute(
                        """
                        INSERT INTO session_changes (
                            change_id, session_id, clause_id, clause_result_id, citation_id,
                            change_kind, impact_level, semantic_effect, confidence_score,
                            search_text, change_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            change_id,
                            session_id,
                            clause.clause_id,
                            clause_result_ids[clause.clause_id],
                            change_citation["citation_id"],
                            record.change_kind,
                            record.impact_level,
                            record.semantic_effect,
                            record.confidence_score,
                            search_text,
                            _safe_json(record.to_dict()),
                        ),
                    )

            for doc_side, chunks in (("A", chunks_a), ("B", chunks_b)):
                for chunk in chunks:
                    chunk_payload = self._serialize_chunk(chunk, doc_side)
                    chunk_citation = self._build_chunk_citation(session_id, chunk_payload)
                    all_citations.append(chunk_citation)
                    self._insert_citation(conn, session_id, chunk_citation)
                    conn.execute(
                        """
                        INSERT INTO session_chunks (
                            chunk_row_id, session_id, citation_id, doc_side, clause_id,
                            page, chunk_id, search_text, chunk_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            _stable_id("chunkrow", session_id, chunk_payload["citation_id"]),
                            session_id,
                            chunk_citation["citation_id"],
                            chunk_payload["doc_side"],
                            chunk_payload["clause_id"],
                            chunk_payload["page"],
                            chunk_payload["chunk_id"],
                            chunk_payload["search_text"],
                            _safe_json(chunk_payload),
                        ),
                    )

            conn.execute(
                """
                UPDATE sessions
                SET status = 'completed',
                    updated_at = ?,
                    completed_at = ?,
                    last_step = 'completed',
                    error_code = NULL,
                    error_message = NULL,
                    summary_counts_json = ?,
                    report_markdown = ?,
                    analysis_json = ?
                WHERE session_id = ?
                """,
                (
                    now,
                    now,
                    _safe_json(stats_dict),
                    report_markdown,
                    _safe_json(analysis_dict),
                    session_id,
                ),
            )
            conn.commit()

        return {
            "clause_citations": clause_citations,
            "all_citations": all_citations,
        }

    def _dedupe_clause_results(self, clauses: list[ClauseResult]) -> list[ClauseResult]:
        """Merge duplicate ClauseResult items by clause_id to avoid deterministic ID collisions."""
        merged: dict[str, ClauseResult] = {}
        seen_change_signatures: dict[str, set[str]] = {}

        for clause in clauses:
            clause_id = clause.clause_id.strip() if clause.clause_id else ""
            if not clause_id:
                # Keep unknown clause IDs as-is; they won't collide with deterministic IDs.
                clause_id = f"__unknown_clause__{len(merged)}"
                clause.clause_id = clause_id

            if clause_id not in merged:
                merged[clause_id] = clause
                seen_change_signatures[clause_id] = {
                    self._change_signature(record)
                    for record in clause.records
                }
                continue

            base = merged[clause_id]
            if clause.semantic_similarity > base.semantic_similarity:
                base.semantic_similarity = clause.semantic_similarity
            if not base.text_a and clause.text_a:
                base.text_a = clause.text_a
            if not base.text_b and clause.text_b:
                base.text_b = clause.text_b
            if not base.source_a and clause.source_a:
                base.source_a = clause.source_a
            if not base.source_b and clause.source_b:
                base.source_b = clause.source_b
            if not base.llm_notes and clause.llm_notes:
                base.llm_notes = clause.llm_notes
            if not base.summary and clause.summary:
                base.summary = clause.summary
            if not base.llm_analysis and clause.llm_analysis:
                base.llm_analysis = dict(clause.llm_analysis)

            for record in clause.records:
                signature = self._change_signature(record)
                if signature in seen_change_signatures[clause_id]:
                    continue
                base.records.append(record)
                seen_change_signatures[clause_id].add(signature)

        if len(merged) != len(clauses):
            logger.warning(
                "Merged duplicate clause results before persistence: %s -> %s",
                len(clauses),
                len(merged),
            )
        return list(merged.values())

    @staticmethod
    def _change_signature(record: ChangeRecord) -> str:
        return _stable_id(
            "changesig",
            record.clause_id,
            record.change_kind,
            record.impact_level,
            record.semantic_effect,
            record.diff_snippet.old or "",
            record.diff_snippet.new or "",
            record.summary,
            "|".join(record.tags),
        )

    def list_completed_sessions(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, file_a_name, file_b_name, created_at, completed_at, summary_counts_json
                FROM sessions
                WHERE status = 'completed'
                ORDER BY COALESCE(completed_at, created_at) DESC
                """
            ).fetchall()
        sessions = []
        for row in rows:
            sessions.append(
                {
                    "session_id": row["session_id"],
                    "file_a_name": row["file_a_name"],
                    "file_b_name": row["file_b_name"],
                    "created_at": row["created_at"],
                    "completed_at": row["completed_at"],
                    "summary_counts": json.loads(row["summary_counts_json"] or "{}"),
                }
            )
        return sessions

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            session = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if not session:
                return None
            citations = conn.execute(
                """
                SELECT citation_json
                FROM session_citations
                WHERE session_id = ?
                ORDER BY citation_type, citation_id
                """,
                (session_id,),
            ).fetchall()
            messages = conn.execute(
                """
                SELECT role, content, citation_ids_json, answer_type, created_at
                FROM session_messages
                WHERE session_id = ?
                ORDER BY message_id
                """,
                (session_id,),
            ).fetchall()

        return {
            "session_id": session["session_id"],
            "status": session["status"],
            "file_a_name": session["file_a_name"],
            "file_b_name": session["file_b_name"],
            "created_at": session["created_at"],
            "completed_at": session["completed_at"],
            "last_step": session["last_step"],
            "summary_counts": json.loads(session["summary_counts_json"] or "{}"),
            "report_markdown": session["report_markdown"] or "",
            "analysis": json.loads(session["analysis_json"] or "{}"),
            "citations": [json.loads(row["citation_json"]) for row in citations],
            "messages": [
                {
                    "role": row["role"],
                    "content": row["content"],
                    "citation_ids": json.loads(row["citation_ids_json"] or "[]"),
                    "answer_type": row["answer_type"],
                    "created_at": row["created_at"],
                }
                for row in messages
            ],
        }

    def get_session_status(self, session_id: str) -> str | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row["status"] if row else None

    def get_citation(self, session_id: str, citation_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT citation_json
                FROM session_citations
                WHERE session_id = ? AND citation_id = ?
                """,
                (session_id, citation_id),
            ).fetchone()
        return json.loads(row["citation_json"]) if row else None

    def save_chat_exchange(
        self,
        session_id: str,
        question: str,
        answer_markdown: str,
        citation_ids: Iterable[str],
        answer_type: str,
    ) -> None:
        now = _utc_now()
        citation_ids_list = list(dict.fromkeys(citation_ids))
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_messages (session_id, role, content, citation_ids_json, answer_type, created_at)
                VALUES (?, 'user', ?, ?, NULL, ?)
                """,
                (session_id, question, "[]", now),
            )
            conn.execute(
                """
                INSERT INTO session_messages (session_id, role, content, citation_ids_json, answer_type, created_at)
                VALUES (?, 'assistant', ?, ?, ?, ?)
                """,
                (
                    session_id,
                    answer_markdown,
                    _safe_json(citation_ids_list),
                    answer_type,
                    now,
                ),
            )
            conn.commit()

    def delete_session(self, session_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()

    def delete_all_sessions(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM sessions")
            conn.commit()

    def get_structured_candidates(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            change_rows = conn.execute(
                """
                SELECT clause_id, citation_id, change_kind, impact_level, semantic_effect, confidence_score,
                       search_text, change_json
                FROM session_changes
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchall()
            clause_rows = conn.execute(
                """
                SELECT clause_id, search_text, result_json
                FROM session_results
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchall()

        candidates: list[dict[str, Any]] = []
        for row in change_rows:
            candidates.append(
                {
                    "candidate_type": "change",
                    "clause_id": row["clause_id"],
                    "citation_id": row["citation_id"],
                    "search_text": row["search_text"],
                    "impact_level": row["impact_level"],
                    "semantic_effect": row["semantic_effect"],
                    "confidence_score": row["confidence_score"],
                    "payload": json.loads(row["change_json"]),
                }
            )
        for row in clause_rows:
            candidates.append(
                {
                    "candidate_type": "clause",
                    "clause_id": row["clause_id"],
                    "citation_id": _stable_id("clause", session_id, row["clause_id"]),
                    "search_text": row["search_text"],
                    "impact_level": None,
                    "semantic_effect": None,
                    "confidence_score": 0.0,
                    "payload": json.loads(row["result_json"]),
                }
            )
        return candidates

    def get_chunk_candidates(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT citation_id, doc_side, clause_id, page, chunk_id, search_text, chunk_json
                FROM session_chunks
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchall()
        return [
            {
                "candidate_type": "chunk",
                "citation_id": row["citation_id"],
                "doc_side": row["doc_side"],
                "clause_id": row["clause_id"],
                "page": row["page"],
                "chunk_id": row["chunk_id"],
                "search_text": row["search_text"],
                "payload": json.loads(row["chunk_json"]),
            }
            for row in rows
        ]

    def get_citations(self, session_id: str, citation_ids: Iterable[str]) -> list[dict[str, Any]]:
        ids = list(dict.fromkeys(citation_ids))
        if not ids:
            return []
        placeholders = ", ".join("?" for _ in ids)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT citation_json
                FROM session_citations
                WHERE session_id = ? AND citation_id IN ({placeholders})
                """,
                [session_id, *ids],
            ).fetchall()
        citations = [json.loads(row["citation_json"]) for row in rows]
        citations.sort(key=lambda item: ids.index(item["citation_id"]) if item["citation_id"] in ids else len(ids))
        return citations

    def _delete_children(self, conn: sqlite3.Connection, session_id: str) -> None:
        for table in (
            "session_results",
            "session_changes",
            "session_citations",
            "session_messages",
            "session_chunks",
        ):
            conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))

    def _insert_citation(self, conn: sqlite3.Connection, session_id: str, citation: dict[str, Any]) -> None:
        conn.execute(
            """
            INSERT INTO session_citations (
                citation_id, session_id, change_id, clause_id, citation_type,
                page_a, page_b, chunk_id_a, chunk_id_b, confidence_band, citation_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                citation["citation_id"],
                session_id,
                citation.get("change_id"),
                citation.get("clause_id"),
                citation["citation_type"],
                citation.get("page_a"),
                citation.get("page_b"),
                citation.get("chunk_id_a"),
                citation.get("chunk_id_b"),
                citation.get("confidence_band"),
                _safe_json(citation),
            ),
        )

    def _build_clause_citation(self, session_id: str, clause: ClauseResult) -> dict[str, Any]:
        best_record = max(
            clause.records,
            key=lambda record: record.confidence_score,
            default=None,
        )
        source_a = clause.source_a
        source_b = clause.source_b
        return {
            "citation_id": _stable_id("clause", session_id, clause.clause_id),
            "clause_id": clause.clause_id,
            "citation_type": "clause",
            "change_kind": clause.clause_change_kind,
            "filename_a": source_a.file if source_a else None,
            "filename_b": source_b.file if source_b else None,
            "page_a": source_a.page if source_a else None,
            "page_b": source_b.page if source_b else None,
            "page_end_a": source_a.page_end if source_a else None,
            "page_end_b": source_b.page_end if source_b else None,
            "line_start_a": source_a.line_start if source_a else None,
            "line_start_b": source_b.line_start if source_b else None,
            "line_end_a": source_a.line_end if source_a else None,
            "line_end_b": source_b.line_end if source_b else None,
            "char_start_a": source_a.char_start if source_a else None,
            "char_start_b": source_b.char_start if source_b else None,
            "char_end_a": source_a.char_end if source_a else None,
            "char_end_b": source_b.char_end if source_b else None,
            "chunk_id_a": source_a.chunk_id if source_a else None,
            "chunk_id_b": source_b.chunk_id if source_b else None,
            "text_a": clause.text_a,
            "text_b": clause.text_b,
            "excerpt_a": _clip(clause.text_a),
            "excerpt_b": _clip(clause.text_b),
            "excerpt": _clip(clause.text_b or clause.text_a),
            "diff_snippet": best_record.diff_snippet.to_dict() if best_record else None,
            "confidence_band": best_record.confidence_band if best_record else None,
        }

    def _build_change_citation(
        self,
        session_id: str,
        clause: ClauseResult,
        record: ChangeRecord,
        change_id: str,
    ) -> dict[str, Any]:
        source_a = record.source_a or clause.source_a
        source_b = record.source_b or clause.source_b
        return {
            "citation_id": _stable_id("citation", session_id, change_id),
            "change_id": change_id,
            "clause_id": clause.clause_id,
            "citation_type": "change",
            "change_kind": record.change_kind,
            "filename_a": source_a.file if source_a else None,
            "filename_b": source_b.file if source_b else None,
            "page_a": source_a.page if source_a else None,
            "page_b": source_b.page if source_b else None,
            "page_end_a": source_a.page_end if source_a else None,
            "page_end_b": source_b.page_end if source_b else None,
            "line_start_a": source_a.line_start if source_a else None,
            "line_start_b": source_b.line_start if source_b else None,
            "line_end_a": source_a.line_end if source_a else None,
            "line_end_b": source_b.line_end if source_b else None,
            "char_start_a": source_a.char_start if source_a else None,
            "char_start_b": source_b.char_start if source_b else None,
            "char_end_a": source_a.char_end if source_a else None,
            "char_end_b": source_b.char_end if source_b else None,
            "chunk_id_a": source_a.chunk_id if source_a else None,
            "chunk_id_b": source_b.chunk_id if source_b else None,
            "text_a": record.diff_snippet.old or clause.text_a,
            "text_b": record.diff_snippet.new or clause.text_b,
            "excerpt_a": _clip(record.diff_snippet.old or clause.text_a),
            "excerpt_b": _clip(record.diff_snippet.new or clause.text_b),
            "excerpt": _clip(record.diff_snippet.new or record.diff_snippet.old),
            "diff_snippet": record.diff_snippet.to_dict(),
            "confidence_band": record.confidence_band,
            "summary": record.summary,
            "tags": list(record.tags),
        }

    def _serialize_chunk(self, chunk: Any, doc_side: str) -> dict[str, Any]:
        metadata = dict(getattr(chunk, "metadata", {}) or {})
        clause_id = (
            getattr(chunk, "article_number", None)
            or getattr(chunk, "title", None)
            or metadata.get("mid_level")
            or metadata.get("top_level")
            or f"Chunk {getattr(chunk, 'raw_index', 0)}"
        )
        chunk_id = (
            metadata.get("chunk_id")
            or metadata.get("uid")
            or (
                f"{getattr(chunk, 'doc_id', 'doc')}__{clause_id}"
                f"__idx{getattr(chunk, 'raw_index', 0)}__sub{getattr(chunk, 'sub_index', 0)}"
            )
        )
        payload = {
            "citation_id": _stable_id(
                "chunk",
                getattr(chunk, "doc_id", "doc"),
                str(getattr(chunk, "raw_index", 0)),
                str(getattr(chunk, "sub_index", 0)),
                str(chunk_id),
            ),
            "doc_side": doc_side,
            "doc_label": getattr(chunk, "doc_label", None),
            "doc_id": getattr(chunk, "doc_id", None),
            "clause_id": clause_id,
            "title": getattr(chunk, "title", None),
            "page": getattr(chunk, "page", None),
            "page_end": getattr(chunk, "page_end", None) or metadata.get("page_end") or metadata.get("page_last"),
            "line_start": getattr(chunk, "line_start", None) or metadata.get("line_start"),
            "line_end": getattr(chunk, "line_end", None) or metadata.get("line_end"),
            "char_start": getattr(chunk, "char_start", None) or metadata.get("char_start"),
            "char_end": getattr(chunk, "char_end", None) or metadata.get("char_end"),
            "raw_index": getattr(chunk, "raw_index", None),
            "sub_index": getattr(chunk, "sub_index", None),
            "chunk_id": chunk_id,
            "content": getattr(chunk, "content", ""),
            "metadata": metadata,
        }
        payload["search_text"] = " ".join(
            filter(
                None,
                [
                    payload["clause_id"],
                    payload["title"],
                    payload["content"],
                    metadata.get("top_level"),
                    metadata.get("mid_level"),
                ],
            )
        )
        return payload

    def _build_chunk_citation(self, session_id: str, chunk_payload: dict[str, Any]) -> dict[str, Any]:
        citation_id = _stable_id("citation", session_id, chunk_payload["citation_id"])
        is_doc_a = chunk_payload["doc_side"] == "A"
        return {
            "citation_id": citation_id,
            "clause_id": chunk_payload["clause_id"],
            "citation_type": "chunk",
            "change_kind": "CHUNK",
            "filename_a": chunk_payload["doc_id"] if is_doc_a else None,
            "filename_b": chunk_payload["doc_id"] if not is_doc_a else None,
            "page_a": chunk_payload["page"] if is_doc_a else None,
            "page_b": chunk_payload["page"] if not is_doc_a else None,
            "page_end_a": chunk_payload.get("page_end") if is_doc_a else None,
            "page_end_b": chunk_payload.get("page_end") if not is_doc_a else None,
            "line_start_a": chunk_payload.get("line_start") if is_doc_a else None,
            "line_start_b": chunk_payload.get("line_start") if not is_doc_a else None,
            "line_end_a": chunk_payload.get("line_end") if is_doc_a else None,
            "line_end_b": chunk_payload.get("line_end") if not is_doc_a else None,
            "char_start_a": chunk_payload.get("char_start") if is_doc_a else None,
            "char_start_b": chunk_payload.get("char_start") if not is_doc_a else None,
            "char_end_a": chunk_payload.get("char_end") if is_doc_a else None,
            "char_end_b": chunk_payload.get("char_end") if not is_doc_a else None,
            "chunk_id_a": chunk_payload["chunk_id"] if is_doc_a else None,
            "chunk_id_b": chunk_payload["chunk_id"] if not is_doc_a else None,
            "text_a": chunk_payload["content"] if is_doc_a else None,
            "text_b": chunk_payload["content"] if not is_doc_a else None,
            "excerpt_a": _clip(chunk_payload["content"]) if is_doc_a else "",
            "excerpt_b": _clip(chunk_payload["content"]) if not is_doc_a else "",
            "excerpt": _clip(chunk_payload["content"]),
            "diff_snippet": None,
            "confidence_band": None,
        }


def is_quote_request(question: str) -> bool:
    normalized = question.lower()
    keywords = (
        "nguyên văn",
        "trích",
        "trích nguyên văn",
        "nội dung",
        "bản mới nói gì",
        "bản cũ nói gì",
    )
    return any(keyword in normalized for keyword in keywords)
