"""
src/ingestion/lsu_chunker.py
====================
Module 2: LsuChunker (Logical Semantic Unit Chunker)

Nhận đầu vào là LegalDocument (output của Module 1) và phân mảnh thành
các LsuChunk dựa trên ranh giới Điều (Article) và Khoản (Clause).

Luồng xử lý:
┌─────────────────────────────────────────────────────────────┐
│  Input: LegalDocument (từ Module 1)                         │
│                                                             │
│  1. Duyệt cây DOM: Document → Section → Article → Clause   │
│  2. Tạo breadcrumb cho từng node                            │
│     VD: '[Chương II > Điều 5 > Khoản 3]'                   │
│  3. Tạo LsuChunk với content_with_prefix = breadcrumb + text│
│  4. Nếu chunk quá lớn (> MAX_CHUNK_CHARS) → split theo câu  │
│  5. Trả về list[LsuChunk]                                   │
└─────────────────────────────────────────────────────────────┘

Quy tắc bất biến:
  - Mỗi chunk LUÔN có breadcrumb prefix để cung cấp ngữ cảnh cho embedding.
  - Tables được serialize thành JSON và đính kèm vào chunk metadata.
  - Chunk ở cấp Article giữ toàn bộ nội dung intro + tóm tắt clauses.
  - Chunk ở cấp Clause giữ đầy đủ nội dung (kể cả điểm a, b, c con).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from .models import (
    ArticleNode,
    ClauseNode,
    ContentType,
    DocumentSection,
    LegalDocument,
    LsuChunk,
    NodeType,
    PointNode,
    TableData,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_CHUNK_CHARS: int = 2_000
_DEFAULT_OVERLAP_CHARS: int = 200

# Regex nhận diện ranh giới câu (tiếng Việt + tiếng Anh)
_RE_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?;])\s+(?=[A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ])",
    re.UNICODE,
)


# ---------------------------------------------------------------------------
# Breadcrumb Builder
# ---------------------------------------------------------------------------


class _BreadcrumbBuilder:
    """
    Tạo chuỗi breadcrumb ngữ cảnh cho mỗi node trong Legal DOM.

    Ví dụ output:
        - '[Điều 5]'
        - '[Chương II > Điều 5]'
        - '[Chương II > Điều 5 > Khoản 3]'
        - '[Chương II > Điều 5 > Khoản 3 > Điểm a]'
    """

    @staticmethod
    def for_article(
        article: ArticleNode,
        section: DocumentSection | None,
    ) -> str:
        parts: list[str] = []
        if section:
            parts.append(section.full_title)
        parts.append(article.full_title)
        return f"[{' > '.join(parts)}]"

    @staticmethod
    def for_clause(
        clause: ClauseNode,
        article: ArticleNode,
        section: DocumentSection | None,
    ) -> str:
        parts: list[str] = []
        if section:
            parts.append(section.full_title)
        parts.append(article.full_title)
        parts.append(f"Khoản {clause.number}")
        return f"[{' > '.join(parts)}]"

    @staticmethod
    def for_point(
        point: PointNode,
        clause: ClauseNode,
        article: ArticleNode,
        section: DocumentSection | None,
    ) -> str:
        parts: list[str] = []
        if section:
            parts.append(section.full_title)
        parts.append(article.full_title)
        parts.append(f"Khoản {clause.number}")
        parts.append(f"Điểm {point.label}")
        return f"[{' > '.join(parts)}]"


# ---------------------------------------------------------------------------
# Text Utilities
# ---------------------------------------------------------------------------


def _flatten_clause_content(clause: ClauseNode) -> str:
    """
    Ghép nội dung đầy đủ của một Khoản:
    - Nội dung header của khoản
    - Các điểm (a, b, c) nếu có
    """
    parts: list[str] = []
    if clause.content.strip():
        parts.append(clause.content.strip())
    for point in clause.points:
        point_text = f"{point.number} {point.content}".strip()
        if point_text:
            parts.append(point_text)
    return "\n".join(parts)


def _serialize_tables(tables: list[TableData]) -> list[dict[str, Any]]:
    """Serialize list[TableData] thành list[dict] (JSON-serializable)."""
    return [t.model_dump(exclude_none=True) for t in tables]


def _detect_content_type(
    text: str, tables: list[TableData]
) -> ContentType:
    """Phân loại loại nội dung của chunk."""
    has_text = bool(text.strip())
    has_tables = len(tables) > 0
    if has_text and has_tables:
        return ContentType.MIXED
    if has_tables:
        return ContentType.TABLE
    return ContentType.TEXT


# ---------------------------------------------------------------------------
# Text Splitter (cho chunk quá lớn)
# ---------------------------------------------------------------------------


class _SentenceSplitter:
    """
    Chia text dài thành các sub-chunk theo ranh giới câu với overlap.

    Strategy:
        1. Split theo câu bằng regex
        2. Gom câu vào chunk, không vượt quá max_chars
        3. Mỗi chunk mới bắt đầu bằng một số câu overlap từ chunk trước
    """

    def __init__(
        self, max_chars: int = _DEFAULT_MAX_CHUNK_CHARS, overlap_chars: int = _DEFAULT_OVERLAP_CHARS
    ) -> None:
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def split(self, text: str) -> list[str]:
        """
        Chia text thành list sub-strings, mỗi cái <= max_chars.

        Nếu text ngắn hơn max_chars, trả về [text].
        """
        if len(text) <= self.max_chars:
            return [text]

        sentences = _RE_SENTENCE_BOUNDARY.split(text)
        # Nếu không split được (text liên tục không có dấu kết câu)
        if len(sentences) == 1:
            return self._hard_split(text)

        chunks: list[str] = []
        current_sentences: list[str] = []
        current_len: int = 0

        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len + 1 > self.max_chars and current_sentences:
                chunks.append(" ".join(current_sentences))
                # Overlap: lấy một phần cuối của chunk vừa tạo
                overlap_text = chunks[-1][-self.overlap_chars :]
                current_sentences = [overlap_text, sent]
                current_len = len(overlap_text) + sent_len + 1
            else:
                current_sentences.append(sent)
                current_len += sent_len + 1

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks if chunks else [text]

    def _hard_split(self, text: str) -> list[str]:
        """Hard split theo số ký tự nếu không có ranh giới câu."""
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.max_chars, len(text))
            chunks.append(text[start:end])
            start = end - self.overlap_chars  # overlap
        return chunks


# ---------------------------------------------------------------------------
# LsuChunker — Main Class
# ---------------------------------------------------------------------------


class LsuChunker:
    """
    Logical Semantic Unit Chunker.

    Duyệt cây Legal DOM và tạo ra các LsuChunk dựa trên ranh giới
    Điều (Article) và Khoản (Clause).

    Mỗi chunk:
        - Có breadcrumb prefix ngữ nghĩa (VD: '[Chương II > Điều 5 > Khoản 3]')
        - Chứa nội dung thuần tuý và nội dung với prefix
        - Kèm metadata: số điều, số khoản, section, page, tables
        - Không vượt quá max_chunk_chars (tự động split nếu cần)

    Usage:
        chunker = LsuChunker(max_chunk_chars=2000)
        chunks: list[LsuChunk] = chunker.chunk(legal_document)
    """

    def __init__(
        self,
        max_chunk_chars: int = _DEFAULT_MAX_CHUNK_CHARS,
        overlap_chars: int = _DEFAULT_OVERLAP_CHARS,
        create_article_level_chunks: bool = True,
        create_clause_level_chunks: bool = True,
    ) -> None:
        """
        Args:
            max_chunk_chars:             Số ký tự tối đa mỗi chunk. Nếu vượt quá,
                                         chunk sẽ được chia thêm theo câu.
            overlap_chars:               Số ký tự overlap khi chia sub-chunk.
            create_article_level_chunks: Tạo chunk cấp Điều (tóm tắt toàn bộ điều).
                                         Hữu ích cho truy vấn "Điều X nói về gì?".
            create_clause_level_chunks:  Tạo chunk cấp Khoản (chi tiết từng khoản).
                                         Hữu ích cho truy vấn chi tiết cụ thể.
        """
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars
        self.create_article_level_chunks = create_article_level_chunks
        self.create_clause_level_chunks = create_clause_level_chunks

        self._splitter = _SentenceSplitter(max_chunk_chars, overlap_chars)
        self._breadcrumb = _BreadcrumbBuilder()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, document: LegalDocument) -> list[LsuChunk]:
        """
        Phân mảnh toàn bộ LegalDocument thành list[LsuChunk].

        Args:
            document: LegalDocument từ Module 1 (LegalDocumentParser).

        Returns:
            list[LsuChunk] — danh sách chunk đã được phân mảnh và gắn breadcrumb.
        """
        logger.info(
            f"[Chunker] Bắt đầu chunk tài liệu: {document.file_name} "
            f"| {len(document.iter_all_articles())} điều"
        )

        all_chunks: list[LsuChunk] = []

        # Duyệt sections có chứa articles
        for section in document.sections:
            section_chunks = self._process_section(
                section=section,
                doc_id=document.doc_id,
            )
            all_chunks.extend(section_chunks)

        # Duyệt orphan articles (không thuộc section nào)
        for article in document.orphan_articles:
            article_chunks = self._process_article(
                article=article,
                section=None,
                doc_id=document.doc_id,
            )
            all_chunks.extend(article_chunks)

        logger.info(
            f"[Chunker] Hoàn thành: {len(all_chunks)} chunks từ {document.file_name}"
        )
        return all_chunks

    # ------------------------------------------------------------------
    # Section-level processing
    # ------------------------------------------------------------------

    def _process_section(
        self,
        section: DocumentSection,
        doc_id: str,
    ) -> list[LsuChunk]:
        """Duyệt một DocumentSection và tạo chunks cho tất cả Articles."""
        chunks: list[LsuChunk] = []
        for article in section.articles:
            article_chunks = self._process_article(
                article=article,
                section=section,
                doc_id=doc_id,
            )
            chunks.extend(article_chunks)
        return chunks

    # ------------------------------------------------------------------
    # Article-level processing
    # ------------------------------------------------------------------

    def _process_article(
        self,
        article: ArticleNode,
        section: DocumentSection | None,
        doc_id: str,
    ) -> list[LsuChunk]:
        """
        Xử lý một ArticleNode:
        1. Tạo chunk cấp Điều (nếu enabled) — chứa intro + tóm tắt các khoản
        2. Tạo chunk cấp Khoản cho từng child ClauseNode (nếu enabled)
        """
        chunks: list[LsuChunk] = []
        breadcrumb = self._breadcrumb.for_article(article, section)

        # ── Chunk cấp Điều ──────────────────────────────────────────
        if self.create_article_level_chunks:
            article_chunk = self._make_article_chunk(
                article=article,
                section=section,
                breadcrumb=breadcrumb,
                doc_id=doc_id,
            )
            chunks.append(article_chunk)

        # ── Chunk cấp Khoản ─────────────────────────────────────────
        if self.create_clause_level_chunks:
            for clause in article.clauses:
                clause_chunks = self._process_clause(
                    clause=clause,
                    article=article,
                    section=section,
                    doc_id=doc_id,
                )
                chunks.extend(clause_chunks)

        return chunks

    def _make_article_chunk(
        self,
        article: ArticleNode,
        section: DocumentSection | None,
        breadcrumb: str,
        doc_id: str,
    ) -> LsuChunk:
        """
        Tạo LsuChunk cấp Điều.

        Nội dung = intro của điều + list số khoản (nếu có clauses).
        Ví dụ:
            [Chương II > Điều 5. Quyền của Bên A]
            Bên A có các quyền sau đây:
            Khoản 1: Quyền yêu cầu thanh toán...
            Khoản 2: Quyền chấm dứt hợp đồng...
        """
        parts: list[str] = []

        # Phần intro của điều
        intro = article.intro.strip()
        if intro:
            parts.append(intro)

        # Tóm tắt các khoản (chỉ lấy đầu khoản để không bị quá dài)
        for clause in article.clauses:
            clause_preview = clause.content.strip()
            if clause_preview:
                preview = (
                    clause_preview[:150] + "..."
                    if len(clause_preview) > 150
                    else clause_preview
                )
                parts.append(f"Khoản {clause.number}: {preview}")

        raw_content = "\n".join(parts).strip()

        # Nếu điều không có intro và không có khoản → dùng tiêu đề làm content
        if not raw_content:
            raw_content = article.full_title

        content_with_prefix = f"{breadcrumb}\n{raw_content}"

        all_tables = list(article.tables)
        content_type = _detect_content_type(raw_content, all_tables)

        return LsuChunk(
            chunk_id=f"chunk_{uuid.uuid4().hex[:16]}",
            doc_id=doc_id,
            source_node_id=article.node_id,
            source_node_type=NodeType.ARTICLE,
            breadcrumb=breadcrumb,
            content_with_prefix=content_with_prefix,
            raw_content=raw_content,
            content_type=content_type,
            tables_json=_serialize_tables(all_tables),
            article_number=article.number,
            clause_number=None,
            section_title=section.full_title if section else None,
            page_number=article.page_number,
        )

    # ------------------------------------------------------------------
    # Clause-level processing
    # ------------------------------------------------------------------

    def _process_clause(
        self,
        clause: ClauseNode,
        article: ArticleNode,
        section: DocumentSection | None,
        doc_id: str,
    ) -> list[LsuChunk]:
        """
        Xử lý một ClauseNode:
        - Ghép toàn bộ nội dung khoản (kể cả điểm con)
        - Nếu quá dài → split theo câu thành nhiều sub-chunks
        """
        breadcrumb = self._breadcrumb.for_clause(clause, article, section)
        raw_content = _flatten_clause_content(clause).strip()

        # Nếu nội dung rỗng → skip
        if not raw_content and not clause.tables:
            logger.debug(
                f"[Chunker] Skip clause rỗng: Điều {article.number}, Khoản {clause.number}"
            )
            return []

        all_tables = list(clause.tables)
        content_type = _detect_content_type(raw_content, all_tables)

        # Nếu nội dung ngắn → tạo 1 chunk
        if len(raw_content) <= self.max_chunk_chars:
            return [
                self._make_clause_chunk(
                    raw_content=raw_content,
                    breadcrumb=breadcrumb,
                    clause=clause,
                    article=article,
                    section=section,
                    doc_id=doc_id,
                    all_tables=all_tables,
                    content_type=content_type,
                )
            ]

        # Nội dung quá dài → split theo câu
        logger.debug(
            f"[Chunker] Clause quá lớn ({len(raw_content)} chars), chia nhỏ: "
            f"Điều {article.number}, Khoản {clause.number}"
        )
        sub_texts = self._splitter.split(raw_content)
        chunks: list[LsuChunk] = []
        for idx, sub_text in enumerate(sub_texts):
            # Breadcrumb cho sub-chunk
            sub_breadcrumb = (
                f"{breadcrumb} [phần {idx + 1}/{len(sub_texts)}]"
                if len(sub_texts) > 1
                else breadcrumb
            )
            chunks.append(
                self._make_clause_chunk(
                    raw_content=sub_text,
                    breadcrumb=sub_breadcrumb,
                    clause=clause,
                    article=article,
                    section=section,
                    doc_id=doc_id,
                    # Tables chỉ đính kèm vào sub-chunk đầu tiên
                    all_tables=all_tables if idx == 0 else [],
                    content_type=content_type,
                )
            )
        return chunks

    def _make_clause_chunk(
        self,
        raw_content: str,
        breadcrumb: str,
        clause: ClauseNode,
        article: ArticleNode,
        section: DocumentSection | None,
        doc_id: str,
        all_tables: list[TableData],
        content_type: ContentType,
    ) -> LsuChunk:
        """Factory method tạo một LsuChunk cấp Khoản."""
        content_with_prefix = f"{breadcrumb}\n{raw_content}"
        return LsuChunk(
            chunk_id=f"chunk_{uuid.uuid4().hex[:16]}",
            doc_id=doc_id,
            source_node_id=clause.node_id,
            source_node_type=NodeType.CLAUSE,
            breadcrumb=breadcrumb,
            content_with_prefix=content_with_prefix,
            raw_content=raw_content,
            content_type=content_type,
            tables_json=_serialize_tables(all_tables),
            article_number=article.number,
            clause_number=clause.number,
            section_title=section.full_title if section else None,
            page_number=clause.page_number,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_stats(self, chunks: list[LsuChunk]) -> dict[str, Any]:
        """
        Trả về thống kê về các chunk đã tạo ra.

        Hữu ích để debug và validate pipeline.
        """
        if not chunks:
            return {"total": 0}

        article_chunks = [c for c in chunks if c.source_node_type == NodeType.ARTICLE]
        clause_chunks = [c for c in chunks if c.source_node_type == NodeType.CLAUSE]
        char_counts = [c.char_count for c in chunks]
        chunks_with_tables = [c for c in chunks if c.tables_json]

        return {
            "total": len(chunks),
            "article_level": len(article_chunks),
            "clause_level": len(clause_chunks),
            "with_tables": len(chunks_with_tables),
            "avg_chars": round(sum(char_counts) / len(char_counts), 1),
            "max_chars": max(char_counts),
            "min_chars": min(char_counts),
            "content_types": {
                ct.value: sum(1 for c in chunks if c.content_type == ct)
                for ct in ContentType
            },
        }