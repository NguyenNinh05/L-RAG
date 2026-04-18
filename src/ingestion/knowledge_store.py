"""
src/ingestion/knowledge_store.py
==========================
Module 3: HybridGraphBuilder

Xây dựng Hybrid Knowledge Graph từ các LsuChunk (output của Module 2).

Luồng xử lý:
┌─────────────────────────────────────────────────────────────────────┐
│  Input: LegalDocument + list[LsuChunk]                              │
│                                                                     │
│  Graph DB (Kuzu Embedded):                                          │
│    1. Khởi tạo schema (NodeTables, EdgeTables) nếu chưa có         │
│    2. Upsert Document node                                          │
│    3. Upsert Article / Clause nodes từ DOM tree                     │
│    4. Tạo CONTAINS edges (parent → child)                           │
│    5. Tạo PRECEDES edges (node[i] → node[i+1]) cùng cấp            │
│    6. Regex scan content → tạo REFERENCES edges ("theo Điều X")    │
│                                                                     │
│  Vector DB (ChromaDB Persistent):                                   │
│    1. Nhúng văn bản = LsuChunk.content_with_prefix                 │
│    2. Lưu VectorRecord với metadata.node_id → liên kết Graph DB    │
│                                                                     │
│  CRITICAL: metadata.node_id là cầu nối duy nhất Vector ↔ Graph.   │
└─────────────────────────────────────────────────────────────────────┘

Ghi chú:
  - Kuzu dùng embedded mode: không cần server, data lưu tại local path.
  - ChromaDB dùng persistent mode: data persist qua các lần chạy.
  - Embedding function là PLACEHOLDER — thay bằng model thực khi cần.
  - Tất cả operations được log đầy đủ để dễ debug.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from .models import (
    ArticleNode,
    ClauseNode,
    DocumentSection,
    EdgeType,
    GraphEdge,
    GraphNode,
    LegalDocument,
    LsuChunk,
    NodeType,
    VectorMetadata,
    VectorRecord,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex Patterns — nhận diện tham chiếu nội bộ trong văn bản pháp lý VN
# ---------------------------------------------------------------------------

# "theo quy định tại Điều 5", "tại Điều 15 Khoản 2", "Điều 3 của hợp đồng này"
_RE_ARTICLE_REF = re.compile(
    r"(?:theo\s+(?:quy\s+định\s+)?tại\s+|tại\s+|theo\s+)?"
    r"(?P<keyword>Điều|điều)\s+(?P<article_num>\d+|[IVXLCDM]+)"
    r"(?:\s*[,.]?\s*(?:Khoản|khoản)\s+(?P<clause_num>\d+))?",
    re.UNICODE | re.IGNORECASE,
)

# "khoản 3 Điều 15" (thứ tự ngược)
_RE_CLAUSE_FIRST_REF = re.compile(
    r"(?:Khoản|khoản)\s+(?P<clause_num>\d+)\s+"
    r"(?:Điều|điều)\s+(?P<article_num>\d+|[IVXLCDM]+)",
    re.UNICODE | re.IGNORECASE,
)

# Cụm "quy định tại" đứng trước reference (để làm context)
_RE_REF_CONTEXT = re.compile(
    r"(?:theo\s+quy\s+định\s+tại|theo\s+quy\s+định\s+của|căn\s+cứ\s+(?:vào\s+)?|"
    r"theo\s+|tại\s+|quy\s+định\s+tại\s+).{0,80}(?:Điều|điều)\s+\d+",
    re.UNICODE | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Embedding Protocol (Placeholder)
# ---------------------------------------------------------------------------


class EmbeddingFunction(Protocol):
    """Protocol cho embedding function. Implement để thay thế placeholder."""

    def __call__(self, texts: list[str]) -> list[list[float]]:
        """Nhận list[str], trả về list[list[float]] (vector embeddings)."""
        ...


def placeholder_embedding_fn(texts: list[str]) -> list[list[float]]:
    """
    PLACEHOLDER Embedding Function.

    Trả về zero vectors có chiều = EMBEDDING_DIM từ config.
    Thay thế bằng model thực (sentence-transformers, OpenAI, etc.) khi sẵn sàng.

    Ví dụ thay thế bằng sentence-transformers:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
        def real_embed(texts):
            return model.encode(texts, normalize_embeddings=True).tolist()
    """
    try:
        from src.config import EMBEDDING_DIM
    except ImportError:
        EMBEDDING_DIM = 1024

    logger.debug(
        f"[Embedding] PLACEHOLDER — trả về zero vectors dim={EMBEDDING_DIM} "
        f"cho {len(texts)} texts"
    )
    return [[0.0] * EMBEDDING_DIM for _ in texts]


# ---------------------------------------------------------------------------
# Kuzu Schema Manager
# ---------------------------------------------------------------------------


class _KuzuSchemaManager:
    """
    Quản lý schema của Kuzu database.

    Schema:
        Node tables:
            - Document(doc_id STRING, file_name STRING, doc_title STRING,
                       doc_number STRING, signing_date STRING, ingested_at STRING)
            - LegalNode(node_id STRING, node_type STRING, doc_id STRING,
                        article_number STRING, clause_number STRING,
                        section_title STRING, content_summary STRING,
                        page_number INT64, char_count INT64, breadcrumb STRING)

        Edge tables:
            - CONTAINS(Document → LegalNode, LegalNode → LegalNode)
            - REFERENCES(LegalNode → LegalNode, context STRING, weight DOUBLE)
            - PRECEDES(LegalNode → LegalNode)
    """

    # DDL Statements
    _DDL_NODE_DOCUMENT = """
        CREATE NODE TABLE IF NOT EXISTS Document(
            doc_id STRING,
            file_name STRING,
            doc_title STRING,
            doc_number STRING,
            signing_date STRING,
            ingested_at STRING,
            PRIMARY KEY (doc_id)
        )
    """

    _DDL_NODE_LEGAL = """
        CREATE NODE TABLE IF NOT EXISTS LegalNode(
            node_id STRING,
            node_type STRING,
            doc_id STRING,
            article_number STRING,
            clause_number STRING,
            section_title STRING,
            content_summary STRING,
            page_number INT64,
            char_count INT64,
            breadcrumb STRING,
            PRIMARY KEY (node_id)
        )
    """

    _DDL_EDGE_CONTAINS = """
        CREATE REL TABLE IF NOT EXISTS CONTAINS(
            FROM Document TO LegalNode,
            FROM LegalNode TO LegalNode
        )
    """

    _DDL_EDGE_REFERENCES = """
        CREATE REL TABLE IF NOT EXISTS REFERENCES(
            FROM LegalNode TO LegalNode,
            context STRING,
            weight DOUBLE
        )
    """

    _DDL_EDGE_PRECEDES = """
        CREATE REL TABLE IF NOT EXISTS PRECEDES(
            FROM LegalNode TO LegalNode
        )
    """

    @classmethod
    def initialize(cls, conn: Any) -> None:
        """
        Tạo tất cả node/edge tables nếu chưa tồn tại.

        Args:
            conn: kuzu.Connection object
        """
        logger.debug("[KuzuSchema] Khởi tạo schema...")
        for ddl in [
            cls._DDL_NODE_DOCUMENT,
            cls._DDL_NODE_LEGAL,
            cls._DDL_EDGE_CONTAINS,
            cls._DDL_EDGE_REFERENCES,
            cls._DDL_EDGE_PRECEDES,
        ]:
            conn.execute(ddl.strip())
        logger.debug("[KuzuSchema] Schema sẵn sàng.")


# ---------------------------------------------------------------------------
# Article-to-NodeId Index
# ---------------------------------------------------------------------------


class _ArticleIndex:
    """
    Index ánh xạ số điều → node_id trong Graph DB.

    Dùng để tra cứu node đích khi tạo REFERENCES edges.
    Ví dụ: article_num="5" → "article_abc123"
    """

    def __init__(self) -> None:
        # {article_num_str: node_id}
        self._article_map: dict[str, str] = {}
        # {(article_num_str, clause_num_str): node_id}
        self._clause_map: dict[tuple[str, str], str] = {}

    def register_article(self, article: ArticleNode) -> None:
        self._article_map[str(article.number)] = article.node_id

    def register_clause(self, article: ArticleNode, clause: ClauseNode) -> None:
        key = (str(article.number), str(clause.number))
        self._clause_map[key] = clause.node_id

    def lookup_article(self, article_num: str) -> str | None:
        return self._article_map.get(article_num.strip())

    def lookup_clause(self, article_num: str, clause_num: str) -> str | None:
        return self._clause_map.get((article_num.strip(), clause_num.strip()))

    def __len__(self) -> int:
        return len(self._article_map) + len(self._clause_map)


# ---------------------------------------------------------------------------
# HybridGraphBuilder — Main Class
# ---------------------------------------------------------------------------


class HybridGraphBuilder:
    """
    Xây dựng Hybrid Knowledge Graph kết hợp:
      - Graph DB (Kuzu Embedded): cấu trúc và quan hệ pháp lý
      - Vector DB (ChromaDB Persistent): semantic search qua embedding

    Usage:
        builder = HybridGraphBuilder(
            kuzu_db_path="./graph_db/legal_graph",
            chroma_db_path="./vector_db/chroma_legal",
            embedding_fn=my_embedding_model,   # hoặc để None → dùng placeholder
        )
        result = builder.build(document, chunks)
        print(result)  # {'nodes': 42, 'edges': 18, 'vectors': 90}
    """

    def __init__(
        self,
        kuzu_db_path: str | Path,
        chroma_db_path: str | Path,
        chroma_collection_name: str = "legal_documents",
        embedding_fn: EmbeddingFunction | None = None,
        embedding_batch_size: int = 32,
    ) -> None:
        """
        Args:
            kuzu_db_path:            Đường dẫn đến Kuzu DB directory.
            chroma_db_path:          Đường dẫn đến ChromaDB persistent directory.
            chroma_collection_name:  Tên collection trong ChromaDB.
            embedding_fn:            Hàm embedding. None → dùng placeholder.
            embedding_batch_size:    Số texts nhúng mỗi batch để tránh OOM.
        """
        self.kuzu_db_path = Path(kuzu_db_path)
        self.chroma_db_path = Path(chroma_db_path)
        self.chroma_collection_name = chroma_collection_name
        self.embedding_fn: EmbeddingFunction = embedding_fn or placeholder_embedding_fn
        self.embedding_batch_size = embedding_batch_size

        # Lazy-initialized connections
        self._kuzu_db: Any = None
        self._kuzu_conn: Any = None
        self._chroma_client: Any = None
        self._chroma_collection: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        document: LegalDocument,
        chunks: list[LsuChunk],
    ) -> dict[str, int]:
        """
        Entry point chính: xây dựng toàn bộ Hybrid Knowledge Graph.

        Args:
            document: LegalDocument (output Module 1)
            chunks:   list[LsuChunk] (output Module 2)

        Returns:
            dict thống kê: {'nodes': int, 'edges': int, 'vectors': int}
        """
        start_ts = time.perf_counter()
        logger.info(
            f"[GraphBuilder] Bắt đầu build: {document.file_name} "
            f"| {len(chunks)} chunks"
        )

        # --- Khởi tạo kết nối ---
        self._init_kuzu()
        self._init_chroma()

        # --- Build Article Index (để resolve REFERENCES edges) ---
        article_index = self._build_article_index(document)

        # --- Graph DB: upsert nodes & structure edges ---
        node_count = self._build_graph(document, article_index)

        # --- Graph DB: REFERENCES edges từ regex scan ---
        ref_edge_count = self._build_reference_edges(document, article_index)

        # --- Vector DB: embed & store chunks ---
        vector_count = self._build_vector_store(chunks)

        elapsed = time.perf_counter() - start_ts
        result = {
            "nodes": node_count,
            "edges": ref_edge_count,
            "vectors": vector_count,
            "elapsed_seconds": round(elapsed, 2),
        }
        logger.info(
            f"[GraphBuilder] Hoàn thành: {result} | {elapsed:.1f}s"
        )
        return result

    def close(self) -> None:
        """Đóng các kết nối database."""
        if self._kuzu_conn is not None:
            try:
                self._kuzu_conn.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        logger.debug("[GraphBuilder] Đã đóng kết nối Kuzu.")

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_kuzu(self) -> None:
        """Khởi tạo Kuzu DB và schema."""
        try:
            import kuzu  # type: ignore[import]
        except ImportError:
            raise RuntimeError(
                "kuzu chưa được cài đặt. Chạy: pip install kuzu"
            )

        # Chỉ tạo thư mục CHA — Kuzu tự tạo thư mục DB của mình.
        # Nếu tạo trước bằng mkdir(), Kuzu sẽ báo lỗi
        # "Database path cannot be a directory".
        self.kuzu_db_path.parent.mkdir(parents=True, exist_ok=True)

        # Nếu path đã là thư mục rỗng (do run trước bị interrupt), xóa đi
        # để Kuzu có thể tạo mới hoàn toàn.
        if self.kuzu_db_path.exists() and self.kuzu_db_path.is_dir():
            import os
            if not any(self.kuzu_db_path.iterdir()):  # rỗng
                self.kuzu_db_path.rmdir()
                logger.debug(f"[Kuzu] Đã xóa thư mục rỗng cũ: {self.kuzu_db_path}")

        logger.debug(f"[Kuzu] Kết nối: {self.kuzu_db_path}")
        self._kuzu_db = kuzu.Database(str(self.kuzu_db_path))
        self._kuzu_conn = kuzu.Connection(self._kuzu_db)

        # Khởi tạo schema
        _KuzuSchemaManager.initialize(self._kuzu_conn)

    def _init_chroma(self) -> None:
        """Khởi tạo ChromaDB persistent client và collection."""
        try:
            import chromadb  # type: ignore[import]
        except ImportError:
            raise RuntimeError(
                "chromadb chưa được cài đặt. Chạy: pip install chromadb"
            )

        self.chroma_db_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"[ChromaDB] Kết nối: {self.chroma_db_path}")
        self._chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_db_path)
        )
        self._chroma_collection = self._chroma_client.get_or_create_collection(
            name=self.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.debug(
            f"[ChromaDB] Collection '{self.chroma_collection_name}' sẵn sàng. "
            f"Hiện có {self._chroma_collection.count()} vectors."
        )

    # ------------------------------------------------------------------
    # Article Index Builder
    # ------------------------------------------------------------------

    def _build_article_index(self, document: LegalDocument) -> _ArticleIndex:
        """
        Xây dựng index ánh xạ số điều/khoản → node_id.
        Dùng để resolve REFERENCES edges sau này.
        """
        index = _ArticleIndex()
        for article in document.iter_all_articles():
            index.register_article(article)
            for clause in article.clauses:
                index.register_clause(article, clause)

        logger.debug(
            f"[GraphBuilder] Article index: {len(index)} entries"
        )
        return index

    # ------------------------------------------------------------------
    # Graph DB Builder
    # ------------------------------------------------------------------

    def _build_graph(
        self, document: LegalDocument, article_index: _ArticleIndex
    ) -> int:
        """
        Upsert toàn bộ structure vào Kuzu:
        - Document node
        - LegalNode cho mỗi Article/Clause
        - CONTAINS edges (parent → child)
        - PRECEDES edges (tuần tự)

        Returns:
            Tổng số nodes đã upsert.
        """
        logger.debug("[Kuzu] Upsert Document node...")
        self._upsert_document_node(document)
        node_count = 1

        # Duyệt sections
        for section in document.sections:
            for i, article in enumerate(section.articles):
                self._upsert_article(document, article, section)
                node_count += 1

                # CONTAINS: Document → Article
                self._create_contains_edge_doc_to_node(
                    document.doc_id, article.node_id
                )

                # PRECEDES: Article[i] → Article[i+1]
                if i > 0:
                    prev_article = section.articles[i - 1]
                    self._create_precedes_edge(
                        prev_article.node_id, article.node_id
                    )

                # Clauses
                for j, clause in enumerate(article.clauses):
                    self._upsert_clause(document, clause, article, section)
                    node_count += 1

                    # CONTAINS: Article → Clause
                    self._create_contains_edge_node_to_node(
                        article.node_id, clause.node_id
                    )

                    # PRECEDES: Clause[j] → Clause[j+1]
                    if j > 0:
                        prev_clause = article.clauses[j - 1]
                        self._create_precedes_edge(
                            prev_clause.node_id, clause.node_id
                        )

        # Orphan articles (không thuộc section nào)
        for i, article in enumerate(document.orphan_articles):
            self._upsert_article(document, article, section=None)
            node_count += 1

            self._create_contains_edge_doc_to_node(
                document.doc_id, article.node_id
            )

            if i > 0:
                prev_article = document.orphan_articles[i - 1]
                self._create_precedes_edge(prev_article.node_id, article.node_id)

            for j, clause in enumerate(article.clauses):
                self._upsert_clause(document, clause, article, section=None)
                node_count += 1

                self._create_contains_edge_node_to_node(
                    article.node_id, clause.node_id
                )

                if j > 0:
                    prev_clause = article.clauses[j - 1]
                    self._create_precedes_edge(
                        prev_clause.node_id, clause.node_id
                    )

        logger.info(f"[Kuzu] Upsert xong {node_count} nodes.")
        return node_count

    # ── Upsert Methods ───────────────────────────────────────────────

    def _upsert_document_node(self, document: LegalDocument) -> None:
        """Upsert Document node vào Kuzu."""
        cypher = """
            MERGE (d:Document {doc_id: $doc_id})
            SET d.file_name = $file_name,
                d.doc_title = $doc_title,
                d.doc_number = $doc_number,
                d.signing_date = $signing_date,
                d.ingested_at = $ingested_at
        """
        self._kuzu_conn.execute(
            cypher,
            parameters={
                "doc_id": document.doc_id,
                "file_name": document.file_name,
                "doc_title": document.doc_title or "",
                "doc_number": document.doc_number or "",
                "signing_date": document.signing_date or "",
                "ingested_at": document.ingested_at.isoformat(),
            },
        )

    def _upsert_article(
        self,
        document: LegalDocument,
        article: ArticleNode,
        section: DocumentSection | None,
    ) -> None:
        """Upsert một ArticleNode vào Kuzu LegalNode table."""
        cypher = """
            MERGE (n:LegalNode {node_id: $node_id})
            SET n.node_type = $node_type,
                n.doc_id = $doc_id,
                n.article_number = $article_number,
                n.clause_number = $clause_number,
                n.section_title = $section_title,
                n.content_summary = $content_summary,
                n.page_number = $page_number,
                n.char_count = $char_count,
                n.breadcrumb = $breadcrumb
        """
        content_summary = (
            (article.intro or article.title or article.full_title)[:300]
        )
        breadcrumb = (
            f"[{section.full_title} > {article.full_title}]"
            if section
            else f"[{article.full_title}]"
        )
        self._kuzu_conn.execute(
            cypher,
            parameters={
                "node_id": article.node_id,
                "node_type": NodeType.ARTICLE.value,
                "doc_id": document.doc_id,
                "article_number": str(article.number),
                "clause_number": "",
                "section_title": section.full_title if section else "",
                "content_summary": content_summary,
                "page_number": article.page_number or 0,
                "char_count": len(article.intro),
                "breadcrumb": breadcrumb,
            },
        )

    def _upsert_clause(
        self,
        document: LegalDocument,
        clause: ClauseNode,
        article: ArticleNode,
        section: DocumentSection | None,
    ) -> None:
        """Upsert một ClauseNode vào Kuzu LegalNode table."""
        cypher = """
            MERGE (n:LegalNode {node_id: $node_id})
            SET n.node_type = $node_type,
                n.doc_id = $doc_id,
                n.article_number = $article_number,
                n.clause_number = $clause_number,
                n.section_title = $section_title,
                n.content_summary = $content_summary,
                n.page_number = $page_number,
                n.char_count = $char_count,
                n.breadcrumb = $breadcrumb
        """
        section_str = section.full_title if section else ""
        breadcrumb = (
            f"[{section_str} > {article.full_title} > Khoản {clause.number}]"
            if section_str
            else f"[{article.full_title} > Khoản {clause.number}]"
        )
        self._kuzu_conn.execute(
            cypher,
            parameters={
                "node_id": clause.node_id,
                "node_type": NodeType.CLAUSE.value,
                "doc_id": document.doc_id,
                "article_number": str(article.number),
                "clause_number": str(clause.number),
                "section_title": section_str,
                "content_summary": clause.content[:300],
                "page_number": clause.page_number or 0,
                "char_count": len(clause.content),
                "breadcrumb": breadcrumb,
            },
        )

    # ── Edge Helpers ─────────────────────────────────────────────────

    def _create_contains_edge_doc_to_node(
        self, doc_id: str, node_id: str
    ) -> None:
        """Tạo CONTAINS edge: Document → LegalNode."""
        cypher = """
            MATCH (d:Document {doc_id: $doc_id}), (n:LegalNode {node_id: $node_id})
            MERGE (d)-[:CONTAINS]->(n)
        """
        try:
            self._kuzu_conn.execute(
                cypher, parameters={"doc_id": doc_id, "node_id": node_id}
            )
        except Exception as exc:
            logger.debug(f"[Kuzu] CONTAINS edge skip (có thể đã tồn tại): {exc}")

    def _create_contains_edge_node_to_node(
        self, parent_id: str, child_id: str
    ) -> None:
        """Tạo CONTAINS edge: LegalNode → LegalNode."""
        cypher = """
            MATCH (p:LegalNode {node_id: $parent_id}), (c:LegalNode {node_id: $child_id})
            MERGE (p)-[:CONTAINS]->(c)
        """
        try:
            self._kuzu_conn.execute(
                cypher,
                parameters={"parent_id": parent_id, "child_id": child_id},
            )
        except Exception as exc:
            logger.debug(f"[Kuzu] CONTAINS edge skip: {exc}")

    def _create_precedes_edge(self, from_id: str, to_id: str) -> None:
        """Tạo PRECEDES edge: LegalNode → LegalNode."""
        cypher = """
            MATCH (a:LegalNode {node_id: $from_id}), (b:LegalNode {node_id: $to_id})
            MERGE (a)-[:PRECEDES]->(b)
        """
        try:
            self._kuzu_conn.execute(
                cypher,
                parameters={"from_id": from_id, "to_id": to_id},
            )
        except Exception as exc:
            logger.debug(f"[Kuzu] PRECEDES edge skip: {exc}")

    # ------------------------------------------------------------------
    # REFERENCES Edges — Regex-based cross-reference detection
    # ------------------------------------------------------------------

    def _build_reference_edges(
        self,
        document: LegalDocument,
        article_index: _ArticleIndex,
    ) -> int:
        """
        Scan nội dung từng node để tìm tham chiếu nội bộ,
        sau đó tạo REFERENCES edges.

        Patterns được nhận diện:
          - "theo quy định tại Điều 5"
          - "tại Điều 15 Khoản 2"
          - "Điều 3 của hợp đồng này"
          - "Khoản 3 Điều 10"

        Returns:
            Số REFERENCES edges đã tạo.
        """
        logger.debug("[Kuzu] Tìm kiếm REFERENCES edges...")
        edge_count = 0

        for article in document.iter_all_articles():
            # Scan intro của Article
            edge_count += self._scan_and_create_refs(
                source_node_id=article.node_id,
                text=article.intro + " " + article.title,
                doc_id=document.doc_id,
                article_index=article_index,
            )

            # Scan content của từng Clause
            for clause in article.clauses:
                full_text = clause.content + " ".join(
                    p.content for p in clause.points
                )
                edge_count += self._scan_and_create_refs(
                    source_node_id=clause.node_id,
                    text=full_text,
                    doc_id=document.doc_id,
                    article_index=article_index,
                )

        logger.info(f"[Kuzu] Tạo {edge_count} REFERENCES edges.")
        return edge_count

    def _scan_and_create_refs(
        self,
        source_node_id: str,
        text: str,
        doc_id: str,
        article_index: _ArticleIndex,
    ) -> int:
        """
        Scan một đoạn text, tìm tất cả tham chiếu đến Điều/Khoản,
        và tạo REFERENCES edges trong Kuzu.

        Returns:
            Số edges đã tạo.
        """
        if not text.strip():
            return 0

        created = 0
        seen_refs: set[str] = set()  # Tránh tạo edge trùng lặp

        # Pattern 1: "Điều X" hoặc "Điều X Khoản Y"
        for match in _RE_ARTICLE_REF.finditer(text):
            article_num = match.group("article_num").strip()
            clause_num_raw = match.group("clause_num")
            clause_num = clause_num_raw.strip() if clause_num_raw else None

            # Lấy context xung quanh match (±50 chars)
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end].strip()

            target_id = self._resolve_reference_target(
                article_index, article_num, clause_num
            )
            if target_id and target_id != source_node_id:
                ref_key = f"{source_node_id}→{target_id}"
                if ref_key not in seen_refs:
                    self._create_references_edge(
                        source_id=source_node_id,
                        target_id=target_id,
                        context=context,
                        doc_id=doc_id,
                    )
                    seen_refs.add(ref_key)
                    created += 1

        # Pattern 2: "Khoản X Điều Y" (thứ tự ngược)
        for match in _RE_CLAUSE_FIRST_REF.finditer(text):
            article_num = match.group("article_num").strip()
            clause_num = match.group("clause_num").strip()

            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end].strip()

            target_id = self._resolve_reference_target(
                article_index, article_num, clause_num
            )
            if target_id and target_id != source_node_id:
                ref_key = f"{source_node_id}→{target_id}"
                if ref_key not in seen_refs:
                    self._create_references_edge(
                        source_id=source_node_id,
                        target_id=target_id,
                        context=context,
                        doc_id=doc_id,
                    )
                    seen_refs.add(ref_key)
                    created += 1

        return created

    def _resolve_reference_target(
        self,
        article_index: _ArticleIndex,
        article_num: str,
        clause_num: str | None,
    ) -> str | None:
        """
        Giải quyết tham chiếu → trả về node_id của node đích.

        Ưu tiên: Clause-level > Article-level.
        """
        if clause_num:
            target = article_index.lookup_clause(article_num, clause_num)
            if target:
                return target

        return article_index.lookup_article(article_num)

    def _create_references_edge(
        self,
        source_id: str,
        target_id: str,
        context: str,
        doc_id: str,
    ) -> None:
        """Tạo REFERENCES edge trong Kuzu."""
        cypher = """
            MATCH (src:LegalNode {node_id: $source_id}),
                  (dst:LegalNode {node_id: $target_id})
            MERGE (src)-[r:REFERENCES {context: $context}]->(dst)
        """
        try:
            self._kuzu_conn.execute(
                cypher,
                parameters={
                    "source_id": source_id,
                    "target_id": target_id,
                    "context": context[:500],  # Giới hạn độ dài context
                },
            )
            logger.debug(
                f"[Kuzu] REFERENCES: {source_id} → {target_id} "
                f"| context: '{context[:80]}...'"
            )
        except Exception as exc:
            logger.warning(f"[Kuzu] Lỗi tạo REFERENCES edge: {exc}")

    # ------------------------------------------------------------------
    # Vector DB Builder
    # ------------------------------------------------------------------

    def _build_vector_store(self, chunks: list[LsuChunk]) -> int:
        """
        Nhúng và lưu tất cả chunks vào ChromaDB.

        ⚠️ CRITICAL: metadata PHẢI chứa `node_id` để kết nối về Graph DB.

        Args:
            chunks: list[LsuChunk] từ Module 2.

        Returns:
            Số records đã lưu vào ChromaDB.
        """
        if not chunks:
            logger.info("[ChromaDB] Không có chunks để lưu.")
            return 0

        logger.info(f"[ChromaDB] Bắt đầu embed và lưu {len(chunks)} chunks...")

        total_stored = 0
        # Xử lý theo batch để tránh OOM với model embedding lớn
        for batch_start in range(0, len(chunks), self.embedding_batch_size):
            batch = chunks[batch_start : batch_start + self.embedding_batch_size]
            total_stored += self._store_chunk_batch(batch)

        logger.info(f"[ChromaDB] Đã lưu {total_stored} vectors.")
        return total_stored

    def _store_chunk_batch(self, batch: list[LsuChunk]) -> int:
        """
        Xử lý một batch chunks: embed → tạo VectorRecord → lưu vào ChromaDB.
        """
        texts_to_embed = [chunk.content_with_prefix for chunk in batch]

        # Gọi embedding function (placeholder hoặc model thực)
        try:
            embeddings: list[list[float]] = self.embedding_fn(texts_to_embed)
        except Exception as exc:
            logger.error(f"[Embedding] Lỗi khi embed batch: {exc}")
            # Fallback: zero vectors
            from src.config import EMBEDDING_DIM  # type: ignore[import]
            embeddings = [[0.0] * EMBEDDING_DIM for _ in batch]

        # Tạo VectorRecord và lưu vào ChromaDB
        records = self._build_vector_records(batch, embeddings)
        self._upsert_to_chroma(records)

        return len(records)

    def _build_vector_records(
        self,
        chunks: list[LsuChunk],
        embeddings: list[list[float]],
    ) -> list[VectorRecord]:
        """
        Tạo VectorRecord từ chunks và embeddings.

        ⚠️ CRITICAL: VectorMetadata.node_id = chunk.source_node_id
            Đây là field duy nhất liên kết Vector DB ↔ Graph DB.
        """
        records: list[VectorRecord] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for chunk, embedding in zip(chunks, embeddings):
            metadata = VectorMetadata(
                # ↓ CRITICAL: liên kết về Graph DB node
                node_id=chunk.source_node_id,
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                node_type=chunk.source_node_type.value,
                breadcrumb=chunk.breadcrumb,
                file_name="",  # Sẽ được điền nếu cần từ document context
                article_number=str(chunk.article_number) if chunk.article_number is not None else "",
                clause_number=str(chunk.clause_number) if chunk.clause_number is not None else "",
                section_title=chunk.section_title or "",
                content_type=chunk.content_type.value,
                has_tables=len(chunk.tables_json) > 0,
                page_number=chunk.page_number or 0,
                char_count=chunk.char_count,
                ingested_at=now_iso,
            )

            record = VectorRecord(
                chroma_id=chunk.chunk_id,
                embedding=embedding,
                document_text=chunk.content_with_prefix,
                metadata=metadata,
            )
            records.append(record)

        return records

    def _upsert_to_chroma(self, records: list[VectorRecord]) -> None:
        """
        Lưu batch VectorRecords vào ChromaDB.

        Dùng upsert (thay vì add) để idempotent — chạy lại không tạo duplicate.
        """
        if not records:
            return

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for record in records:
            ids.append(record.chroma_id)
            embeddings.append(record.embedding)
            documents.append(record.document_text)
            metadatas.append(record.metadata.model_dump(exclude_none=True))

        try:
            self._chroma_collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.debug(
                f"[ChromaDB] Upsert {len(records)} records thành công. "
                f"Total: {self._chroma_collection.count()}"
            )
        except Exception as exc:
            logger.error(f"[ChromaDB] Lỗi upsert: {exc}")
            raise

    # ------------------------------------------------------------------
    # Query Helpers (bonus utilities)
    # ------------------------------------------------------------------

    def query_similar_chunks(
        self,
        query_text: str,
        n_results: int = 5,
        where_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Tìm kiếm chunks tương tự trong ChromaDB theo ngữ nghĩa.

        Args:
            query_text:   Câu hỏi hoặc đoạn văn cần tìm.
            n_results:    Số kết quả trả về.
            where_filter: Filter ChromaDB metadata (vd: {"doc_id": "doc_abc"}).

        Returns:
            list[dict] với keys: chunk_id, node_id, breadcrumb, document, distance
        """
        if self._chroma_collection is None:
            raise RuntimeError("ChromaDB chưa được khởi tạo. Gọi build() trước.")

        query_embedding = self.embedding_fn([query_text])[0]

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = self._chroma_collection.query(**kwargs)

        # Chuẩn hóa kết quả
        formatted: list[dict[str, Any]] = []
        if results and results.get("ids"):
            for i, chroma_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i]
                formatted.append(
                    {
                        "chunk_id": chroma_id,
                        "node_id": meta.get("node_id", ""),  # Graph DB link
                        "breadcrumb": meta.get("breadcrumb", ""),
                        "document": results["documents"][0][i],
                        "distance": results["distances"][0][i],
                        "metadata": meta,
                    }
                )
        return formatted

    def get_node_references(self, node_id: str) -> list[dict[str, Any]]:
        """
        Lấy tất cả REFERENCES edges từ một node trong Kuzu.

        Args:
            node_id: ID của node nguồn trong Graph DB.

        Returns:
            list[dict] với keys: target_node_id, target_type, context
        """
        if self._kuzu_conn is None:
            raise RuntimeError("Kuzu chưa được khởi tạo. Gọi build() trước.")

        cypher = """
            MATCH (src:LegalNode {node_id: $node_id})-[r:REFERENCES]->(dst:LegalNode)
            RETURN dst.node_id AS target_node_id,
                   dst.node_type AS target_type,
                   dst.breadcrumb AS target_breadcrumb,
                   r.context AS context
        """
        result = self._kuzu_conn.execute(
            cypher, parameters={"node_id": node_id}
        )

        references: list[dict[str, Any]] = []
        while result.has_next():
            row = result.get_next()
            references.append(
                {
                    "target_node_id": row[0],
                    "target_type": row[1],
                    "target_breadcrumb": row[2],
                    "context": row[3],
                }
            )
        return references

    def get_graph_stats(self) -> dict[str, Any]:
        """Thống kê Graph DB: số node, edge theo loại."""
        if self._kuzu_conn is None:
            return {}

        stats: dict[str, Any] = {}
        try:
            res = self._kuzu_conn.execute("MATCH (n:LegalNode) RETURN n.node_type, COUNT(*)")
            node_counts: dict[str, int] = {}
            while res.has_next():
                row = res.get_next()
                node_counts[row[0]] = int(row[1])
            stats["nodes_by_type"] = node_counts

            res = self._kuzu_conn.execute("MATCH ()-[r:REFERENCES]->() RETURN COUNT(*)")
            if res.has_next():
                stats["references_edges"] = int(res.get_next()[0])

            res = self._kuzu_conn.execute("MATCH ()-[r:CONTAINS]->() RETURN COUNT(*)")
            if res.has_next():
                stats["contains_edges"] = int(res.get_next()[0])
        except Exception as exc:
            logger.warning(f"[Kuzu] Lỗi lấy stats: {exc}")

        if self._chroma_collection is not None:
            stats["vectors_total"] = self._chroma_collection.count()

        return stats
