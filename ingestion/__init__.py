"""
ingestion/__init__.py
======================
Public API của ingestion package.

Exports chính:
    - LegalDocumentParser  (Module 1)
    - LsuChunker           (Module 2)
    - HybridGraphBuilder   (Module 3)
    - ingest_document()    (pipeline runner tích hợp cả 3 modules)
"""

from ingestion.chunker import LsuChunker
from ingestion.graph_builder import HybridGraphBuilder, placeholder_embedding_fn
from ingestion.models import (
    ArticleNode,
    ClauseNode,
    ContentType,
    DocumentSection,
    EdgeType,
    GraphEdge,
    GraphNode,
    LegalDocument,
    LsuChunk,
    NodeType,
    ParseEngine,
    ParseQualityMetrics,
    PointNode,
    TableCell,
    TableData,
    VectorMetadata,
    VectorRecord,
)
from ingestion.parser import LegalDocumentParser

__all__ = [
    # Parsers & Chunkers
    "LegalDocumentParser",
    "LsuChunker",
    "HybridGraphBuilder",
    "placeholder_embedding_fn",
    # Pipeline
    "ingest_document",
    # Models
    "LegalDocument",
    "DocumentSection",
    "ArticleNode",
    "ClauseNode",
    "PointNode",
    "TableCell",
    "TableData",
    "LsuChunk",
    "GraphNode",
    "GraphEdge",
    "VectorRecord",
    "VectorMetadata",
    "ParseQualityMetrics",
    # Enums
    "NodeType",
    "EdgeType",
    "ParseEngine",
    "ContentType",
]


def ingest_document(
    file_path: str,
    kuzu_db_path: str = "./graph_db/legal_graph",
    chroma_db_path: str = "./vector_db/chroma_legal",
    chroma_collection_name: str = "legal_documents",
    embedding_fn=None,
    confidence_threshold: float = 0.75,
    max_chunk_chars: int = 2000,
) -> dict:
    """
    Pipeline runner tích hợp 3 Modules:

        [Module 1] LegalDocumentParser → LegalDocument
        [Module 2] LsuChunker          → list[LsuChunk]
        [Module 3] HybridGraphBuilder  → Kuzu + ChromaDB

    Args:
        file_path:               Đường dẫn file PDF/DOCX cần ingest.
        kuzu_db_path:            Path đến Kuzu DB directory.
        chroma_db_path:          Path đến ChromaDB directory.
        chroma_collection_name:  Tên collection ChromaDB.
        embedding_fn:            Hàm embedding (None = placeholder zero vectors).
        confidence_threshold:    Ngưỡng confidence để trigger OCR fallback.
        max_chunk_chars:         Kích thước chunk tối đa (ký tự).

    Returns:
        dict với các keys:
            - doc_id:    ID tài liệu đã ingest
            - file_name: Tên file
            - articles:  Số điều trong tài liệu
            - chunks:    Số chunks đã tạo
            - nodes:     Số nodes trong Graph DB
            - edges:     Số REFERENCES edges
            - vectors:   Số vectors trong ChromaDB
            - engine:    Engine đã dùng (docling/marker-pdf)
    """
    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"[Pipeline] Bắt đầu ingest: {file_path}")

    # Module 1: Parse
    parser = LegalDocumentParser(confidence_threshold=confidence_threshold)
    document: LegalDocument = parser.parse(file_path)

    # Module 2: Chunk
    chunker = LsuChunker(max_chunk_chars=max_chunk_chars)
    chunks: list[LsuChunk] = chunker.chunk(document)

    # Module 3: Build Graph
    builder = HybridGraphBuilder(
        kuzu_db_path=kuzu_db_path,
        chroma_db_path=chroma_db_path,
        chroma_collection_name=chroma_collection_name,
        embedding_fn=embedding_fn,
    )
    try:
        graph_result = builder.build(document, chunks)
    finally:
        builder.close()

    result = {
        "doc_id": document.doc_id,
        "file_name": document.file_name,
        "articles": len(document.iter_all_articles()),
        "chunks": len(chunks),
        "nodes": graph_result.get("nodes", 0),
        "edges": graph_result.get("edges", 0),
        "vectors": graph_result.get("vectors", 0),
        "engine": document.quality_metrics.engine_used.value
        if document.quality_metrics
        else "unknown",
    }

    logger.info(f"[Pipeline] Hoàn thành: {result}")
    return result
