"""
config.py
=========
Cấu hình tập trung cho toàn bộ hệ thống L-RAG.
Đọc từ biến môi trường hoặc file .env (nếu có).
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# Parser (Module 1) settings
# ---------------------------------------------------------------------------

# Ngưỡng confidence: nếu avg confidence của docling < threshold này
# thì tự động fallback sang marker-pdf OCR.
DOCLING_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("DOCLING_CONFIDENCE_THRESHOLD", "0.75")
)

# Tỷ lệ tối đa trang có confidence thấp trước khi trigger OCR toàn bộ
DOCLING_LOW_CONF_PAGE_RATIO: float = float(
    os.getenv("DOCLING_LOW_CONF_PAGE_RATIO", "0.3")
)

# marker-pdf CLI / Python API timeout (giây)
MARKER_TIMEOUT_SECONDS: int = int(os.getenv("MARKER_TIMEOUT_SECONDS", "300"))

# ---------------------------------------------------------------------------
# Graph DB (Module 3) settings — Kuzu embedded
# ---------------------------------------------------------------------------
KUZU_DB_PATH: Path = Path(
    os.getenv("KUZU_DB_PATH", str(PROJECT_ROOT / "graph_db" / "legal_graph"))
)

# ---------------------------------------------------------------------------
# Vector DB (Module 3) settings — ChromaDB persistent
# ---------------------------------------------------------------------------
CHROMA_DB_PATH: Path = Path(
    os.getenv("CHROMA_DB_PATH", str(PROJECT_ROOT / "vector_db" / "chroma_legal"))
)

CHROMA_COLLECTION_NAME: str = os.getenv(
    "CHROMA_COLLECTION_NAME", "legal_documents"
)

# ---------------------------------------------------------------------------
# Chunking (Module 2) settings
# ---------------------------------------------------------------------------
# Kích thước chunk tối đa (ký tự). Nếu khoản có nội dung dài hơn,
# sẽ bị chia nhỏ thêm theo câu.
MAX_CHUNK_CHARS: int = int(os.getenv("MAX_CHUNK_CHARS", "2000"))

# Số ký tự overlap giữa các sub-chunk khi khoản bị chia nhỏ
CHUNK_OVERLAP_CHARS: int = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))

# ---------------------------------------------------------------------------
# Embedding (placeholder — sẽ được thay bởi model thực)
# ---------------------------------------------------------------------------
EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "1024"))
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
