"""
src/alignment/__init__.py
==========================
Public API của alignment package — Phase 2: Indexing & Alignment Strategy.

Exports chính:
    - LegalAlignmentEngine (orchestrator toàn bộ Phase 2)
    - AlignmentConfig      (cấu hình trọng số similarity)
    - DiffPairCatalog      (kết quả alignment)
    - BGEM3Manager         (embedding model manager)
    - QdrantManager        (vector store manager)

Usage:
    from src.alignment import LegalAlignmentEngine, AlignmentConfig
    engine = LegalAlignmentEngine(embed_manager=BGEM3Manager())
    catalog = engine.align_documents(doc_v1, doc_v2)
"""

# Data models
from .diff_catalog import (
    DiffPair,
    DiffPairCatalog,
    MatchType,
    NodeEmbeddings,
    NodeVersion,
    QdrantPayload,
)

# Embedding
from .embedder import BGEM3Manager

# Similarity & Matching
from .similarity_matrix import AlignmentConfig, NodeRecord, compute_similarity_matrix
from .hungarian_matcher import hungarian_match, detect_split_merge

# Vector store
from .qdrant_indexer import QdrantManager, QdrantCollectionConfig

# Main orchestrator — lazy import để avoid circular deps
def __getattr__(name: str):
    if name == "LegalAlignmentEngine":
        from comparison.alignment_engine import LegalAlignmentEngine
        return LegalAlignmentEngine
    raise AttributeError(f"module 'src.alignment' has no attribute {name!r}")

__all__ = [
    # Orchestrator
    "LegalAlignmentEngine",
    # Config
    "AlignmentConfig",
    "NodeRecord",
    # Data models
    "DiffPair",
    "DiffPairCatalog",
    "MatchType",
    "NodeEmbeddings",
    "NodeVersion",
    "QdrantPayload",
    # Embedding
    "BGEM3Manager",
    # Matching functions
    "compute_similarity_matrix",
    "hungarian_match",
    "detect_split_merge",
    # Vector store
    "QdrantManager",
    "QdrantCollectionConfig",
]
