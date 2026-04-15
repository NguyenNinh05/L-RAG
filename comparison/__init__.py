"""
comparison/__init__.py
======================
Public API cho Phase 2 — Indexing & Alignment Strategy.

Usage nhanh:
    from comparison import LegalAlignmentEngine, AlignmentConfig, BGEM3Manager

    engine = LegalAlignmentEngine(
        embed_manager=BGEM3Manager(),
        config=AlignmentConfig(match_threshold=0.65),
    )
    catalog = engine.align_documents(doc_v1, doc_v2)
    print(catalog.summary())
"""

from comparison.models import (
    DiffPair,
    DiffPairCatalog,
    MatchType,
    NodeEmbeddings,
    NodeVersion,
    QdrantPayload,
)
from comparison.embedding_manager import BGEM3Manager
from comparison.qdrant_indexer import QdrantManager, QdrantCollectionConfig
from comparison.alignment_engine import LegalAlignmentEngine, AlignmentConfig

__all__ = [
    # Models
    "DiffPair",
    "DiffPairCatalog",
    "MatchType",
    "NodeEmbeddings",
    "NodeVersion",
    "QdrantPayload",
    # Managers
    "BGEM3Manager",
    "QdrantManager",
    "QdrantCollectionConfig",
    # Engine
    "LegalAlignmentEngine",
    "AlignmentConfig",
]
