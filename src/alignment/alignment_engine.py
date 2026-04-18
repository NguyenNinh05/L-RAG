"""
src/alignment/alignment_engine.py
===================================
LegalAlignmentEngine — Orchestrator Phase 2: Indexing & Alignment.

Luồng xử lý:
    1. Embed tất cả ArticleNodes (V1 + V2) bằng BGEM3Manager
    2. Lưu embeddings vào Qdrant (optional)
    3. Xây dựng Similarity Matrix N×M
    4. Áp dụng Hungarian Algorithm → Matched pairs
    5. Phát hiện Split/Merge trong unmatched nodes
    6. Phân loại ADDED / DELETED → trả về DiffPairCatalog

Usage:
    from src.alignment.alignment_engine import LegalAlignmentEngine, AlignmentConfig
    from src.alignment.embedder import BGEM3Manager

    engine = LegalAlignmentEngine(embed_manager=BGEM3Manager())
    catalog = engine.align_documents(doc_v1, doc_v2, collection_name="legal_abc")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .diff_catalog import (
    DiffPair,
    DiffPairCatalog,
    MatchType,
    NodeVersion,
)
from .similarity_matrix import AlignmentConfig, NodeRecord, compute_similarity_matrix
from .hungarian_matcher import hungarian_match, detect_split_merge

if TYPE_CHECKING:
    from src.ingestion.models import LegalDocument, ArticleNode
    from .embedder import BGEM3Manager
    from .qdrant_indexer import QdrantManager

logger = logging.getLogger(__name__)


class LegalAlignmentEngine:
    """
    Orchestrator điều phối toàn bộ Phase 2 — Alignment Strategy.

    Pipeline:
        embed_articles() → build_similarity_matrix() → hungarian_match()
        → detect_split_merge() → classify_unmatched() → DiffPairCatalog
    """

    def __init__(
        self,
        embed_manager: "BGEM3Manager",
        config: AlignmentConfig | None = None,
        qdrant_manager: "QdrantManager | None" = None,
    ) -> None:
        self._embedder = embed_manager
        self._config = config or AlignmentConfig()
        self._qdrant = qdrant_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def align_documents(
        self,
        doc_v1: "LegalDocument",
        doc_v2: "LegalDocument",
        collection_name: str | None = None,
    ) -> DiffPairCatalog:
        """
        Thực hiện alignment giữa hai phiên bản tài liệu pháp lý.

        Args:
            doc_v1:           LegalDocument phiên bản gốc.
            doc_v2:           LegalDocument phiên bản sửa đổi.
            collection_name:  Tên Qdrant collection (None → bỏ qua lưu Qdrant).

        Returns:
            DiffPairCatalog chứa toàn bộ matched/added/deleted/split/merged pairs.
        """
        # ── Bước 1: Thu thập tất cả ArticleNodes ──
        v1_articles: list["ArticleNode"] = list(doc_v1.iter_all_articles())
        v2_articles: list["ArticleNode"] = list(doc_v2.iter_all_articles())

        logger.info(
            "[Alignment] V1: %d articles | V2: %d articles",
            len(v1_articles),
            len(v2_articles),
        )

        if not v1_articles or not v2_articles:
            logger.warning("[Alignment] Một trong hai tài liệu không có articles.")
            return self._empty_catalog(doc_v1.doc_id, doc_v2.doc_id)

        # ── Bước 2: Embed ──
        logger.info("[Alignment] Bắt đầu embedding V1...")
        v1_embeddings = self._embedder.embed_article_nodes(
            articles=v1_articles,
            version=NodeVersion.V1,
            doc_id=doc_v1.doc_id,
        )

        logger.info("[Alignment] Bắt đầu embedding V2...")
        v2_embeddings = self._embedder.embed_article_nodes(
            articles=v2_articles,
            version=NodeVersion.V2,
            doc_id=doc_v2.doc_id,
        )

        # ── Bước 3: Lưu Qdrant (optional) ──
        if self._qdrant is not None and collection_name:
            self._index_to_qdrant(collection_name, v1_embeddings + v2_embeddings)

        # ── Bước 4: Xây dựng NodeRecord cho similarity matrix ──
        v1_records = self._to_node_records(v1_embeddings)
        v2_records = self._to_node_records(v2_embeddings)

        # ── Bước 5: Tính Similarity Matrix ──
        logger.info("[Alignment] Tính similarity matrix %dx%d...", len(v1_records), len(v2_records))
        sim_matrix = compute_similarity_matrix(v1_records, v2_records, self._config)

        # ── Bước 6: Hungarian Matching ──
        matched_triples, v1_unmatched_idx, v2_unmatched_idx = hungarian_match(
            sim_matrix,
            match_threshold=self._config.match_threshold,
        )
        logger.info(
            "[Alignment] Hungarian: %d matched, %d V1-unmatched, %d V2-unmatched",
            len(matched_triples),
            len(v1_unmatched_idx),
            len(v2_unmatched_idx),
        )

        # ── Bước 7: Build DiffPairs cho matched ──
        pairs: list[DiffPair] = []
        for i, j, score in matched_triples:
            pairs.append(
                DiffPair(
                    v1_ids=[v1_records[i].node_id],
                    v2_ids=[v2_records[j].node_id],
                    match_type=MatchType.MATCHED,
                    confidence_score=round(score, 4),
                    v1_texts=[v1_records[i].raw_text],
                    v2_texts=[v2_records[j].raw_text],
                )
            )

        # ── Bước 8: Split/Merge Detection ──
        v1_unmatched_records = [v1_records[i] for i in v1_unmatched_idx]
        v2_unmatched_records = [v2_records[j] for j in v2_unmatched_idx]

        split_merge_pairs, still_v1_idx, still_v2_idx = detect_split_merge(
            v1_unmatched=v1_unmatched_records,
            v2_unmatched=v2_unmatched_records,
            embed_fn=self._embedder.embed_texts_semantic,
            split_merge_threshold=self._config.split_merge_threshold,
        )
        pairs.extend(split_merge_pairs)

        # ── Bước 9: Phân loại DELETED và ADDED ──
        for idx in still_v1_idx:
            rec = v1_unmatched_records[idx]
            pairs.append(
                DiffPair(
                    v1_ids=[rec.node_id],
                    v2_ids=[],
                    match_type=MatchType.DELETED,
                    confidence_score=0.0,
                    v1_texts=[rec.raw_text],
                )
            )

        for idx in still_v2_idx:
            rec = v2_unmatched_records[idx]
            pairs.append(
                DiffPair(
                    v1_ids=[],
                    v2_ids=[rec.node_id],
                    match_type=MatchType.ADDED,
                    confidence_score=0.0,
                    v2_texts=[rec.raw_text],
                )
            )

        catalog = DiffPairCatalog(
            v1_doc_id=doc_v1.doc_id,
            v2_doc_id=doc_v2.doc_id,
            pairs=pairs,
            match_threshold=self._config.match_threshold,
            split_merge_threshold=self._config.split_merge_threshold,
        )

        logger.info("[Alignment] Hoàn thành. %s", catalog.summary())
        return catalog

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_node_records(embeddings) -> list[NodeRecord]:
        """Chuyển đổi list[NodeEmbeddings] → list[NodeRecord] cho similarity matrix."""
        records: list[NodeRecord] = []
        for i, emb in enumerate(embeddings):
            payload = emb.payload
            records.append(
                NodeRecord(
                    node_id=emb.node_id,
                    title=payload.title,
                    raw_text=payload.raw_text,
                    ordinal=payload.ordinal if payload.ordinal is not None else i,
                    semantic_vec=np.array(emb.semantic_dense, dtype=np.float32),
                )
            )
        return records

    def _index_to_qdrant(self, collection_name: str, embeddings) -> None:
        """Tạo collection và upsert embeddings vào Qdrant."""
        try:
            from .qdrant_indexer import QdrantCollectionConfig as QCfg, DENSE_DIM
            cfg = QCfg(collection_name=collection_name, dense_dim=DENSE_DIM)
            self._qdrant.create_collection(cfg, recreate_if_exists=False)
            count = self._qdrant.upsert_embeddings(collection_name, embeddings)
            logger.info("[Alignment] Qdrant: upserted %d points → '%s'", count, collection_name)
        except Exception as e:
            logger.warning("[Alignment] Qdrant upsert thất bại (bỏ qua): %s", e)

    @staticmethod
    def _empty_catalog(v1_doc_id: str, v2_doc_id: str) -> DiffPairCatalog:
        return DiffPairCatalog(v1_doc_id=v1_doc_id, v2_doc_id=v2_doc_id, pairs=[])
