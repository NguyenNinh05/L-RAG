"""
comparison/qdrant_indexer.py
============================
QdrantManager — Khởi tạo và quản lý Qdrant collection hỗ trợ multi-vector
(dense + sparse) cho Phase 2 của hệ thống đối chiếu văn bản pháp lý.

Architecture:
    - Mỗi ArticleNode/ClauseNode được lưu dưới dạng 1 Qdrant Point.
    - Mỗi Point có 2 named dense vectors:
        * "structural" — embedding của title/ordinal (nhận diện vị trí/số hiệu)
        * "semantic"   — embedding của full_text + breadcrumb (nội dung ngữ nghĩa)
    - Sparse vectors (lexical BM25-like từ BGE-M3) lưu dưới dạng SparseVector.
    - Payload tuân theo QdrantPayload schema (version, ordinal, raw_text, ...).

Collection naming convention:
    - legal_v1_{doc_id_short}  — nodes từ phiên bản V1
    - legal_v2_{doc_id_short}  — nodes từ phiên bản V2
    - Hoặc dùng 1 collection chung với filter theo payload.version
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from comparison.models import NodeEmbeddings, QdrantPayload

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DENSE_DIM = 1024          # BGE-M3 dense output dimension
STRUCTURAL_VECTOR_NAME = "structural"
SEMANTIC_VECTOR_NAME = "semantic"

# Sparse vector names (BGE-M3 sparse head outputs BM25-style weights)
STRUCTURAL_SPARSE_NAME = "structural_sparse"
SEMANTIC_SPARSE_NAME = "semantic_sparse"


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class QdrantCollectionConfig:
    """Cấu hình cho một Qdrant collection."""

    collection_name: str
    dense_dim: int = DENSE_DIM
    # Distance metric cho dense vectors
    dense_distance: qmodels.Distance = field(default_factory=lambda: qmodels.Distance.COSINE)
    # HNSW indexing params (có thể tune sau)
    hnsw_m: int = 16
    hnsw_ef_construct: int = 200
    # Số lượng segment song song (tăng throughput lúc concurrent search)
    on_disk_payload: bool = False   # True để lưu payload ra disk (tiết kiệm RAM)


# ---------------------------------------------------------------------------
# QdrantManager
# ---------------------------------------------------------------------------


class QdrantManager:
    """
    Quản lý toàn bộ lifecycle của Qdrant collection cho Phase 2.

    Sử dụng LOCAL in-memory/persistent mode (không cần server riêng):
        manager = QdrantManager()               # in-memory (test)
        manager = QdrantManager(path="./qdrant_db")  # persistent local

    Multi-vector layout mỗi Point:
        vectors={
            "structural": [float * 1024],   # hướng nhận diện vị trí
            "semantic":   [float * 1024],   # hướng đối sánh nội dung
        }
        sparse_vectors={
            "structural_sparse": SparseVector(indices=[...], values=[...]),
            "semantic_sparse":   SparseVector(indices=[...], values=[...]),
        }

    Payload (tuân theo QdrantPayload schema):
        {
            "node_id":       "article_abc123",
            "doc_id":        "doc_v1_xyz",
            "version":       "v1",
            "node_type":     "article",
            "ordinal":       4,
            "raw_text":      "Bên A có quyền...",
            "title":         "Quyền của Bên A",
            "breadcrumb":    "[Chương II > Điều 5]",
            "article_number":"5",
            "clause_number": "",
            "ingested_at":   "2026-04-15T..."
        }
    """

    def __init__(
        self,
        path: str | None = None,
        url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """
        Khởi tạo QdrantClient.

        Args:
            path:    Đường dẫn local DB (None → in-memory). Ưu tiên nếu được chỉ định.
            url:     URL Qdrant server (VD: "http://localhost:6333"). Dùng thay path.
            api_key: API key nếu dùng Qdrant Cloud.
        """
        if path is not None:
            logger.info("QdrantManager: Persistent local mode → %s", path)
            self._client = QdrantClient(path=path)
        elif url is not None:
            logger.info("QdrantManager: Remote server mode → %s", url)
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            logger.info("QdrantManager: In-memory mode (dùng cho testing)")
            self._client = QdrantClient(":memory:")

    # ------------------------------------------------------------------
    # Collection Management
    # ------------------------------------------------------------------

    def create_collection(
        self,
        config: QdrantCollectionConfig,
        recreate_if_exists: bool = False,
    ) -> None:
        """
        Tạo collection hỗ trợ multi-vector (dense + sparse).

        Args:
            config:             Cấu hình collection.
            recreate_if_exists: Nếu True, xoá collection cũ rồi tạo lại.
        """
        existing = [c.name for c in self._client.get_collections().collections]

        if config.collection_name in existing:
            if recreate_if_exists:
                logger.warning("Xoá và tạo lại collection: %s", config.collection_name)
                self._client.delete_collection(config.collection_name)
            else:
                logger.info("Collection '%s' đã tồn tại, bỏ qua.", config.collection_name)
                return

        # --- Dense vector configs (2 named vectors) ---
        dense_params = qmodels.VectorParams(
            size=config.dense_dim,
            distance=config.dense_distance,
            hnsw_config=qmodels.HnswConfigDiff(
                m=config.hnsw_m,
                ef_construct=config.hnsw_ef_construct,
            ),
        )

        # --- Sparse vector configs ---
        sparse_params = qmodels.SparseVectorParams(
            index=qmodels.SparseIndexParams(on_disk=False)
        )

        self._client.create_collection(
            collection_name=config.collection_name,
            vectors_config={
                STRUCTURAL_VECTOR_NAME: dense_params,
                SEMANTIC_VECTOR_NAME:   dense_params,
            },
            sparse_vectors_config={
                STRUCTURAL_SPARSE_NAME: sparse_params,
                SEMANTIC_SPARSE_NAME:   sparse_params,
            },
            on_disk_payload=config.on_disk_payload,
        )

        # Tạo payload index để query nhanh theo version, node_type, doc_id
        self._create_payload_indexes(config.collection_name)

        logger.info(
            "Đã tạo collection '%s' với dense_dim=%d, distance=%s",
            config.collection_name,
            config.dense_dim,
            config.dense_distance,
        )

    def _create_payload_indexes(self, collection_name: str) -> None:
        """Tạo các payload index để filter nhanh."""
        indexed_fields: list[tuple[str, qmodels.PayloadSchemaType]] = [
            ("version",      qmodels.PayloadSchemaType.KEYWORD),
            ("node_type",    qmodels.PayloadSchemaType.KEYWORD),
            ("doc_id",       qmodels.PayloadSchemaType.KEYWORD),
            ("node_id",      qmodels.PayloadSchemaType.KEYWORD),
            ("ordinal",      qmodels.PayloadSchemaType.INTEGER),
            ("article_number", qmodels.PayloadSchemaType.KEYWORD),
        ]
        for field_name, schema_type in indexed_fields:
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema_type,
            )
        logger.debug("Đã tạo payload indexes cho '%s'", collection_name)

    def collection_exists(self, collection_name: str) -> bool:
        """Kiểm tra collection có tồn tại không."""
        existing = [c.name for c in self._client.get_collections().collections]
        return collection_name in existing

    def delete_collection(self, collection_name: str) -> None:
        """Xoá collection."""
        self._client.delete_collection(collection_name)
        logger.info("Đã xoá collection '%s'", collection_name)

    # ------------------------------------------------------------------
    # Upsert / Index
    # ------------------------------------------------------------------

    def upsert_embeddings(
        self,
        collection_name: str,
        embeddings: list[NodeEmbeddings],
        batch_size: int = 64,
    ) -> int:
        """
        Upsert một danh sách NodeEmbeddings vào collection.

        Tự động chia thành batches để tránh OOM khi tập dữ liệu lớn.

        Args:
            collection_name: Tên collection đích.
            embeddings:      Danh sách NodeEmbeddings đã được tính.
            batch_size:      Số points mỗi batch.

        Returns:
            Tổng số points đã upsert thành công.
        """
        total_upserted = 0

        for batch_start in range(0, len(embeddings), batch_size):
            batch = embeddings[batch_start : batch_start + batch_size]
            points = [self._node_emb_to_point(emb) for emb in batch]

            self._client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )
            total_upserted += len(points)
            logger.debug(
                "Upserted batch [%d:%d] → collection '%s'",
                batch_start,
                batch_start + len(batch),
                collection_name,
            )

        logger.info(
            "Upsert hoàn tất: %d points → '%s'",
            total_upserted,
            collection_name,
        )
        return total_upserted

    def _node_emb_to_point(self, emb: NodeEmbeddings) -> qmodels.PointStruct:
        """
        Chuyển đổi NodeEmbeddings sang Qdrant PointStruct.

        Trong qdrant-client >= 1.7, sparse vectors được truyền qua
        vector dict với NamedSparseVector — không còn field sparse_vectors riêng.
        Lưu ý: Qdrant in-memory (local) không hỗ trợ sparse vectors khi collection
        được tạo với cả dense lẫn sparse config — sparse sẽ bị bỏ qua gracefully.
        """
        # Dense named vectors (bắt buộc)
        vector: dict = {
            STRUCTURAL_VECTOR_NAME: emb.structural_dense,
            SEMANTIC_VECTOR_NAME:   emb.semantic_dense,
        }

        # Sparse vectors — thêm vào vector dict dưới dạng NamedSparseVector
        if emb.structural_sparse:
            indices, values = zip(*sorted(emb.structural_sparse.items()))
            vector[STRUCTURAL_SPARSE_NAME] = qmodels.SparseVector(
                indices=list(indices), values=list(values)
            )
        if emb.semantic_sparse:
            indices, values = zip(*sorted(emb.semantic_sparse.items()))
            vector[SEMANTIC_SPARSE_NAME] = qmodels.SparseVector(
                indices=list(indices), values=list(values)
            )

        return qmodels.PointStruct(
            id=self._node_id_to_qdrant_id(emb.node_id),
            vector=vector,
            payload=emb.payload.to_dict(),
        )

    @staticmethod
    def _node_id_to_qdrant_id(node_id: str) -> str:
        """
        Chuyển node_id sang UUID string hợp lệ cho Qdrant.

        Qdrant local mode (in-memory) bắt buộc point ID phải là UUID v4.
        Ta dùng uuid5 (namespace-based) để đảm bảo deterministic:
        cùng node_id luôn ra cùng UUID, tránh duplicate khi upsert lại.
        """
        import uuid as _uuid
        # Dùng uuid5 với namespace DNS để tạo UUID deterministic từ node_id string
        return str(_uuid.uuid5(_uuid.NAMESPACE_DNS, node_id))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_by_semantic(
        self,
        collection_name: str,
        query_vector: list[float],
        version_filter: str | None = None,
        top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> list[qmodels.ScoredPoint]:
        """
        Tìm kiếm theo semantic dense vector với optional filter theo version.

        Args:
            collection_name: Tên collection.
            query_vector:    Dense vector truy vấn (dim = DENSE_DIM).
            version_filter:  Lọc theo version: "v1" hoặc "v2" (None = tất cả).
            top_k:           Số kết quả trả về.
            score_threshold: Ngưỡng score tối thiểu.

        Returns:
            Danh sách ScoredPoint sắp xếp theo score giảm dần.
        """
        query_filter = None
        if version_filter is not None:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="version",
                        match=qmodels.MatchValue(value=version_filter),
                    )
                ]
            )

        return self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=SEMANTIC_VECTOR_NAME,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold if score_threshold > 0 else None,
            with_payload=True,
        ).points

    def search_by_structural(
        self,
        collection_name: str,
        query_vector: list[float],
        version_filter: str | None = None,
        top_k: int = 10,
    ) -> list[qmodels.ScoredPoint]:
        """Tìm kiếm theo structural dense vector."""
        query_filter = None
        if version_filter is not None:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="version",
                        match=qmodels.MatchValue(value=version_filter),
                    )
                ]
            )

        return self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=STRUCTURAL_VECTOR_NAME,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        ).points

    def get_all_points(
        self,
        collection_name: str,
        version_filter: str | None = None,
        with_vectors: bool = True,
    ) -> list[qmodels.Record]:
        """
        Lấy toàn bộ points từ collection (dùng scroll để handle large collections).

        Args:
            collection_name: Tên collection.
            version_filter:  Lọc theo version "v1"/"v2" (None = tất cả).
            with_vectors:    Có trả kèm vectors không (True cần RAM nhiều hơn).

        Returns:
            Danh sách Record theo thứ tự ordinal.
        """
        scroll_filter = None
        if version_filter is not None:
            scroll_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="version",
                        match=qmodels.MatchValue(value=version_filter),
                    )
                ]
            )

        all_records: list[qmodels.Record] = []
        offset = None

        while True:
            records, next_offset = self._client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=with_vectors,
            )
            all_records.extend(records)
            if next_offset is None:
                break
            offset = next_offset

        # Sắp xếp theo ordinal để đảm bảo thứ tự nhất quán
        all_records.sort(key=lambda r: r.payload.get("ordinal", 0) if r.payload else 0)
        return all_records

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_collection_info(self, collection_name: str) -> dict[str, Any]:
        """Trả về thông tin collection dưới dạng dict."""
        info = self._client.get_collection(collection_name)
        # API thay đổi tuỳ version — dùng getattr với fallback
        points_count = (
            getattr(info, "points_count", None)
            or getattr(info, "indexed_vectors_count", None)
            or 0
        )
        vectors_count = getattr(info, "vectors_count", points_count)
        return {
            "name": collection_name,
            "vectors_count": vectors_count,
            "points_count": points_count,
            "status": str(getattr(info, "status", "unknown")),
            "config": {
                "dense_vectors": list(info.config.params.vectors.keys())
                if isinstance(info.config.params.vectors, dict)
                else [],
            },
        }

    @property
    def client(self) -> QdrantClient:
        """Expose raw client nếu cần truy cập nâng cao."""
        return self._client
