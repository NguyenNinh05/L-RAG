"""
comparison/models.py
====================
Pydantic data models cho Phase 2 — Indexing & Alignment Strategy.

Các model này bổ sung thêm vào ingestion/models.py của Phase 1.
Không import vòng — chỉ import từ ingestion.models.

Hierarchy (Phase 2):
    ArticleAlignmentResult
        └── List[ClauseAlignmentResult]   (đệ quy alignment cấp Khoản)
    DiffPairCatalog
        └── Danh sách tất cả cặp đã ghép (matched / added / deleted / split / merged)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums Phase 2
# ---------------------------------------------------------------------------


class MatchType(str, Enum):
    """
    Kết quả phân loại cặp ghép nối giữa hai phiên bản văn bản.

    - matched : Cặp 1-1 rõ ràng, score >= threshold.
    - added   : Node chỉ xuất hiện ở V2 (không có ứng viên nào trong V1).
    - deleted : Node chỉ xuất hiện ở V1 (không còn trong V2).
    - split   : 1 node V1 tách thành 2+ node V2.
    - merged  : 2+ node V1 gộp thành 1 node V2.
    """

    MATCHED = "matched"
    ADDED = "added"
    DELETED = "deleted"
    SPLIT = "split"
    MERGED = "merged"


class NodeVersion(str, Enum):
    """Phân biệt node thuộc phiên bản tài liệu nào."""

    V1 = "v1"
    V2 = "v2"


# ---------------------------------------------------------------------------
# Qdrant Payload — lưu cùng mỗi vector point
# ---------------------------------------------------------------------------


class QdrantPayload(BaseModel):
    """
    Payload gắn liền với mỗi Qdrant point.

    Qdrant hỗ trợ bất kỳ JSON-serializable dict, nhưng ta chuẩn hoá
    thành model này để đảm bảo type safety và dễ query/filter sau.
    """

    node_id: str = Field(..., description="ID node gốc từ Phase 1 (article/clause node_id)")
    doc_id: str = Field(..., description="ID tài liệu nguồn")
    version: NodeVersion = Field(..., description="Phiên bản văn bản: v1 hoặc v2")
    node_type: str = Field(..., description="'article' hoặc 'clause'")

    # Thứ tự xuất hiện trong văn bản (0-indexed)
    ordinal: int = Field(..., ge=0, description="Số thứ tự node trong tài liệu (0-indexed)")

    # Nội dung gốc để tính Jaro-Winkler và hiển thị
    raw_text: str = Field(..., description="Nội dung text thuần không có breadcrumb")
    title: str = Field(default="", description="Tiêu đề / nhãn ngắn (dùng tính string similarity)")
    breadcrumb: str = Field(default="", description="Breadcrumb đầy đủ từ Phase 1")

    # Số điều / số khoản để tính ordinal proximity
    article_number: str = Field(default="", description="Số điều (str)")
    clause_number: str = Field(default="", description="Số khoản (str, rỗng nếu là article-level)")

    ingested_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Timestamp ISO 8601",
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize sang dict phẳng cho Qdrant payload."""
        d = self.model_dump()
        # Qdrant payload values phải là JSON-primitive — enum → str
        d["version"] = self.version.value
        return d


# ---------------------------------------------------------------------------
# DiffPair — đơn vị kết quả matching cho một cặp node
# ---------------------------------------------------------------------------


class DiffPair(BaseModel):
    """
    Kết quả ghép nối giữa một cặp node (V1 ↔ V2).

    Một DiffPair có thể biểu diễn:
      - matched  : v1_ids=[A],   v2_ids=[B]   — cặp 1-1
      - added    : v1_ids=[],    v2_ids=[B]   — chỉ có ở V2
      - deleted  : v1_ids=[A],   v2_ids=[]    — chỉ có ở V1
      - split    : v1_ids=[A],   v2_ids=[B,C] — A tách thành B và C
      - merged   : v1_ids=[A,B], v2_ids=[C]   — A và B gộp thành C
    """

    pair_id: str = Field(
        default_factory=lambda: f"pair_{uuid.uuid4().hex[:12]}",
        description="ID duy nhất của cặp ghép nối này",
    )

    # IDs của các node tham gia (có thể 0, 1, hoặc nhiều phần tử)
    v1_ids: list[str] = Field(default_factory=list, description="Node ID(s) từ phiên bản V1")
    v2_ids: list[str] = Field(default_factory=list, description="Node ID(s) từ phiên bản V2")

    match_type: MatchType = Field(..., description="Phân loại kết quả: matched/added/deleted/split/merged")
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Điểm tương đồng tổng hợp [0, 1]; 0.0 nếu added/deleted",
    )

    # Breakdown điểm thành phần (optional, dùng cho debugging/explainability)
    semantic_score: float | None = Field(default=None, ge=0.0, le=1.0)
    structural_score: float | None = Field(default=None, ge=0.0, le=1.0)
    jaro_winkler_score: float | None = Field(default=None, ge=0.0, le=1.0)
    ordinal_proximity_score: float | None = Field(default=None, ge=0.0, le=1.0)

    # Nội dung text gốc (để render diff sau này)
    v1_texts: list[str] = Field(default_factory=list)
    v2_texts: list[str] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _validate_match_type_consistency(self) -> "DiffPair":
        """Đảm bảo match_type nhất quán với số lượng IDs."""
        n_v1, n_v2 = len(self.v1_ids), len(self.v2_ids)
        if self.match_type == MatchType.MATCHED and not (n_v1 == 1 and n_v2 == 1):
            raise ValueError("MatchType.MATCHED yêu cầu đúng 1 v1_id và 1 v2_id.")
        if self.match_type == MatchType.ADDED and not (n_v1 == 0 and n_v2 >= 1):
            raise ValueError("MatchType.ADDED yêu cầu v1_ids rỗng và ít nhất 1 v2_id.")
        if self.match_type == MatchType.DELETED and not (n_v1 >= 1 and n_v2 == 0):
            raise ValueError("MatchType.DELETED yêu cầu ít nhất 1 v1_id và v2_ids rỗng.")
        if self.match_type == MatchType.SPLIT and not (n_v1 == 1 and n_v2 >= 2):
            raise ValueError("MatchType.SPLIT yêu cầu 1 v1_id và ít nhất 2 v2_ids.")
        if self.match_type == MatchType.MERGED and not (n_v1 >= 2 and n_v2 == 1):
            raise ValueError("MatchType.MERGED yêu cầu ít nhất 2 v1_ids và 1 v2_id.")
        return self


# ---------------------------------------------------------------------------
# DiffPairCatalog — tập hợp tất cả cặp ghép nối của một phiên so sánh
# ---------------------------------------------------------------------------


class DiffPairCatalog(BaseModel):
    """
    Catalogue đầy đủ kết quả ghép nối giữa V1 và V2.

    Đây là đầu ra chính của LegalAlignmentEngine (alignment_engine.py)
    và là đầu vào của Phase 3 (Hybrid Retrieval / Diff Generation).
    """

    catalog_id: str = Field(
        default_factory=lambda: f"catalog_{uuid.uuid4().hex[:12]}",
        description="ID duy nhất của phiên so sánh",
    )
    v1_doc_id: str = Field(..., description="doc_id của tài liệu phiên bản V1")
    v2_doc_id: str = Field(..., description="doc_id của tài liệu phiên bản V2")

    pairs: list[DiffPair] = Field(
        default_factory=list,
        description="Toàn bộ cặp ghép nối (matched + added + deleted + split + merged)",
    )

    # Statistics — tự động tính qua property
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Cấu hình ngưỡng đã dùng khi tạo catalog này (audit trail)
    match_threshold: float = Field(default=0.65, description="Ngưỡng θ đã dùng cho Hungarian matching")
    split_merge_threshold: float = Field(default=0.80, description="Ngưỡng dùng cho split/merge detection")

    @property
    def matched_pairs(self) -> list[DiffPair]:
        return [p for p in self.pairs if p.match_type == MatchType.MATCHED]

    @property
    def added_nodes(self) -> list[DiffPair]:
        return [p for p in self.pairs if p.match_type == MatchType.ADDED]

    @property
    def deleted_nodes(self) -> list[DiffPair]:
        return [p for p in self.pairs if p.match_type == MatchType.DELETED]

    @property
    def split_cases(self) -> list[DiffPair]:
        return [p for p in self.pairs if p.match_type == MatchType.SPLIT]

    @property
    def merged_cases(self) -> list[DiffPair]:
        return [p for p in self.pairs if p.match_type == MatchType.MERGED]

    def summary(self) -> dict[str, int]:
        """Trả về thống kê tổng hợp của catalog."""
        return {
            "total_pairs": len(self.pairs),
            "matched": len(self.matched_pairs),
            "added": len(self.added_nodes),
            "deleted": len(self.deleted_nodes),
            "split": len(self.split_cases),
            "merged": len(self.merged_cases),
        }

    def to_report_dict(self) -> dict[str, Any]:
        """Serialize toàn bộ catalog thành dict cho JSON export / LLM input."""
        return {
            "catalog_id": self.catalog_id,
            "v1_doc_id": self.v1_doc_id,
            "v2_doc_id": self.v2_doc_id,
            "created_at": self.created_at.isoformat(),
            "thresholds": {
                "match": self.match_threshold,
                "split_merge": self.split_merge_threshold,
            },
            "summary": self.summary(),
            "pairs": [p.model_dump(mode="json") for p in self.pairs],
        }


# ---------------------------------------------------------------------------
# Alignment Result — wrapper trung gian (dùng nội bộ trong engine)
# ---------------------------------------------------------------------------


class NodeEmbeddings(BaseModel):
    """
    Lưu trữ cả 2 loại embedding của một node sau khi qua BGEM3Manager.
    Dense vectors ở đây là list[float] — Qdrant sẽ nhận chúng trực tiếp.
    """

    node_id: str = Field(..., description="ID node (article/clause node_id)")
    structural_dense: list[float] = Field(..., description="Dense vector của structural_embed (title/ordinal)")
    semantic_dense: list[float] = Field(..., description="Dense vector của semantic_embed (full_text + breadcrumb)")
    # sparse (lexical) — biểu diễn dạng {index: weight} dict
    structural_sparse: dict[int, float] = Field(
        default_factory=dict,
        description="Sparse (BM25-like) weights từ BGE-M3 cho structural text",
    )
    semantic_sparse: dict[int, float] = Field(
        default_factory=dict,
        description="Sparse (BM25-like) weights từ BGE-M3 cho semantic text",
    )
    payload: QdrantPayload = Field(..., description="Payload đầy đủ đã chuẩn bị cho Qdrant")
