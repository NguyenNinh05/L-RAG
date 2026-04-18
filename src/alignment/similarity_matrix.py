"""
src/alignment/similarity_matrix.py
====================================
Các hàm tính Similarity Matrix cho Phase 2 — Alignment Strategy.

Ma trận N×M:
    S[i][j] = w_sem * Cosine(Semantic_i, Semantic_j)
             + w_jaro * JaroWinkler(Title_i, Title_j)
             + w_ord  * OrdinalProximity(ordinal_i, ordinal_j, N, M)

Module này được trích từ LegalAlignmentEngine để tách biệt logic
xây dựng similarity matrix, giúp dễ kiểm thử và thay thế độc lập.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jellyfish
import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NodeRecord — lightweight wrapper dùng nội bộ trong alignment
# ---------------------------------------------------------------------------


@dataclass
class NodeRecord:
    """Wrapper nhẹ lưu node + embedding + metadata dùng cho alignment."""

    node_id: str
    title: str           # title/label ngắn dùng cho JaroWinkler
    raw_text: str        # full text dùng cho semantic embed
    ordinal: int         # vị trí thứ tự trong tài liệu (0-indexed)
    semantic_vec: np.ndarray = field(default_factory=lambda: np.array([]))

    # Giữ tham chiếu object gốc để lấy clauses sau (hierarchical)
    article_ref: "ArticleNode | None" = None  # type: ignore[name-defined]
    clause_ref: "ClauseNode | None" = None    # type: ignore[name-defined]


# ---------------------------------------------------------------------------
# AlignmentConfig
# ---------------------------------------------------------------------------


@dataclass
class AlignmentConfig:
    """
    Cấu hình trọng số cho Similarity formula.

    S[i][j] = w_semantic * Cosine(Semantic)
             + w_jaro_winkler * JaroWinkler(Title)
             + w_ordinal * OrdinalProximity

    Tổng w_semantic + w_jaro_winkler + w_ordinal phải = 1.0
    """

    w_semantic: float = 0.6
    w_jaro_winkler: float = 0.3
    w_ordinal: float = 0.1
    match_threshold: float = 0.65
    split_merge_threshold: float = 0.80
    embed_batch_size: int = 32

    def __post_init__(self) -> None:
        total = self.w_semantic + self.w_jaro_winkler + self.w_ordinal
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Tổng trọng số phải = 1.0, nhận được {total:.4f}. "
                f"(w_semantic={self.w_semantic}, w_jaro={self.w_jaro_winkler}, "
                f"w_ord={self.w_ordinal})"
            )


# ---------------------------------------------------------------------------
# Similarity Matrix Functions
# ---------------------------------------------------------------------------


@staticmethod
def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """L2 normalize hàng của ma trận."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return mat / norms


def cosine_similarity_matrix(
    v1_records: list[NodeRecord],
    v2_records: list[NodeRecord],
) -> np.ndarray:
    """
    Tính ma trận cosine similarity từ semantic_vec đã L2-normalize sẵn.

    Returns:
        np.ndarray shape (N, M), dtype float32.
    """
    v1_mat = np.stack([r.semantic_vec for r in v1_records], axis=0)  # (N, D)
    v2_mat = np.stack([r.semantic_vec for r in v2_records], axis=0)  # (M, D)

    v1_mat = _l2_normalize(v1_mat)
    v2_mat = _l2_normalize(v2_mat)

    return (v1_mat @ v2_mat.T).astype(np.float32)  # (N, M)


def jaro_winkler_matrix(
    v1_records: list[NodeRecord],
    v2_records: list[NodeRecord],
) -> np.ndarray:
    """
    Tính ma trận Jaro-Winkler title similarity.

    Returns:
        np.ndarray shape (N, M), dtype float32, giá trị trong [0, 1].
    """
    N, M = len(v1_records), len(v2_records)
    mat = np.zeros((N, M), dtype=np.float32)
    for i, r1 in enumerate(v1_records):
        for j, r2 in enumerate(v2_records):
            mat[i, j] = jellyfish.jaro_winkler_similarity(
                r1.title.lower(), r2.title.lower()
            )
    return mat


def ordinal_proximity_matrix(
    v1_records: list[NodeRecord],
    v2_records: list[NodeRecord],
    N: int,
    M: int,
) -> np.ndarray:
    """
    Tính ma trận ordinal proximity dựa trên vị trí tương đối.

    OrdinalProximity(i, j) = 1 - |i/N - j/M|
    Giá trị ∈ [0, 1], tối đa = 1 khi i/N == j/M.

    Returns:
        np.ndarray shape (N, M), dtype float32.
    """
    v1_pos = np.array(
        [r.ordinal / max(N - 1, 1) for r in v1_records], dtype=np.float32
    )  # (N,)
    v2_pos = np.array(
        [r.ordinal / max(M - 1, 1) for r in v2_records], dtype=np.float32
    )  # (M,)

    diff = np.abs(v1_pos[:, np.newaxis] - v2_pos[np.newaxis, :])
    return (1.0 - diff).astype(np.float32)


def compute_similarity_matrix(
    v1_records: list[NodeRecord],
    v2_records: list[NodeRecord],
    config: AlignmentConfig | None = None,
) -> np.ndarray:
    """
    Tính ma trận tương đồng N×M tổng hợp.

    S[i][j] = (w_sem × Cosine) + (w_jaro × JaroWinkler) + (w_ord × Ordinal)

    Args:
        v1_records: List N NodeRecord từ V1.
        v2_records: List M NodeRecord từ V2.
        config:     AlignmentConfig (None → dùng default).

    Returns:
        np.ndarray shape (N, M), dtype float32, giá trị đã clamp về [0, 1].
    """
    if config is None:
        config = AlignmentConfig()

    N, M = len(v1_records), len(v2_records)
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)

    sem_matrix = cosine_similarity_matrix(v1_records, v2_records)
    jaro_matrix = jaro_winkler_matrix(v1_records, v2_records)
    ord_matrix = ordinal_proximity_matrix(v1_records, v2_records, N, M)

    S = (
        config.w_semantic * sem_matrix
        + config.w_jaro_winkler * jaro_matrix
        + config.w_ordinal * ord_matrix
    ).astype(np.float32)

    np.clip(S, 0.0, 1.0, out=S)
    return S
