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

# Bonus cộng thêm khi hai article CÙNG số điều (vd "Điều 5" ↔ "Điều 5").
# Legal articles hiếm khi đổi số trừ khi renumber → đây là tín hiệu nhận diện mạnh.
ARTICLE_NUMBER_BONUS: float = 0.15


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

    # Sparse (lexical, BM25-like) weights từ BGE-M3 — {token_index: weight}
    semantic_sparse_vec: dict[int, float] = field(default_factory=dict)
    # Số điều (str) — dùng cho article-number exact-match bonus
    article_number: str = ""

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

    S[i][j] = w_semantic * Cosine(Semantic_dense)
             + w_jaro_winkler * JaroWinkler(Title)
             + w_ordinal * OrdinalProximity
             + w_sparse * SparseOverlap(lexical, BGE-M3)
             (+ article-number exact-match bonus, xem compute_similarity_matrix)

    Tổng w_semantic + w_jaro_winkler + w_ordinal + w_sparse phải = 1.0
    """

    w_semantic: float = 0.6
    w_jaro_winkler: float = 0.3
    w_ordinal: float = 0.1
    w_sparse: float = 0.0  # mặc định tắt; bật khi NodeRecord có sparse vec
    match_threshold: float = 0.65
    split_merge_threshold: float = 0.80
    embed_batch_size: int = 32

    def __post_init__(self) -> None:
        total = (
            self.w_semantic + self.w_jaro_winkler
            + self.w_ordinal + self.w_sparse
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Tổng trọng số phải = 1.0, nhận được {total:.4f}. "
                f"(w_semantic={self.w_semantic}, w_jaro={self.w_jaro_winkler}, "
                f"w_ord={self.w_ordinal}, w_sparse={self.w_sparse})"
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


def _normalize_sparse(vec: dict[int, float]) -> dict[int, float]:
    """L2-normalize một sparse vector (dict index→weight). Trả về {} nếu rỗng."""
    if not vec:
        return {}
    norm = sum(w * w for w in vec.values()) ** 0.5
    if norm == 0.0:
        return {}
    return {k: w / norm for k, w in vec.items()}


def sparse_overlap_matrix(
    v1_records: list[NodeRecord],
    v2_records: list[NodeRecord],
) -> np.ndarray:
    """
    Ma trận cosine similarity giữa các sparse (lexical) vector của BGE-M3.

    Mỗi entry = cosine(sparse_i, sparse_j) ∈ [0, 1] (trọng số ≥ 0).
    Bù đắp cho dense semantic: bắt các exact-match term (số điều, tên riêng,
    thuật ngữ pháp lý) mà dense embedding có thể làm mờ.

    Returns:
        np.ndarray shape (N, M), dtype float32.
    """
    N, M = len(v1_records), len(v2_records)
    mat = np.zeros((N, M), dtype=np.float32)
    if N == 0 or M == 0:
        return mat

    nv1 = [_normalize_sparse(getattr(r, "semantic_sparse_vec", {}) or {}) for r in v1_records]
    nv2 = [_normalize_sparse(getattr(r, "semantic_sparse_vec", {}) or {}) for r in v2_records]

    for i in range(N):
        ai = nv1[i]
        if not ai:
            continue
        for j in range(M):
            bj = nv2[j]
            if not bj:
                continue
            # dot product qua các key chung; duyệt dict nhỏ hơn
            if len(ai) <= len(bj):
                dot = sum(w * bj.get(k, 0.0) for k, w in ai.items())
            else:
                dot = sum(w * ai.get(k, 0.0) for k, w in bj.items())
            mat[i, j] = dot

    np.clip(mat, 0.0, 1.0, out=mat)
    return mat


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
    )

    # Chiến lược E — sparse (lexical) overlap term
    if config.w_sparse > 0.0:
        sp_matrix = sparse_overlap_matrix(v1_records, v2_records)
        S = S + config.w_sparse * sp_matrix

    S = S.astype(np.float32)

    # Chiến lược E — article-number exact-match bonus (cộng thêm, rồi clamp)
    for i, r1 in enumerate(v1_records):
        an1 = (getattr(r1, "article_number", "") or "").strip()
        if not an1:
            continue
        for j, r2 in enumerate(v2_records):
            an2 = (getattr(r2, "article_number", "") or "").strip()
            if an2 and an1 == an2:
                S[i, j] += ARTICLE_NUMBER_BONUS

    np.clip(S, 0.0, 1.0, out=S)
    return S
