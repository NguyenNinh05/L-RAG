"""
src/alignment/hungarian_matcher.py
====================================
Hungarian Matching + Split/Merge Handler cho Phase 2 — Alignment.

Bước 2: Hungarian Matching
    scipy.optimize.linear_sum_assignment trên Cost Matrix = 1 - S
    Lọc cặp unmatched nếu confidence < θ = 0.65

Bước 3: Split/Merge Handler
    Fallback cho các node unmatched.
    Gộp text 2 nodes nhỏ so sánh với 1 node lớn → threshold 0.8

Module này được trích từ LegalAlignmentEngine để tách biệt logic
matching, giúp dễ kiểm thử và thay thế thuật toán độc lập.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

from .diff_catalog import DiffPair, MatchType

if TYPE_CHECKING:
    from .similarity_matrix import AlignmentConfig, NodeRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hungarian Matching
# ---------------------------------------------------------------------------


def hungarian_match(
    similarity_matrix: np.ndarray,
    match_threshold: float = 0.65,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """
    Áp dụng Hungarian algorithm để tìm matching tối ưu.

    Sử dụng scipy.optimize.linear_sum_assignment trên Cost Matrix = 1 - S.
    Sau đó lọc các cặp có confidence < θ.

    Args:
        similarity_matrix: Ma trận S shape (N, M).
        match_threshold:   Ngưỡng θ — cặp có score < θ bị coi là unmatched.

    Returns:
        matched_pairs:     List of (i, j, score).
        v1_unmatched_idx:  List index i không có cặp trong V2.
        v2_unmatched_idx:  List index j không có cặp trong V1.
    """
    N, M = similarity_matrix.shape
    if N == 0 or M == 0:
        return [], list(range(N)), list(range(M))

    cost_matrix = 1.0 - similarity_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_pairs: list[tuple[int, int, float]] = []
    matched_v1: set[int] = set()
    matched_v2: set[int] = set()

    for i, j in zip(row_ind, col_ind):
        score = float(similarity_matrix[i, j])
        if score >= match_threshold:
            matched_pairs.append((i, j, score))
            matched_v1.add(i)
            matched_v2.add(j)
        else:
            logger.debug(
                "Cặp (%d, %d) bị loại: score=%.3f < θ=%.2f",
                i, j, score, match_threshold,
            )

    v1_unmatched = [i for i in range(N) if i not in matched_v1]
    v2_unmatched = [j for j in range(M) if j not in matched_v2]

    logger.debug(
        "Hungarian: %d matched, %d V1-unmatched, %d V2-unmatched",
        len(matched_pairs), len(v1_unmatched), len(v2_unmatched),
    )

    return matched_pairs, v1_unmatched, v2_unmatched


# ---------------------------------------------------------------------------
# Split/Merge Detection
# ---------------------------------------------------------------------------


def detect_split_merge(
    v1_unmatched: list["NodeRecord"],
    v2_unmatched: list["NodeRecord"],
    embed_fn,
    split_merge_threshold: float = 0.80,
) -> tuple[list[DiffPair], list[int], list[int]]:
    """
    Phát hiện các trường hợp Split/Merge trong nodes chưa matched.

    - SPLIT: 1 node V1 lớn → kiểm tra xem ghép 2 nodes V2 nhỏ
             có tạo ra text tương đồng >= threshold không.
    - MERGE: 2 nodes V1 nhỏ → gộp text → so sánh với 1 node V2 lớn.

    Args:
        v1_unmatched:           NodeRecord V1 chưa matched.
        v2_unmatched:           NodeRecord V2 chưa matched.
        embed_fn:               Callable nhận list[str] → np.ndarray (N, D), normalized.
        split_merge_threshold:  Ngưỡng cosine similarity để xác nhận split/merge.

    Returns:
        split_merge_pairs:     DiffPair với type SPLIT hoặc MERGED.
        still_v1_unmatched:   Index trong v1_unmatched vẫn còn unmatched.
        still_v2_unmatched:   Index trong v2_unmatched vẫn còn unmatched.
    """
    result_pairs: list[DiffPair] = []
    used_v1: set[int] = set()
    used_v2: set[int] = set()

    def text_cosine(text_a: str, text_b: str) -> float:
        vecs = embed_fn([text_a, text_b])
        if vecs.shape[0] < 2:
            return 0.0
        return float(np.dot(vecs[0], vecs[1]))

    # ── Detect MERGE: 2 V1 → 1 V2 ──
    for j, v2_rec in enumerate(v2_unmatched):
        if j in used_v2:
            continue

        best_pair: tuple[int, int, float] | None = None

        for i1 in range(len(v1_unmatched)):
            if i1 in used_v1:
                continue
            for i2 in range(i1 + 1, len(v1_unmatched)):
                if i2 in used_v1:
                    continue

                merged_text = v1_unmatched[i1].raw_text + "\n" + v1_unmatched[i2].raw_text
                score = text_cosine(merged_text, v2_rec.raw_text)

                if score >= split_merge_threshold:
                    if best_pair is None or score > best_pair[2]:
                        best_pair = (i1, i2, score)

        if best_pair is not None:
            i1, i2, score = best_pair
            pair = DiffPair(
                v1_ids=[v1_unmatched[i1].node_id, v1_unmatched[i2].node_id],
                v2_ids=[v2_rec.node_id],
                match_type=MatchType.MERGED,
                confidence_score=round(score, 4),
                v1_texts=[v1_unmatched[i1].raw_text, v1_unmatched[i2].raw_text],
                v2_texts=[v2_rec.raw_text],
            )
            result_pairs.append(pair)
            used_v1.add(i1)
            used_v1.add(i2)
            used_v2.add(j)
            logger.debug("MERGE: V1[%d]+V1[%d] → V2[%d], score=%.3f", i1, i2, j, score)

    # ── Detect SPLIT: 1 V1 → 2 V2 ──
    for i, v1_rec in enumerate(v1_unmatched):
        if i in used_v1:
            continue

        best_pair_split: tuple[int, int, float] | None = None

        for j1 in range(len(v2_unmatched)):
            if j1 in used_v2:
                continue
            for j2 in range(j1 + 1, len(v2_unmatched)):
                if j2 in used_v2:
                    continue

                merged_text = v2_unmatched[j1].raw_text + "\n" + v2_unmatched[j2].raw_text
                score = text_cosine(v1_rec.raw_text, merged_text)

                if score >= split_merge_threshold:
                    if best_pair_split is None or score > best_pair_split[2]:
                        best_pair_split = (j1, j2, score)

        if best_pair_split is not None:
            j1, j2, score = best_pair_split
            pair = DiffPair(
                v1_ids=[v1_rec.node_id],
                v2_ids=[v2_unmatched[j1].node_id, v2_unmatched[j2].node_id],
                match_type=MatchType.SPLIT,
                confidence_score=round(score, 4),
                v1_texts=[v1_rec.raw_text],
                v2_texts=[v2_unmatched[j1].raw_text, v2_unmatched[j2].raw_text],
            )
            result_pairs.append(pair)
            used_v1.add(i)
            used_v2.add(j1)
            used_v2.add(j2)
            logger.debug("SPLIT: V1[%d] → V2[%d]+V2[%d], score=%.3f", i, j1, j2, score)

    still_v1 = [i for i in range(len(v1_unmatched)) if i not in used_v1]
    still_v2 = [j for j in range(len(v2_unmatched)) if j not in used_v2]

    return result_pairs, still_v1, still_v2
