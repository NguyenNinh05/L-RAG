"""
retrieval/matcher.py
====================
So sánh hai danh sách ArticleChunk bằng pipeline 2 tầng:

  Tầng 1 — Anchor (difflib.SequenceMatcher):
    Tìm các chunk giống nhau 100% về text → chốt UNCHANGED ngay, không cần embed.

  Tầng 2 — Semantic NW (chỉ trên các "gap"):
    Với các vùng không khớp, chạy Embedding + Needleman-Wunsch để phân loại
    MODIFIED / ADDED / DELETED.

  Bonus — Text-ratio check:
    Sau NW, nếu embedding sim cao nhưng text thực sự thay đổi (ABC→DEF, 0.1%→0.05%)
    → downgrade UNCHANGED → MODIFIED.

Giảm ~80-90% khối lượng embedding + NW so với pipeline cũ.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import logging

logger = logging.getLogger(__name__)

from ingestion.models import ArticleChunk
from embedding.embedder import embed_chunks, INSTRUCTION_DOC
from config import (
    UNCHANGED_THRESHOLD,
    MODIFIED_THRESHOLD,
    GAP_PENALTY,
    TEXT_UNCHANGED_RATIO,
)


# ── Data model cho kết quả ──────────────────────────────────────────────────
@dataclass
class ComparedPair:
    chunk_a:    Optional[ArticleChunk]   # None nếu ADDED
    chunk_b:    Optional[ArticleChunk]   # None nếu DELETED
    match_type: str                      # "UNCHANGED" | "MODIFIED" | "ADDED" | "DELETED"
    similarity: float                    # cosine similarity (1.0 nếu anchor, 0.0 nếu ADDED/DELETED)

    @property
    def label(self) -> str:
        if self.chunk_a:
            return self.chunk_a.article_number or "N/A"
        return self.chunk_b.article_number or "N/A"


# ── Tầng 1: Anchor detection ─────────────────────────────────────────────────
def _normalize_for_anchor(text: str) -> str:
    """Chuẩn hóa text để so sánh anchor: lowercase + collapse whitespace."""
    return " ".join(text.lower().split())


def _find_anchors(
    chunks_a: list[ArticleChunk],
    chunks_b: list[ArticleChunk],
) -> list[tuple[str, int, int, int, int]]:
    """
    Dùng difflib.SequenceMatcher để chia thành anchor và gap.

    Returns list of opcodes (tag, i1, i2, j1, j2):
      - 'equal'   → anchor: chunks_a[i1:i2] giống chunks_b[j1:j2] → UNCHANGED ngay
      - khác      → gap: cần NW + embedding để phân tích
    """
    norm_a = [_normalize_for_anchor(c.content) for c in chunks_a]
    norm_b = [_normalize_for_anchor(c.content) for c in chunks_b]
    matcher = difflib.SequenceMatcher(None, norm_a, norm_b, autojunk=False)
    return matcher.get_opcodes()


# ── Tầng 2: Similarity Matrix ────────────────────────────────────────────────
def _sim_matrix(
    embeds_a: np.ndarray,
    embeds_b: np.ndarray,
) -> np.ndarray:
    """Cosine similarity matrix (n_a x n_b). Thuc hien L2-normalize de dot product = cosine sim."""
    # L2 normalize moi vector (chia cho norm)
    # factor = 1e-9 de tranh chia cho 0
    norm_a = np.linalg.norm(embeds_a, axis=1, keepdims=True) + 1e-9
    norm_b = np.linalg.norm(embeds_b, axis=1, keepdims=True) + 1e-9
    
    A_norm = embeds_a / norm_a
    B_norm = embeds_b / norm_b
    
    return (A_norm @ B_norm.T).astype(np.float64)


# ── Tầng 2: Needleman-Wunsch (sequence alignment) ────────────────────────────
def _needleman_wunsch(
    sim:         np.ndarray,
    gap_penalty: float,
) -> list[tuple[int, int]]:
    """
    Global sequence alignment — Needleman-Wunsch.
    Trả về list (idx_a, idx_b):
      (i,  j)  → ghép cặp
      (i, -1)  → DELETED (gap ở B)
      (-1, j)  → ADDED   (gap ở A)
    """
    n, m = sim.shape
    dp  = np.full((n + 1, m + 1), -np.inf, dtype=np.float64)
    ptr = np.zeros((n + 1, m + 1), dtype=np.int8)

    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        dp[i, 0]  = dp[i - 1, 0] - gap_penalty
        ptr[i, 0] = 1
    for j in range(1, m + 1):
        dp[0, j]  = dp[0, j - 1] - gap_penalty
        ptr[0, j] = 2

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates = [
                dp[i - 1, j - 1] + sim[i - 1, j - 1],
                dp[i - 1, j    ] - gap_penalty,
                dp[i    , j - 1] - gap_penalty,
            ]
            best      = int(np.argmax(candidates))
            dp[i, j]  = candidates[best]
            ptr[i, j] = best

    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i == 0:
            path.append((-1, j - 1)); j -= 1
        elif j == 0:
            path.append((i - 1, -1)); i -= 1
        else:
            d = int(ptr[i, j])
            if d == 0:
                path.append((i - 1, j - 1)); i -= 1; j -= 1
            elif d == 1:
                path.append((i - 1, -1));  i -= 1
            else:
                path.append((-1, j - 1));  j -= 1

    path.reverse()
    return path


# ── Bonus: Text-ratio check ───────────────────────────────────────────────────
def _text_ratio(text_a: str, text_b: str) -> float:
    """
    So sánh ở mức ký tự (character-level) bằng difflib.
    Dùng character thay vì word vì tiếng Việt đơn âm tiết:
    "hợp đồng," và "hợp đồng" khác nhau 1 ký tự nhưng
    word.split() thì khác nhau toàn bộ token cuối.
    Trả về ratio 0.0 → 1.0.
    """
    return difflib.SequenceMatcher(
        None, text_a.lower(), text_b.lower(), autojunk=False
    ).ratio()


# ── Phân loại một cặp NW ─────────────────────────────────────────────────────
def _classify_pair(
    chunk_a: ArticleChunk,
    chunk_b: ArticleChunk,
    sim_score: float,
    unchanged_threshold: float,
    modified_threshold: float,
) -> list[ComparedPair]:
    """
    Phân loại một cặp đã được NW ghép.
    Tích hợp text-ratio để bắt false negative (thay đổi nhỏ bị embedding bỏ sót).
    """
    if sim_score >= unchanged_threshold:
        # Kiểm tra thêm bằng text-ratio
        ratio = _text_ratio(chunk_a.content, chunk_b.content)
        if ratio >= TEXT_UNCHANGED_RATIO:
            return [ComparedPair(chunk_a, chunk_b, "UNCHANGED", sim_score)]
        else:
            # Embedding nói giống nhau, nhưng text thực sự thay đổi → MODIFIED
            return [ComparedPair(chunk_a, chunk_b, "MODIFIED", sim_score)]

    elif sim_score >= modified_threshold:
        return [ComparedPair(chunk_a, chunk_b, "MODIFIED", sim_score)]

    else:
        # Quá khác nhau → tách làm DELETED + ADDED
        return [
            ComparedPair(chunk_a, None,    "DELETED", sim_score),
            ComparedPair(None,    chunk_b, "ADDED",   0.0),
        ]


# ── Pipeline chính ────────────────────────────────────────────────────────────
def build_comparison_pairs(
    chunks_a: list[ArticleChunk],
    chunks_b: list[ArticleChunk],
    embeds_a: Optional[list[list[float]]] = None,
    embeds_b: Optional[list[list[float]]] = None,
    unchanged_threshold: float = UNCHANGED_THRESHOLD,
    modified_threshold:  float = MODIFIED_THRESHOLD,
    gap_penalty:         float = GAP_PENALTY,
) -> list[ComparedPair]:
    """
    Pipeline 2 tầng: Anchor-based → NW-on-gaps.

    Tầng 1: difflib tìm anchor (exact match) → UNCHANGED ngay, không cần embed.
    Tầng 2: NW + embedding chỉ cho các gap giữa anchors.
    Bonus:  text-ratio check để bắt thay đổi nhỏ (false negative của embedding).
    """
    # Trường hợp biên
    if not chunks_a and not chunks_b:
        return []
    if not chunks_a:
        return [ComparedPair(None, c, "ADDED", 0.0) for c in chunks_b]
    if not chunks_b:
        return [ComparedPair(c, None, "DELETED", 0.0) for c in chunks_a]

    # Embed toàn bộ nếu chưa có (dùng cho các gap)
    if embeds_a is None:
        logger.info("[Matcher] Encoding doc_A chunks...")
        embeds_a = embed_chunks(chunks_a, instruction=INSTRUCTION_DOC)
    if embeds_b is None:
        logger.info("[Matcher] Encoding doc_B chunks...")
        embeds_b = embed_chunks(chunks_b, instruction=INSTRUCTION_DOC)

    A = np.array(embeds_a, dtype=np.float32)
    B = np.array(embeds_b, dtype=np.float32)

    # Tầng 1: tìm anchors và gaps
    opcodes = _find_anchors(chunks_a, chunks_b)

    n_anchors = sum(1 for tag, i1, i2, j1, j2 in opcodes if tag == 'equal' for _ in range(i2-i1))
    n_gaps    = len(chunks_a) + len(chunks_b) - 2 * n_anchors
    logger.info(f"[Matcher] Anchors: {n_anchors} locked | Gaps: ~{n_gaps} chunks need NW")

    pairs: list[ComparedPair] = []

    for tag, i1, i2, j1, j2 in opcodes:

        # ── Anchor: exact text match → UNCHANGED ngay ──
        if tag == 'equal':
            for k in range(i2 - i1):
                # Vẫn check text-ratio để chắc chắn (phòng normalize bỏ qua dấu)
                ratio = _text_ratio(chunks_a[i1+k].content, chunks_b[j1+k].content)
                mtype = "UNCHANGED" if ratio >= TEXT_UNCHANGED_RATIO else "MODIFIED"
                pairs.append(ComparedPair(chunks_a[i1+k], chunks_b[j1+k], mtype, 1.0))
            continue

        # ── Gap: dùng NW + embedding ──
        gap_a = chunks_a[i1:i2]
        gap_b = chunks_b[j1:j2]

        if not gap_a:
            for c in gap_b:
                pairs.append(ComparedPair(None, c, "ADDED", 0.0))
            continue

        if not gap_b:
            for c in gap_a:
                pairs.append(ComparedPair(c, None, "DELETED", 0.0))
            continue

        # NW chỉ trên vùng gap
        gap_A   = A[i1:i2]
        gap_B   = B[j1:j2]
        sim_mat = _sim_matrix(gap_A, gap_B)
        alignment = _needleman_wunsch(sim_mat, gap_penalty)

        for idx_a, idx_b in alignment:
            if idx_a == -1:
                pairs.append(ComparedPair(None, gap_b[idx_b], "ADDED", 0.0))
            elif idx_b == -1:
                pairs.append(ComparedPair(gap_a[idx_a], None, "DELETED", 0.0))
            else:
                s = float(sim_mat[idx_a, idx_b])
                pairs.extend(_classify_pair(
                    gap_a[idx_a], gap_b[idx_b], s,
                    unchanged_threshold, modified_threshold,
                ))

    return pairs


# ── Hiển thị kết quả ─────────────────────────────────────────────────────────
def print_diff_summary(pairs: list[ComparedPair]) -> None:
    """In kết quả phân tích theo dạng git-diff style."""
    ICONS = {
        "UNCHANGED": "   ",
        "MODIFIED":  "~  ",
        "ADDED":     "+  ",
        "DELETED":   "-  ",
    }

    stats = {k: 0 for k in ICONS}

    print(f"\n{'='*70}")
    print("  KẾT QUẢ PHÂN TÍCH THAY ĐỔI (Semantic Git-Diff)")
    print(f"{'='*70}")

    for p in pairs:
        icon = ICONS[p.match_type]
        sim_str = f"  [sim={p.similarity:.3f}]" if p.match_type == "MODIFIED" else ""

        if p.match_type in ("UNCHANGED", "MODIFIED"):
            label_a = p.chunk_a.article_number or "N/A"
            label_b = p.chunk_b.article_number or "N/A"
            label   = label_a if label_a == label_b else f"{label_a} → {label_b}"
        elif p.match_type == "DELETED":
            label = p.chunk_a.article_number or "N/A"
        else:
            label = p.chunk_b.article_number or "N/A"

        print(f"  {icon} {label}{sim_str}")
        stats[p.match_type] += 1

    print(f"\n{'─'*70}")
    print(
        f"  Tổng kết: "
        f"{stats['UNCHANGED']} không đổi  |  "
        f"{stats['MODIFIED']} sửa đổi  |  "
        f"{stats['ADDED']} thêm mới  |  "
        f"{stats['DELETED']} bị xóa"
    )
    print(f"{'='*70}\n")
