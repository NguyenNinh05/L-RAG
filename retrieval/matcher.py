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
import re
import unicodedata
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
    NEAR_UNCHANGED_BAND,
    CLAUSE_HINT_ALPHA,
    CLAUSE_HINT_MIN_GAP,
)


# ── Data model cho kết quả ──────────────────────────────────────────────────
@dataclass
class ComparedPair:
    chunk_a:    Optional[ArticleChunk]   # None nếu ADDED
    chunk_b:    Optional[ArticleChunk]   # None nếu DELETED
    match_type: str                      # "UNCHANGED" | "MODIFIED" | "ADDED" | "DELETED" 
    similarity: float                    # cosine similarity (1.0 nếu anchor, 0.0 nếu ADDED/DELETED)
    near_threshold: bool = False         # True nếu cặp nằm sát ngưỡng UNCHANGED

    @property
    def label(self) -> str:
        if self.chunk_a:
            return self.chunk_a.article_number or "N/A"
        return self.chunk_b.article_number or "N/A"


# ── Tầng 1: Anchor detection ─────────────────────────────────────────────────
def _normalize_for_anchor(text: str) -> str:
    """Chuẩn hóa text để so sánh anchor: lowercase + bỏ dấu + chuẩn hóa punctuation/roman tokens."""
    lowered = text.lower()
    decomposed = unicodedata.normalize("NFKD", lowered)
    no_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    normalized = no_marks.replace("đ", "d").replace("Đ", "D")
    normalized = normalized.replace("—", "-").replace("–", "-")
    normalized = re.sub(r"\b(viii|vii|vi|iv|iii|ii|ix|x|v|i)\b", " ", normalized)
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return " ".join(normalized.split())


def _normalize_clause_key(value: str | None) -> str:
    if not value:
        return ""
    normalized = _normalize_for_anchor(value)
    return normalized.strip()


def _extract_clause_anchor(value: str | None) -> str:
    normalized = _normalize_clause_key(value)
    if not normalized:
        return ""
    match = re.search(r"(dieu\s+\d+|phu\s+luc\s+[a-z0-9]+)", normalized)
    return match.group(1) if match else ""


def _clause_match_bonus(chunk_a: ArticleChunk, chunk_b: ArticleChunk) -> float:
    key_a = _normalize_clause_key(chunk_a.article_number or chunk_a.title)
    key_b = _normalize_clause_key(chunk_b.article_number or chunk_b.title)
    if not key_a or not key_b:
        return 0.0
    if key_a == key_b:
        return 1.0
    anchor_a = _extract_clause_anchor(key_a)
    anchor_b = _extract_clause_anchor(key_b)
    if anchor_a and anchor_a == anchor_b:
        return 0.6
    return 0.0


def _apply_clause_hint_bonus(
    sim: np.ndarray,
    gap_a: list[ArticleChunk],
    gap_b: list[ArticleChunk],
    alpha: float,
    min_gap: int,
) -> np.ndarray:
    """Boost similarity scores softly when clause identity matches in ambiguous long gaps."""
    if alpha <= 0.0 or max(len(gap_a), len(gap_b)) < min_gap:
        return sim

    adjusted = sim.copy()
    for i, chunk_a in enumerate(gap_a):
        for j, chunk_b in enumerate(gap_b):
            bonus = _clause_match_bonus(chunk_a, chunk_b)
            if bonus > 0.0:
                adjusted[i, j] = adjusted[i, j] + alpha * bonus
    return adjusted


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
    near_unchanged_band: float,
) -> list[ComparedPair]:
    """
    Phân loại một cặp đã được NW ghép.
    Tích hợp text-ratio để bắt false negative (thay đổi nhỏ bị embedding bỏ sót).
    """
    near_floor = max(modified_threshold, unchanged_threshold - near_unchanged_band)
    near_ceiling = unchanged_threshold + near_unchanged_band

    if sim_score >= unchanged_threshold:
        # Kiểm tra thêm bằng text-ratio
        ratio = _text_ratio(chunk_a.content, chunk_b.content)
        if ratio >= TEXT_UNCHANGED_RATIO:
            # Vùng sát ngưỡng được xếp MODIFIED để downstream review an toàn hơn.
            if sim_score < near_ceiling:
                return [ComparedPair(chunk_a, chunk_b, "MODIFIED", sim_score, near_threshold=True)]
            return [ComparedPair(chunk_a, chunk_b, "UNCHANGED", sim_score)]
        else:
            # Embedding nói giống nhau, nhưng text thực sự thay đổi → MODIFIED
            return [ComparedPair(chunk_a, chunk_b, "MODIFIED", sim_score)]

    elif sim_score >= near_floor:
        return [ComparedPair(chunk_a, chunk_b, "MODIFIED", sim_score, near_threshold=True)]

    elif sim_score >= modified_threshold:
        return [ComparedPair(chunk_a, chunk_b, "MODIFIED", sim_score)]

    else:
        # Quá khác nhau → tách làm DELETED + ADDED
        return [
            ComparedPair(chunk_a, None,    "DELETED", sim_score),
            ComparedPair(None,    chunk_b, "ADDED",   0.0),
        ]


# ── Gap-only embedding (optimized lazy path) ──────────────────────────────
def _embed_gaps_only(
    chunks_a: list[ArticleChunk],
    chunks_b: list[ArticleChunk],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tim anchors truoc, chi embed cac chunk thuoc gap regions.
    Tra ve full-size arrays (dummy zeros cho anchor positions) de giu index alignment.
    """
    opcodes = _find_anchors(chunks_a, chunks_b)

    # Thu thập chỉ số gap cho doc A và doc B
    gap_indices_a: list[int] = []
    gap_indices_b: list[int] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag != 'equal':
            gap_indices_a.extend(range(i1, i2))
            gap_indices_b.extend(range(j1, j2))

    # Embed chỉ gap chunks
    gap_chunks_a = [chunks_a[i] for i in gap_indices_a]
    gap_chunks_b = [chunks_b[i] for i in gap_indices_b]

    n_anchors_a = len(chunks_a) - len(gap_indices_a)
    n_anchors_b = len(chunks_b) - len(gap_indices_b)
    logger.info(
        f"[Matcher] Optimized embed: {len(gap_indices_a)}/{len(chunks_a)} "
        f"gap chunks in A, {len(gap_indices_b)}/{len(chunks_b)} gap chunks in B "
        f"({n_anchors_a + n_anchors_b} anchors skipped)"
    )

    gap_embeds_a = embed_chunks(gap_chunks_a, instruction=INSTRUCTION_DOC) if gap_chunks_a else []
    gap_embeds_b = embed_chunks(gap_chunks_b, instruction=INSTRUCTION_DOC) if gap_chunks_b else []

    # Xác định embedding dimension từ gap embeds
    dim = 0
    if gap_embeds_a:
        dim = len(gap_embeds_a[0])
    elif gap_embeds_b:
        dim = len(gap_embeds_b[0])

    if dim == 0:
        # Không có gap nào → không cần embed, trả về array rỗng
        logger.info("[Matcher] No gaps to embed — all chunks are anchors.")
        A = np.zeros((len(chunks_a), 1), dtype=np.float32)
        B = np.zeros((len(chunks_b), 1), dtype=np.float32)
        return A, B

    # Xây full-size arrays với dummy zeros cho anchor positions
    A = np.zeros((len(chunks_a), dim), dtype=np.float32)
    B = np.zeros((len(chunks_b), dim), dtype=np.float32)

    # Đặt real embeddings tại đúng vị trí gap
    for idx, emb in zip(gap_indices_a, gap_embeds_a):
        A[idx] = emb
    for idx, emb in zip(gap_indices_b, gap_embeds_b):
        B[idx] = emb

    return A, B


# ── Pipeline chính ────────────────────────────────────────────────────────────
def build_comparison_pairs(
    chunks_a: list[ArticleChunk],
    chunks_b: list[ArticleChunk],
    embeds_a: Optional[list[list[float]]] = None,
    embeds_b: Optional[list[list[float]]] = None,
    unchanged_threshold: float = UNCHANGED_THRESHOLD,
    modified_threshold:  float = MODIFIED_THRESHOLD,
    gap_penalty:         float = GAP_PENALTY,
    near_unchanged_band: float = NEAR_UNCHANGED_BAND,
    clause_hint_alpha:   float = CLAUSE_HINT_ALPHA,
    clause_hint_min_gap: int = CLAUSE_HINT_MIN_GAP,
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

    # ── Build embedding arrays ──────────────────────────────────────────
    # Nếu caller đã truyền precomputed embeds (từ API/main) → dùng luôn.
    # Nếu embeds=None → tối ưu: tìm anchors trước, chỉ embed gap chunks.
    if embeds_a is not None and embeds_b is not None:
        A = np.array(embeds_a, dtype=np.float32)
        B = np.array(embeds_b, dtype=np.float32)
    else:
        A, B = _embed_gaps_only(chunks_a, chunks_b)

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
        sim_mat = _apply_clause_hint_bonus(
            sim_mat,
            gap_a,
            gap_b,
            alpha=clause_hint_alpha,
            min_gap=clause_hint_min_gap,
        )
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
                    near_unchanged_band,
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
