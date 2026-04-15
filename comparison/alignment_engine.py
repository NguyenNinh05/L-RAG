"""
comparison/alignment_engine.py
===============================
LegalAlignmentEngine — Core của Phase 2: Indexing & Alignment Strategy.

Thuật toán 4 bước:
    Bước 1 — Similarity Matrix:
        Ma trận N×M với S[i][j] = 0.6×Cosine(Semantic) + 0.3×JaroWinkler(Title) + 0.1×OrdinalProximity

    Bước 2 — Hungarian Matching:
        scipy.optimize.linear_sum_assignment trên Cost Matrix = 1 - S
        Lọc cặp unmatched nếu confidence < θ = 0.65

    Bước 3 — Split/Merge Handler:
        Fallback cho các node unmatched.
        Gộp text 2 nodes nhỏ so sánh với 1 node lớn → threshold 0.8

    Bước 4 — Hierarchical:
        Đệ quy quy trình trên cho ClauseNode bên trong ArticleNode đã match.

Output: DiffPairCatalog chứa toàn bộ cặp (matched/added/deleted/split/merged).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import jellyfish
import numpy as np
from scipy.optimize import linear_sum_assignment

from comparison.embedding_manager import BGEM3Manager
from comparison.models import (
    DiffPair,
    DiffPairCatalog,
    MatchType,
    NodeEmbeddings,
    NodeVersion,
)
from comparison.qdrant_indexer import QdrantManager
from ingestion.models import ArticleNode, ClauseNode, LegalDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AlignmentConfig:
    """
    Cấu hình cho LegalAlignmentEngine.

    Trọng số trong Similarity formula:
        S[i][j] = w_sem * Cosine(Semantic)
                + w_jaro * JaroWinkler(Title)
                + w_ord  * OrdinalProximity

    Tổng w_sem + w_jaro + w_ord phải = 1.0
    """

    # Trọng số similarity
    w_semantic: float = 0.6       # Cosine similarity của semantic dense vector
    w_jaro_winkler: float = 0.3   # Jaro-Winkler của title text
    w_ordinal: float = 0.1        # Ordinal proximity (vị trí tương đối trong văn bản)

    # Ngưỡng
    match_threshold: float = 0.65    # θ: dưới ngưỡng → coi như unmatched
    split_merge_threshold: float = 0.80  # Ngưỡng chặt hơn cho split/merge detection

    # Embedding batch size cho alignment (có thể nhỏ hơn training batch)
    embed_batch_size: int = 32

    def __post_init__(self) -> None:
        total = self.w_semantic + self.w_jaro_winkler + self.w_ordinal
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Tổng trọng số phải = 1.0, nhận được {total:.4f}. "
                f"(w_semantic={self.w_semantic}, w_jaro={self.w_jaro_winkler}, w_ord={self.w_ordinal})"
            )


# ---------------------------------------------------------------------------
# NodeRecord — lưu trạng thái của một node trong quá trình alignment
# ---------------------------------------------------------------------------


@dataclass
class NodeRecord:
    """Wrapper nhẹ dùng nội bộ trong engine, lưu node + embedding + metadata."""

    node_id: str
    title: str           # title/label ngắn dùng cho JaroWinkler
    raw_text: str        # full text dùng cho semantic embed
    ordinal: int         # vị trí thứ tự trong tài liệu (0-indexed)
    semantic_vec: np.ndarray = field(default_factory=lambda: np.array([]))

    # Giữ tham chiếu object gốc để lấy clauses sau (hierarchical)
    article_ref: ArticleNode | None = None
    clause_ref: ClauseNode | None = None


# ---------------------------------------------------------------------------
# LegalAlignmentEngine
# ---------------------------------------------------------------------------


class LegalAlignmentEngine:
    """
    Công cụ đối chiếu tự động giữa 2 phiên bản văn bản pháp lý.

    Ví dụ sử dụng:
        engine = LegalAlignmentEngine(
            embed_manager=BGEM3Manager(),
            config=AlignmentConfig(),
        )
        catalog = engine.align_documents(doc_v1, doc_v2)
        print(catalog.summary())
    """

    def __init__(
        self,
        embed_manager: BGEM3Manager,
        config: AlignmentConfig | None = None,
        qdrant_manager: QdrantManager | None = None,
    ) -> None:
        """
        Args:
            embed_manager:   BGEM3Manager đã load model (FP16).
            config:          Cấu hình alignment. None → dùng default.
            qdrant_manager:  QdrantManager (optional). Nếu None, engine chạy
                             pure in-memory mà không lưu Qdrant.
        """
        self._emb = embed_manager
        self._cfg = config or AlignmentConfig()
        self._qdrant = qdrant_manager

        logger.info(
            "LegalAlignmentEngine khởi tạo: θ=%.2f, w=(sem=%.1f, jaro=%.1f, ord=%.1f)",
            self._cfg.match_threshold,
            self._cfg.w_semantic,
            self._cfg.w_jaro_winkler,
            self._cfg.w_ordinal,
        )

    # ======================================================================
    # Public API
    # ======================================================================

    def align_documents(
        self,
        doc_v1: LegalDocument,
        doc_v2: LegalDocument,
        collection_name: str | None = None,
    ) -> DiffPairCatalog:
        """
        Align toàn bộ 2 tài liệu, trả về DiffPairCatalog hoàn chỉnh.

        Quy trình:
            1. Thu thập tất cả ArticleNode từ cả 2 document.
            2. Embed tất cả articles (structural + semantic).
            3. Chạy Hungarian alignment ở cấp Article.
            4. Với mỗi cặp matched articles → đệ quy align Clause con.
            5. Gom tất cả DiffPair vào DiffPairCatalog.

        Args:
            doc_v1:           LegalDocument phiên bản V1.
            doc_v2:           LegalDocument phiên bản V2.
            collection_name:  Tên Qdrant collection để index (None → không index).

        Returns:
            DiffPairCatalog với toàn bộ cặp (matched/added/deleted/split/merged).
        """
        logger.info(
            "Bắt đầu align: '%s' (V1) ↔ '%s' (V2)",
            doc_v1.file_name,
            doc_v2.file_name,
        )

        catalog = DiffPairCatalog(
            v1_doc_id=doc_v1.doc_id,
            v2_doc_id=doc_v2.doc_id,
            match_threshold=self._cfg.match_threshold,
            split_merge_threshold=self._cfg.split_merge_threshold,
        )

        # ── Bước A: Thu thập & embed ArticleNodes ──
        v1_articles = doc_v1.iter_all_articles()
        v2_articles = doc_v2.iter_all_articles()

        logger.info(
            "Articles: V1=%d, V2=%d",
            len(v1_articles),
            len(v2_articles),
        )

        v1_records = self._build_article_records(v1_articles, doc_v1)
        v2_records = self._build_article_records(v2_articles, doc_v2)

        # ── Bước B: Optionally index vào Qdrant ──
        if self._qdrant and collection_name:
            self._index_to_qdrant(v1_records, v2_records, doc_v1, doc_v2, collection_name)

        # ── Bước C: Align cấp Article ──
        article_pairs, v1_unmatched_idx, v2_unmatched_idx = self._run_alignment(
            v1_records, v2_records
        )
        catalog.pairs.extend(article_pairs)

        logger.info(
            "Article alignment: %d matched, %d V1-unmatched, %d V2-unmatched",
            len([p for p in article_pairs if p.match_type == MatchType.MATCHED]),
            len(v1_unmatched_idx),
            len(v2_unmatched_idx),
        )

        # ── Bước D: Hierarchical — align Clauses bên trong mỗi matched Article ──
        for pair in article_pairs:
            if pair.match_type != MatchType.MATCHED:
                continue

            v1_art_id = pair.v1_ids[0]
            v2_art_id = pair.v2_ids[0]

            v1_art = next((r for r in v1_records if r.node_id == v1_art_id), None)
            v2_art = next((r for r in v2_records if r.node_id == v2_art_id), None)

            if v1_art is None or v2_art is None:
                continue

            v1_clauses = v1_art.article_ref.clauses if v1_art.article_ref else []
            v2_clauses = v2_art.article_ref.clauses if v2_art.article_ref else []

            if not v1_clauses and not v2_clauses:
                continue

            clause_pairs = self._align_clauses(
                v1_clauses=v1_clauses,
                v2_clauses=v2_clauses,
                v1_article=v1_art.article_ref,
                v2_article=v2_art.article_ref,
                doc_v1=doc_v1,
                doc_v2=doc_v2,
            )
            catalog.pairs.extend(clause_pairs)

        logger.info(
            "Alignment hoàn tất. Tổng: %s",
            catalog.summary(),
        )
        return catalog

    # ======================================================================
    # Bước 1 — Similarity Matrix
    # ======================================================================

    def compute_similarity_matrix(
        self,
        v1_records: list[NodeRecord],
        v2_records: list[NodeRecord],
    ) -> np.ndarray:
        """
        Tính ma trận tương đồng N×M.

        S[i][j] = (0.6 × Cosine(Semantic_i, Semantic_j))
                + (0.3 × JaroWinkler(Title_i, Title_j))
                + (0.1 × OrdinalProximity(ordinal_i, ordinal_j, N, M))

        Args:
            v1_records: List N NodeRecord từ V1.
            v2_records: List M NodeRecord từ V2.

        Returns:
            np.ndarray shape (N, M), dtype float32, giá trị trong [0, 1].
        """
        N, M = len(v1_records), len(v2_records)
        if N == 0 or M == 0:
            return np.zeros((N, M), dtype=np.float32)

        # ── Component 1: Cosine Semantic similarity ──
        sem_matrix = self._cosine_similarity_matrix(v1_records, v2_records)  # (N, M)

        # ── Component 2: Jaro-Winkler Title similarity ──
        jaro_matrix = self._jaro_winkler_matrix(v1_records, v2_records)      # (N, M)

        # ── Component 3: Ordinal Proximity ──
        ord_matrix = self._ordinal_proximity_matrix(v1_records, v2_records, N, M)  # (N, M)

        # ── Weighted sum ──
        S = (
            self._cfg.w_semantic      * sem_matrix
            + self._cfg.w_jaro_winkler * jaro_matrix
            + self._cfg.w_ordinal      * ord_matrix
        ).astype(np.float32)

        # Clamp về [0, 1] để tránh float precision issues
        np.clip(S, 0.0, 1.0, out=S)
        return S

    def _cosine_similarity_matrix(
        self,
        v1: list[NodeRecord],
        v2: list[NodeRecord],
    ) -> np.ndarray:
        """
        Tính ma trận cosine similarity từ semantic_vec đã L2-normalize sẵn.
        Cosine = dot(u, v) khi cả 2 đã normalized.
        """
        v1_mat = np.stack([r.semantic_vec for r in v1], axis=0)  # (N, D)
        v2_mat = np.stack([r.semantic_vec for r in v2], axis=0)  # (M, D)

        # L2 normalize (phòng trường hợp chưa normalize)
        v1_mat = self._l2_normalize(v1_mat)
        v2_mat = self._l2_normalize(v2_mat)

        return (v1_mat @ v2_mat.T).astype(np.float32)  # (N, M)

    def _jaro_winkler_matrix(
        self,
        v1: list[NodeRecord],
        v2: list[NodeRecord],
    ) -> np.ndarray:
        """
        Tính ma trận Jaro-Winkler distance giữa các title strings.
        jellyfish.jaro_winkler_similarity(s1, s2) → float [0, 1].
        """
        N, M = len(v1), len(v2)
        mat = np.zeros((N, M), dtype=np.float32)
        for i, r1 in enumerate(v1):
            for j, r2 in enumerate(v2):
                mat[i, j] = jellyfish.jaro_winkler_similarity(
                    r1.title.lower(), r2.title.lower()
                )
        return mat

    def _ordinal_proximity_matrix(
        self,
        v1: list[NodeRecord],
        v2: list[NodeRecord],
        N: int,
        M: int,
    ) -> np.ndarray:
        """
        Tính ma trận ordinal proximity dựa trên vị trí tương đối.

        Ý tưởng: Hai node có cùng vị trí tương đối (relative position)
        trong tài liệu tương ứng thì có ordinal score cao.

        OrdinalProximity(i, j) = 1 - |i/N - j/M|

        Giá trị ∈ [0, 1], tối đa = 1 khi i/N == j/M.
        """
        v1_positions = np.array([r.ordinal / max(N - 1, 1) for r in v1], dtype=np.float32)  # (N,)
        v2_positions = np.array([r.ordinal / max(M - 1, 1) for r in v2], dtype=np.float32)  # (M,)

        # Broadcasting: |pos_i - pos_j| → (N, M)
        diff = np.abs(v1_positions[:, np.newaxis] - v2_positions[np.newaxis, :])
        return (1.0 - diff).astype(np.float32)

    # ======================================================================
    # Bước 2 — Hungarian Matching
    # ======================================================================

    def hungarian_match(
        self,
        similarity_matrix: np.ndarray,
    ) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
        """
        Áp dụng Hungarian algorithm để tìm matching tối ưu.

        Sử dụng scipy.optimize.linear_sum_assignment trên Cost Matrix = 1 - S.
        Sau đó lọc các cặp có confidence < θ.

        Args:
            similarity_matrix: Ma trận S shape (N, M).

        Returns:
            matched_pairs:     List of (i, j, score) — cặp thoả ngưỡng.
            v1_unmatched_idx:  List index i không có cặp trong V2.
            v2_unmatched_idx:  List index j không có cặp trong V1.
        """
        N, M = similarity_matrix.shape
        if N == 0 or M == 0:
            return [], list(range(N)), list(range(M))

        # Cost matrix = 1 - S (scipy minimize → ta cần đổi sang cost)
        cost_matrix = 1.0 - similarity_matrix

        # Hungarian algorithm — O(n^3)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_pairs: list[tuple[int, int, float]] = []
        matched_v1: set[int] = set()
        matched_v2: set[int] = set()

        for i, j in zip(row_ind, col_ind):
            score = float(similarity_matrix[i, j])
            if score >= self._cfg.match_threshold:
                matched_pairs.append((i, j, score))
                matched_v1.add(i)
                matched_v2.add(j)
            else:
                logger.debug(
                    "Cặp (%d, %d) bị loại: score=%.3f < θ=%.2f",
                    i, j, score, self._cfg.match_threshold,
                )

        v1_unmatched = [i for i in range(N) if i not in matched_v1]
        v2_unmatched = [j for j in range(M) if j not in matched_v2]

        logger.debug(
            "Hungarian: %d matched, %d V1-unmatched, %d V2-unmatched",
            len(matched_pairs),
            len(v1_unmatched),
            len(v2_unmatched),
        )

        return matched_pairs, v1_unmatched, v2_unmatched

    # ======================================================================
    # Bước 3 — Split/Merge Handler
    # ======================================================================

    def detect_split_merge(
        self,
        v1_unmatched: list[NodeRecord],
        v2_unmatched: list[NodeRecord],
    ) -> tuple[list[DiffPair], list[int], list[int]]:
        """
        Phát hiện các trường hợp Split/Merge trong nodes chưa matched.

        Chiến lược:
            - SPLIT: 1 node V1 lớn → kiểm tra xem ghép 2 nodes V2 nhỏ
                     có tạo ra text tương đồng >= 0.8 không.
            - MERGE: 2 nodes V1 nhỏ → gộp text → so sánh với 1 node V2 lớn.
            - Sau khi phát hiện, các node tham gia được loại khỏi remaining lists.

        Args:
            v1_unmatched: NodeRecord V1 chưa matched.
            v2_unmatched: NodeRecord V2 chưa matched.

        Returns:
            split_merge_pairs:    DiffPair với type SPLIT hoặc MERGED.
            still_v1_unmatched:  Index trong v1_unmatched vẫn còn unmatched.
            still_v2_unmatched:  Index trong v2_unmatched vẫn còn unmatched.
        """
        result_pairs: list[DiffPair] = []
        used_v1: set[int] = set()
        used_v2: set[int] = set()

        # ── Detect MERGE: 2 V1 → 1 V2 ──
        # Với mỗi V2 node, thử ghép tất cả cặp V1 nodes
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

                    # Gộp text của 2 V1 nodes
                    merged_text = (
                        v1_unmatched[i1].raw_text
                        + "\n"
                        + v1_unmatched[i2].raw_text
                    )
                    score = self._text_cosine_similarity(merged_text, v2_rec.raw_text)

                    if score >= self._cfg.split_merge_threshold:
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
                logger.debug(
                    "MERGE detected: V1[%d]+V1[%d] → V2[%d], score=%.3f",
                    i1, i2, j, score,
                )

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

                    # Gộp text của 2 V2 nodes → so sánh với V1 node lớn
                    merged_text = (
                        v2_unmatched[j1].raw_text
                        + "\n"
                        + v2_unmatched[j2].raw_text
                    )
                    score = self._text_cosine_similarity(v1_rec.raw_text, merged_text)

                    if score >= self._cfg.split_merge_threshold:
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
                logger.debug(
                    "SPLIT detected: V1[%d] → V2[%d]+V2[%d], score=%.3f",
                    i, j1, j2, score,
                )

        still_v1 = [i for i in range(len(v1_unmatched)) if i not in used_v1]
        still_v2 = [j for j in range(len(v2_unmatched)) if j not in used_v2]

        return result_pairs, still_v1, still_v2

    # ======================================================================
    # Bước 4 — Hierarchical (Clause-level)
    # ======================================================================

    def _align_clauses(
        self,
        v1_clauses: list[ClauseNode],
        v2_clauses: list[ClauseNode],
        v1_article: ArticleNode | None,
        v2_article: ArticleNode | None,
        doc_v1: LegalDocument,
        doc_v2: LegalDocument,
    ) -> list[DiffPair]:
        """
        Đệ quy alignment cho các ClauseNode bên trong 1 cặp Article đã matched.

        Quy trình giống Article-level nhưng:
            - ordinal chỉ tính trong phạm vi article (local ordinal)
            - JaroWinkler dùng "Khoản N" hoặc nội dung đầu của khoản

        Returns:
            Danh sách DiffPair cho các cặp clause.
        """
        if not v1_clauses and not v2_clauses:
            return []

        # Build records cho clauses
        v1_records = self._build_clause_records(v1_clauses, v1_article, doc_v1)
        v2_records = self._build_clause_records(v2_clauses, v2_article, doc_v2)

        return self._run_alignment(v1_records, v2_records)[0]

    # ======================================================================
    # Internal: Orchestration
    # ======================================================================

    def _run_alignment(
        self,
        v1_records: list[NodeRecord],
        v2_records: list[NodeRecord],
    ) -> tuple[list[DiffPair], list[int], list[int]]:
        """
        Chạy full alignment pipeline (Bước 1 → 2 → 3) trên 2 tập records.

        Returns:
            (all_pairs, final_v1_unmatched_idx, final_v2_unmatched_idx)
        """
        all_pairs: list[DiffPair] = []

        if not v1_records and not v2_records:
            return [], [], []

        # Handle degenerate cases
        if not v1_records:
            # Tất cả V2 nodes là ADDED
            all_pairs.extend(
                DiffPair(
                    v2_ids=[r.node_id],
                    match_type=MatchType.ADDED,
                    confidence_score=0.0,
                    v2_texts=[r.raw_text],
                )
                for r in v2_records
            )
            return all_pairs, [], []

        if not v2_records:
            # Tất cả V1 nodes là DELETED
            all_pairs.extend(
                DiffPair(
                    v1_ids=[r.node_id],
                    match_type=MatchType.DELETED,
                    confidence_score=0.0,
                    v1_texts=[r.raw_text],
                )
                for r in v1_records
            )
            return all_pairs, [], []

        # Embed tất cả records (nếu chưa có vector)
        self._ensure_embeddings(v1_records)
        self._ensure_embeddings(v2_records)

        # ── Bước 1: Similarity Matrix ──
        sim_matrix = self.compute_similarity_matrix(v1_records, v2_records)

        # ── Bước 2: Hungarian Matching ──
        matched_tuples, v1_unmatched_idx, v2_unmatched_idx = self.hungarian_match(sim_matrix)

        # Build DiffPair cho matched
        for i, j, score in matched_tuples:
            v1_rec = v1_records[i]
            v2_rec = v2_records[j]

            # Tính breakdown điểm
            sem_score = float(sim_matrix[i, j]) if self._cfg.w_semantic > 0 else None
            jaro_score = float(
                jellyfish.jaro_winkler_similarity(
                    v1_rec.title.lower(), v2_rec.title.lower()
                )
            )
            ord_score = float(
                1.0 - abs(
                    v1_rec.ordinal / max(len(v1_records) - 1, 1)
                    - v2_rec.ordinal / max(len(v2_records) - 1, 1)
                )
            )

            all_pairs.append(
                DiffPair(
                    v1_ids=[v1_rec.node_id],
                    v2_ids=[v2_rec.node_id],
                    match_type=MatchType.MATCHED,
                    confidence_score=round(score, 4),
                    semantic_score=round(sem_score, 4),
                    jaro_winkler_score=round(jaro_score, 4),
                    ordinal_proximity_score=round(ord_score, 4),
                    v1_texts=[v1_rec.raw_text],
                    v2_texts=[v2_rec.raw_text],
                )
            )

        # ── Bước 3: Split/Merge Handler ──
        v1_unmatched_recs = [v1_records[i] for i in v1_unmatched_idx]
        v2_unmatched_recs = [v2_records[j] for j in v2_unmatched_idx]

        if v1_unmatched_recs or v2_unmatched_recs:
            sm_pairs, still_v1_local, still_v2_local = self.detect_split_merge(
                v1_unmatched_recs, v2_unmatched_recs
            )
            all_pairs.extend(sm_pairs)

            # Còn lại sau split/merge → ADDED / DELETED
            final_v1_unmatched = [v1_unmatched_recs[k] for k in still_v1_local]
            final_v2_unmatched = [v2_unmatched_recs[k] for k in still_v2_local]

            for rec in final_v1_unmatched:
                all_pairs.append(
                    DiffPair(
                        v1_ids=[rec.node_id],
                        match_type=MatchType.DELETED,
                        confidence_score=0.0,
                        v1_texts=[rec.raw_text],
                    )
                )
            for rec in final_v2_unmatched:
                all_pairs.append(
                    DiffPair(
                        v2_ids=[rec.node_id],
                        match_type=MatchType.ADDED,
                        confidence_score=0.0,
                        v2_texts=[rec.raw_text],
                    )
                )

        return all_pairs, v1_unmatched_idx, v2_unmatched_idx

    # ======================================================================
    # Internal: Record builders
    # ======================================================================

    def _build_article_records(
        self, articles: list[ArticleNode], doc: LegalDocument
    ) -> list[NodeRecord]:
        """Tạo NodeRecord cho danh sách ArticleNode."""
        records: list[NodeRecord] = []
        for i, art in enumerate(articles):
            records.append(
                NodeRecord(
                    node_id=art.node_id,
                    title=art.full_title,
                    raw_text=self._get_article_raw(art),
                    ordinal=i,
                    article_ref=art,
                )
            )
        return records

    def _build_clause_records(
        self,
        clauses: list[ClauseNode],
        parent_article: ArticleNode | None,
        doc: LegalDocument,
    ) -> list[NodeRecord]:
        """Tạo NodeRecord cho danh sách ClauseNode."""
        records: list[NodeRecord] = []
        art_num = parent_article.number if parent_article else "?"
        for i, clause in enumerate(clauses):
            label = f"Điều {art_num} Khoản {clause.number}"
            raw = clause.content
            for pt in clause.points:
                raw += f"\n{pt.label}) {pt.content}"

            records.append(
                NodeRecord(
                    node_id=clause.node_id,
                    title=label,
                    raw_text=raw.strip(),
                    ordinal=i,
                    clause_ref=clause,
                )
            )
        return records

    # ======================================================================
    # Internal: Embedding helpers
    # ======================================================================

    def _ensure_embeddings(self, records: list[NodeRecord]) -> None:
        """
        Nhúng semantic vector cho các records chưa có vector.
        Dùng batch embedding để hiệu quả.
        """
        missing_idx = [i for i, r in enumerate(records) if r.semantic_vec.size == 0]
        if not missing_idx:
            return

        texts = [records[i].raw_text for i in missing_idx]
        vecs = self._emb.embed_texts_semantic(texts)  # (K, D)

        for local_i, global_i in enumerate(missing_idx):
            records[global_i].semantic_vec = vecs[local_i]

    # ======================================================================
    # Internal: Text similarity (dùng cho split/merge)
    # ======================================================================

    def _text_cosine_similarity(self, text_a: str, text_b: str) -> float:
        """
        Tính cosine similarity trực tiếp từ 2 chuỗi text.
        Gọi model embed → dot product. Chi phí cao, chỉ dùng cho fallback.
        """
        vecs = self._emb.embed_texts_semantic([text_a, text_b])  # (2, D)
        if vecs.shape[0] < 2:
            return 0.0
        # Đã L2-normalize bên trong embed_texts_semantic
        return float(np.dot(vecs[0], vecs[1]))

    # ======================================================================
    # Internal: Qdrant indexing
    # ======================================================================

    def _index_to_qdrant(
        self,
        v1_records: list[NodeRecord],
        v2_records: list[NodeRecord],
        doc_v1: LegalDocument,
        doc_v2: LegalDocument,
        collection_name: str,
    ) -> None:
        """Index toàn bộ embeddings vào Qdrant (nếu qdrant_manager được cung cấp)."""
        if self._qdrant is None:
            return

        from comparison.qdrant_indexer import QdrantCollectionConfig
        from comparison.models import NodeVersion

        if not self._qdrant.collection_exists(collection_name):
            self._qdrant.create_collection(
                QdrantCollectionConfig(collection_name=collection_name)
            )

        # Embed V1 articles
        v1_arts = [r.article_ref for r in v1_records if r.article_ref is not None]
        v2_arts = [r.article_ref for r in v2_records if r.article_ref is not None]

        v1_embs = self._emb.embed_article_nodes(v1_arts, NodeVersion.V1, doc_v1.doc_id)
        v2_embs = self._emb.embed_article_nodes(v2_arts, NodeVersion.V2, doc_v2.doc_id)

        self._qdrant.upsert_embeddings(collection_name, v1_embs + v2_embs)
        logger.info("Đã index %d V1 + %d V2 article embeddings vào Qdrant.", len(v1_embs), len(v2_embs))

    # ======================================================================
    # Utility: static helpers
    # ======================================================================

    @staticmethod
    def _l2_normalize(mat: np.ndarray) -> np.ndarray:
        """L2 normalize theo axis=1 (mỗi hàng là 1 vector)."""
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return mat / norms

    @staticmethod
    def _get_article_raw(art: ArticleNode) -> str:
        """Lấy text thô của toàn bộ Điều."""
        parts: list[str] = []
        if art.title:
            parts.append(art.title)
        if art.intro:
            parts.append(art.intro)
        for c in art.clauses:
            parts.append(f"{c.number}. {c.content}")
            for pt in c.points:
                parts.append(f"  {pt.label}) {pt.content}")
        return "\n".join(parts)
