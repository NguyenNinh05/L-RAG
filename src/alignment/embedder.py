"""
src/alignment/embedder.py
================================
BGEM3Manager — Quản lý toàn bộ quá trình nhúng vector (embedding) cho Phase 2.

Sử dụng BAAI/bge-m3 qua thư viện FlagEmbedding với chế độ FP16 để tiết kiệm VRAM.

BGE-M3 outputs 3 loại biểu diễn:
    1. dense_vecs:  Dense vector 1024 chiều (dùng cho cosine similarity)
    2. lexical_weights: Sparse BM25-style weights (dùng cho lexical matching)
    3. colbert_vecs: ColBERT late-interaction vectors (không dùng trong Phase 2)

Hai loại embedding được tạo cho mỗi node:
    - structural_embed: Chỉ nhúng title/ordinal — giúp nhận diện "Điều 5" ↔ "Điều 5"
                        dù nội dung đã thay đổi.
    - semantic_embed:   Nhúng full_text kèm breadcrumb — so sánh ngữ nghĩa sâu.

Batch processing và FP16 đảm bảo hiệu suất trên GPU với VRAM hạn chế.

Note về import:
    FlagEmbedding được import LAZILY bên trong __init__ (không phải module-level)
    để tránh crash ImportError khi transformers version không tương thích.
    Các module khác hoàn toàn có thể `from comparison.embedding_manager import BGEM3Manager`
    mà không cần FlagEmbedding cài đặt (chỉ cần khi instantiate).
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

import numpy as np

from .diff_catalog import NodeEmbeddings, NodeVersion, QdrantPayload
from src.ingestion.models import ArticleNode, ClauseNode

if TYPE_CHECKING:
    # Chỉ dùng cho type checker, không chạy lúc runtime
    from FlagEmbedding import BGEM3FlagModel as _BGEM3FlagModelType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DENSE_DIM = 1024  # BGE-M3 dense output dimension (cố định)

# Template cho 2 loại embedding text
_STRUCTURAL_TEMPLATE = "Điều {ordinal}: {title}"
_CLAUSE_STRUCTURAL_TEMPLATE = "Khoản {ordinal}: {title}"

# Prefix cho semantic embedding (BGE-M3 dùng instruction prefix để tăng chất lượng)
_SEMANTIC_PREFIX = ""  # BGE-M3 không cần prefix như E5; để trống cho Vietnamese docs


# ---------------------------------------------------------------------------
# BGEM3Manager
# ---------------------------------------------------------------------------


class BGEM3Manager:
    """
    Manager cho BAAI/bge-m3 embedding model trong Phase 2.

    Chạy FP16 để tiết kiệm VRAM (~2x so với FP32) mà hầu như không ảnh hưởng quality.

    Ví dụ sử dụng:
        manager = BGEM3Manager()
        embeddings = manager.embed_article_nodes(
            articles=v1_articles,
            version=NodeVersion.V1,
            doc_id="doc_v1_abc",
        )
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        use_fp16: bool = True,
        batch_size: int = 16,
        max_length: int = 1024,
        device: str | None = None,
    ) -> None:
        """
        Khởi tạo và load BAAI/bge-m3.

        Args:
            model_name:  Tên model HuggingFace hoặc đường dẫn local.
            use_fp16:    Dùng FP16 để tiết kiệm VRAM (khuyến nghị).
            batch_size:  Số texts nhúng mỗi batch (giảm nếu OOM).
            max_length:  Max token length (BGE-M3 hỗ trợ tới 8192).
            device:      "cuda", "cpu", hoặc None (auto-detect).
        """
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self.max_length = max_length

        logger.info(
            "Đang load model %s (fp16=%s, batch_size=%d, max_length=%d) ...",
            model_name,
            use_fp16,
            batch_size,
            max_length,
        )

        # Lazy import — tránh crash khi transformers version không tương thích
        try:
            from FlagEmbedding import BGEM3FlagModel  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "FlagEmbedding chưa được cài đặt hoặc bị lỗi version.\n"
                "Chạy: pip install FlagEmbedding\n"
                f"Chi tiết: {e}"
            ) from e

        self._model = BGEM3FlagModel(
            model_name_or_path=model_name,
            use_fp16=use_fp16,
            device=device,
        )

        logger.info("Load model thành công: %s", model_name)

    # ------------------------------------------------------------------
    # Public: Article-level embedding
    # ------------------------------------------------------------------

    def embed_article_nodes(
        self,
        articles: list[ArticleNode],
        version: NodeVersion,
        doc_id: str,
        section_title: str = "",
        start_ordinal: int = 0,
    ) -> list[NodeEmbeddings]:
        """
        Tạo NodeEmbeddings cho danh sách ArticleNode.

        Mỗi ArticleNode sẽ có:
            - 1 structural embedding (title + số điều)
            - 1 semantic embedding (full text: intro + preview khoản, kèm breadcrumb)

        Args:
            articles:      Danh sách ArticleNode cần embed.
            version:       NodeVersion.V1 hoặc NodeVersion.V2.
            doc_id:        ID tài liệu nguồn.
            section_title: Tiêu đề chương/phần chứa các điều (cho breadcrumb).
            start_ordinal: Giá trị ordinal bắt đầu (để tính thứ tự toàn cục).

        Returns:
            Danh sách NodeEmbeddings tương ứng 1-1 với articles.
        """
        if not articles:
            return []

        structural_texts: list[str] = []
        semantic_texts: list[str] = []
        payloads: list[QdrantPayload] = []

        for i, article in enumerate(articles):
            ordinal = start_ordinal + i
            s_text = self._build_article_structural_text(article)
            sem_text = self._build_article_semantic_text(article, section_title)

            structural_texts.append(s_text)
            semantic_texts.append(sem_text)

            payloads.append(
                QdrantPayload(
                    node_id=article.node_id,
                    doc_id=doc_id,
                    version=version,
                    node_type="article",
                    ordinal=ordinal,
                    raw_text=self._get_article_raw_text(article),
                    title=article.title,
                    breadcrumb=self._build_article_breadcrumb(article, section_title),
                    article_number=str(article.number),
                    clause_number="",
                )
            )

        return self._batch_embed_and_build(
            structural_texts=structural_texts,
            semantic_texts=semantic_texts,
            payloads=payloads,
        )

    # ------------------------------------------------------------------
    # Public: Clause-level embedding
    # ------------------------------------------------------------------

    def embed_clause_nodes(
        self,
        clauses: list[ClauseNode],
        parent_article: ArticleNode,
        version: NodeVersion,
        doc_id: str,
        section_title: str = "",
        start_ordinal: int = 0,
    ) -> list[NodeEmbeddings]:
        """
        Tạo NodeEmbeddings cho danh sách ClauseNode con của một ArticleNode.

        Args:
            clauses:         Danh sách ClauseNode cần embed.
            parent_article:  ArticleNode cha (để lấy context breadcrumb).
            version:         NodeVersion.V1 hoặc NodeVersion.V2.
            doc_id:          ID tài liệu nguồn.
            section_title:   Tiêu đề chương/phần.
            start_ordinal:   Giá trị ordinal bắt đầu (global).

        Returns:
            Danh sách NodeEmbeddings tương ứng 1-1 với clauses.
        """
        if not clauses:
            return []

        structural_texts: list[str] = []
        semantic_texts: list[str] = []
        payloads: list[QdrantPayload] = []

        for i, clause in enumerate(clauses):
            ordinal = start_ordinal + i
            s_text = self._build_clause_structural_text(clause, parent_article)
            sem_text = self._build_clause_semantic_text(clause, parent_article, section_title)

            structural_texts.append(s_text)
            semantic_texts.append(sem_text)

            payloads.append(
                QdrantPayload(
                    node_id=clause.node_id,
                    doc_id=doc_id,
                    version=version,
                    node_type="clause",
                    ordinal=ordinal,
                    raw_text=self._get_clause_raw_text(clause),
                    title=f"Khoản {clause.number}",
                    breadcrumb=self._build_clause_breadcrumb(
                        clause, parent_article, section_title
                    ),
                    article_number=str(parent_article.number),
                    clause_number=str(clause.number),
                )
            )

        return self._batch_embed_and_build(
            structural_texts=structural_texts,
            semantic_texts=semantic_texts,
            payloads=payloads,
        )

    # ------------------------------------------------------------------
    # Public: Raw text embedding (for alignment engine)
    # ------------------------------------------------------------------

    def embed_texts_semantic(self, texts: list[str]) -> np.ndarray:
        """
        Nhúng danh sách text thuần — dùng cho semantic similarity trong engine.

        Returns:
            numpy array shape (N, 1024), đã L2-normalize.
        """
        if not texts:
            return np.zeros((0, DENSE_DIM), dtype=np.float32)

        output = self._model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        vecs = np.array(output["dense_vecs"], dtype=np.float32)
        # L2 normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vecs / norms

    # ------------------------------------------------------------------
    # Private: Text builders (structural)
    # ------------------------------------------------------------------

    def _build_article_structural_text(self, article: ArticleNode) -> str:
        """
        Tạo text ngắn để nhúng cấu trúc của Điều.
        Chỉ chứa số và tiêu đề — không có nội dung.
        Ví dụ: "Điều 5: Quyền và nghĩa vụ của Bên A"
        """
        title = article.title.strip() if article.title else ""
        return f"Điều {article.number}: {title}" if title else f"Điều {article.number}"

    def _build_clause_structural_text(
        self, clause: ClauseNode, parent: ArticleNode
    ) -> str:
        """
        Tạo text ngắn để nhúng cấu trúc của Khoản.
        Ví dụ: "Điều 5 Khoản 2: Thanh toán đúng hạn..."
        """
        # Lấy 80 ký tự đầu của nội dung khoản làm "label ngắn"
        preview = clause.content.strip()[:80] if clause.content else ""
        base = f"Điều {parent.number} Khoản {clause.number}"
        return f"{base}: {preview}" if preview else base

    # ------------------------------------------------------------------
    # Private: Text builders (semantic)
    # ------------------------------------------------------------------

    def _build_article_semantic_text(
        self, article: ArticleNode, section_title: str
    ) -> str:
        """
        Tạo text đầy đủ để nhúng ngữ nghĩa của Điều.
        Format: [breadcrumb]\\nIntro text\\nPreview từng khoản...
        """
        breadcrumb = self._build_article_breadcrumb(article, section_title)
        parts: list[str] = [breadcrumb]

        if article.intro.strip():
            parts.append(article.intro.strip())

        # Preview tối đa 3 khoản đầu (150 ký tự mỗi khoản)
        for clause in article.clauses[:3]:
            preview = clause.content.strip()[:150]
            parts.append(f"{clause.number}. {preview}...")

        return "\n".join(parts)

    def _build_clause_semantic_text(
        self,
        clause: ClauseNode,
        parent: ArticleNode,
        section_title: str,
    ) -> str:
        """
        Tạo text đầy đủ để nhúng ngữ nghĩa của Khoản.
        Format: [breadcrumb]\\nNội dung khoản\\na) Điểm a\\nb) Điểm b...
        """
        breadcrumb = self._build_clause_breadcrumb(clause, parent, section_title)
        parts: list[str] = [breadcrumb, clause.content.strip()]

        for point in clause.points:
            parts.append(f"{point.label}) {point.content.strip()}")

        return "\n".join(filter(None, parts))

    # ------------------------------------------------------------------
    # Private: Breadcrumb builders
    # ------------------------------------------------------------------

    def _build_article_breadcrumb(
        self, article: ArticleNode, section_title: str
    ) -> str:
        """[Chương II. Quyền và... > Điều 5. Tên điều]"""
        parts: list[str] = []
        if section_title.strip():
            parts.append(section_title.strip())
        parts.append(article.full_title)
        return "[" + " > ".join(parts) + "]"

    def _build_clause_breadcrumb(
        self,
        clause: ClauseNode,
        parent: ArticleNode,
        section_title: str,
    ) -> str:
        """[Chương II > Điều 5 > Khoản 2]"""
        parts: list[str] = []
        if section_title.strip():
            parts.append(section_title.strip())
        parts.append(parent.full_title)
        parts.append(f"Khoản {clause.number}")
        return "[" + " > ".join(parts) + "]"

    # ------------------------------------------------------------------
    # Private: Raw text extractors
    # ------------------------------------------------------------------

    def _get_article_raw_text(self, article: ArticleNode) -> str:
        """Lấy text thô gộp của toàn bộ Điều (dùng cho payload lưu Qdrant)."""
        parts: list[str] = []
        if article.title:
            parts.append(article.title)
        if article.intro:
            parts.append(article.intro)
        for clause in article.clauses:
            parts.append(f"{clause.number}. {clause.content}")
            for point in clause.points:
                parts.append(f"  {point.label}) {point.content}")
        return "\n".join(parts)

    def _get_clause_raw_text(self, clause: ClauseNode) -> str:
        """Lấy text thô gộp của Khoản (dùng cho payload lưu Qdrant)."""
        parts: list[str] = [clause.content]
        for point in clause.points:
            parts.append(f"{point.label}) {point.content}")
        return "\n".join(filter(None, parts))

    # ------------------------------------------------------------------
    # Private: Core batch embedding
    # ------------------------------------------------------------------

    def _batch_embed_and_build(
        self,
        structural_texts: list[str],
        semantic_texts: list[str],
        payloads: list[QdrantPayload],
    ) -> list[NodeEmbeddings]:
        """
        Gọi model nhúng 2 loại texts theo batch và ghép kết quả.

        BGE-M3 trả về:
            - dense_vecs:      np.ndarray (N, 1024)
            - lexical_weights: list[dict[str, float]]  {token: weight}

        Sparse weights cần convert từ {word_token: float} → {int_index: float}
        vì Qdrant SparseVector yêu cầu integer indices.
        """
        logger.debug(
            "Embedding %d structural + %d semantic texts...",
            len(structural_texts),
            len(semantic_texts),
        )

        # --- Structural embedding ---
        struct_output = self._model.encode(
            structural_texts,
            batch_size=self.batch_size,
            max_length=512,   # structural text ngắn, không cần max_length lớn
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        struct_dense: np.ndarray = np.array(struct_output["dense_vecs"], dtype=np.float32)
        struct_sparse_raw: list[dict[str, float]] = struct_output["lexical_weights"]

        # --- Semantic embedding ---
        sem_output = self._model.encode(
            semantic_texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        sem_dense: np.ndarray = np.array(sem_output["dense_vecs"], dtype=np.float32)
        sem_sparse_raw: list[dict[str, float]] = sem_output["lexical_weights"]

        # --- Build NodeEmbeddings list ---
        results: list[NodeEmbeddings] = []
        for i, payload in enumerate(payloads):
            results.append(
                NodeEmbeddings(
                    node_id=payload.node_id,
                    structural_dense=struct_dense[i].tolist(),
                    semantic_dense=sem_dense[i].tolist(),
                    structural_sparse=self._convert_sparse(struct_sparse_raw[i]),
                    semantic_sparse=self._convert_sparse(sem_sparse_raw[i]),
                    payload=payload,
                )
            )

        logger.debug("Đã tạo %d NodeEmbeddings.", len(results))
        return results

    @staticmethod
    def _convert_sparse(lexical_weights: dict[str, float]) -> dict[int, float]:
        """
        Chuyển {word_string: weight} → {word_id_int: weight}.

        BGE-M3 lexical_weights dùng string token làm key.
        Qdrant SparseVector yêu cầu integer indices.
        Ta hash string → positive int bằng Python built-in hash + modulo.

        Note: Collision rate với 2^20 buckets (~1M) là negligible với vocab thông thường.
        """
        if not lexical_weights:
            return {}

        BUCKET_SIZE = 1 << 20  # 1,048,576
        converted: dict[int, float] = {}

        for token, weight in lexical_weights.items():
            idx = abs(hash(token)) % BUCKET_SIZE
            # Merge collisions bằng max (giữ weight cao nhất)
            if idx in converted:
                converted[idx] = max(converted[idx], weight)
            else:
                converted[idx] = weight

        return converted
