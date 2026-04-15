"""
ingestion/models.py
===================
Pydantic data models cho hệ thống đối chiếu văn bản pháp lý (Legal RAG).

Hierarchy (DOM Tree):
    LegalDocument
        └── List[DocumentSection]   (Chương / Phần)
                └── List[ArticleNode]   (Điều)
                        └── List[ClauseNode]   (Khoản)
                                └── List[PointNode]   (Điểm a, b, c...)

Chunk & Graph:
    LsuChunk       — đơn vị nhúng vector, có breadcrumb prefix
    GraphNode      — abstract base cho mọi node trong Kuzu/Neo4j
    GraphEdge      — cạnh REFERENCES giữa các node
    VectorRecord   — record lưu ChromaDB (kết nối vector ↔ graph qua node_id)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class NodeType(str, Enum):
    """Loại node trong Legal DOM tree."""
    DOCUMENT = "document"
    SECTION = "section"       # Chương / Phần / Mục
    ARTICLE = "article"       # Điều
    CLAUSE = "clause"         # Khoản
    POINT = "point"           # Điểm
    TABLE = "table"           # Bảng trích xuất từ tài liệu
    ANNEX = "annex"           # Phụ lục


class EdgeType(str, Enum):
    """Loại cạnh (relationship) trong Knowledge Graph."""
    CONTAINS = "CONTAINS"           # Quan hệ cây cha → con
    REFERENCES = "REFERENCES"       # "theo quy định tại Điều X"
    PRECEDES = "PRECEDES"           # Thứ tự tuần tự giữa các node cùng cấp
    AMENDS = "AMENDS"               # Điều này sửa đổi Điều kia (nếu cần)


class ParseEngine(str, Enum):
    """Engine đã được dùng để parse tài liệu."""
    DOCLING = "docling"
    MARKER_PDF = "marker-pdf"       # Fallback OCR


class ContentType(str, Enum):
    """Loại nội dung trong một chunk."""
    TEXT = "text"
    TABLE = "table"
    MIXED = "mixed"                 # Vừa text vừa table


# ---------------------------------------------------------------------------
# Table Representation
# ---------------------------------------------------------------------------


class TableCell(BaseModel):
    """Một ô trong bảng."""
    row: int = Field(..., ge=0, description="Chỉ số hàng (0-indexed)")
    col: int = Field(..., ge=0, description="Chỉ số cột (0-indexed)")
    row_span: int = Field(default=1, ge=1)
    col_span: int = Field(default=1, ge=1)
    content: str = Field(default="", description="Nội dung text của ô")
    is_header: bool = Field(default=False)


class TableData(BaseModel):
    """
    Bảng được trích xuất dưới dạng structured JSON.
    KHÔNG biến bảng thành text phẳng — giữ nguyên cấu trúc 2D.
    """
    table_id: str = Field(
        default_factory=lambda: f"tbl_{uuid.uuid4().hex[:8]}",
        description="ID duy nhất của bảng",
    )
    caption: str | None = Field(default=None, description="Tiêu đề bảng nếu có")
    headers: list[str] = Field(
        default_factory=list,
        description="Danh sách tên cột (row đầu tiên hoặc header row)",
    )
    rows: list[list[Any]] = Field(
        default_factory=list,
        description="Dữ liệu các hàng sau header, mỗi hàng là list[Any]",
    )
    cells: list[TableCell] = Field(
        default_factory=list,
        description="Biểu diễn chi tiết từng ô (hỗ trợ merged cells)",
    )
    num_rows: int = Field(default=0, ge=0)
    num_cols: int = Field(default=0, ge=0)
    source_page: int | None = Field(
        default=None, description="Trang trong tài liệu gốc (1-indexed)"
    )

    @model_validator(mode="after")
    def _sync_dimensions(self) -> "TableData":
        if self.rows and self.num_rows == 0:
            self.num_rows = len(self.rows)
        if self.rows and self.num_cols == 0 and self.rows[0]:
            self.num_cols = len(self.rows[0])
        return self


# ---------------------------------------------------------------------------
# Legal DOM Nodes
# ---------------------------------------------------------------------------


class PointNode(BaseModel):
    """
    Điểm (a, b, c, ...) — cấp thấp nhất trong hệ thống pháp lý VN.
    Ví dụ: 'Điều 5, Khoản 2, Điểm a'.
    """
    node_id: str = Field(
        default_factory=lambda: f"point_{uuid.uuid4().hex[:12]}",
        description="ID duy nhất, dùng làm khóa trong Graph DB",
    )
    node_type: NodeType = Field(default=NodeType.POINT, frozen=True)
    label: str = Field(..., description="Nhãn điểm: 'a', 'b', 'c', ...")
    number: str = Field(default="", description="Ký hiệu đầy đủ nếu có: 'a)', 'b)'")
    content: str = Field(default="", description="Nội dung text của điểm")
    tables: list[TableData] = Field(
        default_factory=list,
        description="Các bảng đính kèm trong điểm này",
    )
    page_number: int | None = Field(default=None, description="Số trang trong file gốc")


class ClauseNode(BaseModel):
    """
    Khoản (1., 2., 3., ...) — cấp con của Điều.
    Ví dụ: 'Điều 5, Khoản 2'.
    """
    node_id: str = Field(
        default_factory=lambda: f"clause_{uuid.uuid4().hex[:12]}",
        description="ID duy nhất, dùng làm khóa trong Graph DB",
    )
    node_type: NodeType = Field(default=NodeType.CLAUSE, frozen=True)
    number: int | str = Field(..., description="Số thứ tự khoản: 1, 2, 3 hoặc 'a'")
    content: str = Field(default="", description="Nội dung đầu khoản (không bao gồm các điểm)")
    points: list[PointNode] = Field(
        default_factory=list,
        description="Các điểm (a, b, c) thuộc khoản này",
    )
    tables: list[TableData] = Field(
        default_factory=list,
        description="Các bảng đính kèm trực tiếp trong khoản này",
    )
    page_number: int | None = Field(default=None)

    @field_validator("number", mode="before")
    @classmethod
    def _coerce_number(cls, v: Any) -> int | str:
        try:
            return int(v)
        except (ValueError, TypeError):
            return str(v)


class ArticleNode(BaseModel):
    """
    Điều — đơn vị cơ bản của văn bản pháp lý VN.
    Ví dụ: 'Điều 15. Quyền và nghĩa vụ của Bên A'.
    """
    node_id: str = Field(
        default_factory=lambda: f"article_{uuid.uuid4().hex[:12]}",
        description="ID duy nhất, dùng làm khóa trong Graph DB",
    )
    node_type: NodeType = Field(default=NodeType.ARTICLE, frozen=True)
    number: int | str = Field(..., description="Số điều: 1, 2, 15, ...")
    title: str = Field(default="", description="Tiêu đề/tên điều nếu có")
    intro: str = Field(
        default="",
        description="Đoạn mở đầu của Điều (trước Khoản 1)",
    )
    clauses: list[ClauseNode] = Field(
        default_factory=list,
        description="Danh sách các Khoản thuộc Điều này",
    )
    tables: list[TableData] = Field(
        default_factory=list,
        description="Bảng gắn trực tiếp với Điều (không thuộc khoản cụ thể)",
    )
    page_number: int | None = Field(default=None)

    @field_validator("number", mode="before")
    @classmethod
    def _coerce_number(cls, v: Any) -> int | str:
        try:
            return int(v)
        except (ValueError, TypeError):
            return str(v)

    @property
    def full_title(self) -> str:
        """Ví dụ: 'Điều 15. Quyền và nghĩa vụ của Bên A'"""
        base = f"Điều {self.number}"
        return f"{base}. {self.title}" if self.title else base


class DocumentSection(BaseModel):
    """
    Chương / Phần / Mục — nhóm các Điều lại với nhau.
    Ví dụ: 'Chương II. Quyền và Nghĩa vụ các Bên'.
    """
    node_id: str = Field(
        default_factory=lambda: f"section_{uuid.uuid4().hex[:12]}",
        description="ID duy nhất, dùng làm khóa trong Graph DB",
    )
    node_type: NodeType = Field(default=NodeType.SECTION, frozen=True)
    section_type: str = Field(
        default="Chương",
        description="Loại phần: 'Chương', 'Phần', 'Mục', 'Tiểu mục'",
    )
    number: int | str = Field(
        default="",
        description="Số thứ tự: 'I', 'II', '1', '2', ...",
    )
    title: str = Field(default="", description="Tiêu đề của chương/phần")
    articles: list[ArticleNode] = Field(
        default_factory=list,
        description="Danh sách Điều thuộc chương/phần này",
    )

    @property
    def full_title(self) -> str:
        """Ví dụ: 'Chương II. Quyền và Nghĩa vụ các Bên'"""
        base = f"{self.section_type} {self.number}".strip()
        return f"{base}. {self.title}" if self.title else base


# ---------------------------------------------------------------------------
# Top-level Document Model
# ---------------------------------------------------------------------------


class ParseQualityMetrics(BaseModel):
    """Thông tin chất lượng quá trình parse."""
    engine_used: ParseEngine = Field(
        ..., description="Engine nào đã được dùng"
    )
    avg_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence trung bình của text recognition (0–1)",
    )
    low_confidence_pages: list[int] = Field(
        default_factory=list,
        description="Danh sách trang có confidence thấp",
    )
    ocr_triggered: bool = Field(
        default=False,
        description="True nếu fallback OCR (marker-pdf) đã được kích hoạt",
    )
    total_pages: int = Field(default=0, ge=0)
    parse_duration_seconds: float = Field(default=0.0, ge=0.0)
    warnings: list[str] = Field(default_factory=list)


class LegalDocument(BaseModel):
    """
    Đỉnh của Legal DOM tree — đại diện cho một file hợp đồng/văn bản pháp lý.
    Đây là đầu ra của Module 1 (LegalDocumentParser) và
    đầu vào của Module 2 (LsuChunker).
    """
    doc_id: str = Field(
        default_factory=lambda: f"doc_{uuid.uuid4().hex[:16]}",
        description="ID toàn cục duy nhất của tài liệu",
    )
    node_type: NodeType = Field(default=NodeType.DOCUMENT, frozen=True)
    source_path: str = Field(..., description="Đường dẫn tuyệt đối file gốc")
    file_name: str = Field(..., description="Tên file (không có đường dẫn)")
    doc_title: str = Field(default="", description="Tiêu đề văn bản nếu trích được")
    doc_number: str = Field(
        default="",
        description="Số hiệu văn bản, VD: 'Hợp đồng số 2024/HĐ-ABC'",
    )
    signing_date: str | None = Field(
        default=None, description="Ngày ký/ ban hành (ISO 8601 hoặc text gốc)"
    )
    parties: list[str] = Field(
        default_factory=list,
        description="Danh sách tên các bên ký kết",
    )
    sections: list[DocumentSection] = Field(
        default_factory=list,
        description="Danh sách Chương/Phần của văn bản",
    )
    orphan_articles: list[ArticleNode] = Field(
        default_factory=list,
        description="Các Điều không thuộc Chương nào (văn bản phẳng)",
    )
    preamble: str = Field(
        default="",
        description="Phần mở đầu/căn cứ trước Điều 1",
    )
    annexes: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Phụ lục đính kèm",
    )
    quality_metrics: ParseQualityMetrics | None = Field(
        default=None, description="Metrics chất lượng parse"
    )
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Thời điểm nạp dữ liệu (UTC)",
    )

    def iter_all_articles(self) -> list[ArticleNode]:
        """Trả về tất cả ArticleNode trong document (kể cả orphan)."""
        articles: list[ArticleNode] = list(self.orphan_articles)
        for section in self.sections:
            articles.extend(section.articles)
        return articles

    def iter_all_clauses(self) -> list[tuple[ArticleNode, ClauseNode]]:
        """Trả về tất cả (article, clause) pairs."""
        pairs: list[tuple[ArticleNode, ClauseNode]] = []
        for article in self.iter_all_articles():
            for clause in article.clauses:
                pairs.append((article, clause))
        return pairs


# ---------------------------------------------------------------------------
# LSU Chunk — Đầu ra của Module 2
# ---------------------------------------------------------------------------


class LsuChunk(BaseModel):
    """
    Logical Semantic Unit Chunk — đơn vị ngữ nghĩa được nhúng vào Vector DB.

    Mỗi chunk tương ứng với một node trong Legal DOM (Điều hoặc Khoản)
    và luôn kèm theo breadcrumb prefix để cung cấp ngữ cảnh cho embedding.

    Ví dụ breadcrumb: '[Chương II > Điều 5 > Khoản 3]'
    """
    chunk_id: str = Field(
        default_factory=lambda: f"chunk_{uuid.uuid4().hex[:16]}",
        description="ID duy nhất của chunk",
    )
    doc_id: str = Field(..., description="ID tài liệu nguồn (khóa ngoại → LegalDocument.doc_id)")
    source_node_id: str = Field(
        ...,
        description="ID node gốc trong Legal DOM (article/clause node_id)",
    )
    source_node_type: NodeType = Field(
        ..., description="Loại node: ARTICLE hoặc CLAUSE"
    )
    breadcrumb: str = Field(
        ...,
        description="Chuỗi ngữ cảnh đầy đủ. VD: '[Chương II > Điều 5 > Khoản 3]'",
    )
    content_with_prefix: str = Field(
        ...,
        description="Nội dung THỰC SỰ được nhúng = breadcrumb + '\\n' + content",
    )
    raw_content: str = Field(
        ..., description="Nội dung thuần túy không có breadcrumb prefix"
    )
    content_type: ContentType = Field(
        default=ContentType.TEXT,
        description="Loại nội dung: text / table / mixed",
    )
    tables_json: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Dữ liệu bảng đính kèm (nếu chunk có table)",
    )
    # Metadata phụ trợ
    article_number: int | str | None = Field(default=None)
    clause_number: int | str | None = Field(default=None)
    section_title: str | None = Field(default=None)
    page_number: int | None = Field(default=None)
    char_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _compute_char_count(self) -> "LsuChunk":
        if self.char_count == 0:
            self.char_count = len(self.raw_content)
        return self


# ---------------------------------------------------------------------------
# Graph Models — Đầu ra của Module 3
# ---------------------------------------------------------------------------


class GraphNode(BaseModel):
    """
    Abstract base cho mọi node được lưu trong Kuzu/Neo4j.
    Có thể serialize trực tiếp thành dict để INSERT vào Graph DB.
    """

    node_id: str = Field(..., description="Primary key trong Graph DB")
    node_type: NodeType = Field(..., description="Label của node trong Graph DB")
    doc_id: str = Field(..., description="Tài liệu chứa node này")
    content_summary: str = Field(
        default="",
        description="Tóm tắt ngắn (dùng cho hiển thị, không dùng để embed)",
    )

    def to_graph_dict(self) -> dict[str, Any]:
        """Serialize thành dict phù hợp để INSERT vào Graph DB."""
        return self.model_dump(exclude_none=True)


class GraphEdge(BaseModel):
    """
    Cạnh trong Knowledge Graph.

    Ví dụ REFERENCES edge:
        source_id = 'clause_abc123'  (Điều 5, Khoản 3)
        target_id = 'article_xyz456' (Điều 10)
        edge_type  = EdgeType.REFERENCES
        context    = 'theo quy định tại Điều 10'
    """
    edge_id: str = Field(
        default_factory=lambda: f"edge_{uuid.uuid4().hex[:12]}",
        description="ID duy nhất của cạnh",
    )
    source_id: str = Field(..., description="node_id của node nguồn")
    target_id: str = Field(..., description="node_id của node đích")
    edge_type: EdgeType = Field(..., description="Loại quan hệ")
    doc_id: str = Field(..., description="Tài liệu chứa cạnh này")
    context: str = Field(
        default="",
        description="Đoạn text gốc kích hoạt việc tạo cạnh này",
    )
    weight: float = Field(
        default=1.0, ge=0.0, description="Trọng số cạnh (dùng cho ranking)"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_graph_dict(self) -> dict[str, Any]:
        """Serialize thành dict để INSERT cạnh vào Graph DB."""
        return self.model_dump(exclude_none=True)


# ---------------------------------------------------------------------------
# Vector DB Record — Đầu ra lưu ChromaDB
# ---------------------------------------------------------------------------


class VectorRecord(BaseModel):
    """
    Record được lưu vào ChromaDB.

    ⚠️  CRITICAL: field `node_id` là cầu nối duy nhất giữa
        Vector DB (ChromaDB) và Graph DB (Kuzu/Neo4j).
        KHÔNG ĐƯỢC bỏ field này khi lưu metadata.
    """
    chroma_id: str = Field(
        ...,
        description="ID dùng cho ChromaDB document (thường = chunk_id)",
    )
    embedding: list[float] = Field(
        ..., description="Vector embedding (output của embedding model)"
    )
    document_text: str = Field(
        ...,
        description="Text được embed (= LsuChunk.content_with_prefix)",
    )
    # ↓ metadata dict sẽ được flatten và lưu vào ChromaDB metadata field
    metadata: "VectorMetadata" = Field(
        ...,
        description="Metadata bắt buộc — phải có node_id để link về Graph DB",
    )

    def to_chroma_dict(self) -> dict[str, Any]:
        """
        Trả về dict sẵn sàng gọi collection.add(**record.to_chroma_dict()).
        ChromaDB không hỗ trợ nested dict trong metadata → flatten.
        """
        return {
            "ids": [self.chroma_id],
            "embeddings": [self.embedding],
            "documents": [self.document_text],
            "metadatas": [self.metadata.model_dump(exclude_none=True)],
        }


class VectorMetadata(BaseModel):
    """
    Metadata phẳng lưu trong ChromaDB.

    ⚠️  ChromaDB chỉ hỗ trợ str | int | float | bool trong metadata dict
        → tất cả các field phải là kiểu primitive.
    """
    # ↓ CRITICAL: Graph DB link
    node_id: str = Field(
        ...,
        description="[CRITICAL] ID node trong Graph DB (Kuzu/Neo4j)",
    )
    doc_id: str = Field(..., description="ID tài liệu nguồn")
    chunk_id: str = Field(..., description="ID của LsuChunk tương ứng")
    node_type: str = Field(..., description="NodeType string: 'article', 'clause', ...")
    breadcrumb: str = Field(..., description="Breadcrumb path: '[Chương II > Điều 5 > Khoản 3]'")
    file_name: str = Field(..., description="Tên file gốc")
    article_number: str = Field(
        default="",
        description="Số điều (str để tương thích ChromaDB)",
    )
    clause_number: str = Field(
        default="",
        description="Số khoản (str để tương thích ChromaDB)",
    )
    section_title: str = Field(default="")
    content_type: str = Field(default="text", description="'text' | 'table' | 'mixed'")
    has_tables: bool = Field(
        default=False, description="True nếu chunk có bảng đính kèm"
    )
    page_number: int = Field(default=0)
    char_count: int = Field(default=0)
    ingested_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Timestamp ISO 8601",
    )
