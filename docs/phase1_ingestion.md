# Phase 1 — Ingestion & Knowledge Representation (End-to-End)

> Tài liệu mô tả chi tiết **Module Ingestion** của hệ thống LegalDiff (L-RAG):
> từ file PDF/DOCX thô → **Legal DOM** → **LSU Chunks** → **Hybrid Knowledge Store**.
> Toàn bộ nội dung được đối chiếu trực tiếp với mã nguồn tại `src/ingestion/`
> và ví dụ thật từ `data_test/` + `tests/output_chunker/`.

---

## Mục lục

1. [Vai trò của Phase 1 trong pipeline](#1-vai-trò-của-phase-1-trong-pipeline)
2. [Sơ đồ luồng dữ liệu end-to-end](#2-sơ-đồ-luồng-dữ-liệu-end-to-end)
3. [Cấu trúc source code & Public API](#3-cấu-trúc-source-code--public-api)
4. [Cấu hình (config)](#4-cấu-hình-config)
5. [Module 1 — `LegalDocumentParser` (Parsing)](#5-module-1--legaldocumentparser-parsing)
6. [Data Models — Legal DOM hierarchy](#6-data-models--legal-dom-hierarchy)
7. [Module 2 — `LsuChunker` (Chunking)](#7-module-2--lsuchunker-chunking)
8. [Module 3 — `HybridGraphBuilder` (Knowledge Store)](#8-module-3--hybridgraphbuilder-knowledge-store)
9. [Orchestration: `ingest_document()` vs `LegalDiffPipeline`](#9-orchestration-ingest_document-vs-legaldiffpipeline)
10. [Production wiring — Backend Celery worker](#10-production-wiring--backend-celery-worker)
11. [Worked example end-to-end (từ dữ liệu thật)](#11-worked-example-end-to-end-từ-dữ-liệu-thật)
12. [Nguyên tắc bất biến (Design invariants)](#12-nguyên-tắc-bất-biến-design-invariants)
13. [Cách chạy & mở rộng](#13-cách-chạy--mở-rộng)

---

## 1. Vai trò của Phase 1 trong pipeline

LegalDiff là hệ thống **đối chiếu 2 phiên bản văn bản pháp lý** (V1 vs V2) thành biên bản
so sánh có trích dẫn, zero-hallucination. Pipeline gồm 3 phase:

| Phase | Tên | Module | Vai trò |
|-------|-----|--------|---------|
| **1** | **Ingestion** | `src.ingestion` | **(tài liệu này)** Parse PDF/DOCX → cấu trúc pháp lý + chunks có ngữ cảnh |
| 2 | Alignment | `src.alignment` | Ghép cặp Điều/Khoản V1↔V2 bằng BGE-M3 + Hungarian |
| 3 | Generative Comparison | `src.comparison` | LLM sinh biên bản so sánh có trích dẫn |

**Mục tiêu cụ thể của Phase 1:**

1. **Biến văn bản thô thành cấu trúc** — nhận diện `Chương → Điều → Khoản → Điểm`
   (cây phân cấp theo Luật Việt Nam) thay vì một mớ text phẳng.
2. **Trích xuất bảng giữ nguyên cấu trúc 2D** — bảng pháp lý (biểu mẫu, định mức)
   phải ra `TableData` JSON, **không bao giờ flatten thành text**.
3. **Tạo đơn vị nhúng (chunk) có ngữ cảnh** — mỗi chunk mang *breadcrumb prefix*
   (VD: `[Chương II > Điều 5 > Khoản 3]`) để embedding model hiểu chunk đang nằm
   ở đâu trong văn bản.
4. **Xây Knowledge Store** — lưu cấu trúc + quan hệ tham chiếu (`theo Điều X`)
   vào Graph DB, và embedding vào Vector DB, **liên kết qua `node_id`**.

> **Đầu ra của Phase 1 là `đầu vào` của Phase 2.** Chất lượng Alignment phụ thuộc
> hoàn toàn vào việc Phase 1 nhận diện đúng ranh giới Điều/Khoản và gắn breadcrumb
> chính xác — nếu sai ở đây, matching bên Phase 2 sẽ recall = 0.

---

## 2. Sơ đồ luồng dữ liệu end-to-end

```
                ┌──────────────────────────────────────────────┐
                │            PDF / DOCX (V1, V2)                │
                └──────────────────────┬───────────────────────┘
                                       │
 ╔═════════════════════════════════════╧═══════════════════════════════════╗
 ║                       PHASE 1 — INGESTION                              ║
 ╠═════════════════════════════════════════════════════════════════════════╣
 ║                                                                        ║
 ║  [Module 1] LegalDocumentParser  (src/ingestion/docling_parser.py)     ║
 ║     │  docling convert  →  tính confidence từng trang                  ║
 ║     │  ├─ confidence ≥ 0.75  → build DOM từ markdown docling            ║
 ║     │  └─ confidence < 0.75  → fallback marker-pdf OCR → build DOM      ║
 ║     │  State machine: Section → Article → Clause → Point               ║
 ║     │  Trích xuất TableData (grid 2D) + metadata (số hiệu, ngày ký)     ║
 ║     ▼                                                                  ║
 ║  LegalDocument  (Pydantic — Legal DOM tree)                            ║
 ║     │                                                                  ║
 ║     ▼                                                                  ║
 ║  [Module 2] LsuChunker  (src/ingestion/lsu_chunker.py)                 ║
 ║     │  Duyệt DOM → gắn breadcrumb cho từng node                        ║
 ║     │  Tạo chunk cấp Điều  (intro + preview các khoản)                 ║
 ║     │  Tạo chunk cấp Khoản (nội dung đầy đủ + các điểm a,b,c)          ║
 ║     │  Chunk quá lớn (>2000 chars) → split theo ranh giới câu + overlap║
 ║     ▼                                                                  ║
 ║  list[LsuChunk]  (có breadcrumb, source_node_id, tables_json)          ║
 ║     │                                                                  ║
 ║     ▼                                                                  ║
 ║  [Module 3] HybridGraphBuilder  (src/ingestion/knowledge_store.py)     ║
 ║     ├── Kuzu (Graph DB):                                               ║
 ║     │     upsert Document + LegalNode (Article/Clause)                 ║
 ║     │     + CONTAINS (cha→con) + PRECEDES (tuần tự)                    ║
 ║     │     + REFERENCES ("theo quy định tại Điều X")                    ║
 ║     └── ChromaDB (Vector DB):                                          ║
 ║           embed chunk.content_with_prefix → lưu VectorRecord           ║
 ║           với metadata.node_id  ← CẦU NỐI duy nhất Vector ↔ Graph     ║
 ║                                                                        ║
 ╚═════════════════════════════════════════════════════════════════════════╝
                                       │
                                       ▼
                         Phase 2 — Alignment (Qdrant)
```

---

## 3. Cấu trúc source code & Public API

```
src/ingestion/
├── __init__.py            # Public API + ingest_document() runner (3 modules)
├── models.py              # Pydantic data models (Legal DOM + Chunk + Graph + Vector)
├── docling_parser.py      # Module 1: LegalDocumentParser
├── lsu_chunker.py         # Module 2: LsuChunker
└── knowledge_store.py     # Module 3: HybridGraphBuilder (Kuzu + ChromaDB)

src/config.py              # Loader YAML + hằng số backward-compatible
src/pipeline.py            # Orchestrator 3-phase (LegalDiffPipeline)
configs/
├── pipeline_config.yaml   # max_chunk_chars, thresholds, paths
└── model_config.yaml      # confidence_threshold, embedding_dim, LLM config
```

**Public API** (từ `src/ingestion/__init__.py`):

```python
from src.ingestion import (
    # 3 module chính
    LegalDocumentParser,     # Module 1
    LsuChunker,              # Module 2
    HybridGraphBuilder,      # Module 3
    ingest_document,         # Runner tích hợp cả 3 module
    # Data models
    LegalDocument, DocumentSection, ArticleNode, ClauseNode, PointNode,
    TableData, TableCell, LsuChunk,
    GraphNode, GraphEdge, VectorRecord, VectorMetadata, ParseQualityMetrics,
    # Enums
    NodeType, EdgeType, ParseEngine, ContentType,
)
```

---

## 4. Cấu hình (config)

Phase 1 đọc config từ `configs/*.yaml` qua `src/config.py` → `get_config()`.
Các tham số liên quan tới Ingestion:

| Tham số | Mặc định | Ý nghĩa | Nguồn |
|---------|----------|---------|-------|
| `ingestion.confidence_threshold` | `0.75` | Avg confidence < ngưỡng → trigger OCR fallback (docling → marker-pdf) | `model_config.yaml` |
| `ingestion.max_chunk_chars` | `2000` | Số ký tự tối đa/chunk; vượt quá → split theo câu | `pipeline_config.yaml` |
| `ingestion.overlap_chars` | `200` | Ký tự overlap khi split sub-chunk | `pipeline_config.yaml` |
| `ingestion.kuzu_db_path` | `./data/processed/graph_db` | Đường dẫn Kuzu DB (embedded) | `pipeline_config.yaml` |
| `ingestion.chroma_db_path` | `./data/processed/chroma_db` | Đường dẫn ChromaDB persistent | `pipeline_config.yaml` |
| `ingestion.chroma_collection_name` | `legal_documents` | Tên collection ChromaDB | `pipeline_config.yaml` |
| `ingestion.embedding_dim` | `1024` | Chiều vector BGE-M3 (dùng cho placeholder) | `model_config.yaml` |

Truy cập nhanh (backward-compatible constants trong `src/config.py`):

```python
from src.config import (
    CONFIDENCE_THRESHOLD,   # 0.75
    MAX_CHUNK_CHARS,        # 2000
    OVERLAP_CHARS,          # 200
    KUZU_DB_PATH,
    CHROMA_DB_PATH,
    EMBEDDING_DIM,          # 1024
)
```

---

## 5. Module 1 — `LegalDocumentParser` (Parsing)

> **File:** `src/ingestion/docling_parser.py`
> **Vào:** `file_path` (PDF/DOCX) → **Ra:** `LegalDocument` (Legal DOM tree)

### 5.1. Chiến lược 2 tầng: docling + marker-pdf OCR fallback

```python
parser = LegalDocumentParser(
    confidence_threshold=0.75,   # trigger OCR nếu avg conf < 0.75
    low_conf_page_ratio=0.30,    # hoặc nếu >30% trang là low-conf
    marker_timeout=300,          # timeout OCR (giây)
)
doc: LegalDocument = parser.parse("data_test/02-tt-tand.signed_v1.docx")
```

Luồng `parse()`:

1. **Thử docling trước** (`_parse_with_docling`) — `DocumentConverter` với
   `do_ocr=False` (dùng text layer), `do_table_structure=True` (bắt buộc để lấy bảng).
2. **Tính confidence** (`_compute_docling_confidence`) — duyệt `docling_doc.texts[*].prov`,
   lấy `prov.confidence` & `prov.page_no`; trang nào có conf < 0.5 được đánh dấu low-conf.
3. **Quyết định OCR** (`_should_trigger_ocr`):
   - `avg_confidence < confidence_threshold`  **HOẶC**
   - `len(low_conf_pages) / total_pages > low_conf_page_ratio`
   - → chuyển sang marker-pdf.
4. **Fallback marker-pdf** (`_parse_with_marker`) — ưu tiên Python API
   (`marker.convert.convert_single_pdf` với `langs=["vi","en"]`), nếu không có
   thì gọi CLI `marker_single` qua subprocess; kết quả là Markdown string.
5. Nếu docling **throw exception** → cũng fallback marker-pdf và ghi warning.

Kết quả chất lượng được gom vào `ParseQualityMetrics`:

```python
class ParseQualityMetrics(BaseModel):
    engine_used: ParseEngine            # DOCLING | MARKER_PDF
    avg_confidence: float | None        # None nếu marker (không cung cấp)
    low_confidence_pages: list[int]
    ocr_triggered: bool
    total_pages: int
    parse_duration_seconds: float
    warnings: list[str]
```

### 5.2. DOM Builder — state machine 4 cấp

Hai engine đều hội tụ về một builder chung `_build_dom_from_lines(lines, path, tables_by_page)`.
Đây là **state machine duyệt từng dòng Markdown**:

```
current_section → current_article → current_clause → current_point
```

Mỗi khi gặp pattern ở **cấp cao hơn**, buffer của cấp thấp hơn được *flush* (đẩy
lên cha). Các pattern nhận diện bằng regex (`re.UNICODE` cho tiếng Việt):

| Cấp | Regex | Ví dụ bắt được |
|-----|-------|----------------|
| Section | `_RE_SECTION` | `Chương II`, `PHẦN 1`, `Mục 3`, `CHƯƠNG I` |
| Article | `_RE_ARTICLE` | `Điều 1.`, `Điều 15:`, `Điều IV` (chấp nhận La Mã) |
| Clause | `_RE_CLAUSE` | `1. nội dung...`, `2. nội dung...` (1–2 chữ số) |
| Point | `_RE_POINT` | `a) ...`, `b) ...` (1 chữ cái thường, có `đ`) |

**Xử lý artifact của docling/marker** trước khi match:
- Bỏ `#`, `##` (markdown heading).
- Bỏ `**bold**` / `*italic*` → giữ text bên trong (VD: `**Điều 5: Tên**` → `Điều 5: Tên`).
- Bỏ separator `---`, `─` (≥3 ký tự).

**Continuation text:** dòng không match pattern nào sẽ được **nối tiếp** vào node
đang mở ở cấp thấp nhất (`point.content += ...`, rồi `clause.content`, rồi
`article.intro`). Dòng trước khi gặp Section/Article đầu tiên → `preamble`.

### 5.3. Numbered fallback — cho văn bản KHÔNG dùng "Điều"

Một số văn bản (Hướng dẫn, Công điện, Quyết định…) đánh số trực tiếp `1. 2. 3.`
thay vì `Điều X`. Nếu primary pass tạo ra **0 article**, parser tự chạy
`_build_dom_numbered_fallback`:

- `I. II. III.` (La Mã) → `DocumentSection`
- `1. 2. 3.` → pseudo `ArticleNode` (nội dung vào `intro`)
- `a) b) c)` → `PointNode` (gom vào 1 clause mặc định `number="1"`)

> Mục đích: **tránh recall = 0** cho toàn bộ cặp đối chiếu khi văn bản không
> có cấu trúc "Điều". Metadata (doc_id, preamble, số hiệu, ngày ký) được giữ
> nguyên từ primary pass.

### 5.4. Trích xuất bảng — `TableData` 2D (KHÔNG flatten)

Đây là **bất biến quan trọng**: bảng pháp lý phải giữ cấu trúc 2D.

- **Từ docling** (`_extract_tables_from_docling`): duyệt `table_item.data.grid`
  (lưới `list[list[GridCell]]`), mỗi cell → `TableCell(row, col, row_span,
  col_span, content, is_header)`. Cell `column_header`/`row_header` ở dòng 0
  → tách thành `headers`; phần còn lại thành `rows`.
- **Từ marker** (`_parse_markdown_tables`): parse pipe-tables Markdown
  (`| col1 | col2 |` / `|---|---|`), rồi `_inject_markdown_tables` phân phối
  vào các article theo thứ tự xuất hiện.
- **Gắn bảng vào node** (`_attach_tables_to_nodes`): heuristic theo `page_number`
  — bảng ở trang X gắn vào Article có `page_number ≤ X` gần nhất.

### 5.5. Trích xuất metadata tài liệu

`_extract_doc_metadata` quét `preamble` bằng regex:

- **Số hiệu văn bản:** `Số: 02/2024/TT-TANDTC` → `doc.doc_number`
- **Ngày ký:** `ngày 31 tháng 12 năm 2024` → `doc.signing_date = "2024-12-31"` (ISO)

---

## 6. Data Models — Legal DOM hierarchy

> **File:** `src/ingestion/models.py` (Pydantic v2)

```
LegalDocument                         # đỉnh cây (1 file = 1 document)
├── preamble: str                     # phần căn cứ trước Điều 1
├── doc_number, signing_date, parties # metadata
├── sections: list[DocumentSection]   # Chương / Phần
│   └── articles: list[ArticleNode]   # Điều
│       ├── intro: str                # đoạn mở đầu Điều
│       ├── clauses: list[ClauseNode] # Khoản
│       │   ├── content: str
│       │   └── points: list[PointNode]  # Điểm a, b, c
│       └── tables: list[TableData]
├── orphan_articles: list[ArticleNode]# Điều không thuộc Chương (văn bản phẳng)
├── annexes: list[dict]
└── quality_metrics: ParseQualityMetrics
```

Đặc điểm thiết kế:

- **Mỗi node có `node_id` duy nhất** (UUID hex) — đây là **khóa chính trong Graph DB**
  và là **cầu nối tới Vector DB**. VD: `article_8c73219b71e1`, `clause_752945f45fd1`.
- **`node_type` frozen** — không thể đổi sau khi tạo (đảm bảo nhất quán type).
- **`number` của Article/Clause** có validator `_coerce_number`: cố gắng `int()`,
  fallback `str` (chấp nhận La Mã hoặc `a`, `b`).
- **`ArticleNode.full_title`** → `"Điều 15. Quyền và nghĩa vụ của Bên A"`.
- **`TableData`** tự sync `num_rows`/`num_cols` từ `rows` qua `model_validator`.

Các helper duyệt cây (trên `LegalDocument`):

```python
doc.iter_all_articles()           # list[ArticleNode] (kể cả orphan)
doc.iter_all_clauses()            # list[(ArticleNode, ClauseNode)]
```

**Enums** (`NodeType`, `EdgeType`, `ParseEngine`, `ContentType`):

```python
class NodeType(str, Enum):    # DOCUMENT | SECTION | ARTICLE | CLAUSE | POINT | TABLE | ANNEX
class EdgeType(str, Enum):    # CONTAINS | REFERENCES | PRECEDES | AMENDS
class ParseEngine(str, Enum): # DOCLING | MARKER_PDF
class ContentType(str, Enum): # TEXT | TABLE | MIXED
```

---

## 7. Module 2 — `LsuChunker` (Chunking)

> **File:** `src/ingestion/lsu_chunker.py`
> **Vào:** `LegalDocument` → **Ra:** `list[LsuChunk]`

### 7.1. Vì sao gọi là "LSU" (Logical Semantic Unit)?

Chunker **không** split theo số ký tự cứng. Nó tôn trọng **ranh giới logic**
của văn bản pháp lý (Điều / Khoản), rồi mới split theo câu nếu quá lớn. Mỗi chunk
đại diện cho **một đơn vị ngữ nghĩa hoàn chỉnh** + ngữ cảnh (`breadcrumb`).

### 7.2. Breadcrumb Builder — tiền tố ngữ cảnh

`_BreadcrumbBuilder` tạo chuỗi path từ gốc tới node hiện tại:

```
[Chương I > Điều 1. Phạm vi điều chỉnh]                              ← article
[Chương I > Điều 1. Phạm vi điều chỉnh > Khoản 1]                   ← clause
[Chương I > Điều 1. Phạm vi điều chỉnh > Khoản 1 > Điểm a]          ← point
```

Breadcrumb được **nối vào đầu nội dung** trước khi embed:
`content_with_prefix = f"{breadcrumb}\n{raw_content}"`.
→ Embedding model biết chunk này nói về "Điều 1, Khoản 1" ngay cả khi nội dung
ngắn (VD: chỉ `"a) Xây dựng văn bản..."`).

### 7.3. Chiến lược tạo chunk 2 tầng

```python
chunker = LsuChunker(
    max_chunk_chars=2000,
    overlap_chars=200,
    create_article_level_chunks=True,   # chunk cấp Điều
    create_clause_level_chunks=True,    # chunk cấp Khoản
)
chunks = chunker.chunk(document)
```

Duyệt `document.sections[*].articles[*]` (+ `orphan_articles`), với mỗi Article:

**a) Chunk cấp Điều** (`_make_article_chunk`):
- Nội dung = `article.intro` + preview mỗi khoản (`Khoản N: <150 ký tự đầu>...`).
- Hữu ích cho truy vấn kiểu *"Điều X nói về gì?"*.

**b) Chunk cấp Khoản** (`_process_clause`):
- Nội dung = `_flatten_clause_content` = header khoản + tất cả điểm con
  (`a) ...`, `b) ...`).
- Hữu ích cho truy vấn chi tiết.

### 7.4. Sentence Splitter — khi chunk vượt `max_chunk_chars`

`_SentenceSplitter.split(text)`:
1. Nếu `len(text) ≤ max_chars` → trả nguyên `[text]`.
2. Split theo ranh giới câu (`_RE_SENTENCE_BOUNDARY` — hỗ trợ `. ! ? ;` và chữ
   hoa tiếng Việt có dấu).
3. Gom câu vào chunk hiện tại cho tới khi gần vượt `max_chars`; chunk mới bắt đầu
   bằng **overlap** = `overlap_chars` ký tự cuối của chunk trước.
4. **Hard split** (không có dấu câu) → cắt thuần theo ký tự + overlap.

Sub-chunk mang breadcrumb mở rộng: `... Khoản 3 [phần 1/2]`, `... [phần 2/2]`.
Bảng chỉ đính kèm vào **sub-chunk đầu tiên** (tránh trùng lặp).

### 7.5. Content type detection

`_detect_content_type(text, tables)`:

| Có text? | Có table? | `content_type` |
|----------|-----------|----------------|
| ✓ | ✓ | `MIXED` |
| ✗ | ✓ | `TABLE` |
| ✓ | ✗ | `TEXT` |

### 7.6. `LsuChunk` — đầu ra của Module 2

```python
class LsuChunk(BaseModel):
    chunk_id: str                       # UUID — cũng là chroma_id
    doc_id: str                         # FK → LegalDocument.doc_id
    source_node_id: str                 # FK → ArticleNode/ClauseNode.node_id
    source_node_type: NodeType          # ARTICLE | CLAUSE
    breadcrumb: str                     # '[Chương I > Điều 3 > Khoản 2]'
    content_with_prefix: str            # ← THỰC SỰ được embed
    raw_content: str                    # nội dung thuần (không prefix)
    content_type: ContentType           # text | table | mixed
    tables_json: list[dict]             # bảng đính kèm (serialize)
    article_number, clause_number       # metadata định vị
    section_title, page_number, char_count
```

Thống kê qua `chunker.get_stats(chunks)` (ví dụ thật từ `01-tand_signed`):

```json
{
  "total": 53, "article_level": 14, "clause_level": 39,
  "with_tables": 1, "avg_chars": 398.4, "max_chars": 1923, "min_chars": 44,
  "content_types": {"text": 52, "table": 0, "mixed": 1}
}
```

---

## 8. Module 3 — `HybridGraphBuilder` (Knowledge Store)

> **File:** `src/ingestion/knowledge_store.py`
> **Vào:** `(LegalDocument, list[LsuChunk])` → **Ra:** `{'nodes', 'edges', 'vectors'}`
> **2 store:** Kuzu (Graph DB, embedded) + ChromaDB (Vector DB, persistent)

### 8.1. Kuzu schema

`_KuzuSchemaManager.initialize()` tạo các table (Cypher DDL):

**Node tables:**
- `Document(doc_id PK, file_name, doc_title, doc_number, signing_date, ingested_at)`
- `LegalNode(node_id PK, node_type, doc_id, article_number, clause_number,
  section_title, content_summary, page_number, char_count, breadcrumb)`

**Edge (REL) tables:**
- `CONTAINS` — `Document→LegalNode`, `LegalNode→LegalNode` (quan hệ cha→con)
- `REFERENCES` — `LegalNode→LegalNode` (+ `context STRING`, `weight DOUBLE`)
- `PRECEDES` — `LegalNode→LegalNode` (thứ tự tuần tự cùng cấp)

> Kuzu chạy **embedded** (không cần server). Lưu ý trong `_init_kuzu`: chỉ tạo
> thư mục **cha**; nếu thư mục DB đã tồn tại và rỗng (do run trước bị interrupt)
> thì xóa đi trước khi `kuzu.Database()` tạo mới — tránh lỗi
> *"Database path cannot be a directory"*.

### 8.2. Upsert nodes + structure edges (`_build_graph`)

Duyệt DOM và `MERGE` (idempotent — chạy lại không trùng lặp):

```
Document ──CONTAINS──▶ Article ──CONTAINS──▶ Clause
                         │ PRECEDES              │ PRECEDES
                         ▼                       ▼
                   Article[i] ──PRECEDES──▶ Article[i+1]
                   Clause[j]  ──PRECEDES──▶ Clause[j+1]
```

`MERGE (n:LegalNode {node_id: $node_id}) SET n.node_type = ...` — upsert theo
`node_id`. `content_summary` = 300 ký tự đầu (dùng hiển thị, **không** embed).

### 8.3. REFERENCES edges — phát hiện tham chiếu nội bộ (`_build_reference_edges`)

Scan nội dung từng Article (`intro + title`) và Clause (`content + points`)
để tìm tham chiếu chéo, qua 2 pattern:

- `_RE_ARTICLE_REF`: `"theo quy định tại Điều 5"`, `"tại Điều 15 Khoản 2"`.
- `_RE_CLAUSE_FIRST_REF`: `"Khoản 3 Điều 10"` (thứ tự ngược).

**Giải quyết đích** (`_resolve_reference_target`): ưu tiên clause-level >
article-level, tra qua `_ArticleIndex` (map `số điều/khoản → node_id`).
Lấy ±50 ký tự quanh match làm `context`. Tự bỏ tham chiếu tự-reference và
khử trùng (`seen_refs`).

→ Tạo `REFERENCES` edge: `src -[:REFERENCES {context}]-> dst`.

### 8.4. Vector DB — ChromaDB (`_build_vector_store`)

1. Embed theo **batch** (`embedding_batch_size=32`) để tránh OOM.
   Text được embed = `chunk.content_with_prefix` (đã có breadcrumb).
2. Mỗi chunk → `VectorRecord`:
   ```python
   VectorRecord(
       chroma_id   = chunk.chunk_id,
       embedding   = <list[float]>,           # từ embedding_fn
       document_text = chunk.content_with_prefix,
       metadata    = VectorMetadata(
           node_id = chunk.source_node_id,     # ⚠️ CẦU NỐI Vector ↔ Graph
           doc_id, chunk_id, node_type, breadcrumb,
           article_number, clause_number, section_title,
           content_type, has_tables, page_number, char_count, ingested_at,
       ),
   )
   ```
3. `collection.upsert(ids, embeddings, documents, metadatas)` — idempotent.
   Collection dùng `hnsw:space = "cosine"`.

> ⚠️ **CRITICAL:** `metadata.node_id` là cầu nối **duy nhất** giữa Vector DB và
> Graph DB. Khi semantic search trả về 1 vector, ta đọc `metadata.node_id` rồi
> query Kuzu để lấy cấu trúc/quan hệ xung quanh. **Không được bỏ field này.**
>
> Lưu ý nhỏ: `_build_vector_records` hiện set `file_name=""` (chưa điền từ
> document context) — field phụ, không ảnh hưởng tới link `node_id`.

### 8.5. Embedding function — placeholder

`placeholder_embedding_fn` trả về **zero vector** `dim=1024` (từ `EMBEDDING_DIM`).
Đây là chỗ để cắm model thực (BGE-M3 / sentence-transformers):

```python
def real_embed(texts: list[str]) -> list[list[float]]:
    return model.encode(texts, normalize_embeddings=True).tolist()
```

> Lưu ý: trong Phase 2, embedding BGE-M3 thực sự được `BGEM3Manager`
> (`src/alignment/embedder.py`) nạp và dùng để index vào **Qdrant** — xem
> [mục 9](#9-orchestration-ingest_document-vs-legaldiffpipeline) về sự khác biệt
> giữa 2 đường dẫn.

### 8.6. Query helpers (bonus)

- `query_similar_chunks(query_text, n_results, where_filter)` — semantic search
  ChromaDB, trả về `{chunk_id, node_id, breadcrumb, document, distance}`.
- `get_node_references(node_id)` — lấy tất cả REFERENCES edges từ 1 node (Cypher).
- `get_graph_stats()` — đếm node/edge theo loại + tổng vectors.

---

## 9. Orchestration: `ingest_document()` vs `LegalDiffPipeline`

Phase 1 có **2 cách gọi**, phục vụ mục đích khác nhau — cần phân biệt rõ:

### 9.1. `ingest_document()` — runner 3-module đầy đủ (Kuzu + ChromaDB)

Hàm tiện ích trong `src/ingestion/__init__.py`, chạy đủ Module 1 → 2 → 3:

```python
from src.ingestion import ingest_document

result = ingest_document(
    file_path="data_test/02-tt-tand.signed_v1.docx",
    kuzu_db_path="./data/processed/graph_db",
    chroma_db_path="./data/processed/chroma_db",
    embedding_fn=None,                # None → placeholder zero vectors
    confidence_threshold=0.75,
    max_chunk_chars=2000,
)
# → {'doc_id', 'file_name', 'articles', 'chunks', 'nodes', 'edges', 'vectors', 'engine'}
```

→ Phù hợp để **xây Hybrid Knowledge Graph độc lập** (RAG pháp lý, tra cứu
tham chiếu chéo). Đây là đường dẫn được mô tả đầy đủ ở [Module 3](#8-module-3--hybridgraphbuilder-knowledge-store).

### 9.2. `LegalDiffPipeline` — orchestrator 3-phase production

`src/pipeline.py` là pipeline đối chiếu V1↔V2 dùng trong production. Trong
`_run_async()`, Phase 1 **chỉ gọi Module 1 + Module 2**:

```python
parser  = LegalDocumentParser(confidence_threshold=cfg.confidence_threshold)
chunker = LsuChunker(max_chunk_chars=cfg.max_chunk_chars)

doc_v1, doc_v2         = parser.parse(cfg.file_v1), parser.parse(cfg.file_v2)
chunks_v1, chunks_v2   = chunker.chunk(doc_v1),     chunker.chunk(doc_v2)
# → truyền doc + chunks THẲNG vào Phase 2 (bỏ qua Module 3 / Kuzu+ChromaDB)
```

Tức là: trong luồng đối chiếu chính, `HybridGraphBuilder` (Kuzu + ChromaDB)
**không được gọi** — Phase 2 tự xây index riêng trên **Qdrant**
(`QdrantManager` + `BGEM3Manager`) từ `doc` và `chunks`. Module 3 tồn tại như
một thành phần của ingestion package dùng cho use-case Knowledge Graph riêng.

> **Tóm tắt architectural:**
> - Mục tiêu *đối chiếu V1/V2* → dùng `LegalDiffPipeline`
>   (Phase 1 = Module 1+2, vector store = Qdrant ở Phase 2).
> - Mục tiêu *Knowledge Graph pháp lý để tra cứu* → dùng `ingest_document()`
>   (đủ 3 module, Kuzu + ChromaDB).

---

## 10. Production wiring — Backend Celery worker

Trong backend, Phase 1 chạy bên trong Celery task `run_pipeline`
(`backend/workers/pipeline_task.py`), có progress reporting qua Redis pub/sub
cho WebSocket:

```
[Upload V1, V2] → API (/jobs) → Celery: run_pipeline(job_id, v1, v2, overrides)
                      │
                      ├─ GPU lock (chỉ 1 job/GPU)
                      ├─ Phase 1 (0–30%):  "Đang phân tích tài liệu..."
                      │     pipeline = LegalDiffPipeline.from_config()
                      │     pipeline.run(file_v1, file_v2)   ← gọi Module 1+2
                      ├─ Phase 2 (30–55%): "Đang nhúng vector BGE-M3..."
                      ├─ Phase 3 (55–95%): "Đang sinh báo cáo so sánh..."
                      └─ Lưu ComparisonReport vào DB (95–100%)
```

`progress_callback` được truyền vào `PipelineRunConfig` để mỗi phase phát
`{event, job_id, progress_pct, current_phase, message}` lên Redis channel
`job:{job_id}:progress` → frontend stepper cập nhật real-time.

---

## 11. Worked example end-to-end (từ dữ liệu thật)

Corpus test: `data_test/` chứa các cặp văn bản pháp lý V1/V2 (Thông tư TANDTC,
Bộ Y Tế, Bộ GD-ĐT, Nghị quyết...) cùng `*.json` golden diff. Bên dưới dùng
`02-tt-tand.signed_v1.docx` (Thông tư 02/2024/TT-TANDTC).

### Bước 1 — Input thô (trích từ `data_test/02-tt-tand.signed.json`)

Văn bản dạng bảng header + preamble + `Chương I` + các `Điều`. Golden dataset
cho thấy các loại thay đổi Phase 1 phải giữ được cấu trúc để Phase 3 nhận diện:

```json
{
  "change_id": 6, "category": "D", "location": "Điều 3, Khoản 2",
  "change_type": "logic_shift", "severity": "high",
  "original_text": "...không sử dụng cho mục đích khác.",
  "modified_text": "...; có thể được sử dụng cho các mục đích khác nếu được Chánh án... phê duyệt..."
}
```
→ Phase 1 phải tách đúng **Điều 3, Khoản 2** thành chunk riêng để Phase 2 match.

### Bước 2 — Module 1 output: `LegalDocument`

(`tests/output_chunker/parsed_document.json`, đã rút gọn)

```json
{
  "doc_id": "doc_02-tt-tand.signed_v1_1777017495",
  "file_name": "02-tt-tand.signed_v1.docx",
  "doc_number": "02/2024/TT-TANDTC",
  "signing_date": "2024-12-31",
  "preamble": "TÒA ÁN NHÂN DÂN TỐI CAO\nSố: 02/2024/TT-TANDTC\n...",
  "sections": [{
    "section_type": "Chương", "number": "I", "title": ": QUY ĐỊNH CHUNG",
    "articles": [
      { "node_id": "article_8c73219b71e1", "number": 1,
        "title": "Phạm vi điều chỉnh",
        "intro": "Thông tư này quy định về tổ chức bộ máy..." },
      { "node_id": "article_7b3b652af5fb", "number": 2,
        "title": "Đối tượng áp dụng", "intro": "..." },
      { "node_id": "article_19697b98f465", "number": 3,
        "title": "Vụ Pháp chế và Quản lý khoa học",
        "clauses": [{
          "node_id": "clause_752945f45fd1", "number": 3,
          "content": "...thực hiện nhiệm vụ, quyền hạn sau đây:",
          "points": [
            { "label": "a", "number": "a)", "content": "Xây dựng văn bản quy phạm pháp luật..." },
            { "label": "b", "number": "b)", "content": "Phối hợp với Chính phủ..." },
            { "label": "c", "number": "c)", "content": "Phát triển án lệ..." },
            { "label": "d", "number": "d)", "content": "Lập đề nghị xây dựng luật..." }
          ]
        }]
      }
    ]
  }],
  "quality_metrics": {
    "engine_used": "docling", "avg_confidence": null,
    "ocr_triggered": false, "parse_duration_seconds": 2.70
  }
}
```

### Bước 3 — Module 2 output: `list[LsuChunk]`

(`tests/output_chunker/chunks_output.json`) — cho Điều 1, 2, 3 tạo **4 chunk**
(article-level + clause-level):

| chunk (rút gọn) | source_node_type | breadcrumb | char_count |
|-----------------|------------------|------------|------------|
| Điều 1 (intro) | article | `[Chương I > Điều 1. Phạm vi điều chỉnh]` | 138 |
| Điều 2 (intro) | article | `[Chương I > Điều 2. Đối tượng áp dụng]` | 116 |
| Điều 3 (preview khoản) | article | `[Chương I > Điều 3. Vụ Pháp chế...]` | 79 |
| Điều 3, Khoản 3 (đầy đủ) | clause | `...Điều 3 > Khoản 3]` | 533 |

Ví dụ **chunk cấp Khoản** — chú ý `content_with_prefix` có breadcrumb prefix,
đây chính là text được embed:

```json
{
  "chunk_id": "chunk_04b7b9529c794967",
  "source_node_id": "clause_752945f45fd1",
  "source_node_type": "clause",
  "breadcrumb": "[Chương I. : QUY ĐỊNH CHUNG > Điều 3. Vụ Pháp chế và Quản lý khoa học > Khoản 3]",
  "content_with_prefix": "[Chương I. : QUY ĐỊNH CHUNG > Điều 3. Vụ Pháp chế và Quản lý khoa học > Khoản 3]\nVụ Pháp chế và Quản lý khoa học thực hiện nhiệm vụ, quyền hạn sau đây:\na) Xây dựng văn bản quy phạm pháp luật...\nb) Phối hợp với Chính phủ...\nc) Phát triển án lệ...\nd) Lập đề nghị xây dựng luật...",
  "raw_content": "Vụ Pháp chế và Quản lý khoa học thực hiện nhiệm vụ...\nd) Lập đề nghị xây dựng luật...",
  "article_number": 3, "clause_number": 3, "char_count": 533
}
```

### Bước 4 — Module 3 output (đường `ingest_document`)

Từ DOM + chunks ở trên, `HybridGraphBuilder` tạo ra:

- **Nodes (Kuzu):** 1 `Document` + 3 `LegalNode` article + 1 `LegalNode` clause = **5 nodes**.
- **Edges (Kuzu):**
  - `CONTAINS`: Document→Điều1, Document→Điều2, Document→Điều3, Điều3→Khoản3 (=4).
  - `PRECEDES`: Điều1→Điều2, Điều2→Điều3 (=2).
  - `REFERENCES`: nếu clause/điều nào nhắc "theo Điều X" (ví dụ này không có → 0).
- **Vectors (ChromaDB):** 4 `VectorRecord` (theo 4 chunk), mỗi record
  `metadata.node_id` trỏ về đúng `article_*` / `clause_*` node.

```
Document ──CONTAINS──▶ Điều1 ──PRECEDES──▶ Điều2 ──PRECEDES──▶ Điều3
                                                                   │
                                                              CONTAINS
                                                                   ▼
                                                               Khoản3 ──embed──▶ ChromaDB vector
                                                                  (metadata.node_id = clause_...)
```

### Bước 5 — Bảng cũng được giữ nguyên (ví dụ Mixed chunk)

Trong test khác (`tests/lsu_chunker_detailed_output/04_...`), Điều 1 của văn bản
`01-tand_signed` có header dạng bảng 2 cột → Phase 1 trích xuất thành
`tables_json` (5 `TableData`, mỗi cái là grid 1×2 với `is_header=true`) và
đánh `content_type = "mixed"`. Bảng **không bị flatten** vào text:

```json
{
  "content_type": "mixed",
  "tables_json": [
    { "table_id": "tbl_4e41f5aa", "num_rows": 1, "num_cols": 2,
      "cells": [
        { "row": 0, "col": 0, "content": "TÒA ÁN NHÂN DÂN TỐI CAO", "is_header": true },
        { "row": 0, "col": 1, "content": "CỘNG HÒA XÃ HỘI...", "is_header": true }
      ] }
    /* ... 4 bảng khác ... */
  ]
}
```

---

## 12. Nguyên tắc bất biến (Design invariants)

Khi sửa code Phase 1, **phải** giữ các bất biến sau (nếu vi phạm → Phase 2/3 hỏng):

1. **Bảng không bao giờ flatten** — luôn ra `TableData` (headers + rows + cells 2D).
2. **Mỗi chunk LUÔN có breadcrumb prefix** trong `content_with_prefix` (text được embed).
3. **`metadata.node_id` luôn có** — đây là cầu nối duy nhất Vector DB ↔ Graph DB.
   Không được bỏ field này khi lưu ChromaDB.
4. **`node_id` là UUID duy nhất** — làm primary key Graph DB và FK trong chunk.
5. **Operations idempotent** — `MERGE` (Kuzu) + `upsert` (ChromaDB) để chạy lại không trùng.
6. **Parser không ghi disk** — chỉ parse và trả về model; việc persist do Module 3 lo.
7. **Confidence check trên từng trang** — OCR fallback chỉ kích hoạt khi thực sự cần
   (tránh chạy marker-pdf tốn GPU không cần thiết).
8. **Numbered fallback** — văn bản không có "Điều" vẫn phải ra pseudo-articles
   (tránh recall = 0 cho toàn bộ cặp).

---

## 13. Cách chạy & mở rộng

### Chạy nhanh (Knowledge Graph độc lập)

```python
from src.ingestion import ingest_document

result = ingest_document("data_test/02-tt-tand.signed_v1.docx")
print(result)
# {'doc_id': ..., 'articles': 3, 'chunks': 4, 'nodes': 5, 'edges': 6, 'vectors': 4, 'engine': 'docling'}
```

### Chạy từng module (debug)

```python
from src.ingestion import LegalDocumentParser, LsuChunker, HybridGraphBuilder

doc     = LegalDocumentParser().parse("data_test/02-tt-tand.signed_v1.docx")
chunks  = LsuChunker().chunk(doc)
builder = HybridGraphBuilder("./data/processed/graph_db", "./data/processed/chroma_db")
stats   = builder.build(doc, chunks)
builder.close()
```

### Chạy trong pipeline đối chiếu đầy đủ

```python
from src.pipeline import LegalDiffPipeline
pipeline = LegalDiffPipeline.from_config(provider="local")  # hoặc "deepseek"
result = pipeline.run(file_v1="data_test/..._v1.docx",
                      file_v2="data_test/..._v2.docx")
```

### Cắm embedding model thực

Thay `placeholder_embedding_fn` bằng model thật và truyền vào `HybridGraphBuilder`:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3")
def embed(texts): return model.encode(texts, normalize_embeddings=True).tolist()

builder = HybridGraphBuilder(..., embedding_fn=embed)
```

### Test artifacts tham khảo

| File | Nội dung |
|------|----------|
| `tests/output_chunker/parsed_document.json` | Output Module 1 (LegalDocument) thật |
| `tests/output_chunker/chunks_output.json` | Output Module 2 (list[LsuChunk]) thật |
| `tests/lsu_chunker_detailed_output/01_breadcrumb_builder_test.json` | Ví dụ breadcrumb 3 cấp |
| `tests/lsu_chunker_detailed_output/02_text_utilities_test.json` | Flatten/serialize/detect content type |
| `tests/lsu_chunker_detailed_output/03_sentence_splitter_test.json` | Split theo câu + hard split |
| `tests/lsu_chunker_detailed_output/04_lsu_chunker_main_test.json` | Stats + sample chunks (có bảng mixed) |
| `data_test/*.docx` + `*.json` | Corpus cặp V1/V2 + golden diff |

---

*Phiên bản tài liệu này được viết lại cho khớp với mã nguồn hiện hành tại
`src/ingestion/`. Phiên bản cũ trong git từng tham chiếu sai đường dẫn
`ingestion/parser.py`, `chunker.py`, `graph_builder.py`, `config.py`,
`evaluate_phase1.py` — các file này không tồn tại trong cấu trúc dự án hiện tại.*
