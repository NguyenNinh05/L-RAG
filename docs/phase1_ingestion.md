# Phase 1 — Ingestion & Knowledge Representation

## Tổng quan

Phase 1 chuyển đổi các file hợp đồng/văn bản pháp lý (PDF/DOCX) thành một **Hybrid Knowledge Graph** kết hợp:
- **Graph DB (Kuzu Embedded)** — lưu cấu trúc pháp lý và quan hệ tham chiếu giữa các điều khoản
- **Vector DB (ChromaDB Persistent)** — lưu embedding ngữ nghĩa để phục vụ semantic search

Hai cơ sở dữ liệu này được **liên kết với nhau qua `node_id`** — mỗi vector trong ChromaDB đều mang `metadata.node_id` trỏ về đúng node tương ứng trong Kuzu.

### Luồng dữ liệu tổng thể

```
PDF / DOCX
    │
    ▼
[Module 1] LegalDocumentParser
    │   docling parse → check confidence → (fallback) marker-pdf OCR
    │   Build Legal DOM tree (Chương → Điều → Khoản → Điểm)
    │   Extract tables → TableData JSON (không flatten)
    ▼
LegalDocument  (Pydantic model)
    │
    ▼
[Module 2] LsuChunker
    │   Duyệt DOM tree → gắn breadcrumb prefix
    │   Tạo chunks cấp Điều + cấp Khoản
    │   Split chunk dài theo ranh giới câu
    ▼
list[LsuChunk]  (có breadcrumb, source_node_id, tables_json)
    │
    ▼
[Module 3] HybridGraphBuilder
    ├── Kuzu: upsert nodes + CONTAINS + PRECEDES + REFERENCES edges
    └── ChromaDB: embed chunks → lưu với metadata.node_id
```

---

## Cấu trúc file

```
L-RAG/
├── config.py                    # Hằng số toàn hệ thống
├── ingestion/
│   ├── __init__.py              # Public API + ingest_document() runner
│   ├── models.py                # Pydantic data models (536 dòng)
│   ├── parser.py                # Module 1: LegalDocumentParser (870 dòng)
│   ├── chunker.py               # Module 2: LsuChunker (572 dòng)
│   └── graph_builder.py         # Module 3: HybridGraphBuilder (1088 dòng)
├── evaluate_phase1.py           # Script đánh giá từng module
└── docs/
    └── phase1_ingestion.md      # File này
```

---

## Cấu hình (`config.py`)

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `DOCLING_CONFIDENCE_THRESHOLD` | `0.75` | Ngưỡng confidence; thấp hơn → trigger OCR |
| `DOCLING_LOW_CONF_PAGE_RATIO` | `0.30` | Tỷ lệ trang low-conf để trigger OCR toàn file |
| `MARKER_TIMEOUT_SECONDS` | `300` | Timeout gọi marker-pdf |
| `KUZU_DB_PATH` | `./graph_db/legal_graph` | Đường dẫn Kuzu DB |
| `CHROMA_DB_PATH` | `./vector_db/chroma_legal` | Đường dẫn ChromaDB |
| `CHROMA_COLLECTION_NAME` | `legal_documents` | Tên collection ChromaDB |
| `MAX_CHUNK_CHARS` | `2000` | Số ký tự tối đa mỗi chunk |
| `CHUNK_OVERLAP_CHARS` | `200` | Overlap khi chia sub-chunk |
| `EMBEDDING_DIM` | `1024` | Chiều vector (phải khớp model thực) |
| `EMBEDDING_BATCH_SIZE` | `32` | Số texts nhúng mỗi batch |

---

## Data Models (`ingestion/models.py`)

Toàn bộ các class Pydantic được định nghĩa tập trung tại đây. Mọi module đều import từ file này.

### Enums

```python
class NodeType(str, Enum):
    DOCUMENT = "document"
    SECTION  = "section"    # Chương / Phần / Mục
    ARTICLE  = "article"    # Điều
    CLAUSE   = "clause"     # Khoản
    POINT    = "point"      # Điểm a, b, c
    TABLE    = "table"
    ANNEX    = "annex"

class EdgeType(str, Enum):
    CONTAINS    = "CONTAINS"    # Quan hệ cha → con
    REFERENCES  = "REFERENCES"  # "theo quy định tại Điều X"
    PRECEDES    = "PRECEDES"    # Thứ tự tuần tự
    AMENDS      = "AMENDS"      # Sửa đổi (tương lai)
```

### Legal DOM Tree

```
LegalDocument
    ├── preamble: str
    ├── sections: list[DocumentSection]
    │       └── articles: list[ArticleNode]
    │               ├── intro: str
    │               ├── clauses: list[ClauseNode]
    │               │       ├── content: str
    │               │       ├── points: list[PointNode]
    │               │       └── tables: list[TableData]
    │               └── tables: list[TableData]
    ├── orphan_articles: list[ArticleNode]   # Điều không thuộc Chương nào
    └── quality_metrics: ParseQualityMetrics
```

### TableData — Quy tắc bất biến

> Bảng **KHÔNG BAO GIỜ** được chuyển thành text phẳng. Luôn lưu dạng `TableData`:

```python
class TableData(BaseModel):
    headers: list[str]          # Tên cột (row đầu tiên)
    rows: list[list[Any]]       # Dữ liệu 2D
    cells: list[TableCell]      # Chi tiết từng ô (hỗ trợ merged cells)
    num_rows: int
    num_cols: int
    source_page: int | None
```

### LsuChunk — Output Module 2

```python
class LsuChunk(BaseModel):
    chunk_id: str
    doc_id: str
    source_node_id: str          # ← link về Graph DB node
    source_node_type: NodeType   # ARTICLE hoặc CLAUSE
    breadcrumb: str              # "[Chương II > Điều 5 > Khoản 3]"
    content_with_prefix: str     # breadcrumb + "\n" + content  ← cái được embed
    raw_content: str             # nội dung thuần
    content_type: ContentType    # TEXT / TABLE / MIXED
    tables_json: list[dict]      # bảng đính kèm (nếu có)
    article_number: int | str | None
    clause_number: int | str | None
    section_title: str | None
    page_number: int | None
    char_count: int
```

### VectorMetadata — Critical Link

```python
class VectorMetadata(BaseModel):
    node_id: str       # ⚠️ CRITICAL: link về Kuzu Graph DB node
    doc_id: str
    chunk_id: str
    node_type: str
    breadcrumb: str
    file_name: str
    article_number: str
    clause_number: str
    section_title: str
    content_type: str
    has_tables: bool
    page_number: int
    char_count: int
    ingested_at: str   # ISO 8601
```

---

## Module 1 — LegalDocumentParser (`ingestion/parser.py`)

### Mục đích
Chuyển đổi file PDF/DOCX thành `LegalDocument` (Legal DOM tree).

### Regex patterns nhận diện cấu trúc

```python
_RE_ARTICLE = r"^(?P<prefix>Điều)\s+(?P<number>\d+|[IVXLCDM]+)\s*[.:]?\s*(?P<title>.*)$"
_RE_CLAUSE  = r"^(?P<number>\d{1,2})\.\s+(?P<content>.+)$"
_RE_POINT   = r"^\s*(?P<label>[a-zđ])\)\s+(?P<content>.+)$"
_RE_SECTION = r"^(?P<section_type>Chương|Phần|Mục|Tiểu mục)\s+(?P<number>[IVXLCDM\d]+)..."
```

### Fallback Logic

```
parse(file_path)
    │
    ├── _parse_with_docling()
    │       │
    │       ├── export_to_markdown()
    │       ├── _compute_docling_confidence()
    │       │
    │       └── _should_trigger_ocr()?
    │               ├── avg_confidence < 0.75  → True
    │               └── low_conf_pages / total > 0.30 → True
    │
    ├── (nếu OCR cần) _parse_with_marker()
    │       ├── _run_marker_python_api()   # ưu tiên
    │       └── _run_marker_cli()          # fallback subprocess
    │
    └── _build_dom_from_lines()  (state machine dùng chung)
```

### State Machine trong `_build_dom_from_lines()`

Parser duyệt từng dòng text theo thứ tự ưu tiên:
1. Gặp `Chương/Phần/Mục` → flush tất cả, tạo `DocumentSection` mới
2. Gặp `Điều X` → flush article hiện tại, tạo `ArticleNode` mới
3. Gặp `1. Nội dung` trong article → flush clause, tạo `ClauseNode` mới
4. Gặp `a) Nội dung` trong clause → flush point, tạo `PointNode` mới
5. Còn lại → nối vào node hiện tại đang mở

> **Fix quan trọng:** Docling export Markdown có `**bold**`/`*italic*`. Parser thêm bước strip trước regex:
> ```python
> line = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", line).strip()
> # "**Điều 5: Tên điều**" → "Điều 5: Tên điều"
> ```

### Table Extraction

Docling trả về bảng qua `TableItem.data.grid` (list of list of GridCell). Parser extract thành `TableData` với đầy đủ `headers`, `rows`, `cells` (hỗ trợ colspan/rowspan).

---

## Module 2 — LsuChunker (`ingestion/chunker.py`)

### Mục đích
Nhận `LegalDocument`, tạo ra `list[LsuChunk]` với breadcrumb prefix cho embedding.

### Breadcrumb Format

```python
# Article-level chunk:
"[Chương II. Quyền và Nghĩa vụ các Bên > Điều 5. Quyền của Bên A]"

# Clause-level chunk:
"[Chương II. Quyền và Nghĩa vụ các Bên > Điều 5. Quyền của Bên A > Khoản 1]"

# Sub-chunk (khi clause > 2000 chars):
"[Chương II > Điều 5 > Khoản 1 [phần 1/3]]"
```

### Chiến lược chunk 2 cấp

| Loại chunk | Nội dung | Dùng cho |
|---|---|---|
| **Article-level** | intro + preview 150 chars của từng khoản | Query "Điều X nói về gì?" |
| **Clause-level** | nội dung đầy đủ khoản + tất cả điểm con | Query chi tiết cụ thể |

```python
# Flatten clause content (khoản + điểm):
"1. Yêu cầu thanh toán theo hợp đồng.\na) Thanh toán đúng hạn.\nb) Cung cấp biên lai."
```

### Split chunk dài

Khi `len(raw_content) > max_chunk_chars` (mặc định 2000):
1. Split theo regex ranh giới câu (Vietnamese-aware)
2. Nếu không split được (không có dấu câu) → hard split theo số ký tự
3. Mỗi chunk mới bắt đầu bằng `overlap_chars` (mặc định 200) ký tự cuối của chunk trước

---

## Module 3 — HybridGraphBuilder (`ingestion/graph_builder.py`)

### Mục đích
Xây dựng đồng thời Kuzu Graph DB và ChromaDB Vector DB từ `list[LsuChunk]`.

### Schema Kuzu

**Node Tables:**

```
Document(
    doc_id STRING PRIMARY KEY,
    file_name STRING,
    doc_title STRING,
    doc_number STRING,
    signing_date STRING,
    ingested_at STRING
)

LegalNode(
    node_id STRING PRIMARY KEY,    ← khớp với VectorMetadata.node_id
    node_type STRING,              ← "article" | "clause"
    doc_id STRING,
    article_number STRING,
    clause_number STRING,
    section_title STRING,
    content_summary STRING,
    page_number INT64,
    char_count INT64,
    breadcrumb STRING
)
```

**Edge Tables:**

```
CONTAINS(Document → LegalNode, LegalNode → LegalNode)
REFERENCES(LegalNode → LegalNode, context STRING, weight DOUBLE)
PRECEDES(LegalNode → LegalNode)
```

### REFERENCES Edge Detection

Regex scan nội dung từng node, nhận diện 2 patterns:

```python
# Pattern 1: "theo quy định tại Điều 5", "tại Điều 15 Khoản 2"
_RE_ARTICLE_REF = r"(?:theo\s+(?:quy\s+định\s+)?tại\s+|tại\s+|theo\s+)?(?P<keyword>Điều|điều)\s+(?P<article_num>\d+|[IVXLCDM]+)(?:\s*[,.]?\s*(?:Khoản|khoản)\s+(?P<clause_num>\d+))?"

# Pattern 2: "Khoản 3 Điều 15" (thứ tự ngược)
_RE_CLAUSE_FIRST_REF = r"(?:Khoản|khoản)\s+(?P<clause_num>\d+)\s+(?:Điều|điều)\s+(?P<article_num>\d+|[IVXLCDM]+)"
```

Resolve strategy:
1. Nếu có cả số điều **và** số khoản → tìm `ClauseNode` trước
2. Nếu chỉ có số điều → tìm `ArticleNode`
3. Dùng `seen_refs` set để tránh tạo edge trùng lặp

### ChromaDB Storage

```python
# Mỗi chunk được lưu vào ChromaDB với:
collection.upsert(
    ids=[chunk.chunk_id],
    embeddings=[embedding_fn([chunk.content_with_prefix])[0]],
    documents=[chunk.content_with_prefix],
    metadatas=[{
        "node_id": chunk.source_node_id,  # ⚠️ CRITICAL LINK
        "doc_id": chunk.doc_id,
        "breadcrumb": chunk.breadcrumb,
        "node_type": "article" | "clause",
        # ... các field phụ khác
    }]
)
```

### Embedding Placeholder

```python
def placeholder_embedding_fn(texts: list[str]) -> list[list[float]]:
    """Zero vectors — thay bằng model thực."""
    return [[0.0] * EMBEDDING_DIM for _ in texts]
```

**Thay bằng model thực:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")

def real_embed(texts: list[str]) -> list[list[float]]:
    return model.encode(texts, normalize_embeddings=True).tolist()

builder = HybridGraphBuilder(
    kuzu_db_path=...,
    chroma_db_path=...,
    embedding_fn=real_embed,   # plug in tại đây
)
```

---

## Pipeline Runner (`ingestion/__init__.py`)

```python
from ingestion import ingest_document

result = ingest_document(
    file_path="contract.pdf",
    kuzu_db_path="./graph_db/legal_graph",
    chroma_db_path="./vector_db/chroma_legal",
    chroma_collection_name="legal_documents",
    embedding_fn=None,           # None = placeholder
    confidence_threshold=0.75,
    max_chunk_chars=2000,
)

# result = {
#     "doc_id":    "doc_v1_xxx",
#     "file_name": "contract.pdf",
#     "articles":  22,
#     "chunks":    78,
#     "nodes":     23,
#     "edges":     5,
#     "vectors":   78,
#     "engine":    "docling"
# }
```

---

## Cách chạy & Đánh giá

### Cài đặt

```bash
pip install kuzu chromadb docling pydantic>=2.0
# hoặc
pip install -r requirements.txt
```

### Lệnh đánh giá

```bash
# Module 1: xem cấu trúc DOM
python evaluate_phase1.py --module 1 --file data_test/01-tand_signed_v1.docx

# Module 2: xem breadcrumbs + thống kê chunks
python evaluate_phase1.py --module 2 --file data_test/01-tand_signed_v1.docx

# Module 3: build Kuzu + ChromaDB
python evaluate_phase1.py --module 3 --file data_test/01-tand_signed_v1.docx

# Toàn bộ pipeline:
python evaluate_phase1.py --file data_test/01-tand_signed_v1.docx

# Dry-run (không ghi DB):
python evaluate_phase1.py --dry-run

# Verbose (xem docling logs):
python evaluate_phase1.py --verbose
```

### Kết quả đánh giá thực tế

| File | Sections | Articles | Clauses | Tables | Chunks | Time |
|---|---|---|---|---|---|---|
| `01-tand_signed_v1.docx` | 3 | 14 | 38 | 5 | 53 | 2.1s |
| `docs_test/v1.docx` | 0 | 22 | 0* | 9 | 22 | 2.3s |

> \* `v1.docx` dùng format khoản `**1.** Nội dung` (bold số) khác format chuẩn `1. Nội dung`. Parser hiện nhận diện format chuẩn.

### Checklist đánh giá

| Tiêu chí | Lệnh | Pass khi |
|---|---|---|
| Parser nhận diện Điều | `--module 1` | `articles > 0` |
| Bảng không bị flatten | `--module 1` | `tables >= 0`, hiện dạng JSON |
| Breadcrumb đúng format | `--module 2` | `[Chương X > Điều Y > Khoản Z]` |
| node_id có trong chunk | `--module 2` | `source_node_id` không rỗng |
| VALIDATION OK | `--module 2` | `✅ N chunks đều hợp lệ` |
| Graph nodes đầy đủ | `--module 3` | `nodes = articles + clauses` |
| Vectors = chunks | `--module 3` | `vectors == total_chunks` |
| node_id link OK | `--module 3` | ChromaDB metadata `node_id` tồn tại |

---

## Vấn đề đã gặp & cách xử lý

### 1. Parser ra 0 điều với file hợp lệ

**Nguyên nhân:** Docling export markdown thêm `**bold**`/`*italic*` → regex `_RE_ARTICLE` không match `**Điều 5: Tên**`.

**Fix (`ingestion/parser.py`):**
```python
# Thêm sau bước strip heading markers:
line = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", line).strip()
# Strip separator unicode của docling:
if re.match(r"^[─━─\-]{3,}$", line):
    continue
```

### 2. KeyError khi chunker ra 0 chunks

**Nguyên nhân:** `LsuChunker.get_stats([])` trả về `{"total": 0}` — không có key `article_level`.

**Fix (`evaluate_phase1.py`):**
```python
if stats['total'] == 0:
    print("⚠️  CẢNH BÁO: Không tạo được chunk nào!")
else:
    print(stats.get('article_level', 0))  # dùng .get() với default
```

### 3. Kuzu — `Database path cannot be a directory`

**Nguyên nhân:** Code `mkdir(path)` tạo thư mục `graph_db/` **trước** khi gọi `kuzu.Database(path)`. Kuzu v0.x không chấp nhận đường dẫn đã là thư mục tồn tại — nó muốn tự tạo thư mục của mình.

**Fix (`ingestion/graph_builder.py` — `_init_kuzu`):**
```python
# ❌ Sai: tạo thư mục trước
self.kuzu_db_path.mkdir(parents=True, exist_ok=True)
kuzu.Database(str(self.kuzu_db_path))  # → RuntimeError!

# ✅ Đúng: chỉ tạo thư mục CHA, để Kuzu tạo ./graph_db/ tự động
self.kuzu_db_path.parent.mkdir(parents=True, exist_ok=True)

# Xóa thư mục rỗng cũ nếu tồn tại (do run trước bị interrupt):
if self.kuzu_db_path.exists() and self.kuzu_db_path.is_dir():
    if not any(self.kuzu_db_path.iterdir()):  # chỉ xóa nếu rỗng
        self.kuzu_db_path.rmdir()

kuzu.Database(str(self.kuzu_db_path))  # OK
```

**Quy tắc:** Kuzu embedded tự quản lý toàn bộ directory structure bên trong `graph_db/` (các file `.catalog`, `.wal`, `data/`...). **Không được** tạo thư mục đó bằng `mkdir` trước.

### 4. Kuzu MERGE syntax (lưu ý)

Kuzu embedded chưa hỗ trợ `MERGE` hoàn toàn như Neo4j. Nếu gặp lỗi, thay bằng:
```python
# Kiểm tra EXISTS trước, sau đó INSERT hoặc UPDATE
result = conn.execute("MATCH (n:LegalNode {node_id: $id}) RETURN n", ...)
if not result.has_next():
    conn.execute("CREATE (:LegalNode {node_id: $id, ...})", ...)
```

---

## Hướng phát triển tiếp theo

### Phase 2 — Embedding thực
- Tích hợp `intfloat/multilingual-e5-large-instruct` (đã có model weights tại `model.safetensors`)
- Refactor `embedding/embedder.py` để dùng với `LsuChunk` thay vì `ArticleChunk` cũ

### Phase 3 — Hybrid Retrieval
- `retrieval/matcher.py`: kết hợp Graph traversal (Kuzu) + Vector search (ChromaDB)
- Dùng `metadata.node_id` để từ vector hit → tra cứu node trong Kuzu → lấy REFERENCES edges → mở rộng context

### Phase 4 — Document Comparison
- So sánh 2 `LegalDocument` theo từng `ArticleNode`
- Dùng REFERENCES graph để phát hiện điều khoản bị sửa đổi/xóa/thêm

### Phase 5 — LLM Generation
- Sinh báo cáo đối chiếu từ kết quả Phase 4
- Input: diff graph + retrieved chunks; Output: báo cáo pháp lý có cấu trúc
