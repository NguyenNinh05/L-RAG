# Phase 2 — Indexing & Alignment Strategy

## Tổng quan

Phase 2 nhận đầu vào là 2 `LegalDocument` (output của Phase 1) và thực hiện **ghép nối tối ưu** (optimal matching) giữa các `ArticleNode` và `ClauseNode` của V1 và V2.

```
LegalDocument (V1)                LegalDocument (V2)
    │                                    │
    ▼                                    ▼
[Module 1] comparison/models.py    — DiffPairCatalog schema
[Module 2] embedding_manager.py    — BGEM3Manager (FP16)
[Module 3] qdrant_indexer.py       — Multi-vector storage
[Module 4] alignment_engine.py     — Hungarian + Split/Merge
    │
    ▼
DiffPairCatalog
  └── List[DiffPair]  (matched / added / deleted / split / merged)
```

---

## Cấu trúc file

```
L-RAG/
├── comparison/
│   ├── __init__.py              # Public API
│   ├── models.py                # Data models Phase 2
│   ├── qdrant_indexer.py        # QdrantManager (multi-vector)
│   ├── embedding_manager.py     # BGEM3Manager (FP16)
│   └── alignment_engine.py      # LegalAlignmentEngine (CORE)
└── evaluate_phase2.py           # Script đánh giá
```

---

## Data Models (`comparison/models.py`)

### Enums

```python
class MatchType(str, Enum):
    MATCHED = "matched"   # 1-1 rõ ràng, score >= θ
    ADDED   = "added"     # Chỉ có ở V2
    DELETED = "deleted"   # Chỉ có ở V1
    SPLIT   = "split"     # 1 V1 → 2+ V2
    MERGED  = "merged"    # 2+ V1 → 1 V2

class NodeVersion(str, Enum):
    V1 = "v1"
    V2 = "v2"
```

### DiffPair — Đơn vị kết quả

```python
class DiffPair(BaseModel):
    pair_id: str              # ID duy nhất
    v1_ids: list[str]         # Node ID(s) từ V1 (0, 1 hoặc nhiều)
    v2_ids: list[str]         # Node ID(s) từ V2 (0, 1 hoặc nhiều)
    match_type: MatchType
    confidence_score: float   # [0, 1]
    # Breakdown thành phần:
    semantic_score: float | None
    jaro_winkler_score: float | None
    ordinal_proximity_score: float | None
    v1_texts: list[str]       # Raw texts (cho render diff)
    v2_texts: list[str]
```

> `DiffPair` có `@model_validator` đảm bảo tính nhất quán:
> - `MATCHED` bắt buộc `len(v1_ids)==1` và `len(v2_ids)==1`
> - `SPLIT` bắt buộc `len(v1_ids)==1` và `len(v2_ids)>=2`
> - `MERGED` bắt buộc `len(v1_ids)>=2` và `len(v2_ids)==1`

### DiffPairCatalog — Đầu ra chính

```python
class DiffPairCatalog(BaseModel):
    catalog_id: str
    v1_doc_id: str
    v2_doc_id: str
    pairs: list[DiffPair]
    match_threshold: float      # θ đã dùng (audit trail)
    split_merge_threshold: float

    def summary(self) -> dict[str, int]: ...
    def to_report_dict(self) -> dict: ...  # cho JSON export / LLM
```

---

## Qdrant Indexer (`comparison/qdrant_indexer.py`)

### Multi-vector Layout

| Vector Name | Loại | Dim | Mục đích |
|---|---|---|---|
| `structural` | Dense (cosine) | 1024 | Nhận diện vị trí / số hiệu |
| `semantic` | Dense (cosine) | 1024 | So sánh nội dung ngữ nghĩa |
| `structural_sparse` | Sparse (BM25-like) | ≤ 2^20 | Lexical matching |
| `semantic_sparse` | Sparse (BM25-like) | ≤ 2^20 | Lexical matching |

### Payload Schema

```json
{
    "node_id":       "article_abc123",
    "doc_id":        "doc_v1_001",
    "version":       "v1",
    "node_type":     "article",
    "ordinal":       4,
    "raw_text":      "Bên A có quyền...",
    "title":         "Điều 5. Quyền của Bên A",
    "breadcrumb":    "[Chương II > Điều 5]",
    "article_number":"5",
    "clause_number": "",
    "ingested_at":   "2026-04-15T..."
}
```

### Payload Indexes (filter nhanh)

```python
["version", "node_type", "doc_id", "node_id", "ordinal", "article_number"]
```

### Cách dùng

```python
from comparison.qdrant_indexer import QdrantManager, QdrantCollectionConfig

# Local persistent
manager = QdrantManager(path="./qdrant_db")

# Tạo collection
config = QdrantCollectionConfig(collection_name="legal_comparison_v1v2")
manager.create_collection(config)

# Index embeddings
manager.upsert_embeddings("legal_comparison_v1v2", embedding_list)

# Search
results = manager.search_by_semantic(
    "legal_comparison_v1v2",
    query_vector=vec,
    version_filter="v2",
    top_k=10,
)
```

---

## Embedding Manager (`comparison/embedding_manager.py`)

### BGEM3Manager

```python
manager = BGEM3Manager(
    model_name="BAAI/bge-m3",
    use_fp16=True,            # Tiết kiệm VRAM ~2x
    batch_size=16,
    max_length=1024,
)
```

### 2 loại embedding cho mỗi node

| Loại | Input | Max Length | Mục đích |
|---|---|---|---|
| `structural_embed` | `"Điều N: {title}"` | 512 | Nhận diện cấu trúc / số hiệu |
| `semantic_embed` | `[breadcrumb]\n{intro}\n{clause previews}` | 1024 | So sánh ngữ nghĩa |

```python
# Embed ArticleNodes
v1_embeddings = manager.embed_article_nodes(
    articles=doc_v1.iter_all_articles(),
    version=NodeVersion.V1,
    doc_id=doc_v1.doc_id,
)

# Embed ClauseNodes trong một Điều
clause_embeddings = manager.embed_clause_nodes(
    clauses=article.clauses,
    parent_article=article,
    version=NodeVersion.V1,
    doc_id=doc_v1.doc_id,
)
```

### Sparse Vector Conversion

BGE-M3 trả về `{word_string: float}`, Qdrant cần `{int_index: float}`.
Conversion dùng `abs(hash(token)) % 2^20` — bucket size 1M, collision negligible.

---

## Alignment Engine (`comparison/alignment_engine.py`)

### AlignmentConfig

```python
@dataclass
class AlignmentConfig:
    w_semantic:      float = 0.6   # Trọng số Cosine(Semantic)
    w_jaro_winkler:  float = 0.3   # Trọng số JaroWinkler(Title)
    w_ordinal:       float = 0.1   # Trọng số OrdinalProximity

    match_threshold:       float = 0.65   # θ: lọc cặp unmatched
    split_merge_threshold: float = 0.80   # Ngưỡng split/merge detection
```

> Validator: `w_semantic + w_jaro_winkler + w_ordinal` phải = 1.0

### Bước 1 — Similarity Matrix

$$S[i][j] = 0.6 \times \text{Cosine}(\text{Sem}_i, \text{Sem}_j) + 0.3 \times \text{JaroWinkler}(\text{Title}_i, \text{Title}_j) + 0.1 \times (1 - |pos_i - pos_j|)$$

Trong đó $pos_i = \frac{ordinal_i}{N-1}$ (vị trí tương đối trong tài liệu).

```python
sim_matrix = engine.compute_similarity_matrix(v1_records, v2_records)
# → np.ndarray shape (N, M), dtype float32, range [0, 1]
```

### Bước 2 — Hungarian Matching

```python
# Cost Matrix = 1 - S
# scipy.optimize.linear_sum_assignment → cặp optimal
row_ind, col_ind = linear_sum_assignment(1 - sim_matrix)

# Lọc: chỉ giữ cặp có score >= θ = 0.65
matched_pairs = [(i, j, score) for i, j, score in ... if score >= θ]
```

Complexity: **O(n³)** — đủ nhanh cho ~100 articles (< 1s).

### Bước 3 — Split/Merge Fallback

```
Với mỗi cặp V1[A], V2[B], V2[C] (unmatched):
    merged_text = V2[B].text + "\n" + V2[C].text
    score = Cosine(embed(V1[A].text), embed(merged_text))
    if score >= 0.80 → MERGED: V1_ids=[A], V2_ids=[B, C]

Với mỗi cặp V1[A], V1[B], V2[C] (unmatched):
    merged_text = V1[A].text + "\n" + V1[B].text
    score = Cosine(embed(merged_text), embed(V2[C].text))
    if score >= 0.80 → SPLIT: V1_ids=[A, B], V2_ids=[C]
```

### Bước 4 — Hierarchical (Clause-level)

```python
# Với mỗi cặp Article đã MATCHED:
for article_pair in catalog.matched_pairs:
    v1_clauses = v1_article.clauses
    v2_clauses = v2_article.clauses

    clause_pairs = engine._align_clauses(
        v1_clauses, v2_clauses, v1_article, v2_article, doc_v1, doc_v2
    )
    catalog.pairs.extend(clause_pairs)
```

Chạy **đệ quy** cùng một pipeline 4-bước ở cấp Khoản — ordinal tính trong phạm vi local của từng Điều.

---

## Cách dùng hoàn chỉnh

```python
from ingestion.parser import LegalDocumentParser
from comparison import LegalAlignmentEngine, AlignmentConfig, BGEM3Manager

# 1. Parse 2 tài liệu (Phase 1)
parser = LegalDocumentParser()
doc_v1 = parser.parse("contract_v1.docx")
doc_v2 = parser.parse("contract_v2.docx")

# 2. Load model
emb = BGEM3Manager(use_fp16=True)

# 3. Cấu hình & chạy alignment
cfg = AlignmentConfig(match_threshold=0.65, split_merge_threshold=0.80)
engine = LegalAlignmentEngine(embed_manager=emb, config=cfg)

catalog = engine.align_documents(doc_v1, doc_v2)

# 4. Xem kết quả
print(catalog.summary())
# {'total_pairs': 28, 'matched': 20, 'added': 3, 'deleted': 2, 'split': 2, 'merged': 1}

# 5. Export JSON
import json
with open("diff_catalog.json", "w") as f:
    json.dump(catalog.to_report_dict(), f, ensure_ascii=False, indent=2, default=str)
```

---

## Lệnh đánh giá

```bash
# Test models (không cần GPU, < 2s)
python evaluate_phase2.py --module models

# Test Qdrant in-memory (không cần GPU, < 5s)
python evaluate_phase2.py --module qdrant

# Test alignment với mock data (không cần GPU, < 3s)
python evaluate_phase2.py --module align --mock

# Full test suite (không cần GPU)
python evaluate_phase2.py --module all

# End-to-end với file thật + model thật
python evaluate_phase2.py --module align \
    --v1 data_test/01-tand_signed_v1.docx \
    --v2 docs_test/v1.docx \
    --output eval_output/phase2_catalog.json
```

---

## Cài đặt dependencies

```bash
pip install qdrant-client FlagEmbedding jellyfish scipy numpy
```

| Package | Version | Mục đích |
|---|---|---|
| `qdrant-client` | >= 1.9 | Local vector DB (multi-vector) |
| `FlagEmbedding` | >= 1.2 | BAAI/bge-m3 với FP16 |
| `jellyfish` | >= 1.0 | Jaro-Winkler string distance |
| `scipy` | >= 1.11 | `linear_sum_assignment` (Hungarian) |
| `numpy` | >= 1.24 | Matrix operations |

---

## Kết quả đánh giá thực tế (mock data)

| Metric | Giá trị |
|---|---|
| Similarity Matrix | (N, M) float32, range [0, 1] |
| Hungarian (4×3) | ~0.001s |
| θ = 0.65 | Lọc tốt cặp không liên quan |
| Split/Merge θ = 0.80 | Precision cao, ít false positive |

---

## Hướng phát triển — Phase 3

- **`retrieval/diff_retriever.py`**: Dùng `DiffPairCatalog` → tìm context relevant cho từng cặp thay đổi
- **`llm/report_generator.py`**: Sinh báo cáo pháp lý từ catalog (matched/diff/added/deleted)
- **Cross-version search**: Query "Điều 5 V1 đã thay đổi gì?" → dùng Qdrant filter `version=v1` + linked pair
