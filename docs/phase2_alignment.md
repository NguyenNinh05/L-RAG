# Phase 2 — Indexing & Alignment (End-to-End)

> Tài liệu mô tả chi tiết **Module Alignment** của hệ thống LegalDiff (L-RAG):
> từ 2 `LegalDocument` (V1, V2) → **embedding BGE-M3** → **Similarity Matrix**
> → **Hungarian matching + Split/Merge** → **`DiffPairCatalog`**.
> Toàn bộ nội dung được đối chiếu trực tiếp với mã nguồn tại `src/alignment/`
> và ví dụ thật từ `tests/e2e_backend/outputs/*/13_catalog.json`.

---

## Mục lục

1. [Vai trò của Phase 2 trong pipeline](#1-vai-trò-của-phase-2-trong-pipeline)
2. [Sơ đồ luồng dữ liệu end-to-end](#2-sơ-đồ-luồng-dữ-liệu-end-to-end)
3. [Cấu trúc source code & Public API](#3-cấu-trúc-source-code--public-api)
4. [Cấu hình (config)](#4-cấu-hình-config)
5. [Data Models — `diff_catalog.py`](#5-data-models--diff_catalogpy)
6. [Embedding — `BGEM3Manager`](#6-embedding--bgem3manager)
7. [Vector Store — `QdrantManager` (multi-vector)](#7-vector-store--qdrantmanager-multi-vector)
8. [Similarity Matrix — `similarity_matrix.py`](#8-similarity-matrix--similarity_matrixpy)
9. [Matching — `hungarian_matcher.py`](#9-matching--hungarian_matcherpy)
10. [Orchestrator — `LegalAlignmentEngine`](#10-orchestrator--legalalignmentengine)
11. [Worked example end-to-end (từ dữ liệu thật)](#11-worked-example-end-to-end-từ-dữ-liệu-thật)
12. [Nguyên tắc bất biến (Design invariants)](#12-nguyên-tắc-bất-biến-design-invariants)
13. [Cách chạy & mở rộng](#13-cách-chạy--mở-rộng)

---

## 1. Vai trò của Phase 2 trong pipeline

| Phase | Tên | Module | Vai trò |
|-------|-----|--------|---------|
| 1 | Ingestion | `src.ingestion` | Parse PDF/DOCX → Legal DOM + LSU chunks |
| **2** | **Alignment** | `src.alignment` | **(tài liệu này)** Ghép cặp tối ưu các Điều V1↔V2 |
| 3 | Generative Comparison | `src.comparison` | LLM sinh biên bản so sánh từng cặp |

**Bài toán Phase 2 giải quyết:** Cho 2 phiên bản văn bản pháp lý (V1 gốc, V2 sửa
đổi), biết **Điều nào của V1 tương ứng với Điều nào của V2** — kể cả khi:
- Nội dung bị **diễn đạt lại** (paraphrase).
- **Đảo vị trí** (Điều 5 đổi chỗ sang sau Điều 6).
- **Đổi số** điều (renumber).
- **Tách / gộp** (1 Điều V1 → 2 Điều V2, hoặc ngược lại).
- **Thêm / xóa** Điều mới.

**Đầu ra chính:** `DiffPairCatalog` — danh sách `DiffPair`, mỗi pair có `match_type`
thuộc `{matched, added, deleted, split, merged}` + `confidence_score`. Đây là
**đầu vào của Phase 3** (mỗi pair → 1 lần gọi LLM để sinh so sánh).

> **Phạm vi matching: cấp Điều (Article).** Engine hiện gọi
> `doc.iter_all_articles()` và match ở cấp Điều. (Embedder *có* hỗ trợ embed
> Khoản, nhưng orchestrator hiện không chạy alignment cấp Khoản — xem [§10](#10-orchestrator--legalalignmentengine).)

---

## 2. Sơ đồ luồng dữ liệu end-to-end

```
 LegalDocument (V1)                          LegalDocument (V2)
 list[ArticleNode] N                         list[ArticleNode] M
        │                                            │
        ▼                                            ▼
 ╔═══════════════════════════════════════════════════════════════════╗
 ║                     PHASE 2 — ALIGNMENT                           ║
 ╠═══════════════════════════════════════════════════════════════════╣
 ║                                                                   ║
 ║  [Embedding] BGEM3Manager (src/alignment/embedder.py)             ║
 ║     mỗi Article → 2 embedding:                                    ║
 ║       • structural  = "Điều N: {title}"          (nhận diện số)   ║
 ║       • semantic    = "[breadcrumb]\n{intro}\n{khoản previews}"   ║
 ║     + sparse lexical weights (BM25-like) từ BGE-M3                ║
 ║     → list[NodeEmbeddings]  (V1) + list[NodeEmbeddings]  (V2)     ║
 ║        │                                                          ║
 ║        ├─(optional)─▶ [QdrantManager] upsert multi-vector points  ║
 ║        │                                                          ║
 ║        ▼                                                          ║
 ║  [Similarity] compute_similarity_matrix  (similarity_matrix.py)   ║
 ║     S[i][j] = w_sem·Cosine(sem) + w_jaro·JaroWinkler(title)       ║
 ║             + w_ord·OrdinalProximity + w_sparse·SparseOverlap     ║
 ║             + 0.15 nếu cùng số Điều        → clamp [0,1]          ║
 ║     → np.ndarray (N, M)                                           ║
 ║        │                                                          ║
 ║        ▼                                                          ║
 ║  [Matching] hungarian_match  (hungarian_matcher.py)               ║
 ║     Cost = 1 − S → scipy.linear_sum_assignment → cặp tối ưu       ║
 ║     giữ cặp có score ≥ θ_match (0.60)       → MATCHED pairs       ║
 ║     node không match → unmatched V1 / unmatched V2                ║
 ║        │                                                          ║
 ║        ▼                                                          ║
 ║  [Split/Merge] detect_split_merge  (hungarian_matcher.py)         ║
 ║     MERGE: text(V1[a]+V1[b]) ≈ text(V2[c])  ≥ θ_sm (0.80)        ║
 ║     SPLIT: text(V1[a]) ≈ text(V2[b]+V2[c])  ≥ θ_sm (0.80)        ║
 ║     (re-embed text ghép → cosine)                                 ║
 ║        │                                                          ║
 ║        ▼                                                          ║
 ║  Vẫn unmatched: V1-only → DELETED, V2-only → ADDED               ║
 ║                                                                   ║
 ╚═══════════════════════════════════════════════════════════════════╝
        │
        ▼
 DiffPairCatalog  (matched + added + deleted + split + merged)
        │
        ▼
 Phase 3 — Generative Comparison (mỗi pair → 1 LLM call)
```

---

## 3. Cấu trúc source code & Public API

```
src/alignment/
├── __init__.py            # Public API
├── diff_catalog.py        # Data models: MatchType, DiffPair, DiffPairCatalog, NodeEmbeddings, QdrantPayload
├── embedder.py            # BGEM3Manager (BGE-M3, FP16)
├── qdrant_indexer.py      # QdrantManager (multi-vector: dense + sparse)
├── similarity_matrix.py   # AlignmentConfig + các hàm tính S[i][j]
├── hungarian_matcher.py   # hungarian_match() + detect_split_merge()
└── alignment_engine.py    # LegalAlignmentEngine (orchestrator 9 bước)
```

**Public API** (từ `src/alignment/__init__.py`):

```python
from src.alignment import (
    # Orchestrator & config
    LegalAlignmentEngine, AlignmentConfig, NodeRecord,
    # Data models
    DiffPair, DiffPairCatalog, MatchType, NodeVersion,
    NodeEmbeddings, QdrantPayload,
    # Embedding & matching functions
    BGEM3Manager, compute_similarity_matrix,
    hungarian_match, detect_split_merge,
    # Vector store
    QdrantManager, QdrantCollectionConfig,
)
```

---

## 4. Cấu hình (config)

Phase 2 đọc config từ `configs/pipeline_config.yaml` (section `alignment`) và
`configs/model_config.yaml` (section `embedding`).

**Trọng số similarity** (phải tổng = 1.0 — có validator kiểm):

| Tham số | Giá trị production | Default code | Ý nghĩa |
|---------|--------------------|--------------|---------|
| `w_semantic` | `0.50` | `0.6` | Cosine(dense semantic) |
| `w_jaro_winkler` | `0.25` | `0.3` | Jaro-Winkler(title) |
| `w_ordinal` | `0.10` | `0.1` | Ordinal proximity (vị trí tương đối) |
| `w_sparse` | `0.15` | `0.0` | Sparse lexical overlap (BM25-like) |

**Ngưỡng:**

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `match_threshold` | `0.60` | θ_match — Hungarian chỉ giữ cặp có score ≥ ngưỡng |
| `split_merge_threshold` | `0.80` | θ_sm — cosine của text ghép để xác nhận split/merge |
| `embed_batch_size` | `32` | Batch embedding (tránh OOM VRAM) |

**Embedding model:**

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `model_name` | `BAAI/bge-m3` | Multilingual embedder (dense 1024 + sparse) |
| `use_fp16` | `true` | Tiết kiệm VRAM ~2× |
| `max_length` | `1024` | Max token (BGE-M3 hỗ trợ tới 8192) |
| `embedding_dim` | `1024` | Dense output dim |

**Qdrant:** `alignment.qdrant.path = null` (in-memory mặc định trong dev — tránh
lock giữa các job chạy tuần tự).

```python
# Cách load trong code (pipeline.py dùng thế này):
from src.config import get_config
acfg = get_config()["alignment"]
config = AlignmentConfig(
    w_semantic=acfg["w_semantic"], w_jaro_winkler=acfg["w_jaro_winkler"],
    w_ordinal=acfg["w_ordinal"], w_sparse=acfg.get("w_sparse", 0.0),
    match_threshold=acfg["match_threshold"],
    split_merge_threshold=acfg["split_merge_threshold"],
)
```

---

## 5. Data Models — `diff_catalog.py`

> **File:** `src/alignment/diff_catalog.py` (Pydantic v2)

### 5.1. Enums

```python
class MatchType(str, Enum):
    MATCHED = "matched"   # 1-1, score ≥ θ
    ADDED   = "added"     # chỉ có ở V2
    DELETED = "deleted"   # chỉ có ở V1
    SPLIT   = "split"     # 1 V1 → 2+ V2
    MERGED  = "merged"    # 2+ V1 → 1 V2

class NodeVersion(str, Enum):
    V1 = "v1"
    V2 = "v2"
```

### 5.2. `QdrantPayload` — metadata lưu cùng mỗi Qdrant point

```python
class QdrantPayload(BaseModel):
    node_id: str            # ID node gốc từ Phase 1 (article/clause node_id)
    doc_id: str
    version: NodeVersion    # v1 | v2
    node_type: str          # 'article' | 'clause'
    ordinal: int            # thứ tự xuất hiện (0-indexed) — cho ordinal proximity
    raw_text: str           # nội dung thuần (dùng tính similarity + render diff)
    title: str              # tiêu đề ngắn (dùng Jaro-Winkler)
    breadcrumb: str         # từ Phase 1
    article_number: str     # số điều (cho article-number bonus)
    clause_number: str
    ingested_at: str        # ISO 8601
```

### 5.3. `NodeEmbeddings` — output của embedder

Mỗi node được nhúng thành **2 dense + 2 sparse** vector:

```python
class NodeEmbeddings(BaseModel):
    node_id: str
    structural_dense: list[float]                 # "Điều N: {title}"
    semantic_dense:   list[float]                 # "[breadcrumb]\n{full text}"
    structural_sparse: dict[int, float]           # lexical weights (structural)
    semantic_sparse:   dict[int, float]           # lexical weights (semantic)
    payload: QdrantPayload
```

> **Lưu ý implementation:** similarity matrix hiện chỉ dùng `semantic_dense` và
> `semantic_sparse`. `structural_dense` được tính và lưu vào Qdrant (hỗ trợ
> search theo structural) nhưng **chưa nằm trong công thức matching**.

### 5.4. `DiffPair` — đơn vị kết quả (có consistency validator)

```python
class DiffPair(BaseModel):
    pair_id: str                       # UUID
    v1_ids: list[str]                  # node ID(s) từ V1
    v2_ids: list[str]                  # node ID(s) từ V2
    match_type: MatchType
    confidence_score: float            # [0,1]; 0.0 cho added/deleted
    # Breakdown (optional, cho explainability):
    semantic_score, structural_score, jaro_winkler_score, ordinal_proximity_score
    v1_texts: list[str]                # raw text để render diff ở Phase 3
    v2_texts: list[str]
```

`@model_validator` ép nhất quán giữa `match_type` và số ID:

| `match_type` | `len(v1_ids)` | `len(v2_ids)` |
|--------------|---------------|---------------|
| `MATCHED` | `== 1` | `== 1` |
| `ADDED` | `== 0` | `≥ 1` |
| `DELETED` | `≥ 1` | `== 0` |
| `SPLIT` | `== 1` | `≥ 2` |
| `MERGED` | `≥ 2` | `== 1` |

→ Vi phạm → `ValueError` khi tạo. Đảm bảo không thể có pair "vô lý".

### 5.5. `DiffPairCatalog` — đầu ra chính

```python
class DiffPairCatalog(BaseModel):
    catalog_id: str
    v1_doc_id: str, v2_doc_id: str
    pairs: list[DiffPair]
    match_threshold: float        # audit trail — ngưỡng đã dùng
    split_merge_threshold: float

    # Properties lọc theo type:
    @property matched_pairs, added_nodes, deleted_nodes, split_cases, merged_cases

    def summary(self) -> dict      # {total, matched, added, deleted, split, merged}
    def to_report_dict(self) -> dict   # serialize cho JSON export / LLM input
```

---

## 6. Embedding — `BGEM3Manager`

> **File:** `src/alignment/embedder.py` — dùng **BAAI/bge-m3** qua `FlagEmbedding`, FP16.

### 6.1. Vì sao BGE-M3?

BGE-M3 trả về **3 loại biểu diễn** cùng lúc:
1. `dense_vecs` — dense 1024 chiều (cosine similarity).
2. `lexical_weights` — sparse BM25-style (bắt exact-match term: số điều, tên riêng).
3. `colbert_vecs` — ColBERT late-interaction (**không dùng** trong Phase 2).

→ Dense bắt ngữ nghĩa, sparse bắt từ khóa chính xác — kết hợp cả hai trong similarity.

### 6.2. Hai loại text được nhúng cho mỗi node

| Loại | Input text | max_length | Mục đích |
|------|------------|------------|----------|
| **structural** | `"Điều {N}: {title}"` (ngắn) | 512 | Nhận diện số hiệu/cấu trúc |
| **semantic** | `"[breadcrumb]\n{intro}\n{preview 3 khoản đầu}"` | 1024 | So sánh ngữ nghĩa sâu |

```python
manager = BGEM3Manager(use_fp16=True, batch_size=16, max_length=1024)

# Embed toàn bộ Article của V1:
v1_embeddings = manager.embed_article_nodes(
    articles=list(doc_v1.iter_all_articles()),
    version=NodeVersion.V1,
    doc_id=doc_v1.doc_id,
)
# → list[NodeEmbeddings], mỗi cái có structural_dense + semantic_dense + 2 sparse
```

> `_build_article_semantic_text` chỉ lấy preview **3 khoản đầu** (150 ký tự/khoản)
> — không nhúng toàn bộ text để giữ vector ổn định khi chỉ 1 khoản thay đổi.

`embed_clause_nodes(...)` cũng tồn tại (nhúng Khoản con của 1 Điều), nhưng
**orchestrator hiện không gọi** (xem [§10](#10-orchestrator--legalalignmentengine)).

### 6.3. Sparse weight conversion

BGE-M3 trả lexical weights dạng `{word_string: float}`, nhưng Qdrant `SparseVector`
yêu cầu **integer indices**. `_convert_sparse` hash string → int:

```python
BUCKET_SIZE = 1 << 20  # ~1M buckets; collision negligible với vocab pháp lý
idx = abs(hash(token)) % BUCKET_SIZE
# Collision → merge bằng max(weight)
```

### 6.4. `embed_texts_semantic` — dùng cho split/merge

Hàm tiện ích cho detect_split_merge: nhận `list[str]` → `np.ndarray (N, 1024)`
đã L2-normalize (để `dot product = cosine`).

---

## 7. Vector Store — `QdrantManager` (multi-vector)

> **File:** `src/alignment/qdrant_indexer.py`

### 7.1. Mode chạy

```python
QdrantManager()                 # in-memory (test, dev default)
QdrantManager(path="./qdrant_db")   # local persistent
QdrantManager(url="http://...", api_key="...")  # remote server / cloud
```

> **Qdrant là OPTIONAL trong Phase 2.** Nếu `align_documents(collection_name=None)`
> thì bỏ qua lưu Qdrant hoàn toàn — matching chạy thuần in-memory qua similarity
> matrix. Qdrant chỉ dùng khi cần query/search lại (hybrid retrieval).

### 7.2. Multi-vector layout — mỗi node = 1 point

| Vector name | Loại | Dim | Mục đích |
|-------------|------|-----|----------|
| `structural` | Dense (cosine) | 1024 | Nhận diện vị trí / số hiệu |
| `semantic` | Dense (cosine) | 1024 | So sánh nội dung ngữ nghĩa |
| `structural_sparse` | Sparse (BM25-like) | ≤ 2²⁰ | Lexical matching |
| `semantic_sparse` | Sparse (BM25-like) | ≤ 2²⁰ | Lexical matching |

```python
@dataclass
class QdrantCollectionConfig:
    collection_name: str
    dense_dim: int = 1024
    dense_distance: Distance = COSINE
    hnsw_m: int = 16
    hnsw_ef_construct: int = 200
    on_disk_payload: bool = False
```

### 7.3. Point ID deterministic

Qdrant local mode bắt buộc point ID là UUID v4. Dùng `uuid5(NAMESPACE_DNS, node_id)`
→ **cùng node_id luôn ra cùng UUID** → upsert lại không tạo duplicate (idempotent).

### 7.4. Payload indexes (filter nhanh)

Tự tạo index trên: `version`, `node_type`, `doc_id`, `node_id`, `ordinal`,
`article_number` → query theo version (`v1`/`v2`) hoặc doc rất nhanh.

### 7.5. Search API

- `search_by_semantic(collection, query_vector, version_filter, top_k)` — dense semantic.
- `search_by_structural(...)` — dense structural.
- `get_all_points(collection, version_filter)` — scroll toàn bộ (phân trang 256),
  sort theo `ordinal`.

---

## 8. Similarity Matrix — `similarity_matrix.py`

> **File:** `src/alignment/similarity_matrix.py`

### 8.1. Công thức tổng hợp

```
S[i][j] = w_sem · Cosine(sem_dense_i, sem_dense_j)
        + w_jaro · JaroWinkler(title_i, title_j)
        + w_ord · OrdinalProximity(i, j)
        + w_sparse · SparseOverlap(sem_sparse_i, sem_sparse_j)   (nếu w_sparse > 0)
        + 0.15                                                      (nếu cùng số Điều)
        →  clamp về [0, 1]
```

### 8.2. Các thành phần

**a) `cosine_similarity_matrix`** — dense semantic. L2-normalize từng hàng rồi
`V1_mat @ V2_mat.T` → `(N, M)`.

**b) `jaro_winkler_matrix`** — string similarity trên `title.lower()` qua `jellyfish`.
Bắt tiêu đề gần giống nhau dù nội dung đổi.

**c) `ordinal_proximity_matrix`** — vị trí tương đối:
```
OrdinalProximity(i,j) = 1 − | i/(N−1) − j/(M−1) |     ∈ [0,1]
```
→ Điều ở đầu V1 ưu tiên match Điều ở đầu V2. Phá vỡ dead-lock khi nhiều điều
giống nhau về ngữ nghĩa.

**d) `sparse_overlap_matrix`** (Strategy E) — cosine giữa 2 sparse vector (dict
`{index: weight}`, L2-normalize). Bù cho dense: bắt **exact-match term** (số điều,
tên riêng, thuật ngữ pháp lý) mà dense embedding có thể làm mờ.

**e) Article-number bonus** (Strategy E) — `ARTICLE_NUMBER_BONUS = 0.15` cộng
thêm (rồi clamp) khi `article_number` V1 == V2. Văn bản pháp lý hiếm khi đổi số
điều trừ khi renumber → đây là tín hiệu nhận diện mạnh.

### 8.3. `AlignmentConfig` (dataclass + validator)

```python
@dataclass
class AlignmentConfig:
    w_semantic: float = 0.6
    w_jaro_winkler: float = 0.3
    w_ordinal: float = 0.1
    w_sparse: float = 0.0          # production: 0.15 (từ config.yaml)
    match_threshold: float = 0.65
    split_merge_threshold: float = 0.80
    embed_batch_size: int = 32
```

`__post_init__` raise `ValueError` nếu `w_sem + w_jaro + w_ord + w_sparse ≠ 1.0`.

### 8.4. `NodeRecord` — wrapper nhẹ cho similarity

```python
@dataclass
class NodeRecord:
    node_id: str
    title: str                    # cho Jaro-Winkler
    raw_text: str                 # cho split/merge re-embed
    ordinal: int                  # cho ordinal proximity
    semantic_vec: np.ndarray      # dense 1024
    semantic_sparse_vec: dict     # sparse lexical
    article_number: str           # cho bonus
    article_ref, clause_ref       # giữ ref gốc (dùng nếu mở rộng hierarchical)
```

---

## 9. Matching — `hungarian_matcher.py`

> **File:** `src/alignment/hungarian_matcher.py`

### 9.1. `hungarian_match` — matching tối ưu 1-1

```python
def hungarian_match(similarity_matrix, match_threshold=0.65):
    cost = 1.0 - similarity_matrix                  # min cost = max similarity
    row_ind, col_ind = linear_sum_assignment(cost)  # scipy, O(n³)
    # Giữ cặp có score ≥ θ, bỏ cặp yếu:
    matched = [(i,j,score) for ... if score >= match_threshold]
    return matched, v1_unmatched_idx, v2_unmatched_idx
```

- Dùng **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`) để tìm
  phép gán 1-1 **tối ưu toàn cục** (tổng similarity lớn nhất).
- Cost matrix = `1 − S` (đổi max ↔ min).
- Sau đó **lọc** theo `match_threshold`: cặp dưới ngưỡng bị đẩy về unmatched.
- Trả về cả danh sách node V1/V2 không match được → đầu vào cho split/merge.

> Complexity O(n³) — đủ nhanh cho ~100 Điều (<1s). Đây là lý do Phase 2 match
> ở cấp Điều (ít node) thay vì cấp Khoản (nhiều node hơn).

### 9.2. `detect_split_merge` — fallback cho node unmatched

Chạy **sau** Hungarian, trên 2 tập unmatched. Brute-force cặp `(i1, i2)` / `(j1, j2)`:

**MERGE** (2 V1 → 1 V2): với mỗi V2 chưa match, xét từng cặp V1 chưa match,
ghép text `V1[i1].raw + "\n" + V1[i2].raw`, re-embed, cosine với `V2[j].raw`.
Nếu ≥ `split_merge_threshold` (0.80) → `MERGED`.

**SPLIT** (1 V1 → 2 V2): đối xứng — `V1[i]` vs `V2[j1].raw + "\n" + V2[j2].raw`.

```python
def detect_split_merge(v1_unmatched, v2_unmatched, embed_fn, split_merge_threshold=0.80):
    # MERGE pass → SPLIT pass, dùng used_v1/used_v2 để không trùng
    # chọn best pair (score cao nhất) nếu nhiều ứng viên
    return result_pairs, still_v1_unmatched, still_v2_unmatched
```

- `embed_fn = embedder.embed_texts_semantic` (đã L2-normalize → `dot = cosine`).
- Mỗi node chỉ dùng 1 lần (`used_v1`, `used_v2` set).
- Còn unmatched sau 2 lượt → sẽ thành `DELETED` (V1-only) hoặc `ADDED` (V2-only).

---

## 10. Orchestrator — `LegalAlignmentEngine`

> **File:** `src/alignment/alignment_engine.py`

`align_documents(doc_v1, doc_v2, collection_name=None)` — pipeline 9 bước:

| Bước | Việc làm | Output |
|------|----------|--------|
| 1 | Thu thập Article: `doc.iter_all_articles()` (V1, V2) | `list[ArticleNode]` N, M |
| 2 | Embed 2 tập qua `BGEM3Manager.embed_article_nodes` | `list[NodeEmbeddings]` |
| 3 | *(optional)* Lưu Qdrant nếu có `collection_name` | upsert points |
| 4 | Chuyển `NodeEmbeddings → NodeRecord` | `list[NodeRecord]` |
| 5 | `compute_similarity_matrix(v1, v2, config)` | `S` shape (N, M) |
| 6 | `hungarian_match(S, match_threshold)` | matched triples + unmatched idx |
| 7 | matched → `DiffPair(MATCHED)` | list pairs |
| 8 | `detect_split_merge(unmatched, embed_fn, split_merge_threshold)` | SPLIT/MERGED pairs |
| 9 | Vẫn unmatched V1 → `DELETED`, V2 → `ADDED` | pairs còn lại |

→ Gom tất cả → `DiffPairCatalog(v1_doc_id, v2_doc_id, pairs, ...)`.

```python
engine = LegalAlignmentEngine(
    embed_manager=BGEM3Manager(),
    config=AlignmentConfig(w_semantic=0.5, w_jaro_winkler=0.25,
                           w_ordinal=0.1, w_sparse=0.15,
                           match_threshold=0.60, split_merge_threshold=0.80),
    qdrant_manager=None,          # None → bỏ qua Qdrant
)
catalog = engine.align_documents(doc_v1, doc_v2, collection_name=None)
print(catalog.summary())
# {'total_pairs': 16, 'matched': 14, 'added': 2, 'deleted': 0, 'split': 0, 'merged': 0}
```

> **Không có alignment cấp Khoản.** Dù `diff_catalog.py` comment nhắc "đệ quy
> cấp Khoản" và embedder có `embed_clause_nodes`, engine hiện **chỉ match Article**.
> Clause alignment là phần dự kiến mở rộng (xem [§13](#13-cách-chạy--mở-rộng)).

---

## 11. Worked example end-to-end (từ dữ liệu thật)

Văn bản `01-tand_signed` (Thông tư 01/2026/TT-TANDTC), cặp V1/V2 từ `data_test/`.
Catalog thật: `tests/e2e_backend/outputs/20260615_121411/13_catalog.json`.

### Kết quả alignment

```json
{
  "v1_doc_id": "doc_..._01-tand_signed_v1_...",
  "v2_doc_id": "doc_..._01-tand_signed_v2_...",
  "thresholds": {"match": 0.65, "split_merge": 0.8},
  "summary": {"total_pairs": 16, "matched": 14, "added": 2,
              "deleted": 0, "split": 0, "merged": 0}
}
```

→ 14 Điều khớp 1-1, 2 Điều mới xuất hiện ở V2 (added), không có xóa/tách/gộp.

### Ví dụ pair MATCHED (confidence cao)

Điều "Đối tượng áp dụng" — V2 paraphrase ("áp dụng" → "có hiệu lực thi hành")
nhưng vẫn match với score `0.9821`:

```json
{
  "pair_id": "pair_c5e11d18a373",
  "match_type": "matched",
  "confidence_score": 0.9821,
  "v1_ids": ["article_b0489445eb1d"],
  "v2_ids": ["article_1f1defb5752d"],
  "v1_texts": ["Đối tượng áp dụng\n Thông tư này áp dụng đối với Thẩm phán Tòa án..."],
  "v2_texts": ["Đối tượng áp dụng\n Thông tư này có hiệu lực thi hành đối với Thẩm phán Tòa án..."]
}
```

→ Phase 2 "hiểu" 2 đoạn là cùng Điều dù đổi cách diễn đạt → Phase 3 sẽ so sánh
chi tiết sự khác biệt那段 này.

### Ví dụ pair ADDED (Điều mới ở V2)

Golden dataset Phase 1 có `change_id=7, category=C, "Thêm điều khoản mới", "Sau Điều 3"`.
Phase 2 phát hiện đúng — 2 Điều mới không có ứng viên ở V1 → `confidence_score=0.0`:

```json
[
  {"match_type": "added", "confidence_score": 0.0,
   "v2_ids": ["article_5d9263d50ad4"],
   "v2_texts": ["a. Bảo mật thông tin trên Giấy chứng minh Thẩm phán..."]},
  {"match_type": "added", "confidence_score": 0.0,
   "v2_ids": ["article_10de01104219"],
   "v2_texts": ["b. Hình thức cấp đổi, cấp lại..."]}
]
```

### Diễn giải luồng cho 1 Điều match

```
Điều 2 V1 ("Đối tượng áp dụng")
   │ embed_article_nodes → semantic_dense (1024) + semantic_sparse
   ▼
S[Điều2_V1][Điều2_V2] = 0.5·cosine(0.97)      ← gần như giống ngữ nghĩa
                      + 0.25·JW("đối tượng áp dụng","đối tượng áp dụng")=1.0
                      + 0.10·ordinal(2/14, 2/14)=1.0
                      + 0.15·sparse_overlap(0.92)
                      + 0.15·(cùng article_number "2")
                      → clamp = 0.9821   ≥ θ_match (0.60) ✓
   ▼
Hungarian chọn cặp (Điều2_V1, Điều2_V2)   → MATCHED, conf=0.9821
```

---

## 12. Nguyên tắc bất biến (Design invariants)

1. **Tổng trọng số = 1.0** — `AlignmentConfig` validator ép buộc; thêm feature mới
   phải rebalance.
2. **`article_number` bonus cộng SAU weighted sum rồi clamp** — không được đưa vào
   tổng trọng số (nó là tín hiệu nhị phân, không phải similarity).
3. **Hungarian trước, split/merge sau** — luôn match 1-1 tối ưu trước, rồi mới
   xét tách/gộp trên phần unmatched (tránh nhận diện nhầm split khi thực ra match).
4. **Mỗi node chỉ tham gia 1 pair** — `used_v1`/`used_v2` set trong split/merge.
5. **`DiffPair` consistency** — validator ép số ID khớp `match_type`.
6. **Threshold audit trail** — `DiffPairCatalog` lưu `match_threshold` +
   `split_merge_threshold` đã dùng (reproducibility).
7. **Qdrant optional** — alignment phải chạy được không cần Qdrant (in-memory).
8. **Cosine trên vector đã L2-normalize** — split/merge dùng `embed_texts_semantic`
   (đã normalize) để `dot product = cosine`.

---

## 13. Cách chạy & mở rộng

### Chạy trong pipeline đầy đủ (cách production dùng)

```python
from src.pipeline import LegalDiffPipeline
pipeline = LegalDiffPipeline.from_config()
result = pipeline.run(file_v1="data_test/01-tand_signed_v1.docx",
                      file_v2="data_test/01-tand_signed_v2.docx")
catalog = result["catalog"]
print(catalog.summary())
```

### Chạy riêng Phase 2 (debug)

```python
from src.ingestion import LegalDocumentParser
from src.alignment import LegalAlignmentEngine, AlignmentConfig, BGEM3Manager

parser = LegalDocumentParser()
doc_v1 = parser.parse("data_test/01-tand_signed_v1.docx")
doc_v2 = parser.parse("data_test/01-tand_signed_v2.docx")

engine = LegalAlignmentEngine(
    embed_manager=BGEM3Manager(use_fp16=True),
    config=AlignmentConfig(w_semantic=0.5, w_jaro_winkler=0.25,
                           w_ordinal=0.1, w_sparse=0.15),
)
catalog = engine.align_documents(doc_v1, doc_v2, collection_name=None)

import json
print(json.dumps(catalog.to_report_dict(), ensure_ascii=False, indent=2))
```

### Dependencies

```bash
pip install qdrant-client FlagEmbedding jellyfish scipy numpy torch
```

| Package | Mục đích |
|---------|----------|
| `FlagEmbedding` | BAAI/bge-m3 (dense + sparse, FP16) |
| `qdrant-client` | Vector DB multi-vector (optional) |
| `scipy` | `linear_sum_assignment` (Hungarian) |
| `jellyfish` | Jaro-Winkler string distance |
| `numpy` | Matrix operations |
| `torch` | Backend cho BGE-M3 (GPU/CPU) |

### Hướng mở rộng

- **Clause-level alignment** — `embed_clause_nodes` đã sẵn sàng; thêm bước đệ quy
  trong `align_documents` để match Khoản trong từng pair MATCHED (tăng granularity
  cho Phase 3, nhưng tăng chi phí O(n³)).
- **Tuning trọng số** — dùng `evaluation/` (golden dataset) để grid-search
  `w_*` và 2 threshold cho tối ưu F1.
- **ColBERT** — BGE-M3 còn output `colbert_vecs` (late-interaction), chưa dùng;
  có thể thêm thành phần similarity thứ 5 cho cặp khó.

### Test artifacts tham khảo

| File | Nội dung |
|------|----------|
| `tests/e2e_backend/outputs/*/13_catalog.json` | `DiffPairCatalog` thật (14 matched + 2 added) |
| `tests/tand_signed_comparison_output/13_catalog.json` | Catalog cho `01-tand_signed` |
| `data_test/*_v1.docx` / `*_v2.docx` | Cặp văn bản đầu vào |
| `data_test/*.json` | Golden diff (ground truth để đánh giá matching) |

---

*Phiên bản tài liệu này được viết lại cho khớp với mã nguồn hiện hành tại
`src/alignment/`. Phiên bản cũ trong git từng tham chiếu sai package `comparison/`
(với các file `models.py`, `embedding_manager.py`, `evaluate_phase2.py` không tồn tại)
và mô tả một bước "hierarchical clause alignment" không có trong engine hiện tại.
Phase 2 thực sự match ở **cấp Điều**; các file thật là
`alignment_engine.py`, `diff_catalog.py`, `embedder.py`, `hungarian_matcher.py`,
`qdrant_indexer.py`, `similarity_matrix.py`.*
