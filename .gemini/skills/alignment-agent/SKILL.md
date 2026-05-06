# ALIGNMENT AGENT — Phase 2: Indexing & Alignment

## Identity
Bạn là chuyên gia về **semantic alignment và optimal matching** cho văn bản pháp lý.
Nhiệm vụ của bạn là giải quyết bài toán cốt lõi nhất của pipeline:
**ghép cặp chính xác Article X của V1 với Article Y của V2**, kể cả khi chúng bị đảo vị trí, thay đổi từ ngữ, hoặc bị tách/gộp.

---

## Bài toán cụ thể

```
INPUT:
  V1: [Art1, Art2, Art3, Art4, Art5]   (N articles)
  V2: [Art2, Art5, Art3_new, Art6]     (M articles — đã bị reorder + thêm/xóa)

OUTPUT (Diff Pair Catalog):
  matched:   V1.Art2 ↔ V2.Art1  (confidence: 0.94)
  matched:   V1.Art5 ↔ V2.Art2  (confidence: 0.91)
  deleted:   V1.Art1             (không có counterpart trong V2)
  deleted:   V1.Art4             (không có counterpart trong V2)
  matched:   V1.Art3 ↔ V2.Art3  (confidence: 0.82 — có thay đổi nội dung)
  added:     V2.Art4             (mới trong V2, không có trong V1)
```

---

## Tại sao KHÔNG dùng Reranker trong pipeline chính

Đây là câu hỏi thường gặp — hiểu rõ để không tự ý thêm vào:

1. **Không có candidate pool lớn để rerank.** Ma trận N×M (ví dụ 40×45=1.800 ô) được Hungarian algorithm đọc toàn bộ. Không có bước "lấy top-k" nào bị bỏ sót.

2. **ColBERT của BGE-M3 đã là late-interaction scoring.** Reranker cross-encoder cũng là late-interaction. Dùng cả hai là redundant.

3. **Hungarian algorithm không phải ranking** — nó là exact solver cho optimal bipartite matching. Reranker không thể thay thế hoặc cải thiện nó.

**Exception hợp lệ duy nhất:** Cross-encoder nhỏ (`ms-marco-MiniLM-L-6-v2`, ~22MB) cho bước verify split/merge articles ở mục 4 bên dưới.

---

## Stack kỹ thuật

```python
# Embedding
from FlagEmbedding import BGEM3FlagModel  # BGE-M3 official wrapper

# Matching
from scipy.optimize import linear_sum_assignment  # Hungarian algorithm

# String similarity (title matching)
import jellyfish  # jaro_winkler_similarity

# Optional: split/merge verification
from sentence_transformers import CrossEncoder  # ms-marco-MiniLM-L-6-v2

# Storage (viết kết quả)
import kuzu       # thêm CrossDoc edges
import json       # serialize DiffPair Catalog
```

---

## Bước 1 — Embedding với BGE-M3

```python
model = BGEM3FlagModel(
    'BAAI/BGE-M3',
    use_fp16=True,      # ~2GB VRAM, đủ margin với Qwen2.5 được load sau
    device='cuda'
)

def embed_lsu_node(node: LSUNode) -> dict:
    """
    Tạo 2 loại embedding riêng biệt cho mỗi LSU node.
    """
    # Structural embed: chỉ title + ordinal
    # Dùng cho: matching theo vị trí/tiêu đề
    structural_text = f"Điều {node.ordinal}. {node.title}"

    # Semantic embed: full text với context header
    # Dùng cho: matching theo nội dung
    semantic_text = f"{node.context_header} {node.raw_text}"

    structural_output = model.encode(
        structural_text,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False  # không cần cho structural
    )

    semantic_output = model.encode(
        semantic_text,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True   # cần ColBERT cho fine-grained matching
    )

    return {
        "node_id": node.node_id,
        "structural_dense": structural_output["dense_vecs"],
        "semantic_dense":   semantic_output["dense_vecs"],
        "semantic_sparse":  semantic_output["lexical_weights"],
        "semantic_colbert": semantic_output["colbert_vecs"],
    }
```

**Quan trọng về VRAM:** BGE-M3 load trước, embed toàn bộ, unload khỏi GPU. Sau đó mới load Qwen2.5 ở Phase 3. Không chạy đồng thời.

---

## Bước 2 — Xây Similarity Matrix

```python
import numpy as np

def build_similarity_matrix(
    v1_articles: list[LSUNode],
    v2_articles: list[LSUNode],
    v1_embeddings: list[dict],
    v2_embeddings: list[dict]
) -> np.ndarray:
    """
    Ma trận N×M: S[i][j] = similarity score giữa v1[i] và v2[j]
    """
    N, M = len(v1_articles), len(v2_articles)
    S = np.zeros((N, M))

    for i, (node_v1, emb_v1) in enumerate(zip(v1_articles, v1_embeddings)):
        for j, (node_v2, emb_v2) in enumerate(zip(v2_articles, v2_embeddings)):

            # Component 1: Semantic similarity (cosine, dense vectors)
            semantic_sim = cosine_similarity(
                emb_v1["semantic_dense"],
                emb_v2["semantic_dense"]
            )

            # Component 2: Title string similarity (JaroWinkler)
            title_sim = jellyfish.jaro_winkler_similarity(
                node_v1.title.lower(),
                node_v2.title.lower()
            )

            # Component 3: Ordinal proximity (normalized distance)
            ordinal_prox = 1 - abs(i/N - j/M)

            # Weighted combination — tune weights trên golden dataset
            S[i][j] = (
                0.60 * semantic_sim +
                0.30 * title_sim +
                0.10 * ordinal_prox
            )

    return S

# WEIGHTS LÀ HYPERPARAMETER — phải tune trên golden dataset
# Default: (0.60, 0.30, 0.10) ưu tiên semantic nhưng có hint từ title và vị trí
```

---

## Bước 3 — Hungarian Algorithm

```python
THRESHOLD = 0.65  # HYPERPARAMETER — tune trên Precision-Recall curve

def run_hungarian_matching(
    S: np.ndarray,
    v1_articles: list[LSUNode],
    v2_articles: list[LSUNode]
) -> list[DiffPair]:
    """
    Tìm optimal bipartite matching.
    scipy xử lý tốt rectangular matrix (N ≠ M).
    """
    row_ind, col_ind = linear_sum_assignment(-S)  # maximize (nên negate)

    pairs = []
    matched_v1 = set()
    matched_v2 = set()

    for i, j in zip(row_ind, col_ind):
        score = S[i][j]

        if score >= THRESHOLD:
            pairs.append(DiffPair(
                v1_node_id=v1_articles[i].node_id,
                v2_node_id=v2_articles[j].node_id,
                match_type="matched",
                confidence=float(score),
                sub_pairs=[]  # filled later in hierarchical alignment
            ))
            matched_v1.add(i)
            matched_v2.add(j)
        # Nếu score < THRESHOLD: cả hai vẫn có thể là added/deleted
        # Xử lý ở bước tiếp theo

    # Unmatched articles
    for i in range(len(v1_articles)):
        if i not in matched_v1:
            pairs.append(DiffPair(
                v1_node_id=v1_articles[i].node_id,
                v2_node_id=None,
                match_type="deleted",
                confidence=1.0,
                sub_pairs=[]
            ))

    for j in range(len(v2_articles)):
        if j not in matched_v2:
            pairs.append(DiffPair(
                v1_node_id=None,
                v2_node_id=v2_articles[j].node_id,
                match_type="added",
                confidence=1.0,
                sub_pairs=[]
            ))

    return pairs
```

---

## Bước 4 — Split/Merge Detection

Áp dụng SAU Hungarian, cho các unmatched articles:

```python
SPLIT_MERGE_THRESHOLD = 0.80  # Cao hơn main threshold vì đây là edge case

def detect_split_merge(
    unmatched_v1: list[LSUNode],
    unmatched_v2: list[LSUNode],
    embeddings: dict
) -> list[DiffPair]:
    """
    Kiểm tra 1-to-many matching cho split/merge cases.

    Split: 1 article V1 → 2+ articles V2
    Merge: 2+ articles V1 → 1 article V2
    """
    # Kiểm tra split: mỗi v1 article vs mọi cặp v2 articles
    for v1_node in unmatched_v1:
        for combo in combinations(unmatched_v2, 2):
            # Combine embeddings của 2 v2 articles
            combined_text = " ".join([n.raw_text for n in combo])
            combined_embed = model.encode(combined_text)["dense_vecs"]

            sim = cosine_similarity(
                embeddings[v1_node.node_id]["semantic_dense"],
                combined_embed
            )

            if sim >= SPLIT_MERGE_THRESHOLD:
                # Optional: verify với cross-encoder
                if cross_encoder_verify(v1_node, combo):
                    yield DiffPair(
                        v1_node_id=v1_node.node_id,
                        v2_node_id=[n.node_id for n in combo],
                        match_type="split",
                        confidence=float(sim),
                        sub_pairs=[]
                    )
```

---

## Bước 5 — Hierarchical Alignment

Với mỗi cặp `matched`, lặp lại toàn bộ quy trình cho cấp Khoản:

```python
def hierarchical_align(pair: DiffPair, all_nodes: dict) -> DiffPair:
    """
    Đệ quy align Khoản và Điểm bên trong mỗi cặp Article đã matched.
    """
    v1_clauses = get_children(pair.v1_node_id, all_nodes, type="clause")
    v2_clauses = get_children(pair.v2_node_id, all_nodes, type="clause")

    if len(v1_clauses) == 0 and len(v2_clauses) == 0:
        return pair  # không có sub-structure

    sub_embeddings = embed_batch(v1_clauses + v2_clauses)
    sub_S = build_similarity_matrix(v1_clauses, v2_clauses, ...)
    sub_pairs = run_hungarian_matching(sub_S, v1_clauses, v2_clauses)

    pair.sub_pairs = sub_pairs
    return pair
```

---

## Output Schema — DiffPair Catalog

```python
from pydantic import BaseModel
from typing import Literal

class DiffPair(BaseModel):
    pair_id: str                    # UUID
    v1_node_id: str | None          # None nếu match_type == "added"
    v2_node_id: str | list[str] | None  # list nếu split/merge
    match_type: Literal[
        "matched",    # có counterpart, có thể có thay đổi nội dung
        "added",      # mới trong V2
        "deleted",    # bị xóa khỏi V2
        "split",      # 1 V1 article → nhiều V2 articles
        "merged",     # nhiều V1 articles → 1 V2 article
        "reordered"   # same content, khác vị trí (detected sau matched)
    ]
    confidence: float
    similarity_score: float         # raw score từ ma trận
    sub_pairs: list["DiffPair"]     # alignment cấp Khoản, Điểm
```

**Output file:** `data/processed/diff_pair_catalog.json`

---

## Viết CrossDoc edges vào Kuzu

```python
def write_cross_doc_edges(catalog: list[DiffPair], db: kuzu.Database):
    CREATE_CROSSDOC = """
        CREATE REL TABLE CROSSDOC(
            FROM LSU TO LSU,
            match_type STRING,
            confidence DOUBLE,
            similarity_score DOUBLE
        )
    """
    for pair in catalog:
        if pair.v1_node_id and pair.v2_node_id:
            conn.execute("""
                MATCH (a:LSU {node_id: $v1}), (b:LSU {node_id: $v2})
                CREATE (a)-[:CROSSDOC {
                    match_type: $mt, confidence: $conf
                }]->(b)
            """, {"v1": pair.v1_node_id, "v2": pair.v2_node_id, ...})
```

---

## Hyperparameters cần tune

| Parameter | Default | Tune bằng cách |
|---|---|---|
| `THRESHOLD` | 0.65 | Precision-Recall curve trên golden dataset |
| `SPLIT_MERGE_THRESHOLD` | 0.80 | Hard test cases (split/merge scenarios) |
| Weights `(0.60, 0.30, 0.10)` | `(semantic, title, ordinal)` | Grid search trên F1-Alignment |

---

## Checklist trước khi bàn giao Phase 3

- [ ] DiffPair Catalog đã serialize vào `data/processed/diff_pair_catalog.json`
- [ ] CrossDoc edges đã được viết vào Kuzu graph
- [ ] F1-Alignment ≥ 0.90 trên golden dataset
- [ ] Tất cả unmatched articles đã được classify đúng (added/deleted/split/merged)
- [ ] Hierarchical alignment đã chạy đến cấp Khoản
- [ ] Similarity matrix đã được lưu (numpy) để audit trail
- [ ] BGE-M3 đã được unload khỏi GPU (giải phóng VRAM cho Qwen2.5)
