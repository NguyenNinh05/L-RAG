# LegalDiff — Hệ thống AI Đối chiếu Văn bản Pháp lý

> **Dành cho AI assistant đọc file này:** Đây là tài liệu context đầy đủ cho dự án. Trước khi viết bất kỳ dòng code nào, hãy đọc toàn bộ file này. Mọi quyết định kiến trúc đều có lý do được giải thích bên dưới. Không tự ý thay thế component bằng thứ khác mà không hiểu rõ trade-off.

---

## 1. Bài toán cốt lõi

**Input:** 2 phiên bản của cùng một văn bản pháp lý tiếng Việt (V1 = bản gốc, V2 = bản sửa đổi). Định dạng: PDF hoặc DOCX.

**Output:** Biên bản so sánh có cấu trúc, liệt kê **mọi** sự thay đổi từ nhỏ nhất (sửa 1 con số, đổi 1 từ) đến lớn nhất (thêm/xóa/đảo vị trí toàn bộ một Điều khoản), kèm trích dẫn nguyên văn từ cả hai phiên bản.

**Ràng buộc tuyệt đối:**
- Chạy **100% offline/local** — không gọi bất kỳ API bên ngoài nào (OpenAI, Anthropic, Google, v.v.)
- **Zero Hallucination** — mọi thay đổi được báo cáo phải có evidence trích dẫn nguyên văn từ source. Nếu không tìm thấy evidence, không báo cáo.
- Bảo mật dữ liệu — văn bản pháp lý của doanh nghiệp không được rời khỏi máy chủ nội bộ.

**Hardware:** NVIDIA RTX 4090 (24GB VRAM), CPU và RAM dồi dào.

---

## 2. Tại sao đây KHÔNG phải RAG thông thường

Đây là điểm quan trọng nhất để hiểu đúng kiến trúc. **Đừng thiết kế hệ thống này theo pattern RAG truyền thống.**

| Pattern | RAG truyền thống | GraphRAG (Microsoft) | **Kiến trúc này** |
|---|---|---|---|
| Input | 1 corpus + 1 query | 1 corpus nhiều tài liệu | Đúng 2 tài liệu V1, V2 |
| Cơ chế chính | ANN search (query → top-k chunks) | Entity extraction + graph traversal | Bipartite alignment (Hungarian) |
| Có "query" không? | Có | Có (global query) | **Không** |
| Vai trò LLM | Synthesize — tạo câu trả lời | Summarize — tóm tắt community | **Report only** — chỉ transcribe diff |
| Reranker cần không? | Có (lọc top-k) | Đôi khi | **Không cần** (xem lý do mục 4.2) |

**Tên chính xác của pattern này:** `Pairwise Document Intelligence Pipeline`

Toàn bộ "retrieval" xảy ra ở **build time** (khi nạp V1+V2), không phải runtime. Không có người dùng "hỏi" gì — hệ thống chủ động so sánh tất cả các cặp article.

---

## 3. Kiến trúc End-to-End

```
[V1: PDF/DOCX]  [V2: PDF/DOCX]
      │                │
      ▼                ▼
┌─────────────────────────────────┐
│  PHASE 1: Ingestion             │
│  Docling Parser → LSU Chunker   │
│  Output: Legal DOM (JSON)       │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  PHASE 2: Alignment             │
│  BGE-M3 Embed → Similarity      │
│  Matrix → Hungarian Matching    │
│  Output: Diff Pair Catalog      │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  PHASE 3: Generative Comparison │
│  Qwen2.5-32B (local) → ACU      │
│  Prompting → Self-Verify        │
│  Output: Biên bản JSON + PDF    │
└─────────────────────────────────┘
```

---

## 4. Chi tiết từng Phase

### Phase 1 — Ingestion & Knowledge Representation

#### 4.1 Document Parsing

**Tool chính:** [Docling](https://github.com/DS4SD/docling) (IBM Research)

**Tại sao Docling, không phải pdfminer/pypdf/python-docx?**
- Docling dùng vision model để phân tích layout, không chỉ extract text thuần
- Tái tạo reading order chính xác kể cả với multi-column layout
- Extract bảng ra dạng JSON có cấu trúc (không phải text phẳng)
- Hỗ trợ tốt văn bản tiếng Việt

**Fallback cho PDF scan:** [Marker](https://github.com/VikParuchuri/marker) với Surya OCR engine

**Quan trọng — Xử lý bảng biểu:**
```python
# ĐÚNG: Bảng lưu dưới dạng JSON có cấu trúc
table_node = {
    "type": "table",
    "headers": ["Loại hàng", "Đơn giá", "Số lượng"],
    "rows": [
        ["Hàng A", "50.000", "100"],
        ["Hàng B", "75.000", "200"]
    ]
}

# SAI: Không stringify bảng thành text
# "Loại hàng | Đơn giá | Số lượng\nHàng A | 50.000 | 100"
```
Lý do: so sánh bảng phải là cell-by-cell, không phải text diff.

#### 4.2 Legal Semantic Unit (LSU) — Đơn vị Chunking

**Không bao giờ chunk theo số ký tự hoặc token cố định.**

Chunking theo cấu trúc pháp lý Việt Nam:

```
Văn bản
├── Chương I
│   ├── Điều 1 (LSU cấp 1 — đơn vị so sánh chính)
│   │   ├── Khoản 1 (LSU cấp 2 — khi Điều > 800 tokens)
│   │   │   ├── Điểm a (LSU cấp 3 — so sánh chi tiết nhất)
│   │   │   └── Điểm b
│   │   └── Khoản 2
│   └── Điều 2
└── Chương II
```

**Schema của mỗi LSU node:**
```python
class LSUNode(BaseModel):
    node_id: str           # "v1_ch1_art5_cl3_pt_a"
    version: str           # "V1" | "V2"
    type: str              # "chapter" | "article" | "clause" | "point" | "table"
    ordinal: str           # "5" (số gốc trong văn bản, giữ nguyên dạng string)
    title: str             # "Điều 5. Quyền và nghĩa vụ của bên mua"
    raw_text: str          # Nội dung nguyên văn, không chỉnh sửa
    context_header: str    # "[Chương II > Điều 5 > Khoản 3]" — context cho embedding
    parent_id: str | None
    children_ids: list[str]
    page_number: int
    char_span: tuple[int, int]  # (start, end) trong raw document
    metadata: dict         # bounding_box, font_info, v.v.
```

**Quy tắc context_header:** Mọi LSU cấp thấp đều mang context của cấp cha khi embedding:
```
Embedding input = "[Chương II > Điều 5 > Khoản 3] Bên mua có nghĩa vụ thanh toán..."
```
Không embed naked text — embedding model cần biết "Khoản 3" này thuộc về điều khoản nào.

#### 4.3 Knowledge Store — Hybrid: Graph + Vector

**Graph engine:** Neo4j (local) hoặc Kuzu (embedded, lightweight hơn)

**Mục đích của Graph** (không phải để retrieval — đây là điểm khác với GraphRAG):
- Lưu cấu trúc phân cấp (parent → children relationships)
- Lưu cross-references nội bộ ("Điều 5 tham chiếu Điều 3")
- Lưu CrossDoc edges — kết quả của Phase 2 (V1.Điều5 ↔ V2.Điều7)
- Audit trail cho toàn bộ pipeline

**Vector store:** Qdrant (local, không cần cloud)

Mỗi LSU node có **2 embedding riêng biệt:**
```python
# structural_embed: chỉ embed title + ordinal
# Dùng cho: matching theo vị trí/tiêu đề
structural_embed = embed(f"Điều {node.ordinal}. {node.title}")

# semantic_embed: embed full text với context
# Dùng cho: matching theo nội dung semantic
semantic_embed = embed(f"{node.context_header} {node.raw_text}")
```

---

### Phase 2 — Indexing & Alignment

#### 4.4 Embedding Model

**Model:** `BAAI/BGE-M3`

**Tại sao BGE-M3:**
- Hỗ trợ tiếng Việt (multilingual, trained trên nhiều ngôn ngữ)
- Cung cấp đồng thời 3 loại representation trong 1 forward pass:
  - Dense vector (cosine similarity — semantic matching)
  - Sparse vector (BM25-style — keyword matching)
  - ColBERT multi-vector (late-interaction — fine-grained matching)
- Kích thước ~570MB, chạy FP16 trên GPU chỉ tốn ~2GB VRAM

**Không dùng reranker (cross-encoder) trong main pipeline — lý do:**
1. Không có "candidate pool" lớn để rerank. Ma trận N×M (40×45 = 1.800 ô) được Hungarian algorithm đọc toàn bộ — không bỏ sót candidate nào.
2. ColBERT của BGE-M3 là late-interaction model, về bản chất đã làm việc tương đương cross-encoder nhưng nhanh hơn nhiều.
3. Hungarian algorithm là exact solver cho bipartite matching — không phải heuristic cần rerank.

**Exception:** Có thể thêm cross-encoder nhỏ (`ms-marco-MiniLM-L-6-v2`, ~22MB) cho bước phân tích split/merge articles — xem mục 4.5.

#### 4.5 Alignment Algorithm

**Bước 1 — Xây Similarity Matrix:**
```python
# S[i][j] = score giữa V1.article[i] và V2.article[j]
S[i][j] = (
    0.6 * cosine_similarity(semantic_embed_v1[i], semantic_embed_v2[j])
  + 0.3 * jaro_winkler(title_v1[i], title_v2[j])
  + 0.1 * (1 - abs(i/N - j/M))  # ordinal proximity
)
```

**Bước 2 — Hungarian Algorithm (Optimal Bipartite Matching):**
```python
from scipy.optimize import linear_sum_assignment
row_ind, col_ind = linear_sum_assignment(-S)  # maximize score
```

**Bước 3 — Threshold & Classification:**
```python
THRESHOLD = 0.65  # Tunable — xem Evaluation Metrics

for i, j in zip(row_ind, col_ind):
    if S[i][j] >= THRESHOLD:
        match_type = "matched"
    else:
        match_type = "added" | "deleted"  # xác định sau
```

**Bước 4 — Split/Merge Detection (Edge Cases):**
Sau Hungarian, với các unmatched articles, kiểm tra 1-to-many matching:
```python
# Ví dụ: V1.Điều3 bị tách thành V2.Điều3a + V2.Điều3b
combined_embed = avg(embed(v2.art3a.text + v2.art3b.text))
if cosine_similarity(v1.art3.semantic_embed, combined_embed) > 0.80:
    match_type = "split"
```
Với edge cases này, có thể dùng cross-encoder nhỏ để verify thêm.

**Bước 5 — Hierarchical Alignment:**
Với mỗi cặp `matched` articles, lặp lại toàn bộ quy trình cho cấp Khoản, sau đó cấp Điểm.

**Output của Phase 2 — Diff Pair Catalog:**
```python
class DiffPair(BaseModel):
    pair_id: str
    v1_node_id: str
    v2_node_id: str | None          # None nếu bị xóa
    match_type: Literal[
        "matched",      # nội dung tương đương, có thể có thay đổi nhỏ
        "added",        # article mới trong V2
        "deleted",      # article bị xóa khỏi V2
        "split",        # 1 article V1 → nhiều articles V2
        "merged",       # nhiều articles V1 → 1 article V2
        "reordered"     # nội dung giống nhưng vị trí khác
    ]
    confidence: float               # 0.0 – 1.0
    sub_pairs: list["DiffPair"]     # cấp Khoản, Điểm
```

---

### Phase 3 — Generative Comparison

#### 4.6 Local LLM

**Model:** `Qwen2.5-7B-Instruct` 


**Serving:** `llama.cpp server`
```bash
./server \
  --model qwen2.5-7b-instruct \
  --n-gpu-layers 999 \    # full GPU offload
  --ctx-size 8192 \
  --parallel 2 \          # 2 article pairs song song
  --host 0.0.0.0 \
  --port 8080
```


#### 4.7 Prompting Architecture — Zero Hallucination by Design

Toàn bộ thiết kế prompt nhằm **constrain LLM chỉ report những gì có trong input**.

**Tầng 1 — Atomic Comparison Unit (ACU) Prompt:**

Mỗi lần gọi LLM chỉ xử lý **một cặp article đã matched**. Không bao giờ feed toàn bộ document.

```python
SYSTEM_PROMPT = """
Bạn là công cụ so sánh văn bản pháp lý chính xác.

QUY TẮC BẮT BUỘC:
1. Chỉ báo cáo những gì THỰC SỰ khác nhau giữa <V1> và <V2>
2. Không diễn giải ý nghĩa pháp lý
3. Không suy đoán ý định của các bên
4. Mọi change phải có verbatim_evidence trích dẫn nguyên văn từ source
5. Nếu hai đoạn giống nhau hoàn toàn, trả về changes = []
6. Output phải là JSON hợp lệ, không có text nào ngoài JSON
"""

USER_PROMPT = """
<V1_ARTICLE id="{v1_node_id}">
{v1_raw_text}
</V1_ARTICLE>

<V2_ARTICLE id="{v2_node_id}">
{v2_raw_text}
</V2_ARTICLE>

Trả về JSON với schema sau, không thêm bất kỳ text nào khác:
{
  "pair_id": "{pair_id}",
  "changes": [
    {
      "change_id": "unique_id",
      "change_type": "numerical|terminology|structural|addition|deletion|reorder",
      "location_v1": "mô tả vị trí trong V1 (e.g. 'Khoản 2, câu đầu')",
      "location_v2": "mô tả vị trí trong V2",
      "original_value": "giá trị gốc trong V1",
      "new_value": "giá trị mới trong V2",
      "verbatim_evidence_v1": "đoạn trích nguyên văn từ V1 chứa thay đổi",
      "verbatim_evidence_v2": "đoạn trích nguyên văn từ V2 chứa thay đổi",
      "confidence": 0.95
    }
  ],
  "no_change": false
}
"""
```

**Tầng 2 — Self-Verification Pass:**

Sau khi collect ACU outputs, chạy verification:
```python
def verify_evidence(change: ACUChange, v1_text: str, v2_text: str) -> bool:
    """
    Kiểm tra verbatim_evidence có thực sự xuất hiện trong source không.
    Nếu không tìm thấy → hallucination → loại bỏ change này.
    """
    v1_found = change.verbatim_evidence_v1 in v1_text
    v2_found = change.verbatim_evidence_v2 in v2_text
    return v1_found and v2_found
```

**Tầng 3 — Numerical Verification (Deterministic):**

```python
import re

def verify_numerical_change(change: ACUChange, v1_text: str, v2_text: str) -> bool:
    """
    KHÔNG tin tưởng LLM cho thay đổi số liệu.
    Dùng regex để verify độc lập.
    """
    if change.change_type != "numerical":
        return True  # skip — chỉ verify numerical

    pattern_v1 = re.escape(change.original_value)
    pattern_v2 = re.escape(change.new_value)

    return (
        bool(re.search(pattern_v1, v1_text)) and
        bool(re.search(pattern_v2, v2_text))
    )
```

Numerical verification phải đạt **100%** — nếu regex không tìm thấy, change bị drop dù LLM tự tin.

**Tầng 4 — Report Aggregation:**

LLM được gọi thêm một lần để sinh executive summary tiếng Việt, nhưng với constraint:
- System prompt yêu cầu: "Chỉ dùng thông tin từ ACU outputs được cung cấp"
- Input là structured ACU JSON, không phải raw document
- Không được thêm nhận xét chủ quan, đánh giá, hay khuyến nghị

---

## 5. Output Schema — Biên bản So sánh

```python
class ComparisonReport(BaseModel):
    report_id: str
    generated_at: datetime
    v1_metadata: DocumentMetadata
    v2_metadata: DocumentMetadata

    summary: ReportSummary           # Tổng quan: X thay đổi, Y articles affected
    diff_pairs: list[DiffPair]       # Toàn bộ pairs với ACU changes
    added_articles: list[LSUNode]    # Articles mới trong V2
    deleted_articles: list[LSUNode]  # Articles bị xóa khỏi V2
    structural_changes: list[str]    # Split/merge/reorder events

    # Audit trail
    alignment_matrix_path: str       # Path đến similarity matrix (numpy)
    acu_outputs_path: str            # Path đến raw ACU JSON trước verification
    verification_log_path: str       # Log của verify steps

class ACUChange(BaseModel):
    change_id: str
    change_type: Literal["numerical", "terminology", "structural", "addition", "deletion", "reorder"]
    location_v1: str
    location_v2: str
    original_value: str
    new_value: str
    verbatim_evidence_v1: str        # Trích dẫn nguyên văn từ V1
    verbatim_evidence_v2: str        # Trích dẫn nguyên văn từ V2
    confidence: float
    verified: bool                   # True nếu pass self-verification
    numerical_verified: bool | None  # True/False/None (None nếu không phải numerical)
```

---

## 6. Evaluation Metrics

### Phase 1 — Ingestion Quality
| Metric | Mục tiêu | Cách đo |
|---|---|---|
| Structure Preservation Rate (SPR) | ≥ 98% text-based, ≥ 90% scanned | So sánh LSU count với DOCX XML ground truth |
| Table Cell Accuracy | ≥ 95% | Cell-by-cell comparison |
| Ordinal Accuracy | 100% | Kiểm tra số thứ tự Điều/Khoản |
| Reading Order Score | ≥ 0.95 | Kendall's Tau với human annotation |

### Phase 2 — Alignment Quality
| Metric | Mục tiêu | Cách đo |
|---|---|---|
| Alignment Precision | ≥ 0.95 | Matched pairs đúng / tổng matched |
| Alignment Recall | ≥ 0.92 | Đúng được tìm thấy / tổng đúng |
| F1-Alignment | ≥ 0.93 | Harmonic mean |
| Split/Merge Detection Rate | ≥ 0.85 | Trên hard test cases |

### Phase 3 — Generation Quality
| Metric | Mục tiêu | Cách đo |
|---|---|---|
| False Positive Rate (FPR) | < 1% | Changes báo cáo sai / tổng changes |
| Change Detection Recall | ≥ 0.95 | Changes thực tế tìm thấy / tổng thực tế |
| Citation Accuracy | 100% | verbatim_evidence tìm thấy trong source |
| Numerical Change Accuracy | 100% | Regex verify |
| Faithfulness Score (RAGAS) | ≥ 0.90 | Claims supported by source |

**FPR là metric quan trọng nhất về mặt pháp lý.** Báo cáo nhầm một thay đổi không tồn tại có thể gây hậu quả nghiêm trọng hơn bỏ sót một thay đổi nhỏ.

---

## 7. Golden Dataset — Tạo Test Data Tự động

```python
# Tạo "V2 giả" programmatically từ V1 thực tế
# Script này generate ground truth labels tự động — zero human labeling

MUTATION_OPS = [
    ("reorder",     reorder_articles),         # Đảo vị trí 2-3 Điều ngẫu nhiên
    ("rename",      rename_ordinals),           # Điều 3 → Điều 5
    ("substitute",  substitute_terms),          # Thay keyword pháp lý
    ("add",         add_articles),              # Thêm 1-2 Điều mới
    ("delete",      delete_articles),           # Xóa 1-2 Điều
    ("split",       split_article),             # Tách 1 Điều thành 2
    ("merge",       merge_articles),            # Gộp 2 Điều thành 1
    ("number",      mutate_numbers),            # Đổi số tiền/ngày/%
    ("typography",  fix_typography),            # "hai mươi triệu" ↔ "20 triệu"
]

def generate_test_pair(v1_doc: Document) -> tuple[Document, GroundTruth]:
    """
    Apply ngẫu nhiên 1-5 mutations lên V1 để tạo V2.
    Ghi lại toàn bộ mutations thành ground truth.
    Trả về (v2_doc, ground_truth_change_log).
    """
    ...

# Trap cases — phải NOT báo cáo là thay đổi
TRAP_CASES = [
    ("hai mươi triệu đồng", "20.000.000 đồng"),   # Tương đương ngữ nghĩa
    ("Bên A", "Bên mua"),                           # Nếu đã định nghĩa tương đương
    ("ngày 01/01/2024", "ngày 1 tháng 1 năm 2024"), # Format khác, giá trị giống
]
```

---

## 8. Tech Stack

```
Parser:         Docling (primary) + Marker/Surya (fallback)
Embedding:      BAAI/BGE-M3 (FP16 on GPU, ~2GB VRAM)
Vector Store:   Qdrant (local, no cloud)
Graph DB:       Kuzu (embedded) hoặc Neo4j (local)
Alignment:      scipy.optimize.linear_sum_assignment (Hungarian)
String Match:   jellyfish (JaroWinkler for title matching)
LLM Server:     llama.cpp server
LLM Model:      Qwen2.5-32B-Instruct Q4_K_M (~21GB VRAM)
API Layer:      FastAPI
Verification:   Python regex + custom evidence checker
Report Output:  JSON (machine-readable) + WeasyPrint/ReportLab (PDF)
```

**Tổng VRAM usage:**
- BGE-M3 (FP16): ~2GB
- Qwen2.5-32B Q4_K_M: ~21GB
- KV Cache + overhead: ~1GB
- **Tổng: ~24GB — vừa đủ RTX 4090**

Không chạy đồng thời embedding và LLM inference. Pipeline là sequential: embed toàn bộ trước → align → rồi mới chạy LLM.

---

## 9. Cấu trúc Thư mục Project

```
legaldiff/
├── README.md                    # File này
├── configs/
│   ├── model_config.yaml        # Model paths, quantization settings
│   ├── pipeline_config.yaml     # Thresholds, batch sizes
│   └── prompt_templates/
│       ├── acu_system.txt
│       ├── acu_user.txt
│       └── summary_user.txt
├── src/
│   ├── ingestion/
│   │   ├── docling_parser.py    # Phase 1: Parse PDF/DOCX
│   │   ├── lsu_chunker.py       # Phase 1: Legal Semantic Unit chunking
│   │   └── knowledge_store.py   # Phase 1: Graph + Vector storage
│   ├── alignment/
│   │   ├── embedder.py          # Phase 2: BGE-M3 embedding
│   │   ├── similarity_matrix.py # Phase 2: Build N×M matrix
│   │   ├── hungarian_matcher.py # Phase 2: Optimal matching
│   │   └── diff_catalog.py      # Phase 2: DiffPair output
│   ├── comparison/
│   │   ├── acu_prompter.py      # Phase 3: ACU prompt generation
│   │   ├── llm_client.py        # Phase 3: llama.cpp HTTP client
│   │   ├── verifier.py          # Phase 3: Self-verify + numerical verify
│   │   └── report_generator.py  # Phase 3: JSON + PDF output
│   └── pipeline.py              # Orchestrator — chạy 3 phases
├── evaluation/
│   ├── golden_dataset_gen.py    # Tạo test data tự động
│   ├── metrics.py               # Tính SPR, F1, FPR, v.v.
│   └── ragas_eval.py            # Faithfulness scoring
├── tests/
│   ├── test_ingestion.py
│   ├── test_alignment.py
│   └── test_comparison.py
└── data/
    ├── raw/                     # V1, V2 input files
    ├── processed/               # LSU JSON DOM
    ├── golden/                  # Ground truth test pairs
    └── reports/                 # Output biên bản
```

---

## 10. Những điều AI assistant KHÔNG được làm

Khi làm việc trong project này:

1. **Không thay Qdrant bằng ChromaDB/FAISS** mà không hỏi — Qdrant được chọn vì hỗ trợ multi-vector per point (dense + sparse + ColBERT cùng lúc).

2. **Không thay Hungarian bằng greedy matching** — greedy không đảm bảo global optimum, dẫn đến cascade errors trong hierarchical alignment.

3. **Không cho LLM đọc toàn bộ document** — luôn feed từng cặp article đã matched. Context window 8K là đủ cho một cặp article pháp lý.

4. **Không bỏ qua bước numerical verification bằng regex** — LLM có thể hallucinate số liệu kể cả khi confidence = 0.99.

5. **Không call bất kỳ external API nào** — kể cả cho mục đích logging, monitoring, hay telemetry. Mọi thứ phải local.

6. **Không stringify bảng thành text** khi parse — mất cấu trúc cell, không thể so sánh chính xác.

7. **Không chunk theo token count** — phải chunk theo LSU (Điều/Khoản/Điểm) dù LSU dài hay ngắn đến đâu.

---

## 11. Câu hỏi thường gặp khi implement

**Q: "Điều X của V1" không match với "Điều X của V2" vì nội dung thay đổi quá nhiều — làm thế nào?"**

A: Hungarian algorithm dùng similarity score, không phải ordinal. Nếu "Điều 5 V1" và "Điều 7 V2" có content gần nhau hơn "Điều 5 V1" và "Điều 5 V2", chúng sẽ được match với nhau. Đây là mục đích của semantic embedding trong similarity matrix.

**Q: Văn bản V1 có 60 Điều, V2 có 55 Điều — Hungarian xử lý thế nào với ma trận không vuông?"**

A: `scipy.optimize.linear_sum_assignment` xử lý tốt rectangular matrix. 5 Điều thừa của V1 sẽ là unmatched → được classify là "deleted". Padding với zeros không cần thiết.

**Q: Confidence threshold 0.65 có hợp lý không?"**

A: Đây là starting point. Phải tune trên golden dataset của bạn. Vẽ Precision-Recall curve, chọn điểm Fbeta với beta=0.5 (precision-focused vì FPR quan trọng hơn recall trong bối cảnh pháp lý).

**Q: Làm sao handle "hai mươi triệu" vs "20.000.000"?"**

A: Trong Phase 1, sau khi parse, chạy normalization layer để convert số viết bằng chữ sang số. Tuy nhiên, giữ raw_text nguyên vẹn — normalized text chỉ dùng cho embedding, không dùng cho evidence citation.

---

*Cập nhật lần cuối: April 2026*
*Kiến trúc: Pairwise Document Intelligence Pipeline*
*Pattern tham khảo: không phải Traditional RAG, không phải GraphRAG*