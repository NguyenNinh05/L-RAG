# L-RAG (LegalDiff) — Phân tích & Hướng dẫn Đọc hiểu Dự án

## Tổng quan một dòng

**LegalDiff** là hệ thống AI chạy **100% offline** để so sánh hai phiên bản văn bản pháp lý tiếng Việt (V1 vs V2), phát hiện **mọi** thay đổi (từ sửa 1 con số đến thêm/xóa/đảo toàn Điều), và xuất biên bản có trích dẫn nguyên văn với **Zero Hallucination**.

Pattern kiến trúc: **Pairwise Document Intelligence Pipeline** — KHÔNG phải RAG truyền thống, KHÔNG phải GraphRAG.

---

## Thứ tự đọc hiểu dự án (5 lớp)

### 🔵 Lớp 1 — Hiểu bài toán & kiến trúc tổng thể

> Đọc đây trước tất cả. Không hiểu lớp này thì code sẽ vô nghĩa.

| # | File | Mục đích đọc |
|---|------|-------------|
| 1 | `README.md` | **Bắt buộc đọc toàn bộ.** Định nghĩa bài toán, tại sao không dùng RAG thường, sơ đồ 3-Phase, công thức similarity, quy tắc Zero-Hallucination, và danh sách những gì KHÔNG được làm |
| 2 | `configs/pipeline_config.yaml` | Các ngưỡng vận hành: `match_threshold=0.65`, `split_merge_threshold=0.80`, trọng số similarity `(0.6, 0.3, 0.1)` |
| 3 | `configs/model_config.yaml` | Model nào được dùng: BGE-M3 cho embedding, Qwen2.5-7B cho LLM |
| 4 | `src/config.py` | Config loader — `get_config()` merge 2 YAML, expose constants `MATCH_THRESHOLD`, `EMBEDDING_DIM`, v.v. |

---

### 🟢 Lớp 2 — Data Models (nền tảng type system)

> Đọc tiếp theo. Toàn bộ pipeline xoay quanh các model này.

| # | File | Mục đích đọc |
|---|------|-------------|
| 5 | `src/ingestion/models.py` | **File quan trọng nhất về data.** Định nghĩa toàn bộ Legal DOM: `LegalDocument → DocumentSection → ArticleNode → ClauseNode → PointNode`. Thêm `LsuChunk` (đơn vị embedding), `TableData` (bảng có cấu trúc), `GraphNode/GraphEdge` (Kuzu), `VectorRecord/VectorMetadata` (ChromaDB) |
| 6 | `src/alignment/diff_catalog.py` | Data models Phase 2: `DiffPair` (1 cặp matched/added/deleted/split/merged), `DiffPairCatalog` (toàn bộ kết quả alignment), `NodeEmbeddings` (2 vector/node), `QdrantPayload` |
| 7 | `src/comparison/models.py` | Data models Phase 3: `ACUOutput` (1 thay đổi nguyên tử có evidence), `VerificationResult`, `ComparisonReport` (output cuối), `ComparisonRequest` (input LLM) |

---

### 🟡 Lớp 3 — Từng Phase theo thứ tự pipeline

#### Phase 1 — Ingestion

| # | File | Vai trò |
|---|------|---------|
| 8 | `src/ingestion/docling_parser.py` | Parse PDF/DOCX → `LegalDocument`. Dùng Docling (vision model) để giữ layout. Fallback: Marker/Surya OCR cho scan PDF. **Không stringify bảng** — giữ JSON có cấu trúc |
| 9 | `src/ingestion/lsu_chunker.py` | Chunk `LegalDocument` → `list[LsuChunk]` theo cấu trúc pháp lý (Điều/Khoản/Điểm). **Không chunk theo token cố định.** Mỗi chunk có `breadcrumb` prefix: `[Chương II > Điều 5 > Khoản 3]` để embedding có context |
| 10 | `src/ingestion/knowledge_store.py` | `HybridGraphBuilder`: Lưu Legal DOM vào **Kuzu** (graph, cạnh CONTAINS/PRECEDES/REFERENCES) và **ChromaDB** (vector, link về graph qua `node_id`). Trích regex cross-references ("theo quy định tại Điều X") |
| 11 | `src/ingestion/__init__.py` | Public API + `ingest_document()` — convenience function chạy cả 3 module trên |

#### Phase 2 — Alignment

| # | File | Vai trò |
|---|------|---------|
| 12 | `src/alignment/embedder.py` | `BGEM3Manager`: Load `BAAI/BGE-M3` FP16. Tạo **2 loại embedding/node**: `structural_embed` (title+ordinal) và `semantic_embed` (full text + breadcrumb). Batch processing, lazy import FlagEmbedding |
| 13 | `src/alignment/similarity_matrix.py` | Tính ma trận N×M: `S[i][j] = 0.6×Cosine + 0.3×JaroWinkler + 0.1×OrdinalProximity`. Expose `AlignmentConfig`, `NodeRecord`, `compute_similarity_matrix()` |
| 14 | `src/alignment/hungarian_matcher.py` | `hungarian_match()` dùng `scipy.optimize.linear_sum_assignment` → optimal bipartite matching. `detect_split_merge()` kiểm tra 1-to-many cho các unmatched nodes |
| 15 | `src/alignment/qdrant_indexer.py` | `QdrantManager`: Lưu embeddings vào Qdrant local (multi-vector: dense + sparse/BM25 per point). Hỗ trợ upsert batch |
| 16 | `src/alignment/diff_catalog.py` | *(đã đọc ở Lớp 2)* — Ghi nhớ: đây là **output** của Phase 2 |
| 17 | `src/alignment/alignment_engine.py` | `LegalAlignmentEngine`: **Orchestrator Phase 2**. 9 bước: thu thập articles → embed → lưu Qdrant → build NodeRecord → similarity matrix → Hungarian → build DiffPairs → split/merge detection → classify ADDED/DELETED → `DiffPairCatalog` |
| 18 | `src/alignment/__init__.py` | Public API của alignment package |

#### Phase 3 — Generative Comparison

| # | File | Vai trò |
|---|------|---------|
| 19 | `configs/prompt_templates/acu_system.txt` | System prompt cho LLM: quy tắc bắt buộc — chỉ report thay đổi thực sự, cần verbatim evidence, output JSON thuần |
| 20 | `configs/prompt_templates/acu_user.txt` | User prompt template: inject `<V1_ARTICLE>`, `<V2_ARTICLE>`, yêu cầu JSON schema ACU |
| 21 | `configs/prompt_templates/summary_user.txt` | Prompt cho Tầng 4 — sinh executive summary từ ACU JSON đã verified |
| 22 | `src/comparison/llm_client.py` | `LocalLLMClient`: HTTP client gọi llama.cpp server (`localhost:8000/v1`). Hỗ trợ async, retry, parse JSON response |
| 23 | `src/comparison/acu_prompter.py` | `ACUPrompter`: Build prompt từ `ComparisonRequest` + templates. Inject breadcrumb, truncate nếu vượt context window |
| 24 | `src/comparison/verifier.py` | `VerificationEngine`: **4 tầng verify**. Tầng 2: exact/fuzzy substring match evidence. Tầng 3: regex verify số liệu (100% required). Loại bỏ ACU hallucinate |
| 25 | `src/comparison/report_generator.py` | `GenerativeComparisonPipeline`: Orchestrator Phase 3. Async batch LLM calls → collect ACU → verify → Tầng 4 summary → `ComparisonReport` (JSON + Markdown) |
| 26 | `src/comparison/__init__.py` | Public API, lazy import cho LLM-dependent modules |

---

### 🟠 Lớp 4 — Orchestrator & Entry Point

| # | File | Vai trò |
|---|------|---------|
| 27 | `src/pipeline.py` | `LegalDiffPipeline`: **Main entry point** toàn hệ thống. `pipeline.run(file_v1, file_v2)` chạy tuần tự Phase 1 → 2 → 3. `PipelineRunConfig` gom tất cả config. Có `skip_phase3=True` để debug |
| 28 | `src/__init__.py` | Package init |

---

### 🔴 Lớp 5 — Testing & Evaluation

| # | File | Vai trò |
|---|------|---------|
| 29 | `tests/test_ingestion.py` | Unit test Phase 1: parse + chunk |
| 30 | `tests/test_alignment.py` | Unit test Phase 2: embedding, similarity matrix, Hungarian |
| 31 | `tests/test_comparison.py` | Unit test Phase 3: verifier, report generation |
| 32 | `tests/test_lsu_chunker.py` | Test chi tiết LsuChunker: breadcrumb, sentence splitting |
| 33 | `tests/test_knowledge_store.py` | Test HybridGraphBuilder: Kuzu + ChromaDB |
| 34 | `tests/test_docling_parser_all.py` | Integration test parser nhiều file |
| 35 | `evaluation/golden_dataset_gen.py` | Sinh "V2 giả" từ V1 bằng 9 mutation ops (reorder, add, delete, split, merge, substitute...) — zero human labeling |
| 36 | `evaluation/metrics.py` | Tính SPR, F1-Alignment, FPR, Citation Accuracy, Faithfulness |
| 37 | `evaluation/ragas_eval.py` | RAGAS Faithfulness scoring |
| 38 | `tests/evaluate_phase1.py` | End-to-end eval Phase 1 |
| 39 | `tests/evaluate_phase2.py` | End-to-end eval Phase 2 |

---

## Luồng dữ liệu end-to-end

```
[V1.pdf] [V2.pdf]
    │         │
    ▼         ▼
LegalDocumentParser           ← docling_parser.py
    │         │
    ▼         ▼
LegalDocument (DOM tree)      ← models.py (LegalDocument/ArticleNode/...)
    │         │
    ▼         ▼
LsuChunker                    ← lsu_chunker.py
    │         │
    ▼         ▼
list[LsuChunk]                ← models.py (LsuChunk, breadcrumb)
    │
    ▼
HybridGraphBuilder            ← knowledge_store.py
    ├── Kuzu DB (graph)        CONTAINS / PRECEDES / REFERENCES edges
    └── ChromaDB (vector)     node_id link → graph

──── Phase 2 ────────────────────────────────────────────

BGEM3Manager                  ← embedder.py
    ├── structural_embed (title/ordinal)
    └── semantic_embed (full_text + breadcrumb)
         │
         ▼
QdrantManager.upsert()        ← qdrant_indexer.py (optional)
         │
         ▼
compute_similarity_matrix()   ← similarity_matrix.py
    S[i][j] = 0.6×Cosine + 0.3×JaroWinkler + 0.1×Ordinal
         │
         ▼
hungarian_match()             ← hungarian_matcher.py
    + detect_split_merge()
         │
         ▼
DiffPairCatalog               ← diff_catalog.py
    matched / added / deleted / split / merged

──── Phase 3 ────────────────────────────────────────────

GenerativeComparisonPipeline  ← report_generator.py
    │
    ├── ACUPrompter.build()   ← acu_prompter.py
    │       inject V1 + V2 text → prompt
    │
    ├── LocalLLMClient.chat() ← llm_client.py
    │       Qwen2.5-7B @ llama.cpp
    │       → list[ACUOutput] (raw)
    │
    ├── VerificationEngine    ← verifier.py
    │       Tầng 2: exact/fuzzy evidence check
    │       Tầng 3: regex numerical verify (100% required)
    │       → list[ACUOutput] (verified, hallucination loại bỏ)
    │
    └── ComparisonReport      ← models.py
            verified_acus + markdown_report + executive_summary
```

---

## Bảng tóm tắt nhanh — vai trò từng file

| File | Thuộc Phase | Vai trò ngắn gọn |
|------|------------|-----------------|
| `README.md` | Tất cả | Spec toàn bộ hệ thống — **đọc trước tiên** |
| `configs/pipeline_config.yaml` | Config | Ngưỡng & trọng số runtime |
| `configs/model_config.yaml` | Config | Tên model, quantization |
| `configs/prompt_templates/*.txt` | Phase 3 | System/user prompt cho LLM |
| `src/config.py` | Config | YAML loader, fallback defaults, constants |
| `src/pipeline.py` | Orchestrator | Entry point toàn hệ thống |
| `src/__init__.py` | — | Package marker |
| `src/ingestion/models.py` | Phase 1 | **Core data models** — Legal DOM |
| `src/ingestion/docling_parser.py` | Phase 1 | PDF/DOCX → LegalDocument |
| `src/ingestion/lsu_chunker.py` | Phase 1 | LegalDocument → list[LsuChunk] |
| `src/ingestion/knowledge_store.py` | Phase 1 | LsuChunk → Kuzu + ChromaDB |
| `src/ingestion/__init__.py` | Phase 1 | Public API + `ingest_document()` |
| `src/alignment/diff_catalog.py` | Phase 2 | Data models: DiffPair, DiffPairCatalog |
| `src/alignment/embedder.py` | Phase 2 | BGE-M3 → structural + semantic embedding |
| `src/alignment/similarity_matrix.py` | Phase 2 | Ma trận N×M (Cosine+JaroWinkler+Ordinal) |
| `src/alignment/hungarian_matcher.py` | Phase 2 | Optimal bipartite matching + split/merge |
| `src/alignment/qdrant_indexer.py` | Phase 2 | Multi-vector Qdrant upsert/query |
| `src/alignment/alignment_engine.py` | Phase 2 | **Orchestrator Phase 2** |
| `src/alignment/__init__.py` | Phase 2 | Public API |
| `src/comparison/models.py` | Phase 3 | ACUOutput, ComparisonReport, VerificationResult |
| `src/comparison/llm_client.py` | Phase 3 | HTTP client → llama.cpp server |
| `src/comparison/acu_prompter.py` | Phase 3 | Build prompt từ DiffPair + templates |
| `src/comparison/verifier.py` | Phase 3 | 4-tầng verification, loại hallucination |
| `src/comparison/report_generator.py` | Phase 3 | **Orchestrator Phase 3** |
| `src/comparison/__init__.py` | Phase 3 | Public API (lazy import LLM modules) |
| `evaluation/golden_dataset_gen.py` | Eval | Sinh test data tự động (9 mutation ops) |
| `evaluation/metrics.py` | Eval | SPR, F1, FPR, Citation Accuracy |
| `evaluation/ragas_eval.py` | Eval | RAGAS Faithfulness |
| `tests/test_*.py` | Test | Unit + integration tests từng module |
| `tests/evaluate_phase*.py` | Eval | End-to-end eval Phase 1 & 2 |

---

## Các ngưỡng quyết định (tunable)

| Tham số | Giá trị mặc định | Ý nghĩa |
|---------|-----------------|---------|
| `match_threshold` | 0.65 | Score tối thiểu để coi 2 node là MATCHED |
| `split_merge_threshold` | 0.80 | Score để phát hiện split/merge |
| `w_semantic` | 0.6 | Trọng số cosine similarity trong S[i][j] |
| `w_jaro_winkler` | 0.3 | Trọng số Jaro-Winkler title similarity |
| `w_ordinal` | 0.1 | Trọng số ordinal proximity |
| `fuzzy_match_threshold` | 0.85 | Ngưỡng fuzzy cho evidence verification |
| `confidence_threshold` | 0.75 | Ngưỡng OCR để trigger fallback marker-pdf |
| `max_chunk_chars` | 2000 | Kích thước LSU chunk tối đa |

---

> **Tip:** Khi debug, bắt đầu từ `src/pipeline.py` và trace ngược từng phase. Mỗi phase có logger `[Phase N]` để theo dõi.
