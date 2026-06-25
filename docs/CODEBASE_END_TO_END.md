# L-RAG (LegalDiff) — End-to-End Codebase Documentation

> A pairwise legal-document intelligence pipeline that compares two versions of a Vietnamese legal document (V1 vs V2) and produces a **verified, zero-hallucination comparison report**.
>
> This document is the single end-to-end reference for the entire codebase: the 3-phase RAG core (`src/`), the FastAPI + Celery + WebSocket backend (`backend/`), the React frontend (`frontend/`), and all data stores and configuration. Each section includes concrete examples.

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [System Architecture (Bird's-Eye View)](#2-system-architecture-birds-eye-view)
3. [Repository Layout](#3-repository-layout)
4. [The 3-Phase RAG Pipeline — Overview](#4-the-3-phase-rag-pipeline--overview)
5. [Phase 1 — Ingestion (Parsing & Knowledge Representation)](#5-phase-1--ingestion-parsing--knowledge-representation)
6. [Phase 2 — Alignment (Embedding + Matching)](#6-phase-2--alignment-embedding--matching)
7. [Phase 3 — Generative Comparison (LLM + Verification)](#7-phase-3--generative-comparison-llm--verification)
8. [Backend (FastAPI + Celery + WebSocket)](#8-backend-fastapi--celery--websocket)
9. [Frontend (React + Vite)](#9-frontend-react--vite)
10. [Data Stores & Persistence](#10-data-stores--persistence)
11. [Configuration & Environment](#11-configuration--environment)
12. [End-to-End Example (Full Walkthrough)](#12-end-to-end-example-full-walkthrough)
13. [Deployment & Running](#13-deployment--running)
14. [Appendix: Configuration Reference & File Map](#14-appendix-configuration-reference--file-map)

---

## 1. What This System Does

### The problem

When a Vietnamese legal document is revised (e.g. a contract, a circular, or a decree goes from version V1 to V2), lawyers must manually diff the two to find **every** meaningful change — a changed amount (`500.000 đồng` → `600.000 đồng`), a shifted deadline (`30 ngày` → `45 ngày`), a flipped legal meaning (`không được chuyển nhượng` → `có thể chuyển nhượng`), a split/merged clause, or entirely added/deleted content.

Manual comparison is slow, error-prone, and — critically for an LLM-based approach — **hallucination-prone**. A raw LLM asked to "compare these two texts" will invent evidence, paraphrase instead of quoting, miss small numerical changes, and confidently invent legal reasoning.

### The solution

L-RAG solves this with a **deterministic-first, LLM-second** pipeline organized into three phases, with a hard zero-hallucination guarantee enforced by multi-tier programmatic verification:

| Phase | Name | Responsibility | Output |
|------|------|----------------|--------|
| **1** | Ingestion | Parse PDF/DOCX into a structured Legal DOM + Logical Semantic Unit (LSU) chunks | `LegalDocument`, `list[LsuChunk]` |
| **2** | Alignment | Embed chunks and match Article X (V1) ↔ Article Y (V2) even when reordered/reworded/split/merged | `DiffPairCatalog` |
| **3** | Generative Comparison | For each matched pair, extract Atomic Comparison Units (ACUs) via LLM, then **verify every claim against source text** | `ComparisonReport[]` (JSON + Markdown) |

### Design principles (the "three absolute constraints")

1. **Structured-first**: legal structure (Chương/Điều/Khoản/Điểm) is parsed deterministically before any model touches it.
2. **Matching by algorithm, not vibes**: article pairing uses BGE-M3 embeddings + the Hungarian algorithm — a provably optimal assignment, not LLM guessing.
3. **Zero hallucination**: every LLM claim (an ACU) must pass evidence + numerical verification against the raw source text before it reaches the report. Anything that cannot be grounded is dropped and logged.

---

## 2. System Architecture (Bird's-Eye View)

```
                         ┌─────────────────────────────────────────────┐
                         │                 FRONTEND (React/Vite)        │
                         │  Login · Dashboard · CreateJob · Report ·   │
                         │  Settings · Eval   (WebSocket live progress)│
                         └───────────────┬─────────────────────────────┘
                                  REST   │   WebSocket (/ws/jobs/{id})
                                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        BACKEND (FastAPI)                                │
│   /api/auth  /api/documents  /api/jobs  /api/reports  /api/settings    │
│   /api/eval  /api/health     ConnectionManager (Redis pub/sub → WS)    │
└──────────┬───────────────────────────────────────┬─────────────────────┘
           │ enqueue task                          │ SQLAlchemy (async)
           ▼                                       ▼
┌─────────────────────────┐              ┌────────────────────────┐
│   CELERY WORKER         │              │   PostgreSQL            │
│   run_pipeline(job_id)  │── progress → │   users, documents,    │
│   • GPU lock (Redis)    │   Redis      │   jobs, reports, eval,  │
│   • LegalDiffPipeline   │   pub/sub    │   user_settings         │
└──────────┬──────────────┘              └────────────────────────┘
           │ calls
           ▼
┌────────────────────────────────────────────────────────────────────────┐
│              CORE RAG PIPELINE  (src/pipeline.py)                       │
│                                                                          │
│   PHASE 1: Ingestion          PHASE 2: Alignment        PHASE 3: Compare│
│   Docling parser → LSU        BGE-M3 embed →            LLM ACU extract│
│   chunker → LegalDOM          Similarity matrix →       → Evidence ver │
│                                Hungarian → Split/Merge   → Number ver  │
│                                → DiffPairCatalog         → Summary LLM │
│                                                          → Report      │
└────────────────────────────────────────────────────────────────────────┘
           │ uses (transient / optional persistent)
           ▼
   Kuzu (graph) · Qdrant (vectors) · ChromaDB (legacy vectors) · FS (reports)
```
**System architecture (Mermaid):**

```mermaid
flowchart TB
    subgraph FE["Frontend · React 19 + Vite + TS"]
        UI["Pages: Login · Dashboard · CreateJob · Report · Settings · Eval<br/>(useWebSocket hook → PhaseStepper)"]
    end
    UI <-->|"REST /api · WebSocket /ws/jobs/{id}?token=JWT"| API
    subgraph BE["Backend · FastAPI (async)"]
        API["Routers: auth · documents · jobs · reports · settings · eval · health"]
        CM["ConnectionManager<br/>Redis pub/sub → WS fan-out"]
        API --> CM
    end
    API -->|"enqueue run_pipeline"| TASK
    API -->|"SQLAlchemy async"| PG[("PostgreSQL<br/>users·documents·jobs·reports·settings")]
    subgraph WK["Celery Worker · worker_concurrency=1, acks_late"]
        TASK["run_pipeline(job_id)<br/>+ GPULock (Redis SET NX, TTL 15m)"]
        PIPE["LegalDiffPipeline (src/pipeline.py)"]
        TASK --> PIPE
    end
    TASK -.->|"progress events → Redis pub/sub"| CM
    subgraph CORE["Core 3-Phase RAG Pipeline · src/"]
        P1["Phase 1 · Ingestion<br/>Docling → LSU Chunker → Legal DOM"]
        P2["Phase 2 · Alignment<br/>BGE-M3 → Similarity → Hungarian → Catalog"]
        P3["Phase 3 · Comparison<br/>LLM ACU → Verify (T2/T3) → Summary → MD"]
        P1 --> P2 --> P3
    end
    PIPE --> CORE
    P2 -.->|"dense+sparse vectors"| QDR[("Qdrant<br/>in-memory default")]
    P1 -.->|"standalone only"| KZ[("Kuzu graph + ChromaDB")]
    PIPE --> FS[("Filesystem<br/>reports JSON + MD")]
```

**Three independently deployable tiers:**
- **Core engine** (`src/`) — pure Python library, no web dependencies. Can run from CLI.
- **Backend** (`backend/`) — wraps the engine as an authenticated, async web service with live progress.
- **Frontend** (`frontend/`) — React UI driving the backend.

---

## 3. Repository Layout

```
L-RAG/
├── src/                         # 🧠 Core 3-phase RAG pipeline (library)
│   ├── pipeline.py              #    LegalDiffPipeline orchestrator (all 3 phases)
│   ├── config.py                #    YAML + env config loader, LLM provider resolution
│   ├── ingestion/               #    Phase 1 (parser, chunker, knowledge store, models)
│   ├── alignment/               #    Phase 2 (embedder, similarity, Hungarian, catalog)
│   └── comparison/              #    Phase 3 (prompter, llm_client, verifier, report)
│
├── backend/                     # 🌐 FastAPI + Celery + WebSocket service
│   ├── main.py                  #    app factory, middleware, router mount, WS endpoint
│   ├── api/                     #    REST routers (auth, documents, jobs, reports, ...)
│   ├── services/                #    business logic (job/document/auth/settings/storage)
│   ├── workers/pipeline_task.py #    Celery task that runs LegalDiffPipeline + GPU lock
│   ├── ws/                      #    WebSocket manager + handlers (Redis pub/sub bridge)
│   ├── models/                  #    SQLAlchemy ORM (User, Document, Job, Report, ...)
│   ├── schemas/                 #    Pydantic request/response schemas
│   ├── celery_app.py            #    Celery config (worker_concurrency=1)
│   └── alembic/                 #    DB migrations
│
├── frontend/                    # 🖼️ React 19 + Vite + TS + Tailwind/shadcn
│   └── src/                     #    pages, components, api clients, useWebSocket hook
│
├── configs/                     # ⚙️ Runtime configuration
│   ├── pipeline_config.yaml     #    thresholds, weights, concurrency
│   ├── model_config.yaml        #    embedding/LLM model paths, provider presets
│   └── prompt_templates/        #    acu_system.txt, acu_user.txt, summary_user.txt
│
├── data/                        # 💾 Runtime data (git-ignored)
│   ├── raw/                     #    original uploaded PDF/DOCX
│   ├── processed/               #    Kuzu graph_db, chroma_db
│   ├── qdrant_*/                #    per-job Qdrant collections (when persistent)
│   ├── pg_data/                 #    PostgreSQL data dir
│   ├── reports/                 #    generated JSON + Markdown reports
│   └── golden/                  #    ground-truth eval dataset
│
├── evaluation/                  # 📊 offline eval scripts (golden dataset, metrics)
├── docs/                        # 📚 this file + phase docs + frontend docs
├── models/                      # downloaded weights (multilingual-e5-large-instruct — legacy)
└── requirements.txt
```

---

## 4. The 3-Phase RAG Pipeline — Overview

The entire core lives in `src/pipeline.py` as `LegalDiffPipeline`. Its docstring states the contract precisely:

```
Input: 2 file PDF/DOCX (V1, V2)

Phase 1 — Ingestion:    LegalDocumentParser → LsuChunker
                        → LegalDocument, list[LsuChunk]

Phase 2 — Alignment:    BGEM3Manager → LegalAlignmentEngine → DiffPairCatalog
                        → N pairs (matched/added/deleted/split/merged)

Phase 3 — Comparison:   GenerativeComparisonPipeline → ComparisonReport[]
                        → comparison report (JSON + Markdown)
```

### Minimal usage (library API)

```python
from src.pipeline import LegalDiffPipeline

pipeline = LegalDiffPipeline.from_config()          # reads configs/*.yaml
result = pipeline.run(
    file_v1="data/raw/contract_v1.docx",
    file_v2="data/raw/contract_v2.docx",
)
print(result["catalog"].summary())   # {'matched': .., 'added': .., 'deleted': .., ...}
print(result["markdown"])            # full Vietnamese comparison report
```

### Run configuration

`PipelineRunConfig` (a dataclass) carries every knob through one run. Key fields:

```python
@dataclass
class PipelineRunConfig:
    # Phase 1
    confidence_threshold: float = 0.75     # OCR fallback trigger
    max_chunk_chars: int = 2000
    # Phase 2
    match_threshold: float = 0.65          # (effective runtime value comes from YAML = 0.60)
    qdrant_path: str | None = "./data/processed/qdrant_db"
    # Phase 3
    llm_base_url: str = "http://localhost:8000/v1"
    llm_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_concurrency: int = 4
    llm_temperature_acu: float = 0.05      # near-zero → deterministic ACU extraction
    llm_temperature_summary: float = 0.3
    enable_second_pass: bool = True
    enable_number_enumeration: bool = True
    min_confidence_to_include: float = 0.2
    skip_phase3: bool = False              # True → only run Phase 1 + 2
    progress_callback: Callable[[int, str, str], None] | None = None
    llm_provider: str = "local"            # "local" | "deepseek"
```

### Progress callback contract

The pipeline emits `(percent, phase, message)` at phase boundaries. The backend wires this into WebSocket messages so the frontend's `PhaseStepper` can render live progress:

```python
_notify(10, "ingestion",  "Đang phân tích tài liệu...")    # Phase 1 start
_notify(30, "alignment",  "Đang nhúng vector BGE-M3...")   # Phase 2 start
_notify(55, "comparison", "Đang sinh báo cáo so sánh...")  # Phase 3 start
_notify(95, "done",       "Đang lưu kết quả...")           # near complete
```

### End-to-end pipeline flow (Mermaid)

```mermaid
flowchart LR
    IN["2× PDF/DOCX<br/>V1 + V2"] --> P1
    subgraph P1["Phase 1 · Ingestion"]
        direction TB
        DL["Docling parser<br/>+ Marker-PDF OCR fallback"] --> DOM["LegalDocument<br/>Chương → Điều → Khoản → Điểm"]
        DOM --> CHK["LsuChunker<br/>breadcrumb prefix + sentence split"]
    end
    P1 -->|"LegalDocument × 2"| P2
    subgraph P2["Phase 2 · Alignment"]
        direction TB
        EM["BGEM3Manager<br/>dense 1024-d + sparse (lexical)"] --> SIM["Similarity Matrix S<br/>w_sem·cos + w_jw·JW + w_ord·ord + w_sp·sparse<br/>+ article-number bonus, clip[0,1]"]
        SIM --> HU["Hungarian 1:1 (cost = 1 − S), θ=0.60"]
        HU --> SM["Split/Merge on unmatched (cos ≥ 0.80)"]
        SM --> CAT["DiffPairCatalog<br/>matched / added / deleted / split / merged"]
    end
    P2 -->|"DiffPairCatalog"| P3
    subgraph P3["Phase 3 · Comparison"]
        direction TB
        T1["Tier 1 · LLM ACU extract<br/>windowing + 2-pass + number-enum + logic-hint"] --> T2["Tier 2 · Evidence verify<br/>exact → whitespace-norm → fuzzy (≥0.85)"]
        T2 --> T3["Tier 3 · Numerical verify<br/>regex extract, 100% strict"]
        T3 --> T4["Tier 4 · Executive summary + Markdown"]
    end
    P3 --> OUT["ComparisonReport[]<br/>machine-readable JSON + human-readable MD"]
```

---

## 5. Phase 1 — Ingestion (Parsing & Knowledge Representation)

**Goal:** turn a raw PDF/DOCX into a structured, machine-readable representation of the legal document, split into retrievable Logical Semantic Units.

**Files:** `src/ingestion/`
- `models.py` — the Legal DOM data models
- `docling_parser.py` — `LegalDocumentParser`
- `lsu_chunker.py` — `LsuChunker`
- `knowledge_store.py` — `HybridGraphBuilder` (Kuzu + ChromaDB)

> **Important scope note:** In the *comparison* pipeline (`pipeline.py`), Phase 1 only runs **parse → chunk**. The `HybridGraphBuilder` (Kuzu graph + ChromaDB) is an *additional, standalone* ingestion capability exposed via `ingest_document()` for building a persistent searchable legal knowledge base — it is **not** invoked during a normal diff run. Phase 2 has its own (Qdrant) vector indexing.

### 5.1 The Legal DOM (`models.py`)

Vietnamese legal documents have a strict hierarchy: **Chương (Chapter) → Điều (Article) → Khoản (Clause) → Điểm (Point)**. The parser reproduces this as a typed tree:

```
LegalDocument
├── doc_id, file_name, doc_title, doc_number, signing_date, parties, preamble
├── sections: list[DocumentSection]            # Chương / Phần / Mục
│   └── articles: list[ArticleNode]            # Điều
│       ├── intro, title, number
│       └── clauses: list[ClauseNode]          # Khoản
│           ├── content
│           ├── points: list[PointNode]        # Điểm a, b, c
│           └── tables: list[TableData]
├── orphan_articles: list[ArticleNode]         # articles outside any section
└── quality_metrics: ParseQualityMetrics       # engine used, OCR confidence, warnings
```

**Legal DOM as a class hierarchy (Mermaid):**

```mermaid
classDiagram
    class LegalDocument {
        +doc_id / file_name
        +doc_title / doc_number
        +signing_date / parties / preamble
        +quality_metrics: ParseQualityMetrics
        +sections: list~DocumentSection~
        +orphan_articles: list~ArticleNode~
        +iter_all_articles()
        +iter_all_clauses()
    }
    class DocumentSection {
        +section_type "Chương/Phần/Mục"
        +number / title
        +full_title
        +articles: list~ArticleNode~
    }
    class ArticleNode {
        +number (int|str, Arabic or Roman)
        +title / intro
        +clauses: list~ClauseNode~
        +full_title "Điều N. title"
    }
    class ClauseNode {
        +number (coerced int|str)
        +content
        +points: list~PointNode~
    }
    class PointNode {
        +label "a/b/c"
        +content
    }
    class TableData {
        +headers / rows (2D grid)
        +cells (row_span/col_span)
        +num_rows / num_cols
    }
    LegalDocument "1" *-- "0..*" DocumentSection : sections
    LegalDocument "1" *-- "0..*" ArticleNode : orphan_articles
    DocumentSection "1" *-- "0..*" ArticleNode : articles
    ArticleNode "1" *-- "0..*" ClauseNode : clauses
    ClauseNode "1" *-- "0..*" PointNode : points
    ArticleNode ..> TableData : tables
    ClauseNode ..> TableData : tables
```

**Key design rule — tables are never flattened.** A `TableData` preserves the full 2D grid (headers, rows, cell spans) so numerical data in tables is never lost to linearization:

```python
class TableData(BaseModel):
    table_id: str
    caption: str | None
    headers: list[str]
    rows: list[list[Any]]        # 2D grid preserved
    cells: list[TableCell]       # row_span / col_span aware
    num_rows: int
    num_cols: int
    source_page: int | None
```

### 5.2 `LegalDocumentParser` — parsing + OCR fallback

The parser uses **Docling** as the primary engine (it returns text, tables, structure, and an OCR confidence score). If quality is poor, it falls back to **Marker-PDF** (heavy OCR):

```python
class LegalDocumentParser:
    def __init__(self, confidence_threshold: float = 0.75,
                 low_conf_page_ratio: float = 0.30, marker_timeout: int = 300): ...

    def parse(self, file_path) -> LegalDocument: ...
```

**OCR fallback decision** — trigger Marker when Docling's average confidence is low *or* too many pages are low-quality:

```python
def _should_trigger_ocr(self, metrics) -> bool:
    if metrics.avg_confidence is None:
        return False
    low_conf_ratio = len(metrics.low_confidence_pages) / metrics.total_pages
    return (metrics.avg_confidence < self.confidence_threshold      # < 0.75
            or low_conf_ratio > self.low_conf_page_ratio)           # > 30%
```

**Article detection** uses regex tuned for Vietnamese legal numbering (supports Arabic + Roman numerals):

```python
_RE_ARTICLE = re.compile(
    r"^(?P<prefix>Điều)\s+(?P<number>\d+|[IVXLCDM]+)\s*[.:]?\s*(?P<title>.*)$",
    re.IGNORECASE | re.UNICODE,
)
_RE_CLAUSE  = re.compile(r"^(?P<number>\d{1,2})\.\s+(?P<content>.+)$", re.UNICODE)
_RE_POINT   = re.compile(r"^\s*(?P<label>[a-zđ])\)\s+(?P<content>.+)$", re.UNICODE)
_RE_SECTION = re.compile(
    r"^(?P<section_type>Chương|Phần|Mục|Tiểu mục)\s+(?P<number>[IVXLCDM\d]+)\.?\s*(?P<title>.*)$",
    re.IGNORECASE | re.UNICODE,
)
```

**Article detection** uses regex tuned for Vietnamese legal numbering (supports Arabic + Roman numerals):

| Pattern | Regex (essence) | Captures |
|--------|------------------|----------|
| `_RE_SECTION` | `^(Chương\|CHƯƠNG\|Phần\|PHẦN\|Mục\|MỤC\|Tiểu mục)\s+([IVXLCDM\d]+)\.?\s*(title)` | section heading + number + title |
| `_RE_ARTICLE` | `^(Điều)\s+(\d+\|[IVXLCDM]+)\s*[.:]?\s*(title)` | "Điều 5" / "Điều V" + optional title |
| `_RE_CLAUSE` | `^(\d{1,2})\.\s+(content)$` | numbered clause `1.` … `99.` |
| `_RE_POINT` | `^\s*([a-zđ])\)\s+(content)$` | lettered point `a)` (Vietnamese `đ` allowed) |
| `_RE_FB_SECTION` *(fallback)* | `^([IVXLCDM]+)\.\s+(title)$` | Roman heading **without** "Chương" prefix |
| `_RE_FB_ARTICLE` *(fallback)* | `^(\d{1,3})\.\s+(content)$` | top-level `1.` for non-Điều documents |

> **Fallback pass:** some documents (Công điện, Quyết định, Hướng dẫn…) number items directly (`1.`, `2.`, `I.`, `II.`) instead of using "Điều". When the primary pass detects **zero** articles, the parser reruns with `_RE_FB_*` and promotes the top-level numbered items to `ArticleNode`s so downstream alignment still works.

**State-machine DOM builder:** each input line is tested in priority order (section → article → clause → point → body text), with cascade *flush* functions that close the currently-open node before opening a new one. Docling's Markdown bold/italic markers (`**Điều 5:**`) are stripped before matching so structure detection isn't confused by formatting.

### 5.3 `LsuChunker` — Logical Semantic Unit chunking

Instead of naive fixed-size splitting (which breaks clauses mid-sentence and destroys context), the chunker respects legal structure and attaches a **breadcrumb** to every chunk.

```python
class LsuChunker:
    def __init__(self, max_chunk_chars: int = 2000, overlap_chars: int = 200,
                 create_article_level_chunks: bool = True,
                 create_clause_level_chunks: bool = True): ...
    def chunk(self, document: LegalDocument) -> list[LsuChunk]: ...
```

**Algorithm:**
1. For each `ArticleNode`, create an **article-level overview chunk** (intro + clause previews).
2. For each `ClauseNode`, flatten its content + points into a **clause-level chunk**.
3. If any chunk exceeds `max_chunk_chars` (2000), split at **Vietnamese sentence boundaries** with `overlap_chars` (200) carried into the next piece, labelled `[phần 1/3]`.

A breadcrumb is the hierarchical path, prepended to the embedded text:

```
[Chương II. Quyền và Nghĩa vụ > Điều 5. Quyền của Bên A > Khoản 3]
```

**Sentence-boundary regex** (aware of accented Vietnamese capitals):

```python
_RE_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?;])\s+(?=[A-ZÁÀẢÃẠ...ĐÉÈẺẼẸ...ÝỲỶỸỴ0-9])", re.UNICODE,
)
```

### 5.4 `LsuChunk` — the Phase 1 output unit

```python
class LsuChunk(BaseModel):
    chunk_id: str
    doc_id: str
    source_node_id: str            # FK → ArticleNode/ClauseNode
    source_node_type: NodeType     # ARTICLE | CLAUSE
    breadcrumb: str                # "[Chương II > Điều 5 > Khoản 3]"
    content_with_prefix: str       # breadcrumb + "\n" + raw_content  (this is embedded)
    raw_content: str
    content_type: ContentType      # TEXT | TABLE | MIXED
    tables_json: list[dict]
    article_number: int | str | None
    clause_number: int | str | None
    section_title: str | None
    page_number: int | None
    char_count: int
```

### 5.5 `HybridGraphBuilder` — optional persistent knowledge store

When you *do* want a searchable legal knowledge base (not just a one-off diff), `ingest_document()` builds two linked stores:

- **Kuzu graph DB** — `Document` and `LegalNode` tables with `CONTAINS` (hierarchy), `PRECEDES` (ordering), and `REFERENCES` (cross-references like *"theo quy định tại Điều 15"*) edges. References are detected by regex and resolved via an article/clause index.
- **ChromaDB** — chunk embeddings. The critical glue field is `VectorMetadata.node_id`, which points each vector back to its Kuzu graph node.

```python
# Standalone entrypoint (NOT used by the comparison pipeline)
from src.ingestion import ingest_document
result = ingest_document("data/raw/decree.pdf")  # → {nodes, edges, vectors, ...}
```

### 5.6 Phase 1 example

**Input article text (parsed):**
```
Điều 5. Quyền của Bên A
1. Yêu cầu Bên B giao hàng đúng hạn
2. Kiểm tra chất lượng hàng hóa
```

**Resulting clause-level `LsuChunk`:**
```python
LsuChunk(
    chunk_id="chunk_f6e5d4c3b2a1",
    source_node_id="clause_khoan2",
    source_node_type=NodeType.CLAUSE,
    breadcrumb="[Chương II > Điều 5 > Khoản 2]",
    content_with_prefix=("[Chương II > Điều 5 > Khoản 2]\nKiểm tra chất lượng hàng hóa"),
    raw_content="Kiểm tra chất lượng hàng hóa",
    article_number=5,
    clause_number=2,
    char_count=28,
)
```

---

## 6. Phase 2 — Alignment (Embedding + Matching)

**Goal:** given two `LegalDocument`s, decide for every article in V1 which article(s) in V2 it corresponds to — even when the text was reworded, the order changed, or clauses were split/merged. **Output:** a `DiffPairCatalog`.

**Files:** `src/alignment/`
- `embedder.py` — `BGEM3Manager`
- `similarity_matrix.py` — the weighted scoring formula
- `hungarian_matcher.py` — optimal assignment + split/merge detection
- `diff_catalog.py` — `DiffPair` / `MatchType` / `DiffPairCatalog`
- `qdrant_indexer.py` — `QdrantManager`
- `alignment_engine.py` — `LegalAlignmentEngine` (orchestrator)

**Alignment algorithm flow (Mermaid):**

```mermaid
flowchart TB
    A["collect ArticleNodes<br/>V1: N nodes · V2: M nodes"] --> B["BGEM3Manager.embed_article_nodes<br/>2 embeds/node: structural + semantic<br/>dense(1024) + sparse(lexical)"]
    B --> C["build NodeRecords (title·raw_text·ordinal·vec·sparse·article_number)"]
    C --> D["compute_similarity_matrix → S (N×M)<br/>sem·cos + jaro + ordinal + sparse<br/>+ 0.15 if same article_number · clip[0,1]"]
    D --> E["hungarian_match<br/>linear_sum_assignment(1 − S)<br/>keep pairs with S ≥ θ = 0.60"]
    E --> F{"unmatched<br/>leftovers?"}
    F -->|yes| G["detect_split_merge<br/>MERGE: cos(v1a+v1b, v2) ≥ 0.80<br/>SPLIT: cos(v1, v2a+v2b) ≥ 0.80"]
    F -->|no| H["classify residue"]
    G --> H
    H --> I["DELETED = V1-only · ADDED = V2-only"]
    E --> J["MATCHED pairs"]
    G --> K["SPLIT / MERGED pairs"]
    J --> CAT["DiffPairCatalog"]
    I --> CAT
    K --> CAT
```

### 6.1 `BGEM3Manager` — embedding

Wraps **BGE-M3** (`BAAI/bge-m3`), which emits three vector kinds per text:

| Vector | Dim | Used for |
|--------|-----|----------|
| **Dense** | 1024 | semantic cosine similarity |
| **Sparse** | ≤ 2²⁰ buckets | BM25-like lexical overlap |
| ColBERT | — | late-interaction (not used in Phase 2) |

```python
class BGEM3Manager:
    def __init__(self, model_name="BAAI/bge-m3", use_fp16=True,
                 batch_size=16, max_length=1024, device=None): ...  # device auto-detects cuda/cpu
```

Each article node is embedded **twice**:

| Embedding | Text built | `max_length` | Role |
|-----------|------------|--------------|------|
| **Structural** | `"Điều {number}: {title}"` (article) / `"Điều {n} Khoản {k}: {first 80 chars}"` (clause) | 512 | identity / position — recognises "Điều 5" ↔ "Điều 5" even when content changed |
| **Semantic** | breadcrumb + intro + up to **3 clause previews** (150 chars each) | 1024 | deep content meaning |

The **semantic** dense vector is what feeds the similarity matrix (`NodeRecord.semantic_vec`). Sparse lexical weights are extracted by the BGE-M3 sparse head and converted for Qdrant:

```python
# {word_token: weight}  →  {int_index: weight}  (Qdrant SparseVector needs int indices)
BUCKET_SIZE = 1 << 20  # 1,048,576 buckets
idx = abs(hash(token)) % BUCKET_SIZE      # collision-merge by max(weight)
```

`embed_texts_semantic()` — a convenience that returns L2-normalized `(N, 1024)` dense vectors only — is the `embed_fn` reused by split/merge detection to cosine-compare concatenated texts.

### 6.2 The similarity matrix — the core algorithm

`compute_similarity_matrix()` builds an N×M matrix `S` combining four signals, then adds an article-number bonus:

```
S[i][j] = w_semantic   · Cosine(dense_i, dense_j)         # semantic content
        + w_jaro_winkler · JaroWinkler(title_i, title_j)    # lexical/structural title match
        + w_ordinal     · OrdinalProximity(i, j)            # relative position
        + w_sparse      · Cosine(sparse_i, sparse_j)        # exact-term overlap (optional)
        + ARTICLE_NUMBER_BONUS   (if article_number_i == article_number_j)   # +0.15
```

then `clip(S, 0, 1)`.

**Runtime weights (from `configs/pipeline_config.yaml`):**

| Weight | YAML value | Meaning |
|--------|-----------|---------|
| `w_semantic` | **0.50** | dense semantic cosine |
| `w_jaro_winkler` | **0.25** | title string similarity (`jellyfish`) |
| `w_ordinal` | **0.10** | position proximity |
| `w_sparse` | **0.15** | sparse lexical overlap |

> The four weights must sum to 1.0 (validated in `AlignmentConfig.__post_init__`). The article-number bonus is *added on top* and is not part of the sum. (The `AlignmentConfig` dataclass defaults are 0.6/0.3/0.1/0.0; the pipeline overrides them with the YAML values above, so **0.50/0.25/0.10/0.15 are the effective runtime weights**.)

**Component details:**

- **Cosine** — L2-normalize dense vectors, then `V1 @ V2.T`.
- **Jaro-Winkler** — `jellyfish.jaro_winkler_similarity(title_i.lower(), title_j.lower())`, range [0,1]. Catches *"Nghĩa vụ B"* ↔ *"Nghĩa vụ của Bên B"*.
- **Ordinal proximity** — `1 - |pos_i − pos_j|` where `pos = ordinal / (count − 1)`. Max 1.0 when relative positions match. Rewards articles that stayed in roughly the same place.
- **Sparse overlap** — cosine on the BGE-M3 sparse dicts; catches exact term matches (article numbers, proper nouns, legal terms) that dense embeddings can blur.
- **Article-number bonus** — `+0.15` when both articles carry the same number (e.g. "Điều 5" ↔ "Điều 5"). Strong signal since legal articles rarely change number unless renumbered.

### 6.3 Hungarian matching + split/merge

**Step 1 — optimal 1:1 assignment.** The Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) finds the globally optimal matching by minimizing `cost = 1 − S`. Pairs below `match_threshold` (**0.60** at runtime) are dropped:

```python
def hungarian_match(similarity_matrix, match_threshold=0.65):
    cost_matrix = 1.0 - similarity_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # keep only (i, j) with S[i,j] >= match_threshold
```

**Step 2 — split/merge on the unmatched leftovers.** For nodes the Hungarian step couldn't pair, `detect_split_merge()` tests whether concatenating two small nodes matches one large node (cosine ≥ `split_merge_threshold` = **0.80**):

- **MERGE** (2 V1 → 1 V2): `cosine(text_v1[i1] + "\n" + text_v1[i2], text_v2[j]) ≥ 0.80`
- **SPLIT** (1 V1 → 2 V2): `cosine(text_v1[i], text_v2[j1] + "\n" + text_v2[j2]) ≥ 0.80`

**Step 3 — classify the rest.** Anything still unmatched becomes `ADDED` (only in V2) or `DELETED` (only in V1).

### 6.4 `DiffPair` and `MatchType`

```python
class MatchType(str, Enum):
    MATCHED = "matched"     # 1 V1 ↔ 1 V2, score ≥ θ
    ADDED   = "added"       # only in V2
    DELETED = "deleted"     # only in V1
    SPLIT   = "split"       # 1 V1 → 2+ V2
    MERGED  = "merged"      # 2+ V1 → 1 V2

class DiffPair(BaseModel):
    pair_id: str                       # "pair_{12 hex}"
    v1_ids: list[str]; v2_ids: list[str]
    match_type: MatchType
    confidence_score: float
    v1_texts: list[str]; v2_texts: list[str]   # raw text fed to Phase 3
    # ... plus optional semantic/jaro/ordinal sub-scores
```

`DiffPairCatalog.summary()` returns counts per type — this is what the backend stores on the job row (`matched_count`, `added_count`, …).

### 6.5 `QdrantManager` (indexing)

Optionally indexes the dense+sparse vectors into Qdrant as a multi-vector collection. The collection schema (`QdrantCollectionConfig`, `DENSE_DIM = 1024`):

| Vector / index | Type | Notes |
|----------------|------|-------|
| `structural` | named dense (1024-d) | structural identity embedding |
| `semantic` | named dense (1024-d) | semantic content embedding |
| `structural_sparse` | named sparse | BGE-M3 lexical weights (int-bucketed) |
| `semantic_sparse` | named sparse | BGE-M3 lexical weights (int-bucketed) |
| payload indexes | `version`, `node_type`, `doc_id`, `node_id`, `article_number` = KEYWORD; `ordinal` = INTEGER | fast filtering when querying across jobs/versions |

> ⚠️ **In-memory caveat:** Qdrant *local in-memory* mode does not support sparse vectors when the collection is created with both dense and sparse configs — the sparse vectors are **dropped gracefully**. Sparse retrieval only works with a persistent (`path:` set) Qdrant deployment. (This is fine: sparse overlap is also computed directly in-process by `sparse_overlap_matrix()` during alignment, independent of Qdrant.)

By default the pipeline runs Qdrant **in-memory** (`qdrant.path: null`) to avoid lock contention between sequential dev jobs; the `QdrantManager` is only instantiated when `PipelineRunConfig.qdrant_path` is set, and Qdrant failures are logged-and-skipped (alignment does not depend on Qdrant succeeding).

### 6.6 Phase 2 worked example

V1 has articles A(ordinal 0), B(1), C(2). V2 reworded B, split A into two, and dropped C:

```
V1:  A "Quyền của A"        V2:  A1 "Quyền của A (phần 1)"
     B "Nghĩa vụ B"               A2 "Quyền bổ sung của A"
     C "Thanh toán 30 ngày"       B' "Nghĩa vụ của Bên B"
```

1. **Embed** all 3 + 3 article nodes with BGE-M3.
2. **Score matrix S** (illustrative), e.g. `S[A, A1]` high (same title, same number → +0.15 bonus), `S[B, B']` high (Jaro-Winkler on the near-identical title), `S[C, *]` low everywhere.
3. **Hungarian** pairs `B ↔ B'` (≥ 0.60). `A`, `A1`, `A2`, `C` remain unmatched.
4. **Split detection:** `cosine(A, A1 + "\n" + A2) = 0.83 ≥ 0.80` → `SPLIT` pair `A → [A1, A2]`.
5. `C` is still unmatched → `DELETED`.

Resulting catalog:
```python
DiffPairCatalog(pairs=[
    DiffPair(match_type=MATCHED, v1_ids=["B"],  v2_ids=["B'"],  confidence_score=0.78),
    DiffPair(match_type=SPLIT,   v1_ids=["A"],  v2_ids=["A1","A2"], confidence_score=0.83),
    DiffPair(match_type=DELETED, v1_ids=["C"],  v2_ids=[],       confidence_score=0.0),
])
# summary() → {'matched': 1, 'split': 1, 'deleted': 1, 'added': 0, 'merged': 0}
```

---

## 7. Phase 3 — Generative Comparison (LLM + Verification)

**Goal:** for each `DiffPair`, produce a structured, **verified** list of changes (Atomic Comparison Units) and a human-readable Vietnamese report — with a hard guarantee that no ungrounded claim survives.

**Files:** `src/comparison/`
- `models.py` — `ACUOutput`, `ComparisonReport`, `ComparisonRequest`, enums
- `acu_prompter.py` — the canonical prompt-engineering module: `ACU_SYSTEM_PROMPT`, `build_acu_user_prompt`, `SUMMARY_SYSTEM_PROMPT`, `build_summary_user_prompt` (templates + builders, extracted from the orchestrator). Plain-text mirrors also live in `configs/prompt_templates/*.txt`.
- `report_generator.py` — carries its own working copies of these prompts (the pipeline uses these at runtime), including the extra `ACU_SYSTEM_PROMPT_DEEPSEEK` variant.
- `llm_client.py` — `LocalLLMClient` (OpenAI-compatible, JSON mode, retry, truncation recovery)
- `logic_detector.py` — deterministic logic-shift hints (Category D)
- `verifier.py` — `VerificationEngine` (Tier 2 + Tier 3)
- `report_generator.py` — `GenerativeComparisonPipeline` (orchestrator + Markdown renderer)

### 7.1 The ACU (Atomic Comparison Unit)

An **ACU** is the smallest verifiable change. The LLM must emit them as structured JSON, each carrying **verbatim evidence copied from the source**:

```python
class ACUOutput(BaseModel):
    acu_id: str                          # "acu_{12 hex}"
    change_type: ChangeType              # numerical|terminology|structural|addition|deletion|reorder
    location_v1: str; location_v2: str   # "Điều X, Khoản Y, Điểm Z"
    original_value: str; new_value: str  # short before/after values
    verbatim_evidence_v1: str            # MUST be copy-paste from V1
    verbatim_evidence_v2: str            # MUST be copy-paste from V2
    confidence: float                    # 0.0–1.0
    reasoning: str                       # DeepSeek-only self-explanation
```

`ChangeType` enum: `numerical` (numbers/dates/%/currency/deadlines), `terminology` (legal wording), `structural` (clause structure), `addition`, `deletion`, `reorder`.

### 7.2 The four-tier pipeline

`GenerativeComparisonPipeline.run_single(request)` runs each pair through four tiers:

**Four-tier zero-hallucination pipeline (Mermaid):**

```mermaid
flowchart TB
    REQ["ComparisonRequest<br/>pair_id · match_type · raw_text_v1/v2 · breadcrumbs"]
    REQ --> T1
    subgraph T1["Tier 1 · ACU Extraction  (LLM, temp≈0.05)"]
        T1a{"v1 == v2 after<br/>whitespace-normalize?"}
        T1a -->|yes| SKIP["skip LLM → no ACUs"]
        T1a -->|no| T1b["windowing if max(len) > 5000  (20k DeepSeek)<br/>+ number-enumeration hint + logic-shift hint"]
        T1b --> T1c["LLM call(s) → ACU JSON (parse + validate)"]
        T1c --> T1d["2nd pass: ask for ADDITIONAL changes<br/>(skipped for added/deleted)"]
        T1d --> T1e["dedupe  (type, norm(orig), norm(new))"]
    end
    T1 --> SV{"provider == deepseek<br/>& self-verify on?"}
    SV -->|flagged dropped, conf adjusted| PF["pre-filter: confidence ≥ 0.2"]
    SV -->|local / ok| PF
    PF --> T2
    subgraph T2["Tier 2 · Evidence Verification  (programmatic)"]
        T2a["exact `in` → whitespace-normalized → fuzzy SequenceMatcher ≥ 0.85<br/>ADDITION checks V2 only · DELETION checks V1 only"]
    end
    T2 -->|"any miss → FAILED_EVIDENCE"| DROP["❌ rejected → hallucination log"]
    T2 -->|pass| T3
    subgraph T3["Tier 3 · Numerical Verification  (regex, 100% strict)"]
        T3a["extract all numbers (date/%/currency/float/int)<br/>each must be found in raw_text"]
    end
    T3 -->|"any miss → FAILED_NUMERICAL"| DROP
    T3 -->|pass| T4
    subgraph T4["Tier 4 · Executive Summary + Markdown  (LLM, temp≈0.3)"]
        T4a["LLM over verified ACUs → ExecutiveSummary<br/>(fallback: deterministic Python summary)"]
        T4a --> T4b["render_markdown_report()"]
    end
    T4 --> REP["ComparisonReport<br/>verified_acus · rejected_acus · hallucination_rate"]
    DROP --> REP
```

```
ComparisonRequest (pair_id, match_type, raw_text_v1/v2, breadcrumbs)
        │
        ▼
┌─ TIER 1 — ACU Extraction (LLM, temp≈0.05) ─────────────────────────┐
│  build_acu_user_prompt() + strict ACU_SYSTEM_PROMPT                 │
│  recall boosters:                                                    │
│    B.  Windowing   — split long texts (>5000 chars) into windows    │
│    C1. Number enum — inject pre-extracted V1/V2 numbers into prompt │
│    C2. Two-pass    — 2nd call asks for "additional changes"          │
│  + logic_detector.build_logic_hint() (Category D awareness)          │
│  ⚡ Skip entirely if normalized(v1) == normalized(v2)                 │
└──────────────────────────────────────────────────────────────────────┘
        │  (DeepSeek only: _deepseek_self_verify — model re-checks its own ACUs)
        ▼
   Pre-filter: drop ACUs with confidence < min_confidence_to_include (0.2)
        │
        ▼
┌─ TIER 2 — Evidence Verification (programmatic) ─────────────────────┐
│  For each ACU: is verbatim_evidence really a substring of raw_text? │
│    1. exact `in` match                                               │
│    2. whitespace-normalized match                                    │
│    3. fuzzy SequenceMatcher sliding-window (threshold 0.85)          │
│  → FAIL_EVIDENCE (hallucination) if not found                       │
└──────────────────────────────────────────────────────────────────────┘
        │  (only if Tier 2 passed)
        ▼
┌─ TIER 3 — Numerical Verification (programmatic, regex) ─────────────┐
│  Extract all numbers from original_value/new_value; each MUST be    │
│  findable in the corresponding raw_text (exact → normalized → value)│
│  strict_numerical=True → 100% must match, else FAIL_NUMERICAL       │
│  (Also runs for non-numerical ACUs that happen to contain numbers.) │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─ TIER 4 — Executive Summary (LLM, temp≈0.3) + Markdown render ──────┐
│  One final LLM call over the verified ACUs → ExecutiveSummary       │
│  (overall assessment, critical changes, numerical summary, risks).  │
│  Fallback: deterministic Python summary if LLM fails.               │
│  render_markdown_report() assembles the Vietnamese report.          │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
   ComparisonReport (verified_acus, rejected_acus, executive_summary, markdown_report)
```

**Batching:** `run_batch()` processes all pairs concurrently with an `asyncio.Semaphore(max_concurrency)`; a single pair crashing returns an error-report instead of failing the batch.

#### Tier 1 internals — recall boosters & robustness

| Strategy | Where | Behaviour |
|----------|-------|-----------|
| **B. Windowing** (`_extract_windowed`, `_split_text_windows`) | `report_generator.py` | If `max(len_v1, len_v2) > windowing_threshold_chars` (**5000** local; auto-raised to **20000** for DeepSeek's 128K context, with `window_max_chars` 2500→8000), each side is split into sentence-boundary-aware windows (`window_overlap_chars=400`). Windows are paired **by position** (window *i* of V1 ↔ window *i* of V2) and each pair gets its own LLM call; ACUs from all windows are merged. Prevents output-token truncation on long articles. |
| **C1. Number enumeration** (`_build_number_hint`) | `report_generator.py` | Runs `extract_numbers` on both sides, lists up to **80** numbers per side, injects them into the prompt as a mandatory cross-check ("mỗi khác biệt phải thành 1 ACU"). |
| **C2. Two-pass** (`_second_pass`) | `report_generator.py` | Sends the pass-1 ACU list back and asks for *additional* changes only — pure recall boost. **Skipped** for `added`/`deleted` (they yield a single whole-text ACU). Merged via `_merge_dedupe`. |
| **S4. Logic-shift hint** (`logic_detector.build_logic_hint`) | `logic_detector.py` | Token-count diff over a meaning-lever lexicon (`không`, `được phép`, `phải`, `cấm`, `trừ khi`, `chỉ khi`, `nếu`, …) → injects `⚠️ thêm 'có thể', bỏ 'không'`. A *nudge*, never a gate. |
| **Dedupe** (`_acu_dedupe_key`) | `report_generator.py` | Key = `(change_type, norm(original_value), norm(new_value))`, where `norm` lowercases, collapses whitespace, strips `,.()[]"'«»/\`. ACUs with empty original+new value (pure addition/deletion) bypass dedupe. |
| **Truncation recovery** (`_extract_complete_json_objects`) | `llm_client.py` | If the LLM response is cut mid-object (`finish_reason=length`), a brace-balanced scan salvages all *complete* ACU objects before the cut (objects containing a `change_type` key). |
| **ACU recovery** (`_attempt_acu_recovery`) | `report_generator.py` | On Pydantic `ValidationError`: invalid `change_type` → fallback `"structural"`; out-of-range/missing `confidence` → `0.5`; then re-validate. Only drops an ACU if recovery itself fails. |
| **DeepSeek self-verification** (`_deepseek_self_verify`) | `report_generator.py` | DeepSeek-only, **on by default**: a second LLM call returns `{verified:[...], flagged:[...]}` against the raw texts; flagged ACUs are dropped and confidences adjusted *before* the programmatic Tiers 2/3 (cuts hallucination ~3–5%). |

### 7.3 The strict system prompt

The Tier-1 system prompt enforces 8 mandatory rules (violation ⇒ whole output rejected). The most important:

- **RULE 1 — verbatim only:** `verbatim_evidence_v1/v2` *must* be copy-paste from the input; paraphrasing/summarizing/inventing text is forbidden.
- **RULE 2 — atomic granularity:** one change per ACU (`30 ngày → 45 ngày` *and* `500.000 → 600.000 đồng` are **two** ACUs).
- **RULE 4 — evidence per type:** `addition` ⇒ `evidence_v1=""`; `deletion` ⇒ `evidence_v2=""`; others ⇒ both non-empty.
- **RULE 5 — no legal inference:** describe the change, do not opine on consequences.
- **RULE 8 — logic/meaning shifts:** flipping `không` ↔ `có thể`, `phải` ↔ `không bắt buộc`, adding `nếu/trừ khi/chỉ khi` are critical, easily-missed changes.

A **DeepSeek-optimized** variant (`ACU_SYSTEM_PROMPT_DEEPSEEK`) adds step-by-step reasoning guidance and a mandatory `reasoning` field, leveraging DeepSeek V4's chain-of-thought. When `provider="deepseek"`, the pipeline also auto-raises the windowing threshold (128K context) and enables self-verification.

### 7.4 Tier 2 — evidence verification (`verifier.py`)

```python
def _evidence_in_text(self, evidence, text) -> tuple[bool, float | None]:
    if evidence in text:                              # 1. exact
        return True, None
    if self._cfg.normalize_whitespace:                # 2. whitespace-normalized
        if self._normalize_ws(evidence) in self._normalize_ws(text):
            return True, None
    score = self._fuzzy_evidence_match(evidence, text) # 3. fuzzy sliding window
    return (score >= self._cfg.fuzzy_match_threshold, score)
```

The fuzzy fallback (`_fuzzy_evidence_match`) is only reached when both exact and whitespace-normalized matching fail. It searches a **capped** copy of the text (first 10 000 chars) with a sliding window whose length ranges from **80% → 150%** of the evidence length, stepping by `max(1, ev_len // 4)`; for each window it computes `difflib.SequenceMatcher(autojunk=False).ratio()` and keeps the max, with an early exit at ratio 1.0. A ratio ≥ `fuzzy_match_threshold` (**0.85**) is accepted.

| Per-`change_type` rule | V1 evidence | V2 evidence |
|---|---|---|
| `ADDITION` | skipped (expected `""`) | verified |
| `DELETION` | verified | skipped (expected `""`) |
| all others | verified | verified |

Evidence shorter than `min_evidence_length` (**5** chars) is skipped (treated as pass) to avoid false positives; a non-applicable side is `None` (not `False`). A side becomes `False` only when it *should* exist and can't be matched — that is what triggers `FAILED_EVIDENCE`.

### 7.5 Tier 3 — numerical verification (`verifier.py`)

A multi-pattern regex extracts every number (dates, percentages, VN/intl currency, floats, integers) from `original_value`/`new_value`. Each must be found in the matching raw text via a 3-strategy search:

```python
def _number_in_raw_text(self, num, text) -> bool:
    if num.raw_str in text: return True                       # 1. exact
    normed = re.sub(r"[\s\.]", "", num.raw_str).replace(",", ".")
    if normed in re.sub(r"\s", "", text): return True         # 2. normalized separators
    if num.value is not None and num.num_type not in ("date","integer_single"):
        if re.search(re.escape(str(num.value)), re.sub(r"[\.,]", "", text)):
            return True                                       # 3. value-based
    return False
```

**Numerical extraction patterns** (`_NUMERICAL_PATTERNS`, applied strictest/longest first, with span-overlap de-duplication so a date isn't re-counted as integers):

| `num_type` | Regex (essence) | Example |
|---|---|---|
| `date` | `\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b` | `01/06/2025` |
| `percentage` | `\b\d+(?:[,\.]\d+)?\s*%` | `10,5%` |
| `currency_vn` | `\b\d{1,3}(?:\.\d{3})+(?:,\d+)?\s*(?:đồng\|VNĐ\|VND\|đ)?` | `500.000 đồng` |
| `currency_intl` | `(?:USD\|EUR\|GBP\|\$\|€\|£)\s*\d{1,3}(?:[,.]\d{3})+(?:\.\d+)?` | `USD 1,000.00` |
| `float` | `\b\d+[,\.]\d+\b` | `3,14` |
| `integer` | `\b\d{2,}\b` | `30` |
| `integer_single` | `\b[1-9]\b` | `7` |

`_parse_numeric_value` reconciles Vietnamese vs. international decimal conventions: when both `,` and `.` appear, the **rightmost** one is the decimal mark (`1.000.000,50` → `1000000.50`; `1,000,000.50` → `1000000.50`); a lone `,` is treated as decimal if it has ≤ 2 trailing digits, otherwise as a thousands separator. Dates return `value=None` and are never compared arithmetically (only matched as strings / normalized strings).

`strict_numerical=True` ⇒ any missing number ⇒ `FAILED_NUMERICAL` and the ACU is dropped. (S5: this also runs for `terminology`/`structural`/`addition` ACUs that happen to contain numbers, catching fabricated figures in table-heavy docs.)

### 7.6 Logic-shift hints (`logic_detector.py`)

A deterministic lexicon (`không`, `được phép`, `phải`, `cấm`, `trừ khi`, `chỉ khi`, `nếu`, …) counts token-frequency differences between V1 and V2 and injects a hint like `⚠️ thêm 'có thể', bỏ 'không'` into the Tier-1 prompt. This is a *recall nudge*, not a gate — it steers the LLM toward meaning-flip changes (Category D) it tends to miss.

### 7.7 Identical-text optimization

Recent commit (`316798a`): if V1 and V2 are identical after whitespace normalization, the LLM call is skipped entirely:

```python
if re.sub(r"\s+", " ", v1).strip() == re.sub(r"\s+", " ", v2).strip():
    return []   # no changes → no ACUs, no LLM cost
```

### 7.8 Phase 3 worked example

**Input pair (match_type=matched):**
```
V1: "Bên A phải thanh toán 500.000 đồng trong vòng 30 ngày. Bên A không được chuyển nhượng."
V2: "Bên A phải thanh toán 600.000 đồng trong vòng 45 ngày. Bên A có thể chuyển nhượng nếu được bên B đồng ý."
```

**Tier 1 — LLM returns 3 ACUs** (JSON):
```json
{ "acus": [
  { "change_type":"numerical", "original_value":"500.000 đồng", "new_value":"600.000 đồng",
    "verbatim_evidence_v1":"Bên A phải thanh toán 500.000 đồng trong vòng 30 ngày",
    "verbatim_evidence_v2":"Bên A phải thanh toán 600.000 đồng trong vòng 45 ngày", "confidence":1.0 },
  { "change_type":"numerical", "original_value":"30 ngày", "new_value":"45 ngày",
    "verbatim_evidence_v1":"trong vòng 30 ngày", "verbatim_evidence_v2":"trong vòng 45 ngày", "confidence":1.0 },
  { "change_type":"terminology", "original_value":"không được chuyển nhượng",
    "new_value":"có thể chuyển nhượng nếu được bên B đồng ý",
    "verbatim_evidence_v1":"Bên A không được chuyển nhượng",
    "verbatim_evidence_v2":"Bên A có thể chuyển nhượng nếu được bên B đồng ý", "confidence":0.95 }
] }
```

**Tier 2** — all 6 evidence strings are exact substrings of the raw texts → all PASS.
**Tier 3** — `500.000`/`600.000`/`30`/`45` all found in the right raw text → all PASS.

**Tier 4 — Executive summary + Markdown** (excerpt):
```markdown
# 📋 Báo cáo So sánh Văn bản Pháp lý
...
## 📊 Thống kê Tổng quan
| Tổng ACU phát hiện | 3 | ACU đã xác minh (PASSED) | 3 | Tỷ lệ ảo giác | 0.0% |

## ✅ Các Thay đổi Đã Xác minh (3 ACU)
### 🔢 Thay đổi Số liệu (2)
1. Điều 5 _(confidence: 100%)_
   - Trước (V1): 500.000 đồng   - Sau (V2): 600.000 đồng
   - Bằng chứng V1: > "Bên A phải thanh toán 500.000 đồng trong vòng 30 ngày"
### 📝 Thay đổi Thuật ngữ (1)
1. Điều 5 _(confidence: 95%)_  - "không được chuyển nhượng" → "có thể chuyển nhượng nếu được bên B đồng ý"
```

If the LLM had invented an ACU whose evidence wasn't in the source, Tier 2 would mark it `FAILED_EVIDENCE` and it would appear instead under **❌ ACU Bị Bác bỏ — Nhật ký Ảo giác** — never in the verified changes.

---

## 8. Backend (FastAPI + Celery + WebSocket)

**Role:** wrap the `src/` engine as an authenticated, asynchronous web service with live progress and persistent results.

### 8.1 Stack

- **FastAPI** — async REST + WebSocket, automatic OpenAPI
- **Celery** — distributed task queue for long-running pipeline jobs
- **Redis** — Celery broker/backend **and** pub/sub bridge for WebSocket progress **and** the GPU lock
- **PostgreSQL** — primary metadata store (SQLAlchemy async for API, sync for the worker)
- **Local filesystem** — uploaded docs + generated reports

### 8.2 REST API (mounted under `/api`)

| Router | Method · Path | Purpose |
|--------|---------------|---------|
| auth | `POST /auth/register` · `POST /auth/login` · `POST /auth/refresh` · `GET /auth/me` | JWT register/login/refresh/profile |
| documents | `POST /documents/upload` · `GET /documents` · `GET /documents/{id}` · `DELETE /documents/{id}` | upload (dedup by SHA256+user), list/get/delete |
| jobs | `POST /jobs` · `GET /jobs` · `GET /jobs/{id}` · `GET /jobs/{id}/status` · `POST /jobs/{id}/cancel` · `DELETE /jobs/{id}` · `GET /jobs/{id}/reports` · `GET /jobs/{id}/report` · `GET /jobs/{id}/catalog` | create/list/status/cancel/report |
| reports | `GET /reports/{id}` · `GET /reports/{id}/markdown` | single report / markdown |
| settings | `GET /settings/llm` · `PUT /settings/llm` · `GET /settings/models` | LLM config (per-user), available models |
| eval | `GET /eval/runs` · `GET /eval/runs/{slug}` · `.../pairs` · `.../pairs/{name}` | evaluation dashboards |
| health | `GET /health` · `GET /health/ready` | basic / deep (DB·Redis·Worker·LLM) |

**`POST /jobs` request:**
```json
{
  "document_v1_id": "<uuid>",
  "document_v2_id": "<uuid>",
  "skip_phase3": false,
  "config_overrides": { "llm_temperature_acu": 0.1 }
}
```
It validates the docs, merges the user's saved settings with per-job overrides, creates a `ComparisonJob` row, and dispatches the Celery task `run_pipeline`.

### 8.3 Database models (SQLAlchemy)

| Model | Key columns |
|-------|-------------|
| `User` | id, email, username, password_hash (bcrypt), is_active/superuser |
| `Document` | id, user_id (FK), original_filename, storage_path, content_hash_sha256, doc_title/number/parties, page_count, article_count, is_processed |
| `ComparisonJob` | id, user_id, document_v1_id/v2_id (FK), **status** (pending/processing/completed/failed/cancelled), **current_phase** (queued/ingestion/alignment/comparison/done), progress_pct, error_message, catalog (JSONB), matched/added/deleted/split/merge counts, config_snapshot (JSONB), celery_task_id |
| `ComparisonReportModel` | id, job_id (FK), pair_id, match_type, total/passed/rejected_acus, hallucination_rate, executive_summary (JSONB), markdown_report, verified_acus (JSONB) |
| `UserSettings` | id, user_id (unique), llm_config (JSONB) |
| `EvalRun` / `EvalPair` | benchmark metrics (aggregate, per-phase, per-category JSONB) |

Migrations: `alembic -c backend/alembic.ini upgrade head`.

### 8.4 The Celery worker — `workers/pipeline_task.py` (critical)

This is where a job actually executes. Flow:

1. **Acquire GPU lock** (`GPULock`, Redis `SET NX` with 15-min TTL) — serializes BGE-M3 jobs so only one runs at a time.
2. Mark job `processing`, set `started_at`.
3. Wire a **progress callback** that both updates the DB (`current_phase`, `progress_pct`) and publishes to Redis.
4. Build `PipelineRunConfig` from `src.config` + the user's settings/overrides.
5. Run `LegalDiffPipeline(...).run()`.
6. Store results: update the job's catalog summary + counts, insert one `ComparisonReportModel` row per pair.
7. Mark job `completed` (or `failed` in `PipelineTask.on_failure`, which also releases the lock).

**Progress → WebSocket bridge:** the callback publishes to Redis channel `job:{job_id}:progress`:
```python
payload = {"event": "progress", "job_id": ..., "progress_pct": pct,
           "current_phase": phase, "message": message, "error": None}
r.publish(f"job:{job_id}:progress", json.dumps(payload))
```
Each uvicorn API process subscribes via `ConnectionManager.listen_redis()` and forwards to the browser sockets connected to that job — so progress works even with multiple API workers behind the Celery worker.

**Celery config** (`celery_app.py`): `worker_concurrency=1`, `worker_prefetch_multiplier=1`, `task_acks_late=True` — one pipeline at a time per worker, GPU-safe.

### 8.5 WebSocket — `ws/`

Endpoint: `WS /ws/jobs/{job_id}?token=<JWT>` (token as query param because WS can't use the `Authorization` header). The handler authenticates the token, verifies the caller owns the job, accepts the connection, and starts the Redis listener. Message shape consumed by the frontend:

```json
{ "event": "progress" | "completed" | "error",
  "job_id": "...", "progress_pct": 55, "current_phase": "comparison",
  "message": "Đang sinh báo cáo so sánh...", "error": null }
```

---

## 9. Frontend (React + Vite)

**Stack:** React 19 + TypeScript + Vite, **Zustand** (auth store), **TanStack Query** (server state), **Tailwind + shadcn/ui**, react-hook-form + zod, react-i18next (Vietnamese UI). Custom diff-oriented color tokens (`diff-addition`, `diff-deletion`, `diff-numerical`, `diff-terminology`, …).

### 9.1 Routes (`router.tsx`)

| Path | Page | Auth |
|------|------|------|
| `/login`, `/register` | LoginPage, RegisterPage | public |
| `/` | DashboardPage (job KPIs + recent jobs + eval) | protected |
| `/compare` | CreateJobPage (upload V1/V2, run) | protected |
| `/report/:id` | ReportPage (live progress + report) | protected |
| `/settings` | SettingsPage (LLM provider/thresholds) | protected |
| `/eval`, `/eval/:slug`, `/eval/:slug/:pairName` | Eval dashboards | protected |

`ProtectedRoute` wraps protected routes and redirects to `/login` on 401.

### 9.2 API client (`api/client.ts`)

A tiny typed wrapper injecting the JWT (from the Zustand store) into every request and handling 401 globally. Per-domain modules: `jobs.ts`, `documents.ts`, `report.ts`, `settings.ts`, `eval.ts`.

```typescript
const api = {
  get:    <T>(p: string) => request<T>(p),
  post:   <T>(p: string, body?: unknown) => request<T>(p, { method:'POST', body: body?JSON.stringify(body):undefined }),
  put:    <T>(p: string, body?: unknown) => request<T>(p, { method:'PUT',  body: body?JSON.stringify(body):undefined }),
  delete: <T>(p: string) => request<T>(p, { method:'DELETE' }),
}
```

### 9.3 `useWebSocket` hook

Connects to `/ws/jobs/{id}?token=…`, parses each message, and **reconnects with exponential backoff** (max 30s). The `ReportPage` feeds messages into `PhaseStepper` and invalidates the job query on terminal events.

### 9.4 Key components

- **`DocumentUploader`** — drag-and-drop PDF/DOCX (≤50MB) upload for V1 and V2.
- **`PhaseStepper`** — 3-step live progress (Giai đoạn 1 Nhập liệu → 2 Căn chỉnh → 3 So sánh), driven by the WS `current_phase` / `progress_pct`.
- **`ExecutiveSummary`** + **`AcuList`** + **`FilterBar`** — render the report: severity/type breakdown, color-coded ACU cards with side-by-side V1/V2 evidence, filters (type/severity) and sort.

### 9.5 `ReportPage` data flow

WebSocket drives live progress; a `useQuery(['job', id])` polls `/jobs/{id}` every 2s as a fallback while running; once `status === completed`, a `useQuery(['report', id])` fetches `/jobs/{id}/report` and the ACU list renders with client-side filtering/sorting.

---

## 10. Data Stores & Persistence

| Store | Location | Purpose | Used by |
|-------|----------|---------|---------|
| **PostgreSQL** | `data/pg_data/` | users, documents, jobs, reports, eval, user_settings | backend (async API + sync worker) |
| **Kuzu (graph)** | `data/processed/graph_db/` | legal DOM graph: `CONTAINS`/`PRECEDES`/`REFERENCES` edges | Phase 1 `HybridGraphBuilder` (standalone ingestion only) |
| **Qdrant** | in-memory (default) or `data/qdrant_*/` | Phase 2 dense+sparse vectors, multi-vector collections | Phase 2 alignment |
| **ChromaDB** | `data/processed/chroma_db/` | Phase 1 chunk vectors (legacy/alt knowledge base) | Phase 1 `HybridGraphBuilder` only |
| **Filesystem** | `data/raw/`, `data/reports/`, `backend/storage/` | uploaded docs, generated JSON+MD reports | backend storage service |

**Embedding models:** **BGE-M3** (`BAAI/bge-m3`, ~2.3GB, downloaded by FlagEmbedding on first use) is the active Phase 2 model — dense(1024)+sparse+colbert. The `models/multilingual-e5-large-instruct/` directory is a **legacy/experimental** weight set not used by the current pipeline.

---

## 11. Configuration & Environment

### 11.1 YAML configs (`configs/`)

**`pipeline_config.yaml`** — runtime thresholds & weights:

```yaml
ingestion:  { max_chunk_chars: 2000, overlap_chars: 200, confidence_threshold: 0.75 }
alignment:
  w_semantic: 0.5      # must sum to 1.0 with the next three
  w_jaro_winkler: 0.25
  w_ordinal: 0.1
  w_sparse: 0.15
  match_threshold: 0.60
  split_merge_threshold: 0.80
  qdrant: { path: null }   # null = in-memory
comparison:
  fuzzy_match_threshold: 0.85
  strict_numerical: true
  min_confidence_to_include: 0.2
  max_concurrency: 1
```

**`model_config.yaml`** — model paths + provider presets (BGE-M3 fp16/batch16; local `qwen2.5:14b` via Ollama at `localhost:11434/v1`; `deepseek-v4-flash` preset with `temperature_acu: 0.0`).

`src/config.py` loads both (merged over safe defaults), resolves the LLM provider (`get_llm_provider()` ← env `LLM_PROVIDER` ← YAML ← `"local"`), and resolves DeepSeek credentials from env.

### 11.2 Environment

There are two layers of environment configuration:

**(a) Root `.env_example` — LLM provider selection only** (loaded by `src/config.py` via python-dotenv). This is the entire contents of the committed template:

```env
# --- DeepSeek API (LLM provider) ---
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-v4-flash        # or deepseek-v4-pro for max quality

# --- LLM Provider Selection ---
LLM_PROVIDER=local                      # "local" (Qwen/Ollama) or "deepseek"

# --- Local LLM (Qwen via Ollama/vLLM) — used when LLM_PROVIDER=local ---
# LOCAL_LLM_BASE_URL=http://localhost:11434/v1
# LOCAL_LLM_MODEL=qwen2.5:14b
```

**DeepSeek requires** `DEEPSEEK_API_KEY` (raises `ValueError` otherwise); `DEEPSEEK_BASE_URL`/`DEEPSEEK_MODEL` fall back to the `provider_presets.deepseek` block in `model_config.yaml`.

**(b) Backend runtime vars** — read from env / a backend `.env` by `backend/config.py` (with safe defaults baked in). The key ones:

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABASE_URL` / `DATABASE_URL_SYNC` | `postgresql+asyncpg://legaldiff:...@localhost:5432/legaldiff` | async (API) / sync (worker, Alembic) |
| `REDIS_URL` | `redis://localhost:6379/0` | Celery broker + WS pub/sub + GPU lock |
| `JWT_SECRET_KEY` / `JWT_ALGORITHM` | `change-me...` / `HS256` | auth tokens (access 60 min, refresh 7 days) |
| `LLM_BASE_URL` / `LLM_MODEL_NAME` | `http://localhost:8000/v1` / `Qwen/Qwen2.5-7B-Instruct` | fallback LLM if user has no saved settings |
| `STORAGE_ROOT` / `MAX_UPLOAD_SIZE_MB` | `./backend/storage` / 100 | file storage |

> The root `.env` (provider) and backend env (infra) are independent — the pipeline only reads the provider keys; the backend service reads the infra keys.

**Config resolution priority** for a job: system defaults (`src/config.py`) → user's saved `UserSettings.llm_config` → per-job `config_overrides`.

---

## 12. End-to-End Example (Full Walkthrough)

A user compares two contract versions end to end:

1. **Auth** — `POST /api/auth/login` → JWT stored in the frontend Zustand store.
2. **Upload** — On `/compare`, two `DocumentUploader`s `POST /api/documents/upload`. Each file is saved, SHA256-hashed (deduped per user), and a `Document` row is created.
3. **Create job** — Click *Bắt đầu so sánh* → `POST /api/jobs {document_v1_id, document_v2_id}`. Backend creates a `ComparisonJob` (status `pending`) and dispatches Celery `run_pipeline`. Frontend navigates to `/report/{jobId}`.
4. **WebSocket** — `useWebSocket` opens `/ws/jobs/{jobId}?token=…`. The worker acquires the GPU lock, sets status `processing`, and starts emitting progress.
5. **Phase 1 (≈10–30%)** — Docling parses both PDFs into Legal DOMs; LSU chunker splits into chunks. WS: `phase1_ingestion`, pct climbing.
6. **Phase 2 (≈30–55%)** — BGE-M3 embeds article nodes; similarity matrix; Hungarian + split/merge → `DiffPairCatalog`. WS: `phase2_alignment`.
7. **Phase 3 (≈55–95%)** — For each pair: LLM extracts ACUs (windowed/two-pass/number-hinted) → Tier 2 evidence check → Tier 3 number check → Tier 4 summary → Markdown. WS: `phase3_comparison`.
8. **Persist** — Worker inserts `ComparisonReportModel` rows, updates job counts + `status=completed`, publishes `event: completed`, releases the GPU lock.
9. **View report** — Frontend receives `completed`, fetches `/api/jobs/{id}/report`, and `ReportPage` renders the executive summary + filterable ACU cards (color-coded, with verbatim V1/V2 evidence).

**Example WS sequence:**
```json
{"event":"progress","current_phase":"phase1_ingestion","progress_pct":0.10,"message":"Đang phân tích tài liệu..."}
{"event":"progress","current_phase":"phase2_alignment","progress_pct":0.40,"message":"Đang đối chiếu điều khoản..."}
{"event":"progress","current_phase":"phase3_comparison","progress_pct":0.75,"message":"Đang so sánh..."}
{"event":"completed","current_phase":"comparison","progress_pct":1.0,"message":"Hoàn thành!"}
```

---

## 13. Deployment & Running

### 13.1 Service topology (dev, tmux)

A `legaldiff` tmux session with 3 windows:

| Window | Service | Command | Notes |
|--------|---------|---------|-------|
| `:0` | FastAPI API | `uvicorn backend.main:app --reload --host 0.0.0.0 --port 8001` | auto-reloads |
| `:1` | Frontend | `cd frontend && npm run dev` (Vite, :5173) | HMR |
| `:2` | Celery worker | `PYTHONPATH=$(pwd) celery -A backend.celery_app worker --concurrency=1 --loglevel=info` | **manual restart** on code change |

Plus infrastructure: **PostgreSQL** (:5432), **Redis** (:6379), and an **LLM server** (Ollama :11434 with `qwen2.5:14b`, or vLLM).

### 13.2 First-time setup

```bash
# env + deps
conda create -n l_rag python=3.10 -y && conda activate l_rag
pip install -r requirements.txt
cp .env_example .env          # fill in secrets

# infra
docker compose -f backend/docker-compose.yml up -d postgres redis
ollama pull qwen2.5:14b       # (local provider)

# db
alembic -c backend/alembic.ini upgrade head

# frontend
cd frontend && npm install
```

### 13.3 Run / test / build

```bash
# backend
uvicorn backend.main:app --reload --port 8001
PYTHONPATH=$(pwd) celery -A backend.celery_app worker --concurrency=1 --loglevel=info
pytest tests/

# frontend
npm run dev          # dev server
npm run build        # production build
npm run test         # vitest
npm run test:e2e     # playwright
```

### 13.4 Production

Docker Compose (Option A) containerizes Postgres + Redis + API + worker (GPU support); the frontend is built and served by **nginx**, which also proxies `/api` → API and `/ws` → the WebSocket endpoint (with TLS). The local Qwen model is replaced by a quantized `Qwen2.5-7B` served by llama.cpp, or by the DeepSeek API (`LLM_PROVIDER=deepseek`).

---

## 14. Appendix: Configuration Reference & File Map

### 14.1 Key parameter reference

| Phase | Parameter | Default (runtime) | Meaning |
|-------|-----------|-------------------|---------|
| 1 | `confidence_threshold` | 0.75 | OCR fallback (Docling → Marker) trigger |
| 1 | `max_chunk_chars` / `overlap_chars` | 2000 / 200 | LSU chunk size / overlap |
| 2 | `w_semantic` / `w_jaro` / `w_ordinal` / `w_sparse` | 0.50 / 0.25 / 0.10 / 0.15 | similarity weights (sum=1.0) |
| 2 | `ARTICLE_NUMBER_BONUS` | +0.15 | same article-number bonus |
| 2 | `match_threshold` (θ) | 0.60 | min score for a MATCHED pair |
| 2 | `split_merge_threshold` | 0.80 | cosine to confirm split/merge |
| 2 | `use_fp16` / `max_length` / dense dim | True / 1024 / 1024 | BGE-M3 settings |
| 3 | `acu_temperature` / `summary_temperature` | 0.05 / 0.3 | LLM temperatures |
| 3 | `fuzzy_match_threshold` (Tier 2) | 0.85 | evidence fuzzy-match cutoff |
| 3 | `strict_numerical` (Tier 3) | True | 100% numbers must match |
| 3 | `min_confidence_to_include` | 0.2 | ACU pre-filter |
| 3 | `windowing_threshold_chars` | 5000 (20k DeepSeek) | long-text windowing |
| 3 | `enable_second_pass` / `enable_number_enumeration` | True / True | recall boosters |
| 3 | `max_concurrency` | 1 (YAML) / 4 (lib default) | parallel pairs |

### 14.2 Core file map

| Concern | File |
|---------|------|
| Pipeline orchestration | `src/pipeline.py` |
| Config / provider resolution | `src/config.py`, `configs/*.yaml` |
| Phase 1 parse / chunk / store | `src/ingestion/{docling_parser,lsu_chunker,knowledge_store,models}.py` |
| Phase 2 embed / score / match / catalog | `src/alignment/{embedder,similarity_matrix,hungarian_matcher,diff_catalog,qdrant_indexer,alignment_engine}.py` |
| Phase 3 prompt / LLM / verify / report | `src/comparison/{report_generator,acu_prompter,llm_client,verifier,logic_detector,models}.py` + `configs/prompt_templates/*` |
| Backend app / routers | `backend/main.py`, `backend/api/*` |
| Celery worker + GPU lock | `backend/workers/{pipeline_task,gpu_lock}.py` |
| WebSocket | `backend/ws/{manager,handlers}.py` |
| ORM models | `backend/models/*` |
| Frontend entry / routing | `frontend/src/{main,App,router}.tsx` |
| Frontend WebSocket | `frontend/src/hooks/useWebSocket.ts` |

---

*This document describes the codebase as of branch `hieuvu`. For per-phase deep dives see `docs/phase1_ingestion.md`, `docs/phase2_alignment.md`, and the inline docstrings in `src/`; for operations see `docs/RUNNING.md`.*
