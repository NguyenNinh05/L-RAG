# Legal RAG Comparison System — QWEN Context

## Project Overview

**Legal RAG Comparison System** is a Vietnamese legal document comparison platform that uses **Retrieval-Augmented Generation (RAG)** and **Local LLM** (via Ollama) to detect, classify, and analyze differences between two versions of legal documents (contracts, decrees, circulars, etc.).

The system automates the comparison of PDF/DOCX files through a multi-stage pipeline:
1. **Ingestion** — Parses PDF/DOCX, recognizes Vietnamese legal document structure (Phần, Chương, Mục, Điều, Khoản, Điểm)
2. **Embedding** — Converts text chunks into vectors via Ollama embedding API, stores in ChromaDB
3. **Retrieval** — Semantic matching using Anchor Detection + Needleman-Wunsch alignment
4. **LLM Analysis** — Qwen3-4B analyzes changed clauses and generates impact assessments
5. **Report** — Generates professional Markdown reports with comparison tables

### Architecture

```
PDF/DOCX → Ingestion → Embedding → Retrieval → LLM → Markdown Report + Web UI
```

Key modules:
- `ingestion/` — Document parsing (pymupdf4llm, python-docx), chunking, normalization
- `embedding/` — Ollama batch embedding + ChromaDB vector storage
- `retrieval/` — Semantic matching (Anchor Detection + Needleman-Wunsch alignment)
- `llm/` — LLM-based analysis via Ollama Chat API
- `comparison/` — Comparison result models, deterministic analysis
- `chat_service.py` — Post-comparison Q&A over session data
- `session_store.py` — SQLite-based session persistence and retrieval
- `api.py` — FastAPI backend with SSE streaming
- `main.py` — CLI entry point

## Tech Stack

| Category | Technology |
|----------|------------|
| Backend | FastAPI, Uvicorn, SSE streaming |
| Embedding | Qwen3-Embedding 0.6B (via Ollama) |
| LLM | Qwen3-4B-Instruct GGUF Q4_K_M (via Ollama) |
| Vector DB | ChromaDB (cosine distance) |
| Document Parsing | pymupdf4llm, python-docx |
| Session Storage | SQLite |
| Async HTTP | aiohttp |
| Alignment | Custom Needleman-Wunsch + difflib |

## Building and Running

### Prerequisites
- **Python 3.10+**
- **Ollama** installed and running (`ollama serve`)
- Required models pulled:
  ```bash
  ollama pull qwen3-embedding:0.6b
  ollama pull hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
  ```

### Setup
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

### Run Modes

**CLI mode:**
```bash
python main.py <file_v1.pdf> <file_v2.docx>
```

**Web UI + API server:**
```bash
python -m uvicorn api:app --reload --port 8000
```
Access at `http://localhost:8000`

**Direct API call:**
```bash
curl -X POST http://localhost:8000/api/compare \
  -F "file_a=@doc1.pdf" -F "file_b=@doc2.pdf"
```

## Key Configuration (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBEDDING_MODEL_NAME` | `qwen3-embedding:0.6b` | Embedding model via Ollama |
| `OLLAMA_LLM_MODEL` | `hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M` | LLM for analysis |
| `UNCHANGED_THRESHOLD` | `0.95` | Cosine similarity threshold for unchanged |
| `MODIFIED_THRESHOLD` | `0.75` | Below this = DELETED+ADDED |
| `GAP_PENALTY` | `0.40` | Needleman-Wunsch gap cost |
| `TEXT_UNCHANGED_RATIO` | `0.998` | Character-level difflib ratio check |
| `LLM_ENABLE_REPORT` | `true` (env: `LLM_ENABLE_REPORT`) | Toggle LLM report generation |
| `LLM_NUM_CTX` | `16384` | Context window size |
| `CHROMA_KEEP_SESSIONS` | `5` | Max ChromaDB collections to retain |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serve Web UI |
| `POST` | `/api/compare` | Compare two documents (SSE stream) |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/sessions` | List completed sessions |
| `GET` | `/api/sessions/{id}` | Get session details |
| `DELETE` | `/api/sessions/{id}` | Delete a session |
| `DELETE` | `/api/sessions` | Delete all sessions |
| `POST` | `/api/sessions/{id}/chat` | Ask Q&A about comparison (SSE) |
| `GET` | `/api/sessions/{id}/citations/{cid}` | Get citation details |

### SSE Events (`/api/compare`)
- `session` — Session ID and status
- `progress` — Step-by-step progress (step, total, status, title, detail)
- `stats` — Change statistics (modified, added, deleted, unchanged)
- `analysis` — Full comparison result
- `citations` — Source citations
- `report` — Full Markdown report
- `done` — Completion signal
- `error` — Error details

## Key Files

| File | Purpose |
|------|---------|
| `api.py` | FastAPI app with SSE streaming, session management, chat endpoint |
| `main.py` | CLI orchestrator running the full pipeline |
| `config.py` | Centralized configuration (models, thresholds, paths) |
| `chat_service.py` | Intelligent Q&A over comparison sessions with query planning |
| `session_store.py` | SQLite-backed session persistence with citation tracking |
| `ingestion/__init__.py` | Pipeline orchestrator: `process_two_documents()` |
| `ingestion/chunker.py` | Vietnamese legal structure recognition & chunking |
| `retrieval/matcher.py` | Anchor detection + Needleman-Wunsch semantic matching |
| `llm/generator.py` | LLM-based comparison report generation |
| `embedding/embedder.py` | Batch embedding + ChromaDB storage |
| `comparison/models.py` | Data models: `ComparisonResult`, `ClauseResult`, `ChangeRecord` |

## Development Conventions

- **Modular design**: Each module (`ingestion/`, `embedding/`, `retrieval/`, `llm/`) is self-contained with `__init__.py` as the public API
- **Async where applicable**: API uses SSE streaming via `AsyncIterator`; LLM calls use `asyncio`
- **Deterministic fallback**: Chat service has rule-based query planning with deterministic fallback if LLM fails
- **Session isolation**: Each comparison creates a timestamped ChromaDB collection; sessions are persisted in SQLite
- **Citation tracking**: All evidence is tracked with stable `citation_id`s for auditability
- **Error resilience**: Interrupted sessions are recovered on startup; chunk disconnects are detected mid-stream
- **Offline-first**: Everything runs locally via Ollama — no external API calls

## Test Documents

Sample test documents are in `docs_test/` directory. Actual PDF/DOCX files are git-ignored (except in `docs_test/`) for security.

## Notable Features

- **Vietnamese legal document structure parsing**: Recognizes Phần, Chương, Mục, Điều, Khoản, Điểm hierarchy
- **Two-tier semantic matching**: Anchor detection (fast, for unchanged text) + Needleman-Wunsch (for changed regions)
- **Grounded Q&A**: Post-comparison chat uses citation-backed evidence, not hallucination
- **SSE real-time UI**: Web interface shows live progress during processing
- **Session history**: All comparisons are persisted and can be revisited/queried later


<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **L-RAG** (653 symbols, 1898 relationships, 56 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows related to the issue
2. `gitnexus_context({name: "<suspect function>"})` — see all callers, callees, and process participation
3. `READ gitnexus://repo/L-RAG/process/{processName}` — trace the full execution flow step by step
4. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` — see what your branch changed

## When Refactoring

- **Renaming**: MUST use `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first. Review the preview — graph edits are safe, text_search edits need manual review. Then run with `dry_run: false`.
- **Extracting/Splitting**: MUST run `gitnexus_context({name: "target"})` to see all incoming/outgoing refs, then `gitnexus_impact({target: "target", direction: "upstream"})` to find all external callers before moving code.
- After any refactor: run `gitnexus_detect_changes({scope: "all"})` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/L-RAG/context` | Codebase overview, check index freshness |
| `gitnexus://repo/L-RAG/clusters` | All functional areas |
| `gitnexus://repo/L-RAG/processes` | All execution flows |
| `gitnexus://repo/L-RAG/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## Keeping the Index Fresh

After committing code changes, the GitNexus index becomes stale. Re-run analyze to update it:

```bash
npx gitnexus analyze
```

If the index previously included embeddings, preserve them by adding `--embeddings`:

```bash
npx gitnexus analyze --embeddings
```

To check whether embeddings exist, inspect `.gitnexus/meta.json` — the `stats.embeddings` field shows the count (0 means no embeddings). **Running analyze without `--embeddings` will delete any previously generated embeddings.**

> Claude Code users: A PostToolUse hook handles this automatically after `git commit` and `git merge`.

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->