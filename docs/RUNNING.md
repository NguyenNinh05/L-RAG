# LegalDiff — End-to-End Run Guide

This guide explains how to run **every service** in the LegalDiff stack locally,
from a clean clone to a working app in the browser. It covers two paths:

- **Option A — Docker Compose** (one command, self-contained, uses `llama.cpp`)
- **Option B — Local dev** (the day-to-day setup, uses **Ollama**, matches the `tmux` workflow)

> Architecture, design rationale, and the 3-phase pipeline theory live in
> [`../README.md`](../README.md) and [`project_overview.md`](project_overview.md).
> This file is purely about **getting it running**.

---

## 1. Services & ports

LegalDiff is a small micro-service stack. In local dev it looks like this:

| Service | Tech | Default port | Purpose |
|---|---|---|---|
| **PostgreSQL 16** | database | `5432` | Jobs, users, reports, eval data |
| **Redis 7** | broker + result backend | `6379` | Celery queue + task results |
| **LLM server** | Ollama (local dev) / llama.cpp (compose) | `11434` / `8000` | Qwen2.5 generation (Phase 3) |
| **Backend API** | FastAPI + Uvicorn | `8001` | REST + WebSocket endpoints |
| **Celery worker** | Celery (GPU, concurrency=1) | — | Runs the 3-phase pipeline |
| **Frontend** | React + Vite | `5173` | UI; proxies `/api` & `/ws` → `8001` |

Embedding model **BGE-M3** (`BAAI/bge-m3`) runs in-process inside the worker
(downloaded automatically on first run via `FlagEmbedding`).

---

## 2. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| OS | Ubuntu 22.04 (or similar Linux) | macOS works for API/FE; worker wants Linux + NVIDIA |
| Python | **3.10** | `l_rag` conda env in this repo |
| Node.js | **20.x** | for the frontend |
| Docker + Compose | recent | easiest way to run Postgres + Redis |
| NVIDIA GPU + CUDA | optional but recommended | BGE-M3 + Qwen are slow on CPU |
| `ollama` | latest | only for **Option B** (local LLM) |

---

## 3. Option A — Docker Compose (fastest, self-contained)

Everything (Postgres, Redis, LLM server, API, worker) is defined in
[`../backend/docker-compose.yml`](../backend/docker-compose.yml).

```bash
# from repo root
export JWT_SECRET_KEY="something-long-and-random"          # required
export POSTGRES_PASSWORD="legaldiff_secret"                # optional override

docker compose -f backend/docker-compose.yml up --build
```

This starts:

- `postgres` (5432), `redis` (6379)
- `llm-server` — **llama.cpp** serving `qwen2.5-7b-instruct-q4_k_m.gguf` on `8000` (GPU)
- `web` — runs `alembic upgrade head`, then `uvicorn … --port 8001`
- `worker` — Celery worker (GPU, concurrency=1)

> ⚠️ The compose `llm-server` uses **llama.cpp on :8000** and a 7B model. If you
> prefer **Ollama** (the local-dev default), use **Option B** instead.

Open the API docs at <http://localhost:8001/docs>. There is no frontend container
in compose — run the frontend separately (see step 8 of Option B) and it will
proxy to `8001`.

To stop: `docker compose -f backend/docker-compose.yml down` (add `-v` to wipe DB volumes).

---

## 4. Option B — Local dev (Ollama, the `tmux` workflow)

This is the setup used for everyday development. Each service runs in its own
terminal/tmux window so logs are easy to watch.

### Step 1 — Clone & create the Python environment

```bash
git clone <repo-url> L-RAG && cd L-RAG

conda create -n l_rag python=3.10 -y
conda activate l_rag

# Install BOTH requirement files:
#   requirements.txt         -> pipeline (Phase 1-3: docling, kuzu, qdrant, FlagEmbedding…)
#   backend/requirements.txt -> API + worker (fastapi, sqlalchemy, celery, redis…)
pip install -r requirements.txt -r backend/requirements.txt
```

### Step 2 — Start Postgres + Redis

Easiest is to reuse the compose definitions for just the infra:

```bash
docker compose -f backend/docker-compose.yml up -d postgres redis
```

This creates DB `legaldiff`, user `legaldiff`, password `legaldiff_secret`
(matching the defaults in [`backend/config.py`](../backend/config.py)).

<details>
<summary>Using a system Postgres instead of Docker</summary>

```bash
sudo -u postgres createuser legaldiff --createdb
sudo -u postgres psql -c "ALTER USER legaldiff WITH PASSWORD 'legaldiff_secret';"
sudo -u postgres createdb legaldiff -O legaldiff
```
</details>

Verify:
```bash
docker compose -f backend/docker-compose.yml ps        # postgres + redis healthy
redis-cli ping                                          # -> PONG
```

### Step 3 — Start Ollama and pull the model

```bash
ollama serve &                  # if not already running
ollama pull qwen2.5:14b         # the model configured in configs/model_config.yaml
```

Verify: `curl http://localhost:11434/api/tags` should list `qwen2.5:14b`.

### Step 4 — Configure environment

The backend reads env vars with sensible defaults ([`backend/config.py`](../backend/config.py)).
For local dev with Ollama, create **`backend/.env`** (git-ignored) and override the
LLM endpoint away from the compose default (`:8000`) to Ollama (`:11434`):

```dotenv
# backend/.env  (git-ignored — never commit real secrets)
DATABASE_URL=postgresql+asyncpg://legaldiff:legaldiff_secret@localhost:5432/legaldiff
DATABASE_URL_SYNC=postgresql://legaldiff:legaldiff_secret@localhost:5432/legaldiff
REDIS_URL=redis://localhost:6379/0

# IMPORTANT for local dev: point the backend at Ollama, not the compose llama.cpp (:8000)
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL_NAME=qwen2.5:14b

JWT_SECRET_KEY=change-me-to-a-long-random-string
STORAGE_ROOT=./backend/storage
LOG_LEVEL=INFO
```

> The actual comparison pipeline (Phase 3) reads `configs/model_config.yaml`,
> which already targets Ollama (`http://localhost:11434/v1`, `qwen2.5:14b`). To
> switch the **whole** pipeline to the DeepSeek API instead, see §6.

### Step 5 — Run database migrations

```bash
# from repo root, l_rag env active
alembic -c backend/alembic.ini upgrade head
```

Alembic reads `DATABASE_URL_SYNC` (see [`backend/alembic/env.py`](../backend/alembic/env.py)).
Current heads: `001_initial`, `002_mime`, `003_user_settings`, `004_eval_tables`.

### Step 6 — Start the Backend API

```bash
# repo root, l_rag env active
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8001
```

Open <http://localhost:8001/docs> → you should see the Swagger UI.

### Step 7 — Start the Celery worker (GPU)

The worker executes the 3-phase pipeline. `concurrency=1` is intentional — only
one GPU-bound job runs at a time.

```bash
# repo root (or frontend/), l_rag env active
PYTHONPATH=$(pwd) celery -A backend.celery_app worker --concurrency=1 --loglevel=info
```

Wait for `celery@<host> ready.` in the logs.

> **Code changes do not hot-reload the worker.** After editing anything under
> `src/` or `backend/workers/`, restart it: `Ctrl+C` (warm shutdown) then re-run
> the command above. The API (`--reload`) reloads itself.

### Step 8 — Start the frontend

```bash
cd frontend
npm install
npm run dev          # Vite dev server on http://localhost:5173
```

Open <http://localhost:5173>. Vite proxies `/api` and `/ws` to
`http://localhost:8001` (see [`frontend/vite.config.ts`](../frontend/vite.config.ts)),
so the API and worker must be running.

---

## 5. Verify the whole stack

| Check | Expected |
|---|---|
| `curl http://localhost:8001/docs` | HTML (Swagger UI) / redirect |
| `curl http://localhost:11434/api/tags` | JSON listing `qwen2.5:14b` |
| `redis-cli ping` | `PONG` |
| Worker log | `celery@<host> ready.` |
| Browser <http://localhost:5173> | LegalDiff login/dashboard |

End-to-end smoke test: **Register a user → Create job → upload a v1 and a v2
PDF/DOCX → watch the worker process pairs → open the generated report.** On
identical v1/v2 documents you will now see the worker log
`Skipping LLM call for pair_id=… because v1 and v2 texts are identical after normalization.`
(cost-saving fast path in `src/comparison/report_generator.py`).

---

## 6. Configuration reference

### 6.1 Backend env vars (`backend/.env`) — defaults from `backend/config.py`

| Var | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://legaldiff:legaldiff_secret@localhost:5432/legaldiff` | Async DB URL (API) |
| `DATABASE_URL_SYNC` | `postgresql://legaldiff:legaldiff_secret@localhost:5432/legaldiff` | Sync DB URL (worker + Alembic) |
| `REDIS_URL` | `redis://localhost:6379/0` | Celery broker + result backend |
| `JWT_SECRET_KEY` | `change-me-in-production` | **Set this in any real deployment** |
| `LLM_BASE_URL` | `http://localhost:8000/v1` | LLM endpoint (set to Ollama `:11434` locally) |
| `LLM_MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | Model name |
| `STORAGE_ROOT` | `./backend/storage` | Uploads / processed / reports |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

### 6.2 Pipeline config (`configs/`)

- [`model_config.yaml`](../configs/model_config.yaml) — embedding (`BAAI/bge-m3`),
  LLM provider + temperatures. **Provider switch:** `llm.provider: local|deepseek`
  (or env `LLM_PROVIDER`, or CLI `--provider`). For DeepSeek, set `DEEPSEEK_API_KEY`
  in `.env` (see [`.env_example`](../.env_example)).
- [`pipeline_config.yaml`](../configs/pipeline_config.yaml) — chunk sizes, alignment
  similarity weights, match thresholds, Qdrant (in-memory by default), comparison
  verification thresholds.

### 6.3 Choosing an LLM provider

| Provider | When | How |
|---|---|---|
| **local** (default) | Free, private, needs GPU | Ollama + `qwen2.5:14b`; set in `configs/model_config.yaml` |
| **deepseek** | Best quality, no GPU, costs $ | `LLM_PROVIDER=deepseek` + `DEEPSEEK_API_KEY` in `.env` |

---

## 7. Useful scripts

```bash
# Ingest a benchmark eval run into the DB (powers the Eval Dashboard)
PYTHONPATH=$(pwd) python backend/scripts/ingest_eval.py \
    --dir evaluation/outputs/eval_results_v3 \
    --slug v3 --label "v3 — Qwen2.5 ACU split"
```

---

## 8. Day-to-day service management (`tmux`)

The team runs the full stack in a tmux session named **`legaldiff`**:

| tmux target | Service | Reload behavior |
|---|---|---|
| `legaldiff:0` | Backend API (`uvicorn --reload`) | Auto-reloads on Python edits |
| `legaldiff:1` | Frontend (`npm run dev`) | Vite HMR |
| `legaldiff:2` | Celery worker | **Manual restart required** |

Restart the worker (only when idle — no in-flight `run_pipeline` task):
```bash
tmux send-keys -t legaldiff:2 C-c
# wait for the shell prompt to return, then:
tmux send-keys -t legaldiff:2 'PYTHONPATH=/home/haipd/L-RAG celery -A backend.celery_app worker --concurrency=1 --loglevel=info' Enter
```

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `connection refused :5432` / `:6379` | Start infra: `docker compose -f backend/docker-compose.yml up -d postgres redis` |
| Worker errors `:8000/v1` LLM | You're on compose defaults — set `LLM_BASE_URL=http://localhost:11434/v1` in `backend/.env` |
| BGE-M3 downloading on first job | Normal — first embedding run fetches ~2.3 GB from HuggingFace |
| GPU OOM during pipeline | Keep `worker_concurrency=1`; reduce `embed_batch_size` in `configs/pipeline_config.yaml` |
| Frontend shows API errors | Ensure API is on `8001` (the Vite proxy target); check `:8001/docs` |
| Worker ignoring code changes | Celery doesn't hot-reload — restart it (see §8) |
| `alembic` can't find DB URL | Set `DATABASE_URL_SYNC` in env / `backend/.env` |
| Port already in use | `ss -ltnp | grep <port>` to find the process; stop it or change the port |

---

## 10. Quick-start cheat-sheet (Option B)

```bash
conda activate l_rag
docker compose -f backend/docker-compose.yml up -d postgres redis   # infra
ollama pull qwen2.5:14b                                              # LLM
alembic -c backend/alembic.ini upgrade head                          # DB

# then, each in its own terminal:
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8001         # API  -> :8001
PYTHONPATH=$(pwd) celery -A backend.celery_app worker --concurrency=1 --loglevel=info  # worker
cd frontend && npm run dev                                           # UI   -> :5173
```
