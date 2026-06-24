# API Contract — Frontend ↔ Backend

> **The exact wire format the frontend must honor.** Derived from the backend source at `backend/` (FastAPI, as of 2026-06-24). This is the contract; if the backend changes, this file changes in the same PR and `src/types/` is updated to match.
>
> Base: everything is prefixed `/api`. WebSocket is `/ws`. All authenticated calls send `Authorization: Bearer <access_token>`.

---

## 0. Conventions

- **IDs:** `uuid` (string in JSON). **Dates:** ISO-8601 UTC strings. **Booleans:** JSON `true/false`.
- **Pagination** is **1-indexed**. `?page=1&page_size=20` (max `page_size=100`).
- **Errors:** HTTP ≥400 → `{ "detail": "…", "error_code": "…" | null }`. Map `detail` to user message; `error_code` to specific UI behavior.
- **Auth:** all `/documents`, `/jobs`, `/reports` routes and the WebSocket require a valid JWT.

---

## 1. Health (no auth) — drives the system-status chip

| Method | Path | Response |
|---|---|---|
| GET | `/api/health` | `{ "status": "ok" }` |
| GET | `/api/health/ready` | `HealthResponse` (below) |

```ts
type HealthResponse = {
  status: 'ok' | 'degraded'
  database:  'connected' | 'disconnected' | 'unknown'
  redis:     'connected' | 'disconnected' | 'unknown'
  worker:    'available' | 'busy' | 'unreachable' | 'unknown'   // 'busy' ⇒ GPU lock held
  llm_server:'connected' | 'disconnected' | 'error' | 'unknown'
}
```

---

## 2. Auth (`/api/auth`)

| Method | Path | Auth | Body | Response (2xx) |
|---|---|---|---|---|
| POST | `/api/auth/register` | ❌ | `RegisterRequest` | 201 `{ id, username, email }` |
| POST | `/api/auth/login` | ❌ | `LoginRequest` | `TokenResponse` |
| POST | `/api/auth/refresh` | ❌ | `RefreshRequest` | `RefreshResponse` |
| GET  | `/api/auth/me` | ✅ | — | `UserResponse` |

```ts
type RegisterRequest = { email: string /* RFC email, 5–255 */; username: string /* 3–100, [a-zA-Z0-9_] */; password: string /* 8–128 */; full_name?: string }
type LoginRequest    = { username: string; password: string }
type RefreshRequest  = { refresh_token: string }

type TokenResponse   = { access_token: string; refresh_token: string; token_type: 'bearer'; expires_in: number /* 3600 */ }
type RefreshResponse = { access_token: string; expires_in: number }
type UserResponse    = { id: string; email: string; username: string; full_name: string | null; is_active: boolean; created_at: string }
```

**Frontend behavior:**
- On login: store access (in-memory) + refresh (storage per `02-architecture.md` §5).
- `expires_in` (seconds) drives a proactive refresh timer shortly before expiry.
- `GET /me` on app boot to hydrate the current user; if it 401s, try refresh, else → `/login`.

---

## 3. Documents (`/api/documents`)

| Method | Path | Auth | Request | Response |
|---|---|---|---|---|
| POST | `/api/documents/upload` | ✅ | `multipart/form-data`, field `file` | `DocumentUploadResponse` |
| GET  | `/api/documents` | ✅ | `?page&page_size` | `PaginatedResponse<DocumentResponse>` |
| GET  | `/api/documents/{id}` | ✅ | — | `DocumentResponse` |
| DELETE | `/api/documents/{id}` | ✅ | — | 204 |

**Constraints (from `backend/config.py`):** allowed extensions `(.pdf, .docx)`; max size `100MB`. **Validate client-side first** to fail fast with a clear message.

```ts
type DocumentUploadResponse = {
  id: string; original_filename: string
  file_size_bytes: number | null; mime_type: string | null
  is_processed: boolean; created_at: string
}

type DocumentResponse = {
  id: string; original_filename: string
  file_size_bytes: number | null; mime_type: string | null
  doc_title: string | null; doc_number: string | null; signing_date: string | null
  parties: string[] | null
  page_count: number | null; article_count: number | null
  is_processed: boolean; created_at: string
}
```

> **Important:** documents are **not** inherently V1 or V2. The V1/V2 role is assigned at **job creation** by which two IDs you pass.

---

## 4. Jobs (`/api/jobs`)

| Method | Path | Auth | Body / Query | Response |
|---|---|---|---|---|
| POST | `/api/jobs` | ✅ | `CreateJobRequest` | `JobResponse` |
| GET  | `/api/jobs` | ✅ | `?page&page_size` | `PaginatedResponse<JobResponse>` |
| GET  | `/api/jobs/{id}` | ✅ | — | `JobResponse` |
| GET  | `/api/jobs/{id}/status` | ✅ | — | `JobStatusResponse` |
| POST | `/api/jobs/{id}/cancel` | ✅ | — | `JobStatusResponse` |
| DELETE | `/api/jobs/{id}` | ✅ | — | 204 |
| GET  | `/api/jobs/{id}/reports` | ✅ | `?page&page_size` | `PaginatedResponse<ReportSummaryResponse>` |
| GET  | `/api/jobs/{id}/catalog` | ✅ | — | `DiffPairCatalog` (raw alignment catalog) |

```ts
type CreateJobRequest = {
  document_v1_id: string; document_v2_id: string   // must differ (server validates)
  skip_phase3?: boolean         // default false — skip detailed comparison
  config_overrides?: Record<string, unknown> | null
}
```

### Job state machine (frontend must handle every state)

```
status:    pending ─▶ processing ─▶ completed
              │            │
              ▼            ▼
          cancelled     failed

current_phase: queued ─▶ ingestion ─▶ alignment ─▶ comparison ─▶ done
```

> **`pending` ≠ idle.** Because the GPU serializes jobs (Redis lock, one at a time, retries every 30s up to 3×), a `pending` job is almost always **queued waiting for the GPU**. The UI must show "Đang xếp hàng chờ GPU" for `pending`, not a generic spinner.

```ts
type JobResponse = {
  id: string; document_v1_id: string; document_v2_id: string
  v1_filename: string; v2_filename: string
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled'
  current_phase: 'queued' | 'ingestion' | 'alignment' | 'comparison' | 'done'
  progress_pct: number            // 0–100
  error_message: string | null
  // catalog summary (populated after alignment):
  total_pairs: number | null; matched_count: number | null
  added_count: number | null; deleted_count: number | null
  split_count: number | null; merge_count: number | null
  created_at: string; started_at: string | null; completed_at: string | null
}

type JobStatusResponse = {
  id: string; status: JobResponse['status']; current_phase: JobResponse['current_phase']
  progress_pct: number; error_message: string | null; celery_task_id: string | null
}
```

---

## 5. WebSocket — live job progress

**URL:** `GET /ws/jobs/{job_id}?token={access_token}` (upgrade to WebSocket).

- Auth via `token` query param (validated server-side; ownership checked).
- Close codes: `4001` invalid/expired token · `4003` not owner · `4004` invalid job id.
- The server publishes progress through a Redis pub/sub fan-out per `job_id`.

**Message schema** (`backend/schemas/ws.py`):

```ts
type WSProgressMessage = {
  event: 'progress' | 'phase_change' | 'completed' | 'error'
  job_id: string
  progress_pct: number         // 0–100
  current_phase: 'queued' | 'ingestion' | 'alignment' | 'comparison' | 'done'
  message: string              // e.g. "Đang phân tích tài liệu…"
  error: string | null
}
```

**Typical stream** (milestone updates, not continuous):
```
{event:'progress', progress_pct:5,  current_phase:'ingestion',  message:'Đang phân tích tài liệu…'}
{event:'progress', progress_pct:30, current_phase:'alignment',  message:'Đang nhúng vector BGE-M3…'}
{event:'progress', progress_pct:55, current_phase:'comparison', message:'Đang sinh báo cáo so sánh…'}
{event:'progress', progress_pct:95, current_phase:'comparison', message:'Đang lưu kết quả…'}
{event:'progress', progress_pct:100,current_phase:'done',       message:'Hoàn thành!'}
```

On failure: `{event:'progress', progress_pct:0, current_phase:'done', message:'Thất bại: <detail>'}` and DB `status=failed`.

**Frontend strategy** (full design in `02-architecture.md` §6): WS primary + `/jobs/{id}/status` polling every 3s as fallback, both writing to the **same React Query cache key** so the UI has one source of truth. A dropped socket never stops the job — reconnect with backoff and do one status resync.

---

## 6. Reports (`/api/reports`)

| Method | Path | Auth | Response |
|---|---|---|---|
| GET | `/api/reports/{id}` | ✅ | `ReportDetailResponse` |
| GET | `/api/reports/{id}/markdown` | ✅ | `text/markdown` raw body |

A **report = one matched DiffPair** (Phase 3 output). A job has **many** reports (list via `/api/jobs/{id}/reports`).

```ts
type ReportSummaryResponse = {
  id: string; job_id: string
  pair_id: string
  match_type: 'matched' | 'added' | 'deleted' | 'split' | 'merged'
  location_context: string            // e.g. "Điều 5, Khoản 2"
  total_acus: number; passed_acus: number; rejected_acus: number
  hallucination_rate: number          // 0–1
  created_at: string
}

type ReportDetailResponse = ReportSummaryResponse & {
  executive_summary: ExecutiveSummary | null
  markdown_report: string | null
  verified_acus: AcuOutput[] | null
  rejected_acus_detail: AcuOutput[] | null
}
```

### Executive summary (top of the report viewer)

```ts
type ExecutiveSummary = {
  overall_assessment: string
  critical_changes: string[]
  numerical_changes_summary: string
  risk_flags: string[]          // → render with destructive/risk treatment
  recommendation: string | null
  generated_at: string
}
```

### ACU — Atomic Comparison Unit (the core rendering target)

This is the most important type in the frontend. **Every ACU must render its verbatim V1/V2 evidence** — that is the zero-hallucination guarantee.

```ts
type ChangeType = 'numerical' | 'terminology' | 'structural' | 'addition' | 'deletion' | 'reorder'

type AcuOutput = {
  acu_id: string
  change_type: ChangeType
  location_v1: string             // "Điều X, Khoản Y, Điểm Z"
  location_v2: string
  original_value: string          // value in V1
  new_value: string               // value in V2
  verbatim_evidence_v1: string    // EXACT quote from V1 containing original_value  ← never truncate
  verbatim_evidence_v2: string    // EXACT quote from V2 containing new_value        ← never truncate
  confidence: number              // 0.0–1.0  (<0.6 ⇒ "low trust" treatment)
  reasoning: string               // explanation (DeepSeek provider only)
  pair_id: string
  created_at: string
}
```

**Rendering rules (binding):**
- `verbatim_evidence_v1/v2` rendered **monospace, `whitespace-pre-wrap`, never truncated**. The changed token (`original_value`/`new_value`) highlighted in place within the quote.
- `change_type` → `ChangeTypeBadge` (icon + label + color, per `01-design-system.md` §2).
- `confidence < 0.6` → amber "low-trust" marker; else emerald "verified".
- `match_type` of the parent report governs layout: `added` (no V1 evidence), `deleted` (no V2 evidence) — show the single available side with the "added/deleted" treatment.

---

## 7. Catalog (`/api/jobs/{id}/catalog`) — optional raw view

Returns the Phase-2 `DiffPairCatalog` (`DiffPair` list with match types + similarity scores). Use for an optional "alignment map" / statistics view. Shape is defined in `src/alignment/diff_catalog.py`; treat as `Record<string, unknown>` until a typed view is needed. **Not required for MVP report viewing.**

---

## 8. End-to-End Flow (what the frontend orchestrates)

```
1. upload V1   POST /api/documents/upload        → doc_v1.id
2. upload V2   POST /api/documents/upload        → doc_v2.id
3. create job  POST /api/jobs {document_v1_id, document_v2_id}  → job.id (status=pending)
4. track       WS  /ws/jobs/{job.id}?token=…     (fallback: GET /api/jobs/{id}/status every 3s)
5. on done     GET /api/jobs/{id}/reports        → report summaries (one per DiffPair)
6. render      GET /api/reports/{report_id}      → ReportDetailResponse (executive_summary + verified_acus)
7. export      GET /api/reports/{report_id}/markdown  (download)  |  print-to-PDF (client)
```

> **Doc processing note:** `is_processed` becomes true after Phase 1 ingestion of an uploaded doc. For the comparison flow the worker ingests V1/V2 itself, so the wizard can create a job immediately after upload — no need to block on `is_processed` for the standard flow.

---

## 9. Type Generation Strategy

`src/types/` mirrors the above by hand (the backend is not OpenAPI-auto-served here, but FastAPI exposes `/openapi.json` at runtime). **Recommended:** during implementation, run a one-off generation from the running backend's `/openapi.json` (e.g. `openapi-typescript`) to seed `src/types/api.ts`, then refine. Keep the hand-written mirror in this doc as the human-readable contract.
