# Frontend Architecture — LegalDiff

> **How the frontend is structured and how it talks to the backend.**
> Reference for implementation. Read with `03-api-contract.md` (the wire format) and `00-spec.md` (scope).

Stack: **Vite 5 + React 18 + TypeScript (strict) + Tailwind + shadcn/ui + React Router 6 + TanStack Query 5**. Separate process; proxies `/api` and `/ws` to FastAPI at `:8001`.

---

## 1. High-Level Topology

```
┌─────────────────────────────┐         ┌──────────────────────────────────────────┐
│  Browser (SPA)              │         │  LegalDiff backend (Python)               │
│  Vite dev :5173 / dist build│         │  FastAPI :8001  ·  Celery worker          │
│                             │         │  PostgreSQL · Redis · Qdrant · GPU        │
│  React Router (pages)       │         │                                           │
│   │                         │         │                                           │
│   ├─ TanStack Query ────────┼─/api──▶ │  REST  (/api/auth /documents /jobs ...)   │
│   │   (server state, cache) │         │                                           │
│   │                         │         │                                           │
│   ├─ WS client ─────────────┼─/ws───▶ │  WebSocket /ws/jobs/{id}?token=…          │
│   │   (live progress)       │         │                                           │
│   │                         │         │                                           │
│   └─ Zustand (UI state)     │         │                                           │
│      theme, toasts, drawer  │         │                                           │
└─────────────────────────────┘         └──────────────────────────────────────────┘
                  │
                  │  Vite dev proxy (vite.config.ts)
                  └─ target: http://localhost:8001  (same-origin in dev → no CORS pain)
```

**Production:** the built static bundle is served by its own static server (nginx / `vite preview`) and proxies `/api` + `/ws` to the backend host configured via `VITE_API_BASE_URL` / `VITE_WS_BASE_URL`. The backend's CORS (`allow_origins=["*"]` today) should be tightened to the frontend origin before any real deployment (see `00-spec.md` Open Questions #2).

---

## 2. Vite Proxy & Environment

```ts
// vite.config.ts (excerpt)
export default defineConfig({
  plugins: [react()],
  resolve: { alias: { '@': path.resolve(__dirname, './src') } },
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://localhost:8001', changeOrigin: true },
      '/ws':  { target: 'ws://localhost:8001', ws: true, changeOrigin: true },
    },
  },
})
```

```bash
# frontend/.env.example
VITE_API_BASE_URL=/api        # same-origin via proxy in dev; absolute URL in prod
VITE_WS_BASE_URL=/ws          # ditto
VITE_APP_NAME=LegalDiff
```

**Rule:** no code reads `localhost:8001` directly. Everything goes through `VITE_API_BASE_URL` / `VITE_WS_BASE_URL`.

---

## 3. Layered Code Model

Three strict layers; dependencies flow one direction only.

```
routes (pages)
   │  use React Query hooks + WS hooks
   ▼
hooks / lib (domain logic)      ← the ONLY layer that knows HTTP/WS/auth
   │  use typed DTOs
   ▼
types (DTOs mirrored from backend/schemas)
```

- **Components never import `fetch` or know URLs.** They call hooks.
- **Hooks never import component code.** They're testable in isolation.
- **`types/`** is the contract mirror; changing a backend schema updates a type here in the same PR.

---

## 4. Typed API Client (`lib/api/`)

A thin, typed wrapper over `fetch`. Each backend router maps to one module. All request/response bodies are typed against `src/types/`.

```ts
// src/lib/api/client.ts — the single fetch primitive
import { getToken, refreshAccessToken, clearSession } from '@/lib/auth'

export class ApiError extends Error {
  constructor(public status: number, public detail: string, public errorCode?: string) {
    super(detail)
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers)
  if (!(init.body instanceof FormData)) headers.set('Content-Type', 'application/json')
  const token = getToken()
  if (token) headers.set('Authorization', `Bearer ${token}`)

  const res = await fetch(`${import.meta.env.VITE_API_BASE_URL}${path}`, { ...init, headers })

  if (res.status === 401) {
    const refreshed = await refreshAccessToken()      // tries /api/auth/refresh once
    if (!refreshed) { clearSession(); throw new ApiError(401, 'Phiên hết hạn') }
    return request<T>(path, init)                     // retry original once
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new ApiError(res.status, body.detail ?? 'Lỗi không xác định', body.error_code)
  }
  if (res.status === 204) return undefined as T
  return (res.status === 204 ? undefined : await res.json()) as T
}
```

```ts
// src/lib/api/jobs.ts — example domain module
export const jobsApi = {
  create: (b: CreateJobRequest) => request<JobResponse>('/jobs', { method: 'POST', body: JSON.stringify(b) }),
  status: (id: string) => request<JobStatusResponse>(`/jobs/${id}/status`),
  reports: (id: string) => request<PaginatedResponse<ReportSummaryResponse>>(`/jobs/${id}/reports`),
  cancel:  (id: string) => request<JobStatusResponse>(`/jobs/${id}/cancel`, { method: 'POST' }),
  // …list, get, delete, catalog
}
```

**File uploads** use `FormData` (the client detects `FormData` and skips the JSON content-type). Client-side validation (extension `.pdf/.docx`, ≤100MB) happens before upload to fail fast with a clear message.

---

## 5. Authentication & Token Lifecycle (`lib/auth/`)

Backend = JWT Bearer (access 60min + refresh 7d). **Decision pending** (spec Open Question #1); default proposal:

```
access token  → in-memory JS variable (lost on reload, but short-lived)
refresh token → sessionStorage (cleared on tab close)
on reload     → silent refresh on app boot; if refresh fails → /login
on 401        → single refresh attempt, then retry the original request once
```

- **`getToken()`** returns the in-memory access token (or triggers a synchronous refresh).
- **`refreshAccessToken()`** posts to `/api/auth/refresh`; on success stores the new access token; on failure clears the session.
- **Route guards** (`requireAuth` loader) gate protected routes; unauthenticated → redirect to `/login?next=…`.
- **Logout** calls session clear + React Query `clear()` + navigate to `/login`.
- ⚠️ Do **not** store access tokens in `localStorage` without an explicit security sign-off — XSS would expose them. In-memory + refresh is the default.

---

## 6. Live Progress: WebSocket + Polling Fallback (`lib/ws/`)

A comparison job is long (minutes) and the GPU serializes jobs (only one runs; others stay `pending`). Progress must feel live and must survive drops.

**Design: WebSocket primary, polling fallback, single source of truth = the server.**

```ts
// src/hooks/useJobProgress.ts
export function useJobProgress(jobId: string) {
  const queryClient = useQueryClient()

  // 1) Seed + fallback: poll /jobs/{id}/status every 3s (cheap, always works)
  const statusQuery = useQuery({
    queryKey: ['job', jobId, 'status'],
    queryFn: () => jobsApi.status(jobId),
    refetchInterval: (q) => isTerminal(q.state) ? false : 3000,
  })

  // 2) Primary: WS for instant phase changes
  useEffect(() => {
    const ws = createJobSocket(jobId, {
      onMessage: (msg) => {
        // Optimistically patch the React Query cache — single UI source of truth
        queryClient.setQueryData(['job', jobId, 'status'], (old) => ({
          ...old, progress_pct: msg.progress_pct,
          current_phase: msg.current_phase, status: msg.event === 'completed' ? 'completed' : old?.status,
          error_message: msg.error ?? null,
        }))
      },
    })
    return () => ws.close()
  }, [jobId])

  return statusQuery.data   // WS patches the same cache the poll reads → coherent UI
}
```

**The `JobSocket` client handles (see `05-uiux-guidelines.md` §3 for the rules):**
- Token in query param (`?token=…`), close codes `4001/4003/4004` (auth/ownership/bad id) → surface the right message, stop reconnecting.
- **Auto-reconnect** with capped exponential backoff (1s → 2s → 5s, max 5 attempts). A dropped socket **never** stops the job server-side — reconnecting resumes updates.
- On every reconnect, immediately do **one** status poll to resync (in case events were missed while down).
- **Polling stays armed** as the safety net; it self-disables (`refetchInterval: false`) once the job reaches a terminal state (`completed`/`failed`/`cancelled`).
- Token may expire mid-job — on a `4001` close, attempt one refresh then reconnect.

**Phase → UI mapping** (single helper, tested):

| `current_phase` | progress band | UI label (vi) |
|---|---|---|
| `queued` | 0 | "Đang xếp hàng chờ GPU…" (only **one** job runs at a time) |
| `ingestion` | 5–30 | "Đang phân tích tài liệu…" |
| `alignment` | 30–55 | "Đang ghép nối điều khoản…" |
| `comparison` | 55–95 | "Đang sinh báo cáo so sánh…" |
| `done` | 95–100 | "Hoàn thành" / failure |

---

## 7. State Management — clear ownership

| State | Tool | Examples |
|---|---|---|
| **Server state** (documents, jobs, reports, user) | **TanStack Query** | `useDocuments()`, `useJob(id)`, `useReport(id)` |
| **UI/global state** | **Zustand** (small stores) | theme (light/dark), sidebar open, locale, toast queue |
| **Form state** | **React Hook Form** + Zod | login, register, job options |
| **Ephemeral/local** | `useState` | dialog open, selected filter tab |

**Rules:**
- Never copy server data into Zustand/local state (single source of truth = the query cache).
- Mutations invalidate the right query keys (`['documents']`, `['jobs']`).
- Query keys are centralized in `lib/query/keys.ts` to avoid drift.

---

## 8. Routing (`lib/router.tsx`)

React Router 6 data router with protected routes and a shared `AppShell`.

```
/                         → redirect to /dashboard or /login
/login                    → public
/register                 → public
/dashboard                → protected (jobs overview + recent)
/documents                → protected (library)
/documents/:id            → protected (detail)
/jobs                     → protected (all jobs)
/jobs/new                 → protected (new-comparison wizard)
/jobs/:id                 → protected (live progress)
/jobs/:id/report          → protected (comparison report viewer)
/reports/:id              → protected (single report detail)
/settings                 → protected (profile, theme, locale)
```

- `requireAuth` loader → redirect to `/login?next=<path>` when unauthenticated.
- Deep links work (skill §9 `deep-linking`) — every screen is URL-reachable.
- After route change, move focus to main content for screen readers (skill §9 `focus-on-route-change`).
- The report viewer is bookmarkable/shareable by URL within the app.

---

## 9. Error, Loading & Empty States (architecture-level)

Every async surface renders four states explicitly (skill §3/§8):

| State | Pattern |
|---|---|
| Loading (<300ms) | nothing / optimistic |
| Loading (>300ms) | **Skeleton** matching layout (no blank screen, no layout shift) |
| Empty | **EmptyState** with a helpful message + primary action (e.g. "Tải lên văn bản đầu tiên") |
| Error | **ErrorState** with `role="alert"`, cause + **recovery action** (retry / help) (skill §8 `error-recovery`) |

A global **error boundary** + a React Query `onError` → toast pipeline ensures no failure is silent.

---

## 10. Performance Strategy (architecture-level, see `05-uiux-guidelines.md` for rules)

- **Route-level code splitting** (`React.lazy` + Suspense) — report viewer is heavy; don't load it on the dashboard.
- **Virtualize** the ACU list (and any list >50 rows) with `@tanstack/react-virtual`.
- **Debounce** search/filter input (250ms) and resize.
- **Images/fonts:** self-hosted (offline), `font-display: swap`, preload critical fonts only.
- **Tabular numbers** everywhere a number can shift layout (skill §6 `number-tabular`).
- Reserve layout space for async content to keep **CLS < 0.1**.

---

## 11. Cross-Cutting Concerns

- **i18n:** `react-i18next`. All copy in `public/locales/{vi,en}/translation.json`. Backend WS messages are already Vietnamese; surface them verbatim or map via a phase→key table.
- **Theming:** `class="dark"` toggle on `<html>`, persisted in Zustand + `localStorage`, respects `prefers-color-scheme` on first visit.
- **Logging:** structured `console` in dev only; no secrets/tokens logged.
- **Security:** no token in URL logs (WS query param is unavoidable server-side — keep it out of our own analytics); sanitize before rendering any user-controlled markdown (`react-markdown` with `rehype-sanitize`).
- **Observability (later):** hook into `/api/health/ready` to drive a small **system-status chip** in the topbar (db/redis/worker/llm).

---

## 12. Build & Deploy Shape

- `npm run build` → static `frontend/dist/`.
- Deploy = serve `dist/` from a static host; configure `/api` + `/ws` reverse-proxy to the backend; set `VITE_API_BASE_URL`/`VITE_WS_BASE_URL` at build time.
- Optional future: containerize (`Dockerfile.frontend`) + `docker-compose` with the backend — out of scope for the first plan unless requested.
