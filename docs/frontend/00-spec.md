# Spec — LegalDiff Frontend

> **Status:** Draft for review (Spec-Driven Development · Phase 1: Specify)
> **Last updated:** 2026-06-24
> **Decisions locked:** Vite + React + TS + Tailwind + shadcn/ui · separate frontend proxying to FastAPI `:8001` · full-app scope · full multi-user auth (register + login).

This is the **shared source of truth** for the frontend between the engineer and the AI. Code written before this spec is approved is guessing. Read it fully before writing a single component.

---

## 1. Objective

**What we are building.** A desktop-first, responsive **web frontend** for **LegalDiff** — a 100%-offline system that compares two versions of a Vietnamese legal document (V1 vs V2) and produces an evidence-backed comparison report with **zero hallucination**.

**The backend already exists and is complete** (FastAPI + Celery + WebSocket + PostgreSQL + Redis + Qdrant). This spec covers only the **frontend** that consumes it.

**Who is the user.** Legal professionals, compliance officers, and contract reviewers working inside an enterprise/on-prem deployment. They are domain experts, not technical users. They care about:

1. **Trust** — every reported change must be backed by a verbatim quote from V1 and V2. The UI's job is to make that evidence unmissable.
2. **Completeness** — "did I miss a single number or clause?" They must trust nothing was silently dropped.
3. **Speed to insight** — an executive summary up top, then drill into specific Điều/Khoản/Điểm changes.

**What success looks like.** A user can: register/log in → upload two legal documents (PDF/DOCX) → start a comparison → watch live, phase-by-phase progress (including "queued for GPU") → open a structured report that lists every change with color-coded type, severity, side-by-side V1/V2 quotes, and export it. All in Vietnamese, accessible, on their laptop.

**Scope (full app):** authentication, document library, upload wizard, job creation, live job tracking, rich diff-report viewer with evidence, executive summary, filtering/search, and export (Markdown/PDF).

---

## 2. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Framework | **React 18** (SPA) | Mature, matches shadcn/ui ecosystem |
| Build tool | **Vite 5** | Fast HMR, trivial proxy config, static output for on-prem |
| Language | **TypeScript 5** (strict) | The backend speaks precise types (UUIDs, enums, ISO dates); TS mirrors them |
| Styling | **Tailwind CSS 3** | Maps 1:1 to the navy "Trust & Authority" design tokens |
| Components | **shadcn/ui** + **Radix UI** primitives | Accessible (WAI-ARIA), themeable, ownable code (not a black-box lib) |
| Routing | **React Router 6** (data router) | File-free but explicit; supports loaders/actions + protected routes |
| Server state | **TanStack Query (React Query) 5** | Caching, background polling, mutations, 401 retry → token refresh |
| WebSocket | Native `WebSocket` + small client wrapper | Thin layer; no heavy socket lib needed |
| Forms | **React Hook Form** + **Zod** | Zod schemas are the single source of truth for request validation, shared with types |
| Icons | **Lucide React** | Matches shadcn/ui; SVG, themeable, no emoji |
| Markdown | **react-markdown** + **remark-gfm** | Render `markdown_report` field from reports API |
| Export/PDF | **client print** + existing backend Markdown | Print-to-PDF via stylesheet; download Markdown from `/api/reports/{id}/markdown` |
| i18n | **react-i18next** | Vietnamese primary, English toggle; strings externalized |
| Testing | **Vitest** + **React Testing Library** + **Playwright** | Unit/component + E2E |
| Lint/format | **ESLint** + **Prettier** | Consistent style |
| Node | **Node 20 LTS** | Matches Vite 5 requirements |

**Pinned versions** are recorded in `package.json` at implementation time. No pinned versions are invented here.

---

## 3. Commands

All commands run from `frontend/` (a new top-level directory beside `backend/`).

```bash
# Setup
node -v               # expect v20.x
npm install            # install deps

# Development (starts Vite on :5173, proxies /api + /ws -> :8001)
npm run dev

# Type-check + lint + unit tests
npm run typecheck      # tsc --noEmit
npm run lint           # eslint . --max-warnings=0
npm run test           # vitest run
npm run test:watch     # vitest (watch)

# Build production static bundle -> frontend/dist
npm run build          # tsc -b && vite build
npm run preview        # serve the dist build locally

# End-to-end (requires backend + worker + db + redis running)
npm run test:e2e       # playwright test
```

**Backend must be running** for dev: `uvicorn backend.main:app --host 0.0.0.0 --port 8001` plus the Celery worker, Postgres, Redis. See `backend/docs/run_backend.md`.

---

## 4. Project Structure

A **new `frontend/` directory** at the repo root (sibling of `backend/` and `src/`). The backend is untouched.

```
frontend/
├── index.html                 # Vite entry
├── vite.config.ts             # dev server + proxy to :8001
├── tsconfig.json
├── tailwind.config.ts         # design tokens -> CSS vars
├── postcss.config.js
├── package.json
├── .env.example               # VITE_API_BASE_URL, VITE_WS_BASE_URL
├── public/
│   └── locales/               # vi/translation.json, en/translation.json
└── src/
    ├── main.tsx               # app bootstrap (QueryClient, Router, ThemeProvider, i18n)
    ├── App.tsx                # root layout + routes
    ├── routes/                # route pages (one folder per screen)
    │   ├── (auth)/            # login, register
    │   ├── dashboard/
    │   ├── documents/
    │   ├── jobs/
    │   │   ├── new/           # new-comparison wizard
    │   │   ├── [id]/          # job detail + live progress
    │   │   └── [id]/report/   # comparison report viewer
    │   └── settings/
    ├── components/
    │   ├── ui/                # shadcn/ui primitives (Button, Card, Dialog…)
    │   ├── layout/            # AppShell, Sidebar, Topbar, StatusBar
    │   ├── documents/         # UploadDropzone, DocumentCard, DocMetaBadge
    │   ├── jobs/              # PhaseStepper, ProgressBar, GpuQueueNotice
    │   ├── report/            # DiffPairList, AcuCard, EvidenceQuote, ChangeTypeBadge,
    │   │                      #   ExecutiveSummary, SeverityFilter, ReportExport
    │   └── common/            # ErrorState, EmptyState, Skeletons, StatusChip
    ├── lib/
    │   ├── api/               # typed REST client (auth, documents, jobs, reports)
    │   ├── ws/                # WebSocket client w/ reconnect + polling fallback
    │   ├── auth/              # token storage, refresh, guards
    │   ├── query/             # React Query keys + hooks
    │   └── utils/             # date, file, format helpers
    ├── types/                 # generated/mirrored DTO types matching backend schemas
    ├── hooks/                 # useJobProgress, useAuth, useMediaQuery…
    ├── stores/                # zustand stores (UI state only: theme, toasts)
    └── styles/                # globals.css (tokens), tailwind layers
docs/frontend/                 # ← this planning (you are here)
```

**Where things live — rules of thumb**
- `routes/` = pages (one file per URL). `components/` = reusable UI. If a component is used by ≥1 page, it lives in `components/`.
- `lib/api` is the **only** place that knows about `fetch`/HTTP. Components never call `fetch`.
- `types/` mirrors `backend/schemas/*.py`. When a backend schema changes, the matching type changes here in the same PR.

---

## 5. Code Style

One real snippet beats three paragraphs. The codebase style is: typed end-to-end, small composable components, server state via React Query, no magic.

```tsx
// src/components/report/AcuCard.tsx
// One Atomic Comparison Unit (ACU) — the heart of the zero-hallucination report.
import { memo } from 'react'
import { ShieldCheck, AlertTriangle } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { ChangeTypeBadge } from './ChangeTypeBadge'
import { EvidenceQuote } from './EvidenceQuote'
import type { AcuOutput } from '@/types/report'

interface AcuCardProps {
  acu: AcuOutput
}

/** Render a single verified change with its verbatim V1/V2 evidence. */
export const AcuCard = memo(function AcuCard({ acu }: AcuCardProps) {
  const risky = acu.confidence < 0.6
  return (
    <Card data-testid="acu-card" data-acu-id={acu.acu_id}>
      <header className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <ChangeTypeBadge type={acu.change_type} />
          <span className="text-sm text-muted-foreground">{acu.location_v1}</span>
        </div>
        {risky ? (
          <AlertTriangle aria-label="Độ tin cậy thấp" className="size-4 text-amber-600" />
        ) : (
          <ShieldCheck aria-label="Đã xác minh" className="size-4 text-emerald-600" />
        )}
      </header>

      {/* Evidence is never optional — it is the entire value proposition. */}
      <div className="mt-3 grid gap-2 md:grid-cols-2">
        <EvidenceQuote side="v1" location={acu.location_v1} value={acu.original_value} quote={acu.verbatim_evidence_v1} />
        <EvidenceQuote side="v2" location={acu.location_v2} value={acu.new_value} quote={acu.verbatim_evidence_v2} />
      </div>
    </Card>
  )
})
```

**Conventions**
- **Naming:** `PascalCase` components, `camelCase` functions/vars, `kebab-case` file names for pages (`job-detail.tsx`), `PascalCase` for component files. Hooks prefixed `use`.
- **Imports:** path alias `@/` → `src/`. Group: external → `@/` → relative.
- **Types:** no `any`. Backend DTOs are the source of truth; derive UI props from them. Use `type` for DTO shapes, `interface` for component props.
- **State:** server state → React Query. Cross-component UI state → Zustand. Component-local state → `useState`. Never duplicate server data into local state.
- **Accessibility is a code-review gate, not an afterthought:** every interactive element has a label, every icon-only button an `aria-label`, errors use `role="alert"`, focus is managed on route/modal transitions.
- **i18n:** no hardcoded Vietnamese strings in components — all copy goes through `t('...')` in locale JSON.

---

## 6. Testing Strategy

| Level | Framework | What it covers | Where |
|---|---|---|---|
| Unit | Vitest | `lib/` pure functions (formatters, Zod schemas, token refresh logic, phase→progress mapping) | `*.test.ts` next to source |
| Component | Vitest + RTL + jsdom | Component behavior + a11y assertions (rendering, form validation, empty/error/loading states) | `*.test.tsx` next to source |
| Integration | Vitest + MSW | API client + React Query hooks against **mocked** backend (MSW handlers mirroring the real contract) | `lib/api/*.test.ts` |
| E2E | Playwright | The full critical path against a **real** backend+worker: register → upload 2 docs → create job → watch progress → view report → export | `e2e/*.spec.ts` |
| Visual a11y | Playwright + `@axe-core/playwright` | Automated WCAG checks on every page | inside E2E suite |
| Type | `tsc --noEmit` | The type system as a test | CI gate |

**Coverage expectations:**
- `lib/` (api, ws, auth, utils): **≥85%** — this is where bugs are expensive and cheap to test.
- `components/`: behavior-covered via component tests for the critical paths (upload, report rendering, auth). Visual nitpicks are reviewed manually.
- Critical user flow is **E2E-covered end to end** (the upload→compare→report→export journey).

**Test data:** a small fixture pair (a 2-page V1 + V2) committed under `frontend/src/test/fixtures/`, used by both component and E2E tests. Never hit the real GPU pipeline in unit/component tests.

**Test levels by concern:** business logic (ACU formatting, phase mapping, confidence thresholds) → unit; UI behavior & a11y → component; API contract conformance → integration (MSW); "does the whole product work" → E2E.

---

## 7. Boundaries

**Always do:**
- Run `npm run typecheck && npm run lint && npm run test` before considering a task done.
- Mirror backend schema/type changes into `src/types/` in the same change.
- Validate all user input with Zod before it reaches the API.
- Show loading/empty/error states for every async surface (skeletons for >300ms).
- Keep ACU evidence quotes **verbatim** and prominent — never summarize or hide them.
- Use Vietnamese copy via i18n; keep 4.5:1 contrast and keyboard operability.
- Commit the spec and keep it in sync with decisions.

**Ask first:**
- Adding a new runtime dependency (justify vs. existing stack).
- Changing the design tokens in `01-design-system.md` (ripples across the app).
- Changing the routing model or global state strategy.
- Introducing a new top-level route or navigation section.
- Any change to how tokens are stored (security-sensitive).
- Backend API changes that require a frontend migration.

**Never do:**
- Commit `.env` or any real token/secret.
- Store JWT access tokens in `localStorage` without acknowledging the trade-off (see Open Questions) — the default is **httpOnly cookie via a thin proxy OR in-memory + refresh token in storage**; decide before Phase 4.
- Hardcode the backend URL — read from `VITE_API_BASE_URL` / `VITE_WS_BASE_URL`.
- Render an ACU without its evidence quote.
- Disable zoom, remove focus rings, or use emoji as structural icons.
- Skip the loading/empty/error states to "save time".
- Edit anything under `backend/` or `src/` from the frontend task branch (separate concern — open a backend task).

---

## 8. Success Criteria

Specific and testable. A task is "done" when its criteria pass, not when it "looks done."

### Functional
- [ ] A new user can register, log in, and log out. Access/refresh token lifecycle works (auto-refresh on 401, redirect to login on expired refresh).
- [ ] A user can upload a PDF and a DOCX (≤100MB) via drag-drop **or** file picker; unsupported types/sizes are rejected client-side with a clear message matching the backend rules.
- [ ] A user can create a comparison job from any two of their documents; identical-document selection is prevented (mirrors backend `_documents_must_differ`).
- [ ] Job progress updates **live** via WebSocket, advancing through phases `queued → ingestion → alignment → comparison → done`; a dropped socket reconnects and the UI never silently freezes.
- [ ] When the GPU is busy, a job shows a distinct **"queued / đang xếp hàng chờ GPU"** state (from `status=pending`) — not a silent spinner.
- [ ] On completion, the report viewer renders: executive summary, the catalog summary counts (total/matched/added/deleted/split/merged), and a filterable list of ACUs.
- [ ] Every rendered ACU shows its **verbatim V1 and V2 evidence quotes**, change type, location, and confidence — never summarized away.
- [ ] A user can filter ACUs by change type and severity, search by location (Điều/Khoản), and sort.
- [ ] A user can export the report as Markdown (download from `/api/reports/{id}/markdown`) and as PDF (print stylesheet).
- [ ] A user can cancel a running job and delete documents/jobs; destructive actions are confirmed.

### Non-functional
- [ ] **Accessibility:** WCAG 2.1 AA — 4.5:1 contrast, full keyboard nav, visible focus, screen-reader-announced errors, `prefers-reduced-motion` respected. Axe: zero critical/serious violations on core pages.
- [ ] **Responsive:** usable 375px → 1440px+, no horizontal scroll, desktop-first optimized for 1280px+.
- [ ] **Performance:** initial route LCP < 2.5s on the dev machine; report list with 100+ ACUs virtualizes; no layout shift during async load (CLS < 0.1).
- [ ] **i18n:** 100% of user-facing strings externalized; toggling `vi`/`en` changes the whole UI.
- [ ] **Theming:** light and dark mode both pass contrast independently.
- [ ] **Type safety:** `tsc --noEmit` and ESLint pass with zero errors; `any` count is zero in new code.

---

## 9. Open Questions

These need a human answer before or during implementation. None block writing the spec, but each blocks a Phase-4 decision.

1. **Token storage model (security-sensitive).** JWT access tokens: in-memory + refresh in storage, OR httpOnly cookie issued through a backend-aided flow? The current backend is header-Bearer only. → *Default proposal: in-memory access token + refresh token in `sessionStorage`, auto-refresh on 401. Confirm before implementation.*
2. **Offline/on-prem topology.** Is the frontend served from the same host as the backend (so the proxy is same-origin) or a different host/port in production? Affects CORS hardening and `VITE_*_BASE_URL` defaults.
3. **Registration gating.** Should `/register` be publicly reachable, or invitation/admin-provisioned only? (Backend exposes it; UX decision.)
4. **Export fidelity.** Is the existing backend Markdown report the canonical export, or do we need a bespoke branded PDF template? If bespoke, that is a separate design effort.
5. **Report rendering for large jobs.** Is there a hard ceiling on ACU count per report we should design the virtualization around? (No backend pagination on `verified_acus` — confirm expected max.)
6. **Multi-tenancy.** Single organization, or multiple orgs/users with shared documents? Current backend is per-user only — frontend assumes per-user unless told otherwise.
7. **Branding assets.** Do we have a logo, wordmark, and brand color beyond the skill's recommended palette? If yes, swap tokens accordingly.

---

## How to proceed

Once a human reviews and approves this spec:
1. `02-architecture.md` and `03-api-contract.md` become the implementation reference.
2. `06-implementation-plan.md` defines the build order; `07-task-breakdown.md` breaks it into single-session tasks.
3. Each task is implemented per `incremental-implementation` + `test-driven-development`, loading the relevant spec section via `context-engine`.
