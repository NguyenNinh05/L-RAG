# Implementation Plan — LegalDiff Frontend

> **Phase 2: Plan.** The build order, dependencies, risks, and verification checkpoints.
> Derived from the spec (`00`), architecture (`02`), and contract (`03`). The task list (`07`) breaks this into single-session units.

**Approach: vertical slices.** Each milestone delivers a usable slice end-to-end (one screen or one flow working against the real backend), rather than building all of one layer at a time. This de-risks integration early — the riskiest unknown (does the frontend actually work against the live FastAPI + WS + GPU pipeline?) is hit in M2, not at the end.

---

## Milestones (build order)

```
M0  Foundations          ──▶  app boots, design tokens, routing shell, lint/test/typecheck gates
M1  Auth                 ──▶  register/login/logout, token lifecycle, protected routes       (Flow: gate)
M2  Upload + Create Job  ──▶  upload V1/V2, create comparison job (FIRST end-to-end hit)     (Flow A start)
M3  Live Progress        ──▶  WS + polling, phase stepper, GPU-queue state, cancel          (Flow A mid)
M4  Report Viewer        ──▶  executive summary + ACU list + evidence + filter/export       (Flow A end)  ★ signature
M5  Library & Dashboard  ──▶  documents table, jobs list, dashboard KPIs                    (Flows B/C)
M6  Polish & a11y        ──▶  dark mode, i18n EN, virtualization, axe pass, perf
M7  Ship                 ──▶  prod build, proxy/nginx, runbook, E2E green
```

### Dependency graph
```
M0 ── M1 ── M2 ── M3 ── M4 ── M6 ── M7
                 │      │
                 └─ M5 (can parallelize with M4 once M3 done)
```
- **M5 is parallelizable** with M4 (different files/screens) once M3 is in place.
- M4 (report) is the highest-value, highest-risk slice; it gets its own focused milestone.
- M6 and M7 are sequential and gate release.

---

## M0 — Foundations
**Deliverable:** app boots with the design system, an empty `AppShell`, routing, and all quality gates green.

- Vite + React + TS (strict) + Tailwind + shadcn/ui scaffold under `frontend/`.
- `globals.css` + `tailwind.config.ts` with **Trust & Authority** tokens (light + dark) — `01-design-system.md` §9.
- Self-host EB Garamond / Lato / JetBrains Mono (offline).
- Path alias `@/`, ESLint + Prettier, Vitest + RTL + jsdom, Playwright installed.
- `vite.config.ts` proxy `/api` + `/ws` → `:8001`. `.env.example`.
- `AppShell` (sidebar + topbar + status-chip placeholder), React Router with `/login` stub + protected `/dashboard` stub.
- Theme (dark toggle) + i18n (vi) bootstrap.

**Verify:** `npm run dev` boots on :5173; dashboard stub renders with correct fonts/colors; `typecheck && lint && test` pass; `npm run build` succeeds.

---

## M1 — Authentication
**Deliverable:** register → login → protected app; auto-refresh on 401; logout.

- `types/auth.ts` + `lib/api/auth.ts` (register/login/refresh/me).
- `lib/auth` token store (in-memory access + refresh in storage), `refreshAccessToken`, `clearSession`.
- `lib/api/client.ts` with 401 → single refresh → retry.
- `useAuth` hook + `requireAuth` route loader (redirect `/login?next=`).
- Login + Register pages (RHF + Zod, inline errors, password toggle).
- `GET /me` hydrate on boot.

**Verify:** register a user, log in, refresh page stays logged in (silent refresh), hit an expired token → auto-refresh, logout clears session. Component tests for forms; integration test with MSW for the 401 path.

---

## M2 — Upload + Create Job  ⚡ first end-to-end hit
**Deliverable:** upload V1 + V2 (drag-drop/picker), create a comparison job, see it created (status=pending).

- `types/document.ts` + `types/job.ts`; `lib/api/documents.ts`, `lib/api/jobs.ts`.
- `UploadDropzone` (drag-drop + picker, client validation `.pdf/.docx` ≤100MB, progress).
- New-comparison wizard (`/jobs/new`): Step1 V1 → Step2 V2 → Step3 options → POST `/api/jobs`.
- On success → navigate to `/jobs/:id` (M3 screen, stubbed progress for now).

**Verify against live backend:** upload two real PDFs, create job, confirm `JobResponse` with `status=pending`. This is the first time we touch the real API + worker — surface any contract mismatches here and fix `types/` + `03-api-contract.md`.

---

## M3 — Live Progress
**Deliverable:** real-time phase/percent via WS, robust to drops, GPU-queue distinction, cancel.

- `lib/ws/JobSocket` (token query param, close-code handling 4001/4003/4004, reconnect w/ backoff, resync poll).
- `useJobProgress` (WS primary + `/status` polling fallback → same React Query key).
- `PhaseStepper` + progress bar + `current_phase`→label table; **distinct "queued for GPU" state for `pending`**.
- Live message line + start time; Cancel (confirm → POST `/cancel`).
- Terminal states (completed/failed/cancelled) → stop polling, route to report on success.

**Verify:** start a job, watch phases advance live; kill the WS (network throttle) → UI keeps updating via poll + reconnects; submit two jobs → second shows "queued for GPU"; cancel a running job. (Requires GPU worker running.)

---

## M4 — Report Viewer  ★ signature milestone
**Deliverable:** the comparison report rendered end-to-end with evidence.

- `types/report.ts` (ReportSummary, ReportDetail, ExecutiveSummary, **AcuOutput**).
- `lib/api/reports.ts` (`/jobs/{id}/reports`, `/reports/{id}`, `/reports/{id}/markdown`).
- Report page (`/jobs/:id/report`): tabs Summary / Changes; **executive summary** (risk_flags styled); catalog-count legend.
- `AcuCard` + `EvidenceQuote` pair (verbatim, monospace, highlighted token, **never truncated**), `ChangeTypeBadge` (icon+label+color), confidence marker.
- `added`/`deleted` single-side rendering; rejected-ACUs section; `hallucination_rate` metric.
- Filter (type/severity) + search (location) + sort; debounce.
- Export: Markdown download + print-to-PDF stylesheet.

**Verify:** open a completed job's report; every ACU shows both evidence quotes; filter/search/sort work; export Markdown matches backend; print produces a clean PDF. Component tests for `AcuCard`/evidence rendering; a11y assertions.

---

## M5 — Library & Dashboard  (parallelizable with M4)
**Deliverable:** documents table, jobs list, dashboard.

- `useDocuments`/`useJobs` hooks (paginated).
- Documents library (`/documents`): sortable table, search, pagination, row actions (view/delete, "compare with…"), delete confirm.
- Jobs list (`/jobs`): filter by status, links to live/report.
- Dashboard: KPI cards (tabular nums) + recent jobs; status chip wired to `/api/health/ready`.
- Empty states for all.

**Verify:** pagination/sort/search work; delete confirms; status chip reflects worker busy/idle; deep links resolve.

---

## M6 — Polish & Accessibility
**Deliverable:** dark mode solid, EN locale, perf budgets, axe clean.

- Dark-mode contrast audit (every change-type color in both themes).
- Complete EN translation; language toggle.
- Virtualize ACU list + large tables (`@tanstack/react-virtual`).
- Skeleton/empty/error states everywhere; `prefers-reduced-motion` paths.
- `@axe-core/playwright` on every page → zero critical/serious.
- Perf: LCP <2.5s, CLS <0.1 on dev machine; route-splitting confirmed in build.

**Verify:** axe report clean; toggle vi/en + light/dark; load a 100+ ACU report without jank.

---

## M7 — Ship
**Deliverable:** production build + deploy runbook.

- `npm run build` → `dist/`; configure static host + `/api`,`/ws` reverse proxy; set `VITE_*_BASE_URL`.
- (Optional, if chosen) `Dockerfile.frontend` + compose.
- Tighten backend CORS to the frontend origin (coordinate — backend change).
- Full E2E (register→upload→compare→progress→report→export) green against real stack.
- Runbook: how to build, serve, point at backend, rotate if backend moves.

**Verify:** production build served; E2E green end-to-end; runbook reviewed.

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Contract drift (FE types vs backend) | Med | High | M2 hits the real API early; mirror types from `/openapi.json`; update `03-api-contract.md` in-lockstep |
| WebSocket reliability (drops, token expiry mid-job) | High | High | Polling fallback + reconnect + resync from M3 design; close-code handling |
| `pending` looks stuck (GPU serializes) | High | Med | Distinct "queued for GPU" UI + explain one-job-at-a-time |
| Token storage security | Med | High | Default in-memory access + refresh in storage; get sign-off (spec OQ#1) before Phase 4 |
| Large reports (many ACUs) jank | Med | Med | Virtualize in M6; design around it from M4 |
| Evidence truncation/liability | Low | **Critical** | Hard rule: never truncate quotes; checklist gate |
| Offline font/asset loading | Med | Med | Self-host fonts; no runtime CDN |
| Backend CORS `*` in prod | Med | High | M7 hardening (coordinate with backend) |

---

## Verification Checkpoints (gates between milestones)

1. **After M0:** gates green (`typecheck/lint/test/build`) — or don't proceed.
2. **After M2:** first real API round-trip works; `types/` match backend reality.
3. **After M3:** a job runs to completion live through the UI with a drop recovered.
4. **After M4:** a completed report renders with full evidence (the product's core promise demonstrated).
5. **After M6:** axe clean, perf budgets met, both themes/locales.
6. **After M7:** E2E green against the real stack; runbook complete.

Each checkpoint = a human review, per spec-driven-development.
