# Task Breakdown — LegalDiff Frontend

> **Phase 3: Tasks.** Discrete, single-session units with explicit acceptance criteria, a verification step, and the files touched.
> Ordered by dependency (not importance). Each task ≤~5 files. Implement one at a time per `incremental-implementation` + `test-driven-development`; load the relevant spec section via `context-engine`.
> Milestones (M0–M7) defined in `06-implementation-plan.md`.

**Legend:** ✅ Acceptance · 🔎 Verify · 📁 Files. Mark `- [ ]` done only when ✅ **and** 🔎 pass.

---

## M0 — Foundations

- [ ] **T0.1 — Scaffold Vite + React + TS app under `frontend/`**
  - ✅ `npm run dev` boots on :5173; `npm run build` produces `dist/`; TS strict on.
  - 🔎 `node -v` (20); `npm run build` exits 0; `tsc --noEmit` clean.
  - 📁 `frontend/{package.json, vite.config.ts, tsconfig.json, index.html, src/main.tsx, src/App.tsx}`

- [ ] **T0.2 — Tailwind + design tokens + self-hosted fonts**
  - ✅ Light + dark tokens from `01-design-system.md` §2/§9 in `globals.css` + `tailwind.config.ts`; EB Garamond / Lato / JetBrains Mono bundled locally; `darkMode:'class'`.
  - 🔎 Toggle `.dark` on `<html>` → palette swaps; no network request for fonts (offline).
  - 📁 `frontend/{tailwind.config.ts, postcss.config.js, src/styles/globals.css, public/fonts/*}`

- [ ] **T0.3 — Tooling: ESLint, Prettier, Vitest, Playwright, path alias**
  - ✅ `lint`/`typecheck`/`test`/`test:e2e` scripts exist and pass on a sample test.
  - 🔎 `npm run lint --max-warnings=0` clean; sample `*.test.tsx` runs green.
  - 📁 `frontend/{.eslintrc.cjs, prettier config, vitest.config.ts, playwright.config.ts, src/test/setup.ts}`

- [ ] **T0.4 — Vite proxy + env + i18n + theme bootstrap**
  - ✅ `/api` + `/ws` proxy to `:8001`; `VITE_API_BASE_URL`/`VITE_WS_BASE_URL` read; `react-i18next` init with `vi`; theme store persists.
  - 🔎 Dev request to `/api/health` returns `{status:ok}` (backend running); `public/locales/vi/translation.json` loads.
  - 📁 `frontend/{.env.example, vite.config.ts, src/main.tsx, src/lib/i18n.ts, public/locales/vi/translation.json}`

- [ ] **T0.5 — AppShell + routing skeleton + status chip placeholder**
  - ✅ Sidebar (icon+label) + topbar + main; routes `/login`(stub), `/dashboard`(stub); active-route highlight; skip-link; focus to main on route change.
  - 🔎 Keyboard-tab through shell; axe: zero critical on shell.
  - 📁 `src/components/layout/{AppShell,Sidebar,Topbar}.tsx, src/lib/router.tsx, src/App.tsx`

**Checkpoint M0:** gates green + dashboard stub renders correctly.

---

## M1 — Authentication

- [ ] **T1.1 — Auth types + API client + token store**
  - ✅ `types/auth.ts` mirrors contract; `lib/api/client.ts` does 401→refresh→retry once; in-memory access + refresh in storage.
  - 🔎 MSW integration test: 401 triggers refresh then succeeds; failed refresh clears session.
  - 📁 `src/types/auth.ts, src/lib/api/{client,auth}.ts, src/lib/auth/{tokens,session}.ts`

- [ ] **T1.2 — `useAuth` hook + `requireAuth` guard**
  - ✅ Guard redirects unauthenticated → `/login?next=`; `GET /me` hydrates user on boot.
  - 🔎 Visit protected route logged-out → redirect; logged-in → renders.
  - 📁 `src/hooks/useAuth.ts, src/lib/auth/guard.ts, src/lib/router.tsx`

- [ ] **T1.3 — Login + Register pages**
  - ✅ RHF + Zod; inline errors `role=alert`; password show/hide; one primary CTA; loading state; success → `/dashboard`.
  - 🔎 Submit invalid → field errors; valid → navigates; a11y assertions pass.
  - 📁 `src/routes/(auth)/{login,register}.tsx, src/components/auth/*, src/lib/query/keys.ts`

**Checkpoint M1:** register→login→reload (silent refresh)→logout works against live backend.

---

## M2 — Upload + Create Job  ⚡ first end-to-end

- [ ] **T2.1 — Document + Job types and API modules**
  - ✅ `types/{document,job}.ts`; `lib/api/{documents,jobs}.ts` typed; `PaginatedResponse` generic.
  - 🔎 Unit tests for client wrappers against MSW mirroring `03-api-contract.md`.
  - 📁 `src/types/{document,job,common}.ts, src/lib/api/{documents,jobs}.ts`

- [ ] **T2.2 — `UploadDropzone` with client-side validation**
  - ✅ Drag-drop + picker; reject non-`.pdf/.docx` and >100MB with clear inline message; progress; success toast.
  - 🔎 Drop a `.txt` → rejected; drop 120MB → rejected; valid PDF → `DocumentUploadResponse`.
  - 📁 `src/components/documents/{UploadDropzone,FileMeta}.tsx, src/lib/utils/file.ts`

- [ ] **T2.3 — New-comparison wizard (`/jobs/new`)**
  - ✅ 3 steps (V1, V2, options); step indicator; back preserves state; identical-doc blocked; options: `skip_phase3` toggle + collapsed `config_overrides`; Start → POST `/api/jobs` → `/jobs/:id`.
  - 🔎 Create job from two real uploads → navigates to `/jobs/:id` with `status=pending`.
  - 📁 `src/routes/jobs/new/*.tsx, src/components/jobs/{WizardStepper,JobOptionsForm}.tsx`

**Checkpoint M2:** two real docs uploaded → job created against live backend (contract confirmed).

---

## M3 — Live Progress

- [ ] **T3.1 — `JobSocket` WebSocket client**
  - ✅ Connects `/ws/jobs/{id}?token=`; handles close 4001/4003/4004 (distinct messages, stop reconnect); reconnect w/ backoff (1/2/5s, ≤5); resync poll on reconnect; one refresh on token-expiry close.
  - 🔎 Unit test the state machine (fake WS) for close codes + backoff.
  - 📁 `src/lib/ws/JobSocket.ts, src/lib/ws/JobSocket.test.ts`

- [ ] **T3.2 — `useJobProgress` hook (WS + poll → one cache key)**
  - ✅ WS patches `['job',id,'status']`; poll `/status` every 3s; poll self-disables on terminal state.
  - 🔎 Render `<JobProgress>`; assert WS message updates cache; poll fallback covered.
  - 📁 `src/hooks/useJobProgress.ts, src/lib/query/keys.ts`

- [ ] **T3.3 — Job progress page (`PhaseStepper` + GPU-queue + cancel)**
  - ✅ Phase stepper `queued→ingestion→alignment→comparison→done`; **`pending` → "Đang xếp hàng chờ GPU"**; live message + start time; progress bar (reduced-motion aware); cancel confirm → POST `/cancel`; on `completed` → report route.
  - 🔎 Run a job live: phases advance; throttle network → poll keeps it moving; two jobs → 2nd queued; cancel works.
  - 📁 `src/routes/jobs/[id]/*.tsx, src/components/jobs/{PhaseStepper,ProgressBar,GpuQueueNotice}.tsx`

**Checkpoint M3:** a job runs to completion live through the UI; a dropped socket recovers.

---

## M4 — Report Viewer  ★ signature

- [ ] **T4.1 — Report types + API + React Query hooks**
  - ✅ `types/report.ts` (ReportSummary/Detail, ExecutiveSummary, **AcuOutput**); `lib/api/reports.ts`; `useJobReports`, `useReport`.
  - 🔎 Open a completed job's reports list; fetch one detail → typed object.
  - 📁 `src/types/report.ts, src/lib/api/reports.ts, src/hooks/{useJobReports,useReport}.ts`

- [ ] **T4.2 — `AcuCard` + `EvidenceQuote` + `ChangeTypeBadge`**
  - ✅ Both evidence quotes verbatim, monospace, `whitespace-pre-wrap`, **never truncated**, changed token highlighted; location above each; change-type icon+label+color; confidence marker; `added`/`deleted` single-side.
  - 🔎 Component test: renders all 6 change types; quote with long text wraps not truncates; a11y labels present.
  - 📁 `src/components/report/{AcuCard,EvidenceQuote,ChangeTypeBadge,ConfidenceMark}.tsx`

- [ ] **T4.3 — Report page: summary + changes tab + filters**
  - ✅ Executive summary (risk_flags styled); catalog-count legend; changes list filter (type/severity) + search (location) + sort (debounced); rejected-ACUs section; `hallucination_rate` metric.
  - 🔎 Filter/search/sort behave; rejected section visible; counts correct.
  - 📁 `src/routes/jobs/[id]/report/*.tsx, src/components/report/{ExecutiveSummary,CatalogLegend,AcuFilterBar,RejectedAcus}.tsx`

- [ ] **T4.4 — Export (Markdown download + print-to-PDF)**
  - ✅ Download from `/api/reports/{id}/markdown`; print stylesheet includes evidence; print button.
  - 🔎 Downloaded `.md` matches backend; browser print preview shows clean layout with quotes.
  - 📁 `src/components/report/ReportExport.tsx, src/styles/print.css`

**Checkpoint M4:** completed report renders with full evidence — core product promise demonstrated.

---

## M5 — Library & Dashboard  (parallel with M4 after M3)

- [ ] **T5.1 — Documents library (`/documents`)**
  - ✅ Sortable table (`aria-sort`), search, pagination, row actions (view/delete/compare-with), delete confirm; empty state.
  - 🔎 Paginate/sort/search; delete confirms + invalidates cache.
  - 📁 `src/routes/documents/{index,detail}.tsx, src/components/documents/{DocumentTable,DocumentRowActions}.tsx`

- [ ] **T5.2 — Jobs list + Dashboard**
  - ✅ Jobs list filter-by-status + deep links; dashboard KPI cards (tabular nums) + recent jobs; empty states.
  - 🔎 Status filter; KPIs reflect real counts; recent-job links resolve.
  - 📁 `src/routes/jobs/index.tsx, src/routes/dashboard/*.tsx, src/components/jobs/{JobTable,JobKpiCards}.tsx`

- [ ] **T5.3 — System status chip wired to `/api/health/ready`**
  - ✅ Polls health; shows worker `available/busy/unreachable`; db/redis/llm tooltips; never blocks UI on failure.
  - 🔎 Stop worker → chip shows busy/unreachable; restart → available.
  - 📁 `src/components/layout/StatusChip.tsx, src/hooks/useSystemHealth.ts`

---

## M6 — Polish & Accessibility

- [ ] **T6.1 — Dark-mode contrast audit + fixes**
  - ✅ Every change-type/severity color meets 4.5:1/3:1 in both themes; borders visible.
  - 🔎 Manual + axe contrast check per theme.
  - 📁 `src/styles/globals.css, src/components/report/ChangeTypeBadge.tsx`

- [ ] **T6.2 — EN translation + language toggle**
  - ✅ 100% user-facing strings externalized; `en/translation.json` complete; toggle persists.
  - 🔎 Toggle vi↔en flips entire UI; no hardcoded vi in components (`grep` check).
  - 📁 `public/locales/en/translation.json, src/components/layout/LanguageToggle.tsx`

- [ ] **T6.3 — Virtualize ACU list + large tables**
  - ✅ `@tanstack/react-virtual` on ACU list and big tables; 200 rows scroll without jank.
  - 🔎 Load 200-item fixture; frame budget <16ms while scrolling.
  - 📁 `src/components/report/AcuList.tsx, src/components/documents/DocumentTable.tsx`

- [ ] **T6.4 — Axe pass + reduced-motion + perf budgets**
  - ✅ `@axe-core/playwright` zero critical/serious on all pages; reduced-motion paths work; LCP<2.5s, CLS<0.1; route-splitting confirmed in build.
  - 🔎 Run E2E+axe; Lighthouse on dashboard + report.
  - 📁 `e2e/a11y.spec.ts, src/styles/globals.css`

---

## M7 — Ship

- [ ] **T7.1 — Production build + reverse-proxy config + CORS hardening**
  - ✅ `npm run build` → `dist/`; static host config proxies `/api`,`/ws`; backend CORS tightened to FE origin (coord. backend).
  - 🔎 Served build hits backend same-origin; cross-origin blocked where expected.
  - 📁 `frontend/nginx.conf` (or host config), `.env.production`, backend CORS change (separate PR)

- [ ] **T7.2 — End-to-end E2E green + runbook**
  - ✅ Playwright: register→upload→compare→progress→report→export passes against real stack; runbook documents build/serve/point-at-backend.
  - 🔎 `npm run test:e2e` green; runbook reviewed by a human.
  - 📁 `e2e/full-flow.spec.ts, docs/frontend/08-runbook.md`

---

## Open Questions to resolve before the marked task

| Task | Blocking question (from `00-spec.md` §9) |
|---|---|
| T1.1 | OQ#1 — confirm token storage model (in-memory + refresh vs cookie) |
| T2.1 | OQ#5 — expected max ACU count (drives virtualization) |
| T4.4 | OQ#4 — bespoke branded PDF vs backend Markdown export |
| T7.1 | OQ#2/#3 — prod topology + registration gating |
| any branding | OQ#7 — logo/wordmark/brand color beyond the palette |

Resolve these in spec review, then implement.
