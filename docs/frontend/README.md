# Frontend Planning — LegalDiff

This folder is the **spec, design system, and build plan** for the LegalDiff frontend — a desktop-first web UI that lets legal professionals upload two versions of a Vietnamese legal document, run a zero-hallucination comparison, and review an evidence-backed report.

It is produced with the **spec-driven-development** workflow (Specify → Plan → Tasks) and applies the **`ui-ux-pro-max`** design skill ("Trust & Authority" system for legal/enterprise products).

The backend (FastAPI + Celery + WebSocket + Postgres + Redis + Qdrant) already exists and is complete; this planning covers **only the frontend** that consumes it.

---

## Decisions (locked)

| Area | Choice |
|---|---|
| Stack | **Vite 5 + React 18 + TypeScript (strict) + Tailwind + shadcn/ui** |
| Serve/deploy | **Separate frontend**, proxies `/api` + `/ws` → FastAPI `:8001` |
| Scope | **Full app** (auth, library, upload wizard, live progress, report viewer, export) |
| Auth | **Full multi-user** (register + login + token refresh) |

---

## Read order

| # | Doc | Purpose |
|---|---|---|
| 0 | [`00-spec.md`](./00-spec.md) | **The spec** — objective, stack, commands, structure, code style, testing, boundaries, success criteria, open questions |
| 1 | [`01-design-system.md`](./01-design-system.md) | Design tokens (navy "Trust & Authority"), typography, components, dark mode, the evidence-rendering pattern |
| 2 | [`02-architecture.md`](./02-architecture.md) | Layered architecture, API client, auth/token lifecycle, WebSocket + polling, state, routing |
| 3 | [`03-api-contract.md`](./03-api-contract.md) | **Exact** REST + WebSocket contract and data models the frontend must honor |
| 4 | [`04-ux-flows-and-wireframes.md`](./04-ux-flows-and-wireframes.md) | IA, sitemap, user flows, page-by-page ASCII wireframes |
| 5 | [`05-uiux-guidelines.md`](./05-uiux-guidelines.md) | Binding UI/UX rules by priority (a11y, touch, perf, forms, nav, evidence UX) + checklists |
| 6 | [`06-implementation-plan.md`](./06-implementation-plan.md) | Milestones (M0–M7), dependency graph, risks, verification checkpoints |
| 7 | [`07-task-breakdown.md`](./07-task-breakdown.md) | Discrete tasks with acceptance criteria, verify steps, files |

**If you read only one thing:** `00-spec.md`.
**Before writing a component:** `01` (look) + `05` (rules).
**Before calling the API:** `03` (contract).

---

## Status

- ✅ Phase 1 (Specify) — `00-spec.md`
- ✅ Phase 2 (Plan) — `06-implementation-plan.md`
- ✅ Phase 3 (Tasks) — `07-task-breakdown.md`
- ⏳ Phase 4 (Implement) — **pending human approval of this spec.**

The spec contains **Open Questions** (`00-spec.md` §9) that should be answered before implementation begins; the task list flags which task each question blocks.

---

## How to start (after approval)

1. Resolve the open questions in `00-spec.md` §9.
2. Begin at **M0 / T0.1** in `07-task-breakdown.md`.
3. Implement one task at a time, test-first, verifying each 🔎 step before marking done.
4. Treat every milestone checkpoint as a human review gate.
