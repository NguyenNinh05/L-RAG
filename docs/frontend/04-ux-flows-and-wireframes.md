# UX Flows & Wireframes — LegalDiff Frontend

> Information architecture, primary user journeys, and page-level wireframes.
> Applies the `ui-ux-pro-max` "Trust & Authority" system and UX rules (see `01-design-system.md`, `05-uiux-guidelines.md`).
> Desktop-first, responsive; copy in Vietnamese.

---

## 1. Personas & Jobs-to-be-Done

| Persona | Goal | Pain we remove |
|---|---|---|
| **Linh — Legal Counsel** | Confirm what changed between a contract draft and the signed version, with proof | Manual redline reading; missing a clause |
| **Minh — Compliance Officer** | Audit a regulation update against the prior version | No audit trail of *where* the change is |
| **Hoa — Contract Reviewer** | Get an executive summary fast, then drill into specifics | Reading 40 pages to find 3 changes |

**JTBD:** *"When two versions of a legal document exist, help me see every change — down to a single number — each backed by the exact original wording, so I can trust nothing was missed."*

---

## 2. Design Principles in action (recap)

- **Evidence first.** Quotes are never collapsed, summarized, or hidden behind a click without a visible affordance.
- **One primary action per screen** (skill §4 `primary-action`).
- **Predictable navigation + deep links** — every screen is URL-reachable (skill §9).
- **Always-on feedback** — loading skeletons, empty states, error recovery (skill §3/§8).

---

## 3. Sitemap / Information Architecture

```
LegalDiff
├── /login                       (public)
├── /register                    (public)
├── AppShell  (sidebar + topbar + status chip)   ← everything below is protected
│   ├── /dashboard               jobs overview + system status + quick "New comparison"
│   ├── /documents               library (table) of uploaded docs
│   │   └── /documents/:id       detail (metadata + which jobs used it)
│   ├── /jobs                    all comparison jobs (filter by status)
│   │   ├── /jobs/new            new-comparison wizard (Step 1 V1 → Step 2 V2 → Step 3 options → run)
│   │   └── /jobs/:id            live progress (phase stepper, GPU-queue notice, cancel)
│   │       └── /jobs/:id/report comparison report viewer
│   │           └── /reports/:id single DiffPair report detail (deep-linked)
│   └── /settings                profile, theme (light/dark), language (vi/en)
```

**Navigation:** persistent **left sidebar** (desktop `lg+`, icons + labels — skill §9 `nav-label-icon`), collapses to a **bottom/top bar** on mobile (skill §9 `adaptive-navigation`). Active route is visually marked (`nav-state-active`). A **top bar** carries: app wordmark, global search (later), system-status chip, theme toggle, user menu.

---

## 4. Primary User Flows

### Flow A — First comparison (the money path)
```
Login → Dashboard
  └─[New comparison]─▶ Wizard
        Step 1: upload/select V1 (drag-drop or pick from library)
        Step 2: upload/select V2
        Step 3: options (skip Phase 3? config) → [Start comparison]
              ▼
        Job live progress (/jobs/:id)
        queued → ingestion → alignment → comparison → done
              ▼  (on completed)
        Report viewer (/jobs/:id/report)
          Executive summary → change list (filter/search) → drill ACU → export
```

### Flow B — Resume a queued/running job
```
Dashboard/Jobs list → job row shows "Đang xếp hàng chờ GPU" or live %
  └─ open → live progress (WS) → on done → report
```

### Flow C — Re-run / compare existing docs
```
Documents library → select V1 → "Compare with…" → pick V2 → job
```

### Flow D — Review a finished report (deep-linked, shareable)
```
/jobs/:id/report → tabs: Summary | Changes | Export
  Changes: filter by type/severity, search "Điều 5", click ACU → evidence drawer
```

---

## 5. Wireframes (ASCII)

> Not pixel specs — layout intent. Apply `01-design-system.md` tokens. `lg` (1280px+) is the target; notes call out mobile stacking.

### 5.1 Auth — Login / Register
```
┌─────────────────────────────────────────────────────────────┐
│                          (centered card)                     │
│              ╭───────────────────────────╮                  │
│              │  ⚖ LegalDiff               │  ← EB Garamond   │
│              │  So sánh văn bản pháp lý   │                  │
│              │  ────────────────────────  │                  │
│              │  Tên đăng nhập ___________ │                  │
│              │  Mật khẩu      ___________👁│  ← show/hide     │
│              │  ⚠ (error under field)     │                  │
│              │  [  Đăng nhập  ] (primary) │                  │
│              │  Chưa có tài khoản? Đăng ký │                  │
│              ╰───────────────────────────╯                  │
└─────────────────────────────────────────────────────────────┘
```
- One primary action (`Đăng nhập`). Inline errors below fields, `role="alert"` (skill §8).
- Password show/hide toggle (skill §8 `password-toggle`).

### 5.2 AppShell — layout chrome
```
┌─────┬───────────────────────────────────────────────────────────┐
│ ⚖LD │ Topbar:  [🔍 search]          ● Worker: sẵn sàng   🌓  👤▾ │
├─────┼───────────────────────────────────────────────────────────┤
│ 🏠  │                                                           │
│ 📄  │                                                           │
│ ⚙   │                   <route page content>                   │
│ Jobs│                                                           │
│ 📊  │                                                           │
│ ⚙️⚙ │                                                           │
│     │                                                           │
│ + Mới│                                                          │
└─────┴───────────────────────────────────────────────────────────┘
 sidebar      main (focus target on route change)
```
- Sidebar items: icon **+ label** (skill §9). `+ Mới` is the persistent primary CTA to start a comparison.
- **Status chip** (`/api/health/ready`): "Worker: sẵn sàng / đang bận (GPU) / không kết nối".

### 5.3 Dashboard
```
┌────────────────────────────────────────────────────────────────┐
│ Xin chào, Linh                                  [+ So sánh mới] │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│ 📄 Tài liệu  │ 🔄 Đang chạy │ ✅ Hoàn thành│ ⚠ Thất bại         │
│    12        │    1 (35%)   │    24        │    2               │  ← KPI cards
├──────────────┴──────────────┴──────────────┴───────────────────┤
│ Việc so sánh gần đây                          (xem tất cả →)    │
│ ┌────────────────────────────────────────────────────────────┐ │
│ │ HĐ_supply_v2 vs v3 · so sánh · 35% alignment · 2 phút trước│ │
│ │ Nghị định 123 → 124    · ✅ hoàn thành · 18 thay đổi        │ │
│ │ …                                                          │ │
│ └────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```
- KPI cards with tabular numbers; "Đang chạy" card links to the live job.
- Each job row is a deep-link (`/jobs/:id`).

### 5.4 New-comparison Wizard (`/jobs/new`)
```
Step 1 of 3  ●━━○━━○      ← progress indicator (skill §8 multi-step-progress)

  ┌─ Văn bản gốc (V1) ──────────────────────────┐
  │  ┌───────────────────────────────────────┐  │
  │  │   ⬆  Kéo thả file PDF/DOCX vào đây    │  │  ← UploadDropzone (drag-drop + picker)
  │  │      hoặc [ Chọn tệp ]   (≤ 100MB)    │  │
  │  └───────────────────────────────────────┘  │
  │  hoặc chọn từ thư viện:  [ Doc ▾ ]          │
  └─────────────────────────────────────────────┘
                                  [ Hủy ]   [ Tiếp theo → ]
```
- Step 2 = identical for V2. Step 3 = options (`skip_phase3` toggle + advanced `config_overrides` collapsed by default — progressive disclosure).
- Back navigation preserves state (skill §9 `state-preservation`). Identical-doc selection blocked (mirrors backend).

### 5.5 Job — Live Progress (`/jobs/:id`)
```
HĐ_supply_v2.pdf  ⟷  HĐ_supply_v3.pdf
┌──────────────────────────────────────────────────────────────┐
│ Trạng thái: Đang xếp hàng chờ GPU  ⓘ  (1 việc chạy trước)    │  ← distinct QUEUED state
│                                                              │
│  ●━━━━●━━━━○━━━━○━━━━○   35%                                 │
│  Xếp hàng  Phân tích  Ghép nối  So sánh  Xong                │  ← PhaseStepper
│            ▲ current                                          │
│  Đang nhúng vector BGE-M3…                                   │  ← WS message
│  ────────────────────────────────────────────────────────    │
│  Bắt đầu: 10:32   ·   [ Hủy việc so sánh ] (destructive)     │
└──────────────────────────────────────────────────────────────┘
```
- **`pending` → "Đang xếp hàng chờ GPU"** with an explainer (one job at a time). `processing` → phase stepper advances.
- WS drives the % and message live; polling is the silent fallback.
- Cancel = destructive, confirmed (skill §8 `confirmation-dialogs`).

### 5.6 Comparison Report Viewer (`/jobs/:id/report`) — the signature screen
```
HĐ_supply_v2 ⟷ v3   ·   18 thay đổi   ·   [⬇ Tải Markdown] [🖨 PDF]
┌─ Tabs: [ Tóm tắt ] [ Các thay đổi (18) ] [ Trích xuất ] ─────┐
│                                                              │
│  ╭ Tóm tắt điều hành ────────────────────────────────────╮  │
│  │ Tổng quan: phát hiện 18 thay đổi đã xác minh…        │  │
│  │ ⚠ Rủi ro: thay đổi điều khoản thanh toán (3)         │  │  ← risk_flags
│  │ Thay đổi số liệu: 500.000đ → 600.000đ …              │  │
│  ╰──────────────────────────────────────────────────────╯  │
│                                                              │
│  Lọc: [Loại ▾] [Mức độ ▾]   Tìm: [Điều ____]   Sắp xếp ▾    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ ACU · [Đổi số liệu] Điều 5, K1            ✓ đã xác minh│   │
│  │ ┌── V1 (gốc) ──────────┐ ┌── V2 (sửa) ─────────────┐ │   │
│  │ │ …thanh toán »500.000đ│ │ …thanh toán »600.000đ   │ │   │  ← EvidenceQuote pair
│  │ │ trong 30 ngày«       │ │ trong 45 ngày«          │ │   │
│  │ └──────────────────────┘ └──────────────────────────┘ │   │
│  │ Lý do: số tiền và hạn thanh toán thay đổi            │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌ ACU · [Thêm mới] Điều 7 ─────────────────────────────┐   │  ← addition: single side
│  │ ┌── V2 (sửa) ──────────────────────────────────────┐ │   │
│  │ │ «(mới) Bên A có quyền gia hạn thêm 15 ngày…»     │ │   │
│  │ └──────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────┘   │
│   … (virtualized when >50)                                  │
└──────────────────────────────────────────────────────────────┘
```
- **Executive summary** pinned on top (`risk_flags` get destructive/risk styling).
- **Catalog summary counts** (total/matched/added/deleted/split/merged) as a small legend.
- **Change list:** filter by type + severity, search by location, sortable. Virtualized.
- **Each ACU** renders both evidence quotes side-by-side (`lg+`) / stacked (mobile), changed token highlighted, monospace, never truncated.
- `added`/`deleted` match types show a single side with the correct treatment.

### 5.7 Documents Library (`/documents`)
```
┌─────────────────────────────────────────────────────────────────┐
│ Thư viện tài liệu                          [ + Tải lên ]        │
│ Tìm: [________]   Lọc: [Loại ▾]                                 │
│ ┌──────────┬──────────┬──────┬───────┬──────────┬─────────────┐ │
│ │Tên       │Số hiệu   │Trang │Điều   │Đã tải lên│              │ │
│ ├──────────┼──────────┼──────┼───────┼──────────┼─────────────┤ │
│ │HĐ_v2.pdf │HĐ-2024-1 │ 12   │ 24    │ 2 ngày   │ ⋯ (xem/xóa) │ │  ← sortable, aria-sort
│ │ND_123.doc│123/2024  │ 40   │ 86    │ 5 ngày   │ ⋯           │ │
│ └──────────┴──────────┴──────┴───────┴──────────┴─────────────┘ │
│  ◀ 1 2 3 ▶                                                     │
└─────────────────────────────────────────────────────────────────┘
```
- Sortable table (`aria-sort`), pagination, row actions. "So sánh với…" quick action from a row (Flow C).

---

## 6. State Matrix (every async surface has all 4 states)

| Screen | Loading | Empty | Error | Success |
|---|---|---|---|---|
| Dashboard | skeleton KPI + table | "Chưa có việc so sánh" + CTA | retry | data |
| Library | skeleton rows | "Chưa có tài liệu" + upload CTA | retry | rows |
| Job progress | skeleton + connecting WS | n/a (always has a job) | "Không kết nối được — đang thử lại" (reconnect) | phase advance |
| Report | skeleton summary + ACU cards | "Chưa có thay đổi" (0 ACUs) | retry | report |
| Upload | progress bar on dropzone | n/a | inline "Sai định dạng/kích thước" | toast + row appears |

---

## 7. Mobile adaptations (skill §5)

- Sidebar → bottom nav (≤5 items) or hamburger drawer.
- Report evidence quotes **stack vertically** (V1 above V2), each full-width.
- Wizard becomes full-height steps with a sticky "Next" footer.
- Tables → cards on `< md`.
- All touch targets ≥44px; no content under fixed bars (safe padding).

---

## 8. Accessibility-critical flows

- **Keyboard-only compare:** tab to ACU list, arrow-key to move between ACUs, Enter to open evidence drawer; focus visible throughout.
- **Screen reader:** job status changes announced via `aria-live="polite"` region; ACU evidence read with location + change type + both quotes.
- **Reduced motion:** progress bar fills without animation; no entrance staggers.
- **Contrast:** every change-type color verified in **both** themes with icon+label backup.
