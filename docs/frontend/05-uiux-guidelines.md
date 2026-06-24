# UI/UX Guidelines — LegalDiff Frontend

> Applied, binding rules for this product — distilled from the `ui-ux-pro-max` skill and tailored to a legal-document comparison tool.
> Organized by the skill's priority categories (1→10). Each rule has the **do**, the **don't**, and **how it applies here**.
> This is the checklist code is reviewed against. See `01-design-system.md` for tokens.

Priority legend: 🔴 CRITICAL · 🟠 HIGH · 🟡 MEDIUM · 🔵 LOW

---

## 1. Accessibility 🔴

| Rule | Do | Don't | In LegalDiff |
|---|---|---|---|
| `color-contrast` | 4.5:1 body text, 3:1 large/UI | Gray-on-gray, low-contrast captions | Verify each change-type color in **light + dark** |
| `focus-states` | Visible 2–4px ring (`--ring`) | Remove focus outlines | Every button/link/input/dropdown; never `outline:none` without replacement |
| `aria-labels` | Label icon-only buttons (close, cancel, show-pwd, ⋯ menus) | Naked icon buttons | Topbar icons, ACU row actions, password toggle |
| `keyboard-nav` | Tab order = visual order; full keyboard support | Trap focus, skip links missing | Filter/search/report operable without mouse |
| `heading-hierarchy` | Sequential h1→h6, no skips | Two h1s, h3 under h1 | Report: h1 title → h2 section → h3 ACU |
| `color-not-only` | Icon + text + color for every change type/status | Red row with no label | `ChangeTypeBadge` always icon+label+color |
| `reduced-motion` | Respect `prefers-reduced-motion` | Force animations | Progress bar, toast, modal entrance all gate on it |
| `skip-links` | "Skip to main content" | Force tabbing through nav | On every protected page |
| `aria-live-errors` | `role="alert"` / `aria-live` for errors & status | Visual-only error | Login errors, job failure, upload errors, WS status |

**Target: WCAG 2.1 AA.** Axe in E2E: zero critical/serious on core pages.

---

## 2. Touch & Interaction 🔴

| Rule | Do | Don't | Here |
|---|---|---|---|
| `touch-target-size` | ≥44×44px; pad small icons | <44px hit areas | Sidebar items, table row actions, filter chips |
| `loading-buttons` | Disable + spinner during async | Double-submit | Login, Start comparison, Upload, Cancel |
| `error-feedback` | Clear message **near** the problem | Toast-only errors | Inline field errors + summary |
| `cursor-pointer` | `cursor: pointer` on clickables | Pointer on non-clickable | All buttons/links/rows |
| `press-feedback` | Visual press state (opacity/scale 0.97) | No response | Cards, buttons, nav |
| `tap-feedback-speed` | <100ms visual response | Laggy taps | Optimistic UI on filters |

---

## 3. Performance 🟠

| Rule | Do | Don't | Here |
|---|---|---|---|
| `loading-states` | Skeleton for >300ms ops | Blank/frozen screen | Report, library, dashboard |
| `lazy-loading` | Route-split heavy views | One giant bundle | Report viewer + report detail lazy-loaded |
| `virtualize-lists` | Virtualize >50 items | Render 500 ACU nodes | ACU list, large job/document lists (`@tanstack/react-virtual`) |
| `content-jumping` | Reserve space for async | Layout shift on load | Skeletons match layout; CLS < 0.1 |
| `debounce-throttle` | Debounce input/resize 250ms | Request per keystroke | Report search/filter, library search |
| `number-tabular` | `tabular-nums` on numbers | Shifting number widths | KPIs, counts, progress %, evidence values |
| `font-loading` | `font-display: swap`; self-host | FOIT, CDN dependency (offline!) | Bundle EB Garamond/Lato/JetBrains Mono locally |
| `main-thread-budget` | Keep frames <16ms | Layout thrashing | Avoid animating width/height; use transform/opacity |

---

## 4. Style Selection 🟠

- **"Trust & Authority"** consistently across every screen — never mix in glassmorphism/brutalism/neon (skill `consistency`, `effects-match-style`).
- **SVG icons only (Lucide)**, 1.5px stroke, one family, no filled/outline mixing at one hierarchy (`no-emoji-icons`, `icon-style-consistent`).
- **One primary CTA per screen** (`primary-action`). Secondary actions are `outline`/`ghost`.
- **Consistent elevation/radius scale** — no random shadow values (`elevation-consistent`).
- **Light + dark designed together** — contrast verified independently (`dark-mode-pairing`).

---

## 5. Layout & Responsive 🟠

- **Mobile-first**, optimize for `lg/xl` (1280–1440) = the lawyer's laptop.
- **Breakpoints** `640/768/1024/1280/1440`. Test at **375 / 768 / 1024 / 1440**.
- **16px body min** on mobile (avoids iOS auto-zoom); line-height 1.5–1.6.
- **No horizontal scroll** on mobile; report reading column 65–75 chars.
- **8dp spacing rhythm** throughout.
- **z-index scale** defined (0/10/20/40/50/60/100) — no ad-hoc `z-[9999]`.
- **Reserve space** under fixed topbar/sidebar so content isn't hidden.
- **Visual hierarchy** via size/spacing/contrast — not color alone.

---

## 6. Typography & Color 🟡

- Type roles: EB Garamond (display/H1–H3), Lato (body/label), JetBrains Mono (evidence values, locations, numbers).
- Weight hierarchy: headings 600, body 400, labels 500.
- **Semantic color tokens** (not raw hex) in components — `bg-primary`, `text-destructive`, etc. (`color-semantic`).
- **Functional colors** (red error, green success) always carry icon + text (`color-not-decorative-only`).
- **Truncation:** prefer **wrap**; if a location must truncate, ellipsis + tooltip with full text. **Never truncate evidence quotes.**

---

## 7. Animation 🟡

- **Duration 150–300ms** micro-interactions; state transitions smooth not snap; exits shorter than enters (~70%).
- **`transform`/`opacity` only** — never animate `width/height/top/left`.
- **Motion = meaning:** progress fill, modal-from-trigger, phase-step advance. No decorative loops.
- **`prefers-reduced-motion`:** reduce/disable. Progress still shows value (just no tween).
- **Interruptible:** user input cancels in-progress animation; UI stays interactive.
- **Stagger** list entrances ≤30–50ms/item, max 1–2 key elements per view.

---

## 8. Forms & Feedback 🟡

- **Visible label per input** (not placeholder-only). Required fields marked (`*`).
- **Inline validation on blur** (not keystroke); error below the field, `role="alert"`.
- **Error clarity:** cause + how to fix (e.g. "File phải nhỏ hơn 100MB" not "Invalid").
- **Submit feedback:** button loading → success toast / error.
- **Helper text** under complex inputs (e.g. "Hỗ trợ PDF, DOCX — tối đa 100MB").
- **Confirmation dialogs** for destructive actions (delete doc/job, cancel running job) — danger styling, spatially separated (`destructive-emphasis`).
- **Toasts** auto-dismiss ~4s, `aria-live=polite`, never steal focus, include an action where useful (Undo).
- **Empty states** helpful + primary action ("Chưa có tài liệu — Tải lên").
- **Multi-step wizard:** step indicator, back preserves state, confirm-before-leave with unsaved input.

---

## 9. Navigation Patterns 🟠

- **Sidebar (desktop `lg+`)** with icon **+ label**; collapses to bottom nav/drawer on mobile.
- **Active route highlighted** (accent bar/weight). Navigation placement stable across pages.
- **Predictable back** — preserves scroll/filter/input state.
- **Deep links** for every key screen (job, report) — shareable by URL.
- **Breadcrumbs** for >2-level depth (report under job).
- **Overflow menu** when actions exceed space (document row `⋯`).
- **Destructive nav separated** (logout/delete) from normal items.
- **Don't mix** sidebar + bottom nav + tabs at the same hierarchy.

---

## 10. Charts & Data 🔵

(Used for catalog summary / future analytics.)

- Match chart type to data (counts → bar, trend → line). Avoid pie >5 categories.
- **Accessible colors** — no red/green-only pairs; supplement with icon/pattern.
- **Tooltips** on hover **and** tap; keyboard-reachable.
- **Legend** near the chart, clickable to toggle series.
- **Empty/loading/error states** for charts too (skeleton, "No data", retry).
- For the **catalog summary**, prefer a compact **legend of counts** (total/matched/added/deleted/split/merged) over a chart unless a distribution view adds value.

---

## 11. Product-Specific: Zero-Hallucination Evidence UX

These rules exist because **evidence is the product**. They override convenience.

1. **Every ACU shows both verbatim quotes** (`verbatim_evidence_v1`, `verbatim_evidence_v2`), monospace, `whitespace-pre-wrap`, **never truncated**. The changed token is highlighted in place.
2. **Exact source location** (`location_v1`/`location_v2`) always visible above each quote.
3. **`confidence`** shown as a subtle verified/low-trust marker; `< 0.6` → amber "cần xem xét lại", never silently hidden.
4. **Rejected ACUs** (`rejected_acus_detail`) are shown in a distinct "đã loại bỏ (nghi ảo giác)" section with the rejection reason — transparency about what the model *didn't* trust.
5. **`hallucination_rate`** per report shown as a small, honest metric — not buried.
6. **Export** (Markdown/PDF) must include the same evidence, not a summary.

---

## 12. Pre-Delivery Checklist (run before merging any screen)

**Visual**
- [ ] No emoji icons; Lucide only, 1.5px stroke, one family.
- [ ] Semantic tokens used (no raw hex in components).
- [ ] Press states don't shift layout.

**Interaction**
- [ ] All clickables ≥44px + `cursor-pointer` + press feedback.
- [ ] Micro-interactions 150–300ms; transform/opacity only.
- [ ] Disabled states clear + non-interactive.
- [ ] Focus order matches visual order; labels descriptive.

**Light/Dark**
- [ ] Body ≥4.5:1, secondary ≥3:1 in **both** themes.
- [ ] Borders/states visible in both; modal scrim 40–60%.
- [ ] Both themes visually tested.

**Layout**
- [ ] Checked 375 / 768 / 1024 / 1440.
- [ ] No content under fixed bars; 8dp rhythm; reading column ≤75 chars.

**States**
- [ ] Loading skeleton (>300ms), empty, error (with recovery) present.
- [ ] Errors `role="alert"` / `aria-live`; toasts non-focusing.

**Evidence**
- [ ] ACU quotes verbatim, monospace, wrap (never truncate).
- [ ] Change type/severity always icon + label + color.
- [ ] Locations, confidence, rejected-ACU section present.

**A11y**
- [ ] Reduced-motion + large-text scaling don't break layout.
- [ ] Axe: zero critical/serious on the page.
