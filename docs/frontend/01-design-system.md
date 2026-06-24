# Design System — LegalDiff Frontend

> **Source of truth for how the product looks, feels, and moves.**
> Generated from the `ui-ux-pro-max` skill (`--design-system` for "legal / enterprise / trustworthy") and specialized for LegalDiff.
> Pair with `00-spec.md` (what we build) and `05-uiux-guidelines.md` (the rules).

The skill selected the **"Trust & Authority"** style for this product class — exactly right for a legal-document comparison tool whose entire value proposition is *evidence you can trust*. We adopt it wholesale and extend it with **semantic change-type colors** (diff vocabulary) and a **monospace evidence font**.

---

## 1. Design Principles (what guides every decision)

1. **Evidence is the hero.** The verbatim V1/V2 quotes are the product. Layout, type, and color all serve making them unmissable.
2. **Calm authority, not flash.** Navy + slate + restrained blue. No gradients, no glow, no emoji. The UI should feel like a notary's office, not a startup landing page.
3. **Density without noise.** Legal pros scan hundreds of changes. Compact, scannable, sortable — but always readable (≥16px body, 1.5 line-height).
4. **Never decide by color alone.** Every change type, severity, and status carries an **icon + text label**, not just a hue (colorblind-safe).
5. **Motion conveys meaning.** Transitions exist to show cause-effect (progress, state change), never to decorate.

---

## 2. Color Tokens

From the skill's "Trust & Authority" palette, expressed as Tailwind theme tokens via CSS variables (so dark mode swaps them in one place).

### Light mode (default)

| Token | Hex | Tailwind equiv | Use |
|---|---|---|---|
| `--background` | `#F8FAFC` | slate-50 | app background |
| `--foreground` | `#0F172A` | slate-900 | primary text |
| `--primary` | `#1E293B` | slate-800 | brand, topbar/sidebar, primary surfaces |
| `--primary-foreground` | `#FFFFFF` | white | text on primary |
| `--secondary` | `#334155` | slate-700 | secondary text / surfaces |
| `--accent` | `#2563EB` | blue-600 | **primary CTA**, links, focus, active nav |
| `--accent-foreground` | `#FFFFFF` | white | text on accent |
| `--muted` | `#F1F5F9` | slate-100 | subtle fills, hover rows |
| `--muted-foreground` | `#64748B` | slate-500 | secondary/caption text |
| `--border` | `#E2E8F0` | slate-200 | hairlines, card borders |
| `--destructive` | `#DC2626` | red-600 | errors, destructive actions |
| `--ring` | `#1E293B` | slate-800 | focus ring |

### Dark mode

Dark mode uses **desaturated, elevated** tonal variants — *not* inverted colors. Test contrast independently.

| Token | Hex | Notes |
|---|---|---|
| `--background` | `#0B1220` | near-black navy |
| `--foreground` | `#E2E8F0` | slate-200 (≥4.5:1 on bg) |
| `--primary` | `#1E293B` | unchanged brand navy |
| `--primary-foreground` | `#F8FAFC` | |
| `--accent` | `#3B82F6` | blue-500 — brighter for dark-bg contrast |
| `--muted` | `#1E293B` | elevated surface |
| `--muted-foreground` | `#94A3B8` | slate-400 (≥3:1 secondary) |
| `--border` | `#334155` | slate-700 — must stay visible |
| `--destructive` | `#F87171` | red-400 — lifted for contrast |
| `--ring` | `#3B82F6` | |

> **Scope note (from skill):** dark mode text must meet 4.5:1 (primary) / 3:1 (secondary); borders/states must stay visible in both modes; modal scrim 40–60% black. Verify each — don't assume light values port.

### Semantic change-type colors (diff vocabulary) — *additive, never alone*

These are **functional** colors for the diff engine. Each is always paired with an icon + label.

| Change type | Color | Hex (light) | Icon (Lucide) | Label (vi) |
|---|---|---|---|---|
| `addition` | green | `#059669` emerald-600 | `Plus` | "Thêm mới" |
| `deletion` | red | `#DC2626` red-600 (→ `#F87171` dark) | `Minus` | "Bị xóa" |
| `numerical` | blue | `#2563EB` blue-600 | `Hash` | "Đổi số liệu" |
| `terminology` | amber | `#D97706` amber-600 | `Type` | "Đổi thuật ngữ" |
| `structural` | violet | `#7C3AED` violet-600 | `GitBranch` | "Đổi cấu trúc" |
| `reorder` | slate | `#475569` slate-600 | `ArrowUpDown` | "Đổi vị trí" |

### Severity (risk) colors

| Severity | Color | Use |
|---|---|---|
| Critical / risk-flag | `destructive` red + `AlertTriangle` | surfaced in executive summary + on the ACU |
| Warning / low-confidence | amber + `AlertTriangle` | `confidence < 0.6` |
| Verified / passed | emerald + `ShieldCheck` | evidence-verified ACU |

---

## 3. Typography

From the skill: **EB Garamond** (headings) + **Lato** (body) — "legal, professional, traditional, trustworthy, formal, authoritative." We add a **monospace** face for evidence values and numbers.

```css
/* index.html or globals.css — self-host for offline/on-prem (no Google Fonts CDN at runtime) */
@font-face { /* EB Garamond, Lato, JetBrains Mono — bundled locally */ }
```

| Role | Font | Weights | Size scale |
|---|---|---|---|
| Display / H1 | EB Garamond | 600 | 32px |
| H2 section | EB Garamond | 600 | 24px |
| H3 | EB Garamond | 500 | 20px |
| Body | Lato | 400 | 16px |
| Body small / label | Lato | 400/500 | 14px |
| Caption / meta | Lato | 400 | 12–13px |
| **Evidence value / numbers / locations** | **JetBrains Mono** | 400/500 | 14–15px, tabular-nums |

**Type rules (skill §6):**
- Base body **16px** on mobile (prevents iOS auto-zoom); line-height **1.5–1.6**.
- Line length **65–75 chars** in the report reading column.
- Weight hierarchy: headings 600, body 400, labels 500.
- **Tabular figures** (`font-variant-numeric: tabular-nums`) on all numbers — amounts, dates, counts, progress % — to prevent layout shift.
- Use semantic type roles, not raw sizes, in components (`<h1>`, `text-label`, etc.).

**Type scale (Tailwind `fontSize` extension):** `12 14 16 18 20 24 32` — consistent rhythm.

---

## 4. Spacing, Layout & Elevation

- **Spacing scale:** 4px base (`4 8 12 16 24 32 48 64`). 8dp rhythm everywhere (skill §5).
- **Breakpoints (mobile-first):** `sm 640 · md 768 · lg 1024 · xl 1280 · 2xl 1440`. Optimize for **lg/xl (1280–1440)** = the lawyer's laptop.
- **Container:** `max-w-7xl` (1280px) centered with comfortable gutters; report reading column `max-w-4xl`.
- **Radii:** `sm 6 · md 8 · lg 12 · xl 16`. Cards `lg`. Consistent — no random values.
- **Elevation (consistent shadow scale, skill §4):**
  - `shadow-sm` — resting cards
  - `shadow-md` — hover / raised
  - `shadow-lg` — dropdowns, popovers
  - `shadow-xl` — modals/dialogs
- **z-index scale (skill §5):** `base 0 · dropdown 10 · sticky 20 · drawer 40 · modal 50 · toast 60 · overlay 100`.
- **Touch targets ≥44×44px** even on desktop-rich UIs (skill §2); `hitSlop`-equivalent via padding on small icon buttons.

---

## 5. Component Foundations (shadcn/ui mapping)

We adopt shadcn/ui (ownable, Radix-based) and theme it to the tokens above. **Do not** mix in a second component library.

| Component | Notes specific to LegalDiff |
|---|---|
| `Button` | One **primary** CTA per screen (skill §4). Variants: `primary` (accent), `secondary`, `outline`, `ghost`, `destructive`. Loading state disables + spinner (skill §2 `loading-buttons`). |
| `Card` | Report sections, ACUs, document cards. `shadow-sm` resting, hover lift. |
| `Table` | Document library, jobs list, ACU list (sortable, `aria-sort`, skill §10). Virtualize when >50 rows. |
| `Dialog` / `Sheet` | Confirm destructive actions, export options, ACU detail drill-down. Animate from trigger (skill §7 `modal-motion`). |
| `Form` (RHF + Zod) | Login, register, job options. Visible labels, inline errors below fields, `role="alert"` (skill §8). |
| `Badge` / `ChangeTypeBadge` | Change type + severity. Always icon+label+color. |
| `Alert` | Inline error/success with icon + recovery action (skill §8 `error-recovery`). |
| `Skeleton` | For any async >300ms (skill §3/§7). Report & library loading. |
| `Toast` (`sonner`) | Auto-dismiss 4s, `aria-live=polite`, never steal focus (skill §8). |
| `Progress` + custom `PhaseStepper` | Job progress with phase labels + % (skill §8 `multi-step-progress`). |
| `Tabs` | Report: Summary / Changes / Raw. Active state visible (skill §9 `nav-state-active`). |
| `Tooltip` | Truncated locations, icon meanings. Keyboard-reachable (skill §10 `tooltip-keyboard`). |
| `Command` (cmdk) | Quick search across documents/jobs (optional, later phase). |

**Icons:** Lucide only. **Consistent stroke width (1.5px)**, one icon set, no filled/outline mixing at the same hierarchy (skill §4). Never emoji as structural icons.

---

## 6. The Evidence Rendering Pattern (signature UI)

This is the component pattern that defines the product. It must be executed precisely.

```
┌─ ACU · [Đổi số liệu] · Điều 5, Khoản 1 ───────  ✓ Đã xác minh ─┐
│                                                                  │
│  ┌── V1 (bản gốc) ──────────┐  ┌── V2 (bản sửa đổi) ─────────┐ │
│  │ …phải thanh toán »»»500.000│  │ …phải thanh toán »»»600.000│ │
│  │ đồng««« trong vòng 30 ngày │  │ đồng««« trong vòng 45 ngày │ │
│  │ 500.000 đồng               │  │ 600.000 đồng               │ │
│  └────────────────────────────┘  └────────────────────────────┘ │
│  Lý do: Số tiền và thời hạn thanh toán thay đổi.                │
└──────────────────────────────────────────────────────────────────┘
```

Rules:
- **Side-by-side** on `lg+`, stacked on mobile.
- The changed token (`original_value` / `new_value`) is **highlighted inside the verbatim quote** (e.g. mark element), not just repeated below. Both highlight + icon + label convey the change.
- Quotes are **monospace**, `whitespace-pre-wrap`, **never truncated** (skill §6 `truncation-strategy` — prefer wrap; if must truncate, tooltip with full text). This is legal evidence — truncation is a liability.
- Exact source location (`location_v1`/`location_v2`) always visible above each quote.
- `confidence` shown as a subtle verified/low-trust icon, not a scary gauge.

---

## 7. Anti-Patterns to Avoid (from skill §4 + product rules)

- ❌ **AI purple/pink gradients**, neon glow, "cyberpunk" anything. (Skill explicitly flags for legal/enterprise.)
- ❌ **Emoji as icons** (🎨 ⚖️ ⚙️). Use Lucide SVG.
- ❌ **Color-only meaning** (a red row with no label/icon).
- ❌ **Placeholder-only labels** in forms; **errors only at top** of a form.
- ❌ **Instant state changes (0ms)** or **>500ms** animations; animating `width/height/top/left`.
- ❌ **Truncating evidence quotes.**
- ❌ **Disabling zoom**, removing focus rings, `text < 12px`, gray-on-gray.
- ❌ **Mixing flat + skeuomorphic** styles; random shadow/radius values.
- ❌ **More than one primary CTA** per screen.

---

## 8. Pre-Delivery Checklist (run before any screen ships)

- [ ] No emoji icons; all icons Lucide, 1.5px stroke, one family.
- [ ] `cursor-pointer` on every clickable; hover states 150–300ms.
- [ ] Light **and** dark mode contrast verified independently (4.5:1 body, 3:1 secondary).
- [ ] Visible focus rings; keyboard order matches visual order.
- [ ] `prefers-reduced-motion` respected.
- [ ] Responsive checked at 375 / 768 / 1024 / 1440.
- [ ] Loading skeleton (>300ms), empty state, error state all present.
- [ ] Touch targets ≥44px; nothing hidden behind fixed bars.
- [ ] Every change type/severity has icon + label + color.
- [ ] Evidence quotes verbatim, monospace, wrap (not truncate).

---

## 9. Token Implementation (Tailwind + CSS vars)

`tailwind.config.ts` maps the CSS variables so dark mode is a single `class="dark"` toggle (skill §6 `color-semantic` / `token-driven-theming`):

```ts
// tailwind.config.ts (excerpt)
export default {
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        border: 'hsl(var(--border))',
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        primary: { DEFAULT: 'hsl(var(--primary))', foreground: 'hsl(var(--primary-foreground))' },
        accent:  { DEFAULT: 'hsl(var(--accent))',  foreground: 'hsl(var(--accent-foreground))' },
        muted:   { DEFAULT: 'hsl(var(--muted))',   foreground: 'hsl(var(--muted-foreground))' },
        destructive: { DEFAULT: 'hsl(var(--destructive))' },
        ring: 'hsl(var(--ring))',
      },
      fontFamily: {
        sans: ['Lato', 'system-ui', 'sans-serif'],
        serif: ['"EB Garamond"', 'Georgia', 'serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      borderRadius: { sm: '6px', md: '8px', lg: '12px', xl: '16px' },
      fontSize: { /* 12 14 16 18 20 24 32 */ },
    },
  },
}
```

> **Note:** tokens are stored as HSL channels in `globals.css` (`--primary: 215 28% 17%;` for slate-800) to allow Tailwind opacity modifiers (`bg-primary/80`). Convert the hex values above to HSL at implementation. This file is the design source of truth — keep it versioned.
