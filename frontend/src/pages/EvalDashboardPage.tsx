import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { listEvalRuns } from '@/api/eval'
import type { EvalRunSummary } from '@/types/eval'

const RUN_COLORS = ['bg-accent', 'bg-diff-addition', 'bg-diff-terminology', 'bg-diff-numerical']

type Better = 'high' | 'low' | null
interface MetricDef {
  key: keyof EvalRunSummary
  label: string
  better: Better
  fmt: (v: number) => string
}

const pct = (v: number) => `${(v * 100).toFixed(1)}%`
const secs = (v: number) => `${v.toFixed(0)}s`
const int = (v: number) => `${Math.round(v)}`

const METRICS: MetricDef[] = [
  { key: 'change_f1', label: 'F1', better: 'high', fmt: pct },
  { key: 'change_f1_semantic', label: 'F1 (semantic)', better: 'high', fmt: pct },
  { key: 'change_recall', label: 'Recall', better: 'high', fmt: pct },
  { key: 'change_precision', label: 'Precision', better: 'high', fmt: pct },
  { key: 'hallucination_rate', label: 'Hallucination', better: 'low', fmt: pct },
  { key: 'avg_time_per_pair_seconds', label: 'Avg time / pair', better: 'low', fmt: secs },
  { key: 'total_acus_passed', label: 'ACUs passed', better: 'high', fmt: int },
  { key: 'num_pairs', label: 'Pairs', better: null, fmt: int },
]

function bestValue(runs: EvalRunSummary[], key: keyof EvalRunSummary, better: Better) {
  if (!better || runs.length === 0) return null
  const vals = runs.map((r) => Number(r[key]))
  return better === 'high' ? Math.max(...vals) : Math.min(...vals)
}

export function EvalDashboardPage() {
  const { t } = useTranslation()
  const { data: runs, isLoading } = useQuery({ queryKey: ['eval-runs'], queryFn: listEvalRuns })

  if (isLoading) {
    return (
      <div className="mx-auto max-w-6xl px-4 py-8">
        <p className="text-sm text-muted-foreground">{t('common.loading')}</p>
      </div>
    )
  }

  if (!runs || runs.length === 0) {
    return (
      <div className="mx-auto max-w-6xl px-4 py-8">
        <h1 className="font-serif text-2xl font-semibold">{t('eval.title')}</h1>
        <p className="mt-2 text-sm text-muted-foreground">{t('eval.noRuns')}</p>
      </div>
    )
  }

  // Union of categories across runs (sorted).
  const categories = Array.from(
    new Set(runs.flatMap((r) => Object.keys(r.per_category_semantic || {}))),
  ).sort()

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <h1 className="font-serif text-2xl font-semibold">{t('eval.title')}</h1>
      <p className="mt-1 text-sm text-muted-foreground">{t('eval.subtitle')}</p>

      {/* Model cards */}
      <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {runs.map((r, i) => (
          <Link
            key={r.slug}
            to={`/eval/${r.slug}`}
            className="rounded-lg border bg-white p-5 transition-colors hover:border-accent"
          >
            <div className="flex items-center gap-2">
              <span
                className={`inline-block h-2.5 w-2.5 rounded-full ${RUN_COLORS[i % RUN_COLORS.length]}`}
              />
              <p className="font-serif text-base font-medium">{r.label || r.slug}</p>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">{r.num_pairs} cặp tài liệu</p>
            <div className="mt-3 flex items-end justify-between">
              <div>
                <p className="text-xs uppercase text-muted-foreground">Thay đổi phát hiện</p>
                <p className="font-serif text-2xl font-semibold tabular-nums">{r.total_acus_passed}</p>
              </div>
              <div className="text-right">
                <p className="text-xs uppercase text-muted-foreground">F1</p>
                <p className="font-serif text-2xl font-semibold tabular-nums text-diff-addition">
                  {pct(r.change_f1)}
                </p>
              </div>
            </div>
          </Link>
        ))}
      </div>

      {/* Run comparison table */}
      <h2 className="mt-8 font-serif text-lg font-medium">{t('eval.comparison')}</h2>
      <div className="mt-3 overflow-hidden rounded-lg border bg-white">
        <table className="w-full text-sm">
          <thead className="border-b bg-muted/50">
            <tr>
              <th className="px-4 py-3 text-left font-medium text-muted-foreground"></th>
              {runs.map((r, i) => (
                <th key={r.slug} className="px-4 py-3 text-left font-medium">
                  <Link to={`/eval/${r.slug}`} className="hover:underline">
                    <span className="flex items-center gap-2">
                      <span className={`inline-block h-2.5 w-2.5 rounded-full ${RUN_COLORS[i % RUN_COLORS.length]}`} />
                      {r.label || r.slug}
                    </span>
                  </Link>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y">
            {METRICS.map((m) => {
              const best = bestValue(runs, m.key, m.better)
              return (
                <tr key={m.key} className="hover:bg-muted/30">
                  <td className="px-4 py-2.5 font-medium text-muted-foreground">{m.label}</td>
                  {runs.map((r) => {
                    const v = Number(r[m.key])
                    const isBest = best !== null && v === best
                    return (
                      <td
                        key={r.slug}
                        className={`px-4 py-2.5 tabular-nums ${
                          isBest ? 'font-semibold text-diff-addition' : ''
                        }`}
                      >
                        {m.fmt(v)}
                      </td>
                    )
                  })}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Per-category recall (semantic) */}
      <h2 className="mt-8 font-serif text-lg font-medium">{t('eval.perCategory')}</h2>
      <p className="mt-1 text-xs text-muted-foreground">{t('eval.perCategoryHint')}</p>
      <div className="mt-4 space-y-4 rounded-lg border bg-white p-6">
        {categories.map((cat) => (
          <div key={cat} className="grid grid-cols-[3rem_1fr] items-center gap-3">
            <span className="font-mono text-sm font-medium">{cat}</span>
            <div className="space-y-1.5">
              {runs.map((r, i) => {
                const stat = r.per_category_semantic?.[cat]
                const recall = stat?.recall ?? 0
                return (
                  <div key={r.slug} className="flex items-center gap-2">
                    <div className="h-4 flex-1 overflow-hidden rounded bg-muted">
                      <div
                        className={`h-full ${RUN_COLORS[i % RUN_COLORS.length]}`}
                        style={{ width: `${Math.min(recall * 100, 100)}%` }}
                      />
                    </div>
                    <span className="w-14 text-right text-xs tabular-nums text-muted-foreground">
                      {pct(recall)}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        ))}
        <div className="mt-2 flex flex-wrap gap-4 border-t pt-3 text-xs text-muted-foreground">
          {runs.map((r, i) => (
            <span key={r.slug} className="flex items-center gap-1.5">
              <span className={`inline-block h-2.5 w-2.5 rounded-full ${RUN_COLORS[i % RUN_COLORS.length]}`} />
              {r.label || r.slug}
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}
