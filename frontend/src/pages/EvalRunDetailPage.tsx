import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { getEvalRun, listEvalPairs } from '@/api/eval'
import type { CategoryStat } from '@/types/eval'

const pct = (v: number) => `${(v * 100).toFixed(1)}%`

function KpiCard({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="rounded-lg border bg-white p-5">
      <p className="text-xs font-medium uppercase text-muted-foreground">{label}</p>
      <p className={`mt-2 font-serif text-3xl font-semibold tabular-nums ${accent ?? ''}`}>{value}</p>
    </div>
  )
}

export function EvalRunDetailPage() {
  const { slug = '' } = useParams<{ slug: string }>()
  const { t } = useTranslation()

  const { data: run } = useQuery({ queryKey: ['eval-run', slug], queryFn: () => getEvalRun(slug), enabled: !!slug })
  const { data: pairs, isLoading } = useQuery({
    queryKey: ['eval-pairs', slug],
    queryFn: () => listEvalPairs(slug),
    enabled: !!slug,
  })

  const agg = (run?.aggregate ?? {}) as Record<string, unknown>
  const num = (k: string, d = 0) => Number(agg[k] ?? d)
  const perCategory = (agg.per_category_semantic ?? {}) as Record<string, CategoryStat>
  const hallucFlagged = (agg.halluc_flagged_pairs ?? []) as Array<{
    pair: string
    hallucination_rate: number
    rejected: number
    total: number
  }>

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <Link to="/eval" className="text-xs font-medium text-accent hover:underline">
        {t('eval.backToEval')}
      </Link>
      <h1 className="mt-2 font-serif text-2xl font-semibold">
        {run?.label || slug}
        <span className="ml-2 align-middle font-sans text-sm font-normal text-muted-foreground">
          {slug}
        </span>
      </h1>

      {/* Aggregate KPIs */}
      {run && (
        <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <KpiCard label="F1" value={pct(num('change_f1'))} accent="text-diff-addition" />
          <KpiCard label="F1 (semantic)" value={pct(num('change_f1_semantic'))} />
          <KpiCard label="Hallucination" value={pct(num('hallucination_rate'))} accent="text-diff-deletion" />
          <KpiCard label="Avg time / pair" value={`${num('avg_time_per_pair_seconds').toFixed(0)}s`} />
        </div>
      )}

      {/* Per-category recall (semantic) */}
      {run && Object.keys(perCategory).length > 0 && (
        <div className="mt-6 rounded-lg border bg-white p-5">
          <h2 className="font-serif text-lg font-medium">{t('eval.perCategory')}</h2>
          <table className="mt-3 w-full text-sm">
            <thead className="border-b text-left text-muted-foreground">
              <tr>
                <th className="py-2 pr-4 font-medium">Category</th>
                <th className="py-2 pr-4 font-medium">GT</th>
                <th className="py-2 pr-4 font-medium">Detected</th>
                <th className="py-2 font-medium">Recall</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {Object.entries(perCategory)
                .sort(([a], [b]) => a.localeCompare(b))
                .map(([cat, st]) => (
                  <tr key={cat}>
                    <td className="py-2 pr-4 font-mono">{cat}</td>
                    <td className="py-2 pr-4 tabular-nums">{st.gt_total}</td>
                    <td className="py-2 pr-4 tabular-nums">{st.detected}</td>
                    <td className="py-2 tabular-nums">{pct(st.recall ?? 0)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Hallucination-flagged pairs */}
      {run && hallucFlagged.length > 0 && (
        <div className="mt-6 rounded-lg border border-diff-deletion/30 bg-diff-deletion/5 p-5">
          <h2 className="font-serif text-lg font-medium text-diff-deletion">
            {t('eval.hallucFlagged')}
          </h2>
          <ul className="mt-2 space-y-1 text-sm">
            {hallucFlagged.map((h) => (
              <li key={h.pair}>
                <Link to={`/eval/${slug}/${h.pair}`} className="font-mono text-accent hover:underline">
                  {h.pair}
                </Link>{' '}
                — {pct(h.hallucination_rate)} ({h.rejected}/{h.total} {t('eval.rejected')})
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Pair table */}
      <h2 className="mt-8 font-serif text-lg font-medium">{t('eval.pairList')}</h2>
      <div className="mt-3 overflow-hidden rounded-lg border bg-white">
        <table className="w-full text-sm">
          <thead className="border-b bg-muted/50">
            <tr className="text-left text-muted-foreground">
              <th className="px-4 py-3 font-medium">{t('eval.pair')}</th>
              <th className="px-4 py-3 font-medium">Recall</th>
              <th className="px-4 py-3 font-medium">F1</th>
              <th className="px-4 py-3 font-medium">Halluc</th>
              <th className="px-4 py-3 font-medium">ACUs</th>
              <th className="px-4 py-3 font-medium">{t('eval.time')}</th>
            </tr>
          </thead>
          <tbody className="divide-y">
            {(pairs ?? []).map((p) => (
              <tr key={p.pair_name} className="hover:bg-muted/30">
                <td className="px-4 py-2.5">
                  <Link to={`/eval/${slug}/${p.pair_name}`} className="font-mono text-accent hover:underline">
                    {p.pair_name}
                  </Link>
                </td>
                <td className="px-4 py-2.5 tabular-nums">{pct(p.recall)}</td>
                <td className="px-4 py-2.5 tabular-nums">{pct(p.f1)}</td>
                <td className="px-4 py-2.5 tabular-nums">{pct(p.hallucination_rate)}</td>
                <td className="px-4 py-2.5 tabular-nums">{p.num_acus}</td>
                <td className="px-4 py-2.5 tabular-nums">{p.total_time.toFixed(0)}s</td>
              </tr>
            ))}
            {isLoading && (
              <tr>
                <td colSpan={6} className="px-4 py-8 text-center text-muted-foreground">
                  {t('common.loading')}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
