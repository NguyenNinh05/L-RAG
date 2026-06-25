import { useMemo, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { getEvalPair } from '@/api/eval'
import { AcuList } from '@/components/report/AcuList'
import { FilterBar } from '@/components/report/FilterBar'
import type { AcuType, Severity } from '@/types/diff'

const pct = (v: number) => `${(v * 100).toFixed(1)}%`

function PhaseCard({ title, stats }: { title: string; stats: Record<string, number> }) {
  const entries = Object.entries(stats)
  return (
    <div className="rounded-lg border bg-white p-5">
      <h3 className="font-serif text-base font-medium">{title}</h3>
      <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
        {entries.map(([k, v]) => (
          <div key={k} className="flex justify-between">
            <dt className="text-muted-foreground">{k}</dt>
            <dd className="tabular-nums">{typeof v === 'number' ? (v < 10 ? v : v.toFixed(v > 100 ? 0 : 2)) : v}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}

export function EvalPairDetailPage() {
  const { slug = '', pairName = '' } = useParams<{ slug: string; pairName: string }>()
  const { t } = useTranslation()
  const [typeFilter, setTypeFilter] = useState<AcuType | null>(null)
  const [severityFilter, setSeverityFilter] = useState<Severity | null>(null)
  const [sortBy, setSortBy] = useState<'type' | 'severity'>('type')

  const { data: pair, isLoading } = useQuery({
    queryKey: ['eval-pair', slug, pairName],
    queryFn: () => getEvalPair(slug, pairName),
    enabled: !!slug && !!pairName,
  })

  const filteredAcus = useMemo(() => {
    if (!pair) return []
    let result = [...pair.acus]
    if (typeFilter) result = result.filter((a) => a.type === typeFilter)
    if (severityFilter) result = result.filter((a) => a.severity === severityFilter)
    result.sort((a, b) => {
      if (sortBy === 'type') return a.type.localeCompare(b.type)
      const order = { high: 0, medium: 1, low: 2 }
      return (order[a.severity] ?? 0) - (order[b.severity] ?? 0)
    })
    return result
  }, [pair, typeFilter, severityFilter, sortBy])

  if (isLoading) {
    return (
      <div className="mx-auto max-w-5xl px-4 py-8">
        <p className="text-sm text-muted-foreground">{t('common.loading')}</p>
      </div>
    )
  }

  if (!pair) return null

  const cd = pair.change_detection

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <Link to={`/eval/${slug}`} className="text-xs font-medium text-accent hover:underline">
        {t('eval.backToRun')}
      </Link>
      <h1 className="mt-2 font-mono text-xl font-semibold">{pair.pair_name}</h1>
      <p className="mt-1 text-xs text-muted-foreground">
        {slug} · {pair.acus.length} {t('eval.acus')} · {pair.total_time.toFixed(0)}s
      </p>

      {/* Phase stats */}
      <div className="mt-6 grid gap-4 sm:grid-cols-3">
        <PhaseCard title={t('eval.phase1')} stats={pair.phase1} />
        <PhaseCard title={t('eval.phase2')} stats={pair.phase2} />
        <PhaseCard title={t('eval.phase3')} stats={pair.phase3} />
      </div>

      {/* Change detection metrics */}
      <div className="mt-6 rounded-lg border bg-white p-5">
        <h2 className="font-serif text-lg font-medium">{t('eval.changeDetection')}</h2>
        <div className="mt-3 grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-4">
          <Metric label="Recall" value={pct(cd.recall)} />
          <Metric label="Precision" value={pct(cd.precision)} />
          <Metric label="Recall (semantic)" value={pct(cd.recall_semantic)} />
          <Metric label="Recall (article)" value={pct(cd.recall_article)} />
          <Metric label={t('eval.gtDetected')} value={`${cd.gt_detected}`} />
          <Metric label={t('eval.pipelineTotal')} value={`${cd.pipeline_total}`} />
          <Metric label={t('eval.pipelineMatched')} value={`${cd.pipeline_matched}`} />
          <Metric label={t('eval.gtChanges')} value={`${(pair.ground_truth as { total_changes?: number }).total_changes ?? '-'}`} />
        </div>
      </div>

      {/* ACU list (reused) */}
      <div className="mt-6 rounded-lg border bg-white p-4">
        <h2 className="mb-4 font-serif text-lg font-medium">
          {t('eval.acuList')} ({filteredAcus.length})
        </h2>
        <FilterBar
          selectedType={typeFilter}
          onTypeChange={setTypeFilter}
          selectedSeverity={severityFilter}
          onSeverityChange={setSeverityFilter}
          sortBy={sortBy}
          onSortChange={setSortBy}
        />
        <div className="mt-4">
          <AcuList acus={filteredAcus} />
        </div>
      </div>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="font-serif text-lg font-semibold tabular-nums">{value}</p>
    </div>
  )
}
