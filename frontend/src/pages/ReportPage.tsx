import { useState, useMemo, useCallback } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { getReport } from '@/api/report'
import { getJob } from '@/api/jobs'
import { useWebSocket } from '@/hooks/useWebSocket'
import { PhaseStepper } from '@/components/progress/PhaseStepper'
import { QueueIndicator } from '@/components/progress/QueueIndicator'
import { ExecutiveSummary } from '@/components/report/ExecutiveSummary'
import { AcuList } from '@/components/report/AcuList'
import { FilterBar } from '@/components/report/FilterBar'
import type { AcuType, Severity } from '@/types/diff'
import type { JobProgress } from '@/types/job'

export function ReportPage() {
  const { id } = useParams<{ id: string }>()
  const [wsProgress, setWsProgress] = useState<JobProgress | null>(null)
  const [typeFilter, setTypeFilter] = useState<AcuType | null>(null)
  const [severityFilter, setSeverityFilter] = useState<Severity | null>(null)
  const [sortBy, setSortBy] = useState<'type' | 'severity'>('type')

  useWebSocket(id ?? null, useCallback((data: JobProgress) => {
    setWsProgress(data)
  }, []))

  const { data: job, isLoading: jobLoading } = useQuery({
    queryKey: ['job', id],
    queryFn: () => getJob(id!),
    enabled: !!id,
  })

  const { data: report, isLoading } = useQuery({
    queryKey: ['report', id],
    queryFn: () => getReport(id!),
    enabled: !!id && job?.status === 'completed',
  })

  const filteredAcus = useMemo(() => {
    if (!report) return []
    let result = [...report.acus]

    if (typeFilter) result = result.filter((a) => a.type === typeFilter)
    if (severityFilter) result = result.filter((a) => a.severity === severityFilter)

    result.sort((a, b) => {
      if (sortBy === 'type') return a.type.localeCompare(b.type)
      const order = { high: 0, medium: 1, low: 2 }
      return (order[a.severity] ?? 0) - (order[b.severity] ?? 0)
    })

    return result
  }, [report, typeFilter, severityFilter, sortBy])

  if (!id) return null

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <h1 className="font-serif text-2xl font-semibold">
        {job?.name || 'Báo cáo so sánh'}
      </h1>

      {/* Progress section */}
      {wsProgress && wsProgress.phase !== 'completed' && wsProgress.phase !== 'failed' && (
        <div className="mt-6 space-y-4">
          <PhaseStepper progress={wsProgress} />
          <QueueIndicator
            position={wsProgress.gpu_queue_position}
            estimatedWaitMinutes={wsProgress.estimated_wait_minutes}
          />
        </div>
      )}

      {/* Report section */}
      {(report || isLoading || jobLoading) && (
        <div className="mt-8 space-y-6">
          {report && <ExecutiveSummary summary={report.summary} />}

          {report && (
            <div className="rounded-lg border bg-white p-4">
              <h2 className="mb-4 font-serif text-lg font-medium">Thay đổi ({filteredAcus.length})</h2>
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
          )}

          {(isLoading || jobLoading) && (
            <div className="py-12 text-center">
              <p className="text-sm text-muted-foreground">Đang tải báo cáo...</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
