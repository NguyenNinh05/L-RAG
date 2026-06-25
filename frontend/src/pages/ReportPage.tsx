import { useState, useMemo, useCallback } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { getReport } from '@/api/report'
import { getJob } from '@/api/jobs'
import { useWebSocket } from '@/hooks/useWebSocket'
import { PhaseStepper } from '@/components/progress/PhaseStepper'
import { ExecutiveSummary } from '@/components/report/ExecutiveSummary'
import { AcuList } from '@/components/report/AcuList'
import { FilterBar } from '@/components/report/FilterBar'
import type { AcuType, Severity } from '@/types/diff'
import type { WSProgressMessage } from '@/types/job'

export function ReportPage() {
  const { id } = useParams<{ id: string }>()
  const queryClient = useQueryClient()
  const [wsProgress, setWsProgress] = useState<WSProgressMessage | null>(null)
  const [typeFilter, setTypeFilter] = useState<AcuType | null>(null)
  const [severityFilter, setSeverityFilter] = useState<Severity | null>(null)
  const [sortBy, setSortBy] = useState<'type' | 'severity'>('type')

  useWebSocket(
    id ?? null,
    useCallback(
      (data: WSProgressMessage) => {
        setWsProgress(data)
        // When the pipeline finishes (or fails), refresh the job so its status
        // flips to terminal — that status is what gates the report query below.
        if (data.event === 'completed' || data.event === 'error') {
          queryClient.invalidateQueries({ queryKey: ['job', id] })
        }
      },
      [queryClient, id],
    ),
  )

  const { data: job, isLoading: jobLoading } = useQuery({
    queryKey: ['job', id],
    queryFn: () => getJob(id!),
    enabled: !!id,
    // Poll until the job reaches a terminal state. The WS message is the fast
    // path; this is the reliable fallback when a socket frame is missed.
    refetchInterval: (query) => {
      const status = query.state.data?.status
      if (status === 'completed' || status === 'failed' || status === 'cancelled') {
        return false
      }
      return 2000
    },
  })

  const { data: report, isLoading, isError } = useQuery({
    queryKey: ['report', id],
    queryFn: () => getReport(id!),
    // Load the report as soon as either signal says the pipeline is done.
    enabled: !!id && (job?.status === 'completed' || wsProgress?.event === 'completed'),
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

  const completed = job?.status === 'completed' || wsProgress?.event === 'completed'
  const failed =
    job?.status === 'failed' ||
    job?.status === 'cancelled' ||
    wsProgress?.event === 'error'
  const running = !completed && !failed

  if (!id) return null

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <h1 className="font-serif text-2xl font-semibold">
        {job
          ? job.v1_filename && job.v2_filename
            ? `${job.v1_filename} ↔ ${job.v2_filename}`
            : 'Báo cáo so sánh'
          : 'Báo cáo so sánh'}
      </h1>

      {/* Initial load — before we know the job's status */}
      {jobLoading && !wsProgress && (
        <div className="mt-8 py-12 text-center">
          <p className="text-sm text-muted-foreground">Đang tải...</p>
        </div>
      )}

      {/* Progress section — shown while the pipeline is running */}
      {running && wsProgress && (
        <div className="mt-6 space-y-4">
          <PhaseStepper message={wsProgress} />
        </div>
      )}

      {/* Error section */}
      {failed && (
        <div className="mt-6 space-y-4">
          {wsProgress ? (
            <PhaseStepper message={wsProgress} />
          ) : (
            <div className="rounded-md bg-destructive/10 px-4 py-3 text-sm text-destructive">
              {job?.error_message || 'Phiên so sánh đã thất bại. Vui lòng thử lại.'}
            </div>
          )}
        </div>
      )}

      {/* Report section — shown as soon as the pipeline completes */}
      {completed && (
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

          {isLoading && (
            <div className="py-12 text-center">
              <p className="text-sm text-muted-foreground">Đang tải báo cáo...</p>
            </div>
          )}

          {isError && (
            <div className="rounded-md bg-destructive/10 px-4 py-3 text-sm text-destructive">
              Không tải được báo cáo.{' '}
              <button
                type="button"
                onClick={() => queryClient.invalidateQueries({ queryKey: ['report', id] })}
                className="font-medium underline"
              >
                Thử lại
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
