import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { listJobs } from '@/api/jobs'
import { listEvalRuns, listEvalPairs } from '@/api/eval'

const pct = (v: number) => `${(v * 100).toFixed(1)}%`

export function DashboardPage() {
  const { data: jobs, isLoading } = useQuery({
    queryKey: ['jobs'],
    queryFn: listJobs,
  })

  const { data: evalRuns } = useQuery({ queryKey: ['eval-runs'], queryFn: listEvalRuns })
  const [slug, setSlug] = useState<string>('v3')
  const activeSlug = evalRuns?.some((r) => r.slug === slug) ? slug : (evalRuns?.[0]?.slug ?? 'v3')
  const { data: pairs } = useQuery({
    queryKey: ['eval-pairs', activeSlug],
    queryFn: () => listEvalPairs(activeSlug),
    enabled: !!activeSlug,
  })

  const kpis = {
    totalJobs: jobs?.length ?? 0,
    completed: jobs?.filter((j) => j.status === 'completed').length ?? 0,
    inProgress: jobs?.filter((j) => j.status === 'processing').length ?? 0,
    failed: jobs?.filter((j) => j.status === 'failed').length ?? 0,
  }

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <h1 className="font-serif text-2xl font-semibold">Tổng quan</h1>

      {/* KPI Cards */}
      <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-lg border bg-white p-5">
          <p className="text-xs font-medium text-muted-foreground uppercase">Tổng phiên</p>
          <p className="mt-2 font-serif text-3xl font-semibold tabular-nums">
            {isLoading ? '-' : kpis.totalJobs}
          </p>
        </div>
        <div className="rounded-lg border bg-white p-5">
          <p className="text-xs font-medium text-muted-foreground uppercase">Hoàn tất</p>
          <p className="mt-2 font-serif text-3xl font-semibold text-diff-addition tabular-nums">
            {isLoading ? '-' : kpis.completed}
          </p>
        </div>
        <div className="rounded-lg border bg-white p-5">
          <p className="text-xs font-medium text-muted-foreground uppercase">Đang xử lý</p>
          <p className="mt-2 font-serif text-3xl font-semibold text-accent tabular-nums">
            {isLoading ? '-' : kpis.inProgress}
          </p>
        </div>
        <div className="rounded-lg border bg-white p-5">
          <p className="text-xs font-medium text-muted-foreground uppercase">Thất bại</p>
          <p className="mt-2 font-serif text-3xl font-semibold text-diff-deletion tabular-nums">
            {isLoading ? '-' : kpis.failed}
          </p>
        </div>
      </div>

      {/* Per-pair detailed reports (from evaluation data) */}
      {evalRuns && evalRuns.length > 0 && (
        <div className="mt-8">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="font-serif text-lg font-medium">Báo cáo chi tiết theo cặp</h2>
              <p className="mt-1 text-sm text-muted-foreground">
                {pairs?.length ?? 0} cặp · nhấn vào một cặp để xem báo cáo đầy đủ.
              </p>
            </div>
            <select
              value={activeSlug}
              onChange={(e) => setSlug(e.target.value)}
              className="rounded-md border bg-white px-3 py-1.5 text-sm focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent"
            >
              {evalRuns.map((r) => (
                <option key={r.slug} value={r.slug}>
                  {r.label || r.slug}
                </option>
              ))}
            </select>
          </div>

          <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {(pairs ?? []).map((p) => (
              <Link
                key={p.pair_name}
                to={`/eval/${activeSlug}/${p.pair_name}`}
                className="rounded-lg border bg-white p-4 transition-colors hover:border-accent"
              >
                <p className="font-mono text-sm font-semibold">{p.pair_name}</p>
                <div className="mt-3 grid grid-cols-2 gap-y-1.5 text-xs">
                  <span className="text-muted-foreground">Recall</span>
                  <span className="text-right tabular-nums">{pct(p.recall)}</span>
                  <span className="text-muted-foreground">F1</span>
                  <span className="text-right font-medium tabular-nums text-diff-addition">
                    {pct(p.f1)}
                  </span>
                  <span className="text-muted-foreground">Halluc</span>
                  <span className="text-right tabular-nums">{pct(p.hallucination_rate)}</span>
                  <span className="text-muted-foreground">ACUs</span>
                  <span className="text-right tabular-nums">{p.num_acus}</span>
                </div>
              </Link>
            ))}
            {(pairs?.length ?? 0) === 0 && (
              <p className="col-span-full py-6 text-center text-sm text-muted-foreground">
                Không có cặp nào.
              </p>
            )}
          </div>
        </div>
      )}

      {/* Recent jobs table */}
      <div className="mt-8">
        <h2 className="font-serif text-lg font-medium">Phiên gần đây</h2>
        <div className="mt-3 overflow-hidden rounded-lg border bg-white">
          <table className="w-full text-sm">
            <thead className="border-b bg-muted/50">
              <tr>
                <th className="px-4 py-3 text-left font-medium text-muted-foreground">Tên</th>
                <th className="px-4 py-3 text-left font-medium text-muted-foreground">Trạng thái</th>
                <th className="px-4 py-3 text-left font-medium text-muted-foreground">Ngày tạo</th>
                <th className="px-4 py-3 text-right font-medium text-muted-foreground"></th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {jobs?.slice(0, 10).map((job) => (
                <tr key={job.id} className="hover:bg-muted/30">
                  <td className="px-4 py-3 font-medium">
                    {job.v1_filename && job.v2_filename
                      ? `${job.v1_filename} ↔ ${job.v2_filename}`
                      : job.id.slice(0, 8)}
                  </td>
                  <td className="px-4 py-3">
                    <StatusBadge status={job.status} />
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {new Date(job.created_at).toLocaleDateString('vi-VN')}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <Link
                      to={`/report/${job.id}`}
                      className="text-xs font-medium text-accent hover:underline"
                    >
                      Xem báo cáo
                    </Link>
                  </td>
                </tr>
              ))}
              {(!jobs || jobs.length === 0) && !isLoading && (
                <tr>
                  <td colSpan={4} className="px-4 py-8 text-center text-sm text-muted-foreground">
                    Chưa có phiên so sánh nào.{' '}
                    <Link to="/compare" className="text-accent hover:underline">
                      Tạo phiên mới
                    </Link>
                  </td>
                </tr>
              )}
              {isLoading && (
                <tr>
                  <td colSpan={4} className="px-4 py-8 text-center text-sm text-muted-foreground">
                    Đang tải...
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    completed: 'bg-diff-addition/10 text-diff-addition',
    failed: 'bg-destructive/10 text-diff-deletion',
    cancelled: 'bg-muted text-muted-foreground',
    pending: 'bg-muted text-muted-foreground',
    processing: 'bg-accent/10 text-accent',
  }
  const labels: Record<string, string> = {
    completed: 'Hoàn tất',
    failed: 'Thất bại',
    cancelled: 'Đã hủy',
    pending: 'Đang chờ',
    processing: 'Đang xử lý',
  }

  const style = styles[status] || 'bg-muted text-muted-foreground'

  return (
    <span className={`inline-flex rounded-full px-2 py-0.5 text-xs font-medium ${style}`}>
      {labels[status] || status}
    </span>
  )
}
