import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { listJobs } from '@/api/jobs'

export function DashboardPage() {
  const { data: jobs, isLoading } = useQuery({
    queryKey: ['jobs'],
    queryFn: listJobs,
  })

  const kpis = {
    totalJobs: jobs?.length ?? 0,
    completed: jobs?.filter((j) => j.status === 'completed').length ?? 0,
    inProgress: jobs?.filter((j) => ['phase1', 'phase2', 'phase3'].includes(j.status)).length ?? 0,
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
                  <td className="px-4 py-3 font-medium">{job.name}</td>
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
    idle: 'bg-muted text-muted-foreground',
    pending: 'bg-muted text-muted-foreground',
  }
  const labels: Record<string, string> = {
    completed: 'Hoàn tất',
    failed: 'Thất bại',
    pending: 'Đang chờ',
    idle: 'Chờ',
    phase1: 'Phase 1',
    phase2: 'Phase 2',
    phase3: 'Phase 3',
  }

  const style = styles[status] || 'bg-accent/10 text-accent'

  return (
    <span className={`inline-flex rounded-full px-2 py-0.5 text-xs font-medium ${style}`}>
      {labels[status] || status}
    </span>
  )
}
