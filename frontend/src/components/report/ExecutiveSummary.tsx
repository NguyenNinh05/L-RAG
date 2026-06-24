import type { ReportSummary } from '@/types/diff'
import { ACU_TYPE_LABELS } from '@/types/diff'

interface ExecutiveSummaryProps {
  summary: ReportSummary
}

export function ExecutiveSummary({ summary }: ExecutiveSummaryProps) {
  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-3">
        <div className="rounded-lg border bg-white p-4">
          <p className="text-xs font-medium text-muted-foreground uppercase">Tổng thay đổi</p>
          <p className="mt-1 font-serif text-3xl font-semibold tabular-nums">{summary.total_acus}</p>
          <p className="mt-1 text-xs text-muted-foreground">
            {summary.v1_label} &rarr; {summary.v2_label}
          </p>
        </div>

        <div className="rounded-lg border bg-white p-4">
          <p className="text-xs font-medium text-muted-foreground uppercase">Mức cao</p>
          <p className="mt-1 font-serif text-3xl font-semibold text-diff-deletion tabular-nums">
            {summary.by_severity.high}
          </p>
        </div>

        <div className="rounded-lg border bg-white p-4">
          <p className="text-xs font-medium text-muted-foreground uppercase">Trung bình</p>
          <p className="mt-1 font-serif text-3xl font-semibold tabular-nums">
            {summary.by_severity.medium}
          </p>
        </div>
      </div>

      <div className="rounded-lg border bg-white p-4">
        <h3 className="mb-3 font-serif text-lg font-medium">Phân bố theo loại thay đổi</h3>
        <div className="flex flex-wrap gap-2">
          {Object.entries(summary.by_type).map(([type, count]) => (
            <span
              key={type}
              className="inline-flex items-center gap-1.5 rounded-full bg-muted px-3 py-1 text-xs"
            >
              <span
                className="h-2 w-2 rounded-full"
                style={{ backgroundColor: `hsl(var(--diff-${type}))` }}
              />
              {ACU_TYPE_LABELS[type as keyof typeof ACU_TYPE_LABELS] || type}
              <span className="font-medium tabular-nums">{count}</span>
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}
