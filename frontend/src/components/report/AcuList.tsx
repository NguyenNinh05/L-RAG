import type { AcuItem, AcuType } from '@/types/diff'
import { ACU_TYPE_LABELS, SEVERITY_LABELS } from '@/types/diff'

const DIFF_BG: Record<AcuType, string> = {
  addition: 'border-l-diff-addition',
  deletion: 'border-l-diff-deletion',
  numerical: 'border-l-diff-numerical',
  terminology: 'border-l-diff-terminology',
  structural: 'border-l-diff-structural',
  reorder: 'border-l-diff-reorder',
}

const SEVERITY_BORDER: Record<string, string> = {
  high: 'bg-diff-deletion/10 border-diff-deletion/30',
  medium: 'bg-muted/50 border-muted-foreground/20',
  low: 'border-muted-foreground/10',
}

interface AcuListProps {
  acus: AcuItem[]
}

function AcuCard({ item }: { item: AcuItem }) {
  return (
    <div
      className={`rounded-lg border bg-white p-4 border-l-4 ${DIFF_BG[item.type]} ${SEVERITY_BORDER[item.severity] || ''}`}
    >
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="flex items-center gap-2">
            <span className="rounded bg-muted px-1.5 py-0.5 text-xs font-medium text-muted-foreground">
              {ACU_TYPE_LABELS[item.type]}
            </span>
            <span className="text-xs text-muted-foreground">
              {SEVERITY_LABELS[item.severity]}
            </span>
          </div>
          <h4 className="mt-1.5 font-serif text-base font-semibold">{item.title}</h4>
          <p className="mt-1 text-sm text-muted-foreground">{item.description}</p>
        </div>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <div className="rounded-md border bg-muted/30 p-3">
          <p className="text-xs font-medium text-muted-foreground uppercase">V1 (Cũ)</p>
          <p className="mt-1 text-sm">{item.v1_evidence.text}</p>
          {item.v1_evidence.article && (
            <p className="mt-1 text-xs text-muted-foreground">
              Điều {item.v1_evidence.article}
              {item.v1_evidence.citation && ` — ${item.v1_evidence.citation}`}
            </p>
          )}
        </div>
        <div className="rounded-md border bg-muted/30 p-3">
          <p className="text-xs font-medium text-muted-foreground uppercase">V2 (Mới)</p>
          <p className="mt-1 text-sm">{item.v2_evidence.text}</p>
          {item.v2_evidence.article && (
            <p className="mt-1 text-xs text-muted-foreground">
              Điều {item.v2_evidence.article}
              {item.v2_evidence.citation && ` — ${item.v2_evidence.citation}`}
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

export function AcuList({ acus }: AcuListProps) {
  if (acus.length === 0) {
    return (
      <div className="py-12 text-center">
        <p className="text-sm text-muted-foreground">Không có thay đổi nào để hiển thị</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {acus.map((item) => (
        <AcuCard key={item.id} item={item} />
      ))}
    </div>
  )
}
