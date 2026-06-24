import type { AcuType, Severity } from '@/types/diff'
import { ACU_TYPE_LABELS, SEVERITY_LABELS } from '@/types/diff'

interface FilterBarProps {
  selectedType: AcuType | null
  onTypeChange: (t: AcuType | null) => void
  selectedSeverity: Severity | null
  onSeverityChange: (s: Severity | null) => void
  sortBy: 'type' | 'severity'
  onSortChange: (s: 'type' | 'severity') => void
}

export function FilterBar({
  selectedType,
  onTypeChange,
  selectedSeverity,
  onSeverityChange,
  sortBy,
  onSortChange,
}: FilterBarProps) {
  const types: (AcuType | null)[] = [null, 'addition', 'deletion', 'numerical', 'terminology', 'structural', 'reorder']
  const severities: (Severity | null)[] = [null, 'high', 'medium', 'low']

  return (
    <div className="flex flex-wrap items-center gap-3 rounded-lg border bg-white p-3">
      <span className="text-xs font-medium text-muted-foreground">Lọc:</span>

      {/* Type filter */}
      <div className="flex flex-wrap gap-1">
        {types.map((t) => (
          <button
            key={t ?? 'all'}
            onClick={() => onTypeChange(t)}
            className={`rounded px-2 py-1 text-xs font-medium transition-colors
              ${selectedType === t
                ? 'bg-primary text-primary-foreground'
                : 'bg-muted text-muted-foreground hover:bg-muted/70'
              }`}
          >
            {t ? ACU_TYPE_LABELS[t] : 'Tất cả'}
          </button>
        ))}
      </div>

      <span className="text-xs text-muted-foreground/30">|</span>

      {/* Severity filter */}
      <div className="flex gap-1">
        {severities.map((s) => (
          <button
            key={s ?? 'all'}
            onClick={() => onSeverityChange(s)}
            className={`rounded px-2 py-1 text-xs font-medium transition-colors
              ${selectedSeverity === s
                ? 'bg-primary text-primary-foreground'
                : 'bg-muted text-muted-foreground hover:bg-muted/70'
              }`}
          >
            {s ? SEVERITY_LABELS[s] : 'Tất cả'}
          </button>
        ))}
      </div>

      <span className="text-xs text-muted-foreground/30">|</span>

      {/* Sort */}
      <select
        value={sortBy}
        onChange={(e) => onSortChange(e.target.value as 'type' | 'severity')}
        className="rounded border bg-muted px-2 py-1 text-xs text-foreground"
      >
        <option value="type">Sắp xếp: Loại</option>
        <option value="severity">Sắp xếp: Mức độ</option>
      </select>
    </div>
  )
}
