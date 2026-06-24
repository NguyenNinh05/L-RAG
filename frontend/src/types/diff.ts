export type AcuType = 'addition' | 'deletion' | 'numerical' | 'terminology' | 'structural' | 'reorder'
export type Severity = 'low' | 'medium' | 'high'

export interface Evidence {
  text: string
  article?: string
  citation?: string
}

export interface AcuItem {
  id: string
  type: AcuType
  severity: Severity
  title: string
  description: string
  v1_evidence: Evidence
  v2_evidence: Evidence
}

export interface ReportSummary {
  total_acus: number
  by_type: Record<AcuType, number>
  by_severity: Record<Severity, number>
  v1_label: string
  v2_label: string
  generated_at: string
}

export interface DiffReport {
  summary: ReportSummary
  acus: AcuItem[]
}

export const ACU_TYPE_LABELS: Record<AcuType, string> = {
  addition: 'Bổ sung',
  deletion: 'Bãi bỏ',
  numerical: 'Số liệu',
  terminology: 'Thuật ngữ',
  structural: 'Cấu trúc',
  reorder: 'Sắp xếp lại',
}

export const SEVERITY_LABELS: Record<Severity, string> = {
  high: 'Cao',
  medium: 'Trung bình',
  low: 'Thấp',
}

export const DIFF_COLORS: Record<AcuType, string> = {
  addition: 'text-diff-addition',
  deletion: 'text-diff-deletion',
  numerical: 'text-diff-numerical',
  terminology: 'text-diff-terminology',
  structural: 'text-diff-structural',
  reorder: 'text-diff-reorder',
}
