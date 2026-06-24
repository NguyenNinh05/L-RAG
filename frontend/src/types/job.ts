// Backend WS message format: { event, job_id, progress_pct, current_phase, message, error }

export interface WSProgressMessage {
  event: string // "progress" | "phase_change" | "completed" | "error"
  job_id: string
  progress_pct: number
  current_phase: string // "queued" | "ingestion" | "alignment" | "comparison"
  message: string
  error: string | null
}

export const PHASE_ORDER = ['queued', 'ingestion', 'alignment', 'comparison'] as const
export type PipelinePhase = (typeof PHASE_ORDER)[number]

export const PHASE_LABELS: Record<string, string> = {
  queued: 'Đang chờ',
  ingestion: 'Phase 1 — Trích xuất & Phân tích tài liệu',
  alignment: 'Phase 2 — Ghép cặp (Alignment)',
  comparison: 'Phase 3 — So sánh & Đối chiếu',
  completed: 'Hoàn tất',
  failed: 'Thất bại',
}
