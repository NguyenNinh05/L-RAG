export type Phase = 'idle' | 'phase1' | 'phase2' | 'phase3' | 'completed' | 'failed'

export interface PhaseStatus {
  phase: Phase
  label: string
  progress: number // 0-100
  message?: string
  started_at?: string
  completed_at?: string
}

export interface JobProgress {
  job_id: string
  phase: Phase
  phases: [PhaseStatus, PhaseStatus, PhaseStatus]
  gpu_queue_position?: number
  estimated_wait_minutes?: number
}

export const PHASE_LABELS: Record<Phase, string> = {
  idle: 'Đang chờ',
  phase1: 'Phase 1 — Trích xuất ACU',
  phase2: 'Phase 2 — Ghép cặp (Alignment)',
  phase3: 'Phase 3 — So sánh & Đối chiếu',
  completed: 'Hoàn tất',
  failed: 'Thất bại',
}
