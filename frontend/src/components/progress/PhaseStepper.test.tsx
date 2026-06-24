import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { PhaseStepper } from './PhaseStepper'
import type { JobProgress } from '@/types/job'

const mockProgress: JobProgress = {
  job_id: 'test-job',
  phase: 'phase1',
  phases: [
    { phase: 'phase1', label: 'Phase 1 — Trích xuất ACU', progress: 45, message: 'Đang xử lý...' },
    { phase: 'phase2', label: 'Phase 2 — Ghép cặp', progress: 0 },
    { phase: 'phase3', label: 'Phase 3 — So sánh', progress: 0 },
  ],
}

describe('PhaseStepper', () => {
  it('renders all three phases', () => {
    render(<PhaseStepper progress={mockProgress} />)
    expect(screen.getByText(/Trích xuất ACU/)).toBeInTheDocument()
    expect(screen.getByText(/Ghép cặp/)).toBeInTheDocument()
    expect(screen.getByText(/So sánh/)).toBeInTheDocument()
  })

  it('shows active phase message', () => {
    render(<PhaseStepper progress={mockProgress} />)
    expect(screen.getByText(/Đang xử lý/)).toBeInTheDocument()
  })

  it('shows completion message when done', () => {
    const done: JobProgress = {
      ...mockProgress,
      phase: 'completed',
    }
    render(<PhaseStepper progress={done} />)
    expect(screen.getByText(/So sánh hoàn tất/)).toBeInTheDocument()
  })

  it('shows failure message when failed', () => {
    const failed: JobProgress = {
      ...mockProgress,
      phase: 'failed',
    }
    render(<PhaseStepper progress={failed} />)
    expect(screen.getByText(/Đã xảy ra lỗi/)).toBeInTheDocument()
  })
})
