import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { PhaseStepper } from './PhaseStepper'
import type { WSProgressMessage } from '@/types/job'

const inProgressMsg: WSProgressMessage = {
  event: 'progress',
  job_id: 'test',
  progress_pct: 45,
  current_phase: 'alignment',
  message: 'Đang ghép cặp...',
  error: null,
}

const completedMsg: WSProgressMessage = {
  event: 'completed',
  job_id: 'test',
  progress_pct: 100,
  current_phase: 'comparison',
  message: 'Hoàn tất',
  error: null,
}

const errorMsg: WSProgressMessage = {
  event: 'error',
  job_id: 'test',
  progress_pct: 30,
  current_phase: 'alignment',
  message: 'Lỗi',
  error: 'Qdrant lock conflict',
}

describe('PhaseStepper', () => {
  it('renders all four phases', () => {
    render(<PhaseStepper message={inProgressMsg} />)
    expect(screen.getByText(/Trích xuất/)).toBeInTheDocument()
    expect(screen.getByText(/Ghép cặp/)).toBeInTheDocument()
    expect(screen.getByText(/So sánh/)).toBeInTheDocument()
    expect(screen.getByText(/Đang chờ/)).toBeInTheDocument()
  })

  it('shows completion message when done', () => {
    render(<PhaseStepper message={completedMsg} />)
    expect(screen.getByText(/So sánh hoàn tất/)).toBeInTheDocument()
  })

  it('shows error message when failed', () => {
    render(<PhaseStepper message={errorMsg} />)
    expect(screen.getByText(/Đã xảy ra lỗi/)).toBeInTheDocument()
    expect(screen.getAllByText(/Qdrant lock conflict/).length).toBeGreaterThanOrEqual(1)
  })
})
