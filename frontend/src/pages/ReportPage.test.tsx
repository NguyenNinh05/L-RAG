import { render, screen } from '@testing-library/react'
import { createMemoryRouter, RouterProvider } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { describe, it, expect, vi } from 'vitest'
import { ReportPage } from './ReportPage'

vi.mock('@/api/report', () => ({
  getReport: vi.fn(),
}))

vi.mock('@/api/jobs', () => ({
  getJob: vi.fn(),
}))

function renderPage(id = 'test-job-id') {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const router = createMemoryRouter(
    [{ path: '/report/:id', element: <ReportPage /> }],
    { initialEntries: [`/report/${id}`] },
  )
  render(
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  )
}

describe('ReportPage', () => {
  it('renders the page heading', () => {
    renderPage()
    expect(screen.getByText('Báo cáo so sánh')).toBeInTheDocument()
  })

  it('shows loading state initially', () => {
    renderPage()
    expect(screen.getByText('Đang tải báo cáo...')).toBeInTheDocument()
  })
})
