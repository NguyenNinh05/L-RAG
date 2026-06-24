import { render, screen } from '@testing-library/react'
import { createMemoryRouter, RouterProvider } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { describe, it, expect, beforeEach, vi } from 'vitest'
import { CreateJobPage } from './CreateJobPage'

// Mock auth store for protected route access
vi.mock('@/stores/auth', () => ({
  useAuthStore: vi.fn(() => ({ token: 'test-token', user: null })),
}))

function renderPage() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const router = createMemoryRouter(
    [{ path: '/', element: <CreateJobPage /> }],
    { initialEntries: ['/'] },
  )
  render(
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  )
}

describe('CreateJobPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the page heading', () => {
    renderPage()
    expect(screen.getByText('So sánh mới')).toBeInTheDocument()
  })

  it('renders V1 and V2 upload sections', () => {
    renderPage()
    expect(screen.getByText('Tài liệu gốc (V1)')).toBeInTheDocument()
    expect(screen.getByText('Tài liệu sửa đổi (V2)')).toBeInTheDocument()
  })

  it('submit button is disabled when no documents uploaded', () => {
    renderPage()
    expect(screen.getByRole('button', { name: /Bắt đầu so sánh/ })).toBeDisabled()
  })
})
