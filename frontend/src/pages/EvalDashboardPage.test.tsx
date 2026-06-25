import { render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createMemoryRouter, RouterProvider } from 'react-router-dom'
import { describe, it, expect, vi } from 'vitest'
import { EvalDashboardPage } from './EvalDashboardPage'

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k: string) => k }),
}))

vi.mock('@/api/eval', () => ({
  listEvalRuns: vi.fn().mockResolvedValue([
    {
      slug: 'v3',
      label: 'v3',
      num_pairs: 23,
      change_f1: 0.297,
      change_f1_semantic: 0.519,
      change_recall: 0.28,
      change_precision: 0.31,
      hallucination_rate: 0.033,
      avg_time_per_pair_seconds: 997,
      total_acus_generated: 547,
      total_acus_passed: 529,
      total_acus_rejected: 18,
      per_category: {},
      per_category_semantic: { A: { gt_total: 10, detected: 6, recall: 0.6 } },
    },
    {
      slug: 'v2',
      label: 'v2',
      num_pairs: 23,
      change_f1: 0.179,
      change_f1_semantic: 0.627,
      change_recall: 0.2,
      change_precision: 0.2,
      hallucination_rate: 0.075,
      avg_time_per_pair_seconds: 154,
      total_acus_generated: 292,
      total_acus_passed: 270,
      total_acus_rejected: 22,
      per_category: {},
      per_category_semantic: { A: { gt_total: 10, detected: 4, recall: 0.4 } },
    },
  ]),
}))

function renderPage() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const router = createMemoryRouter([{ path: '/eval', element: <EvalDashboardPage /> }], {
    initialEntries: ['/eval'],
  })
  render(
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  )
}

describe('EvalDashboardPage', () => {
  it('renders the heading', async () => {
    renderPage()
    expect(await screen.findByText('eval.title')).toBeInTheDocument()
  })

  it('renders the comparison section with both runs', async () => {
    renderPage()
    // Each run label appears in both the comparison table header and the legend.
    expect((await screen.findAllByText('v3')).length).toBeGreaterThan(0)
    expect((await screen.findAllByText('v2')).length).toBeGreaterThan(0)
  })
})
