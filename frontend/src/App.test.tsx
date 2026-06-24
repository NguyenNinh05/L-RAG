import { render } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createMemoryRouter, RouterProvider } from 'react-router-dom'
import { describe, it, expect } from 'vitest'

function TestApp({ initialRoute = '/' }: { initialRoute?: string }) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const router = createMemoryRouter(
    [{ path: '/', element: <div>LegalDiff</div> }],
    { initialEntries: [initialRoute] },
  )
  return (
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  )
}

describe('App', () => {
  it('renders without crashing', () => {
    const { container } = render(<TestApp />)
    expect(container).toBeTruthy()
  })
})
