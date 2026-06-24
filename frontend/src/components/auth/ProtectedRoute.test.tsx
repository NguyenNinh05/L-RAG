import { render, screen } from '@testing-library/react'
import { createMemoryRouter, RouterProvider } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { describe, it, expect, beforeEach } from 'vitest'
import { ProtectedRoute } from './ProtectedRoute'
import { useAuthStore } from '@/stores/auth'

function renderWithRouter(token: string | null) {
  useAuthStore.setState({ token, user: null, isLoading: false })

  const router = createMemoryRouter(
    [
      {
        element: <ProtectedRoute />,
        children: [
          { path: '/', element: <div>Protected Content</div> },
        ],
      },
      { path: '/login', element: <div>Login Page</div> },
    ],
    { initialEntries: ['/'] },
  )

  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  )
}

describe('ProtectedRoute', () => {
  beforeEach(() => {
    useAuthStore.setState({ token: null, user: null, isLoading: false })
  })

  it('redirects to /login when not authenticated', () => {
    renderWithRouter(null)
    expect(screen.getByText('Login Page')).toBeInTheDocument()
  })

  it('renders children when authenticated', () => {
    renderWithRouter('valid-token')
    expect(screen.getByText('Protected Content')).toBeInTheDocument()
  })
})
