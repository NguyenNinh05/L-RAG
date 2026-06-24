import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import { LoginPage } from './LoginPage'

// Mock i18n
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
}))

// Mock auth store
const mockLogin = vi.fn()
vi.mock('@/stores/auth', () => ({
  useAuthStore: (selector: (s: unknown) => unknown) => {
    if (typeof selector === 'function') {
      return selector({ login: mockLogin, token: null })
    }
  },
}))

function renderPage() {
  render(
    <MemoryRouter initialEntries={['/login']}>
      <LoginPage />
    </MemoryRouter>,
  )
}

describe('LoginPage', () => {
  beforeEach(() => {
    mockLogin.mockReset()
  })

  it('renders login form with all fields', () => {
    renderPage()
    expect(screen.getByLabelText(/Email/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/auth.password/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /auth.login/i })).toBeInTheDocument()
  })

  it('shows validation errors on empty submit', async () => {
    const user = userEvent.setup()
    renderPage()

    await user.click(screen.getByRole('button', { name: /auth.login/i }))

    expect(await screen.findByText(/Vui lòng nhập email/i)).toBeInTheDocument()
    expect(await screen.findByText(/Vui lòng nhập mật khẩu/i)).toBeInTheDocument()
  })

  it('calls login on valid submit', async () => {
    mockLogin.mockResolvedValueOnce(undefined)
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText(/Email/i), 'testuser')
    await user.type(screen.getByLabelText(/auth.password/i), 'password123')
    await user.click(screen.getByRole('button', { name: /auth.login/i }))

    expect(mockLogin).toHaveBeenCalledWith({
      username: 'testuser',
      password: 'password123',
    })
  })
})
