import { describe, it, expect, beforeEach, vi } from 'vitest'
import { useAuthStore } from './auth'

describe('authStore', () => {
  beforeEach(() => {
    useAuthStore.setState({ token: null, user: null, isLoading: false })
    vi.restoreAllMocks()
  })

  it('has null token and user initially', () => {
    const { token, user } = useAuthStore.getState()
    expect(token).toBeNull()
    expect(user).toBeNull()
  })

  it('login sets token and user on success', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            access_token: 'test-token',
            token_type: 'bearer',
            user: { id: '1', username: 'test', email: 'test@example.com', role: 'user' },
          }),
      }),
    )

    await useAuthStore.getState().login({ username: 'test', password: 'pw' })
    const { token, user } = useAuthStore.getState()
    expect(token).toBe('test-token')
    expect(user?.username).toBe('test')
  })

  it('logout clears token and user', () => {
    useAuthStore.setState({
      token: 'some-token',
      user: { id: '1', username: 't', email: 'e@e.com', role: 'user' },
    })

    useAuthStore.getState().logout()

    const { token, user } = useAuthStore.getState()
    expect(token).toBeNull()
    expect(user).toBeNull()
  })
})
