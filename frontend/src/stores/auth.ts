import { create } from 'zustand'
import { api, setTokenGetter } from '@/api/client'
import type { User, LoginPayload, RegisterPayload, AuthResponse } from '@/types/auth'

interface AuthState {
  token: string | null
  user: User | null
  isLoading: boolean

  login: (payload: LoginPayload) => Promise<void>
  register: (payload: RegisterPayload) => Promise<void>
  logout: () => void
}

export const useAuthStore = create<AuthState>((set) => ({
  token: null,
  user: null,
  isLoading: false,

  login: async (payload: LoginPayload) => {
    set({ isLoading: true })
    try {
      const data = await api.post<AuthResponse>('/auth/login', payload)
      set({ token: data.access_token, user: data.user })
    } finally {
      set({ isLoading: false })
    }
  },

  register: async (payload: RegisterPayload) => {
    set({ isLoading: true })
    try {
      const data = await api.post<AuthResponse>('/auth/register', payload)
      set({ token: data.access_token, user: data.user })
    } finally {
      set({ isLoading: false })
    }
  },

  logout: () => {
    set({ token: null, user: null })
  },
}))

// Wire up the token getter so api client can read the current token
setTokenGetter(() => useAuthStore.getState().token)
