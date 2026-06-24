export interface User {
  id: string
  username: string
  email: string
  role: 'user' | 'admin'
}

export interface AuthResponse {
  access_token: string
  token_type: 'bearer'
  user: User
}

export interface LoginPayload {
  username?: string
  email?: string
  password: string
}

export interface RegisterPayload {
  username: string
  email: string
  password: string
}
