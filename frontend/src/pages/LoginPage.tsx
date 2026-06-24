import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { useTranslation } from 'react-i18next'
import { useAuthStore } from '@/stores/auth'
import { useState } from 'react'

const loginSchema = z.object({
  login: z.string().min(1, 'Vui lòng nhập email hoặc tên đăng nhập'),
  password: z.string().min(1, 'Vui lòng nhập mật khẩu'),
})

type LoginFormData = z.infer<typeof loginSchema>

export function LoginPage() {
  const { t } = useTranslation()
  const login = useAuthStore((s) => s.login)
  const [error, setError] = useState<string | null>(null)

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<LoginFormData>({
    resolver: zodResolver(loginSchema),
  })

  const onSubmit = async (data: LoginFormData) => {
    setError(null)
    try {
      await login({
        ...(data.login.includes('@') ? { email: data.login } : { username: data.login }),
        password: data.password,
      })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Đăng nhập thất bại')
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="w-full max-w-md rounded-lg border bg-white p-8 shadow-card">
        <h1 className="font-serif text-2xl font-semibold text-foreground">{t('app.title')}</h1>
        <p className="mt-1 text-sm text-muted-foreground">{t('app.tagline')}</p>

        <form onSubmit={handleSubmit(onSubmit)} className="mt-6 space-y-4">
          <div>
            <label htmlFor="login" className="block text-sm font-medium text-foreground">
              Email / Tên đăng nhập
            </label>
            <input
              id="login"
              type="text"
              autoComplete="username"
              className="mt-1 block w-full rounded-md border px-3 py-2 text-sm focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent"
              {...register('login')}
            />
            {errors.login && <p className="mt-1 text-xs text-destructive">{errors.login.message}</p>}
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-medium text-foreground">
              {t('auth.password')}
            </label>
            <input
              id="password"
              type="password"
              autoComplete="current-password"
              className="mt-1 block w-full rounded-md border px-3 py-2 text-sm focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent"
              {...register('password')}
            />
            {errors.password && (
              <p className="mt-1 text-xs text-destructive">{errors.password.message}</p>
            )}
          </div>

          {error && (
            <div className="rounded-md bg-destructive/10 px-3 py-2 text-sm text-destructive">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={isSubmitting}
            className="w-full rounded-md bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            {isSubmitting ? 'Đang đăng nhập...' : t('auth.login')}
          </button>
        </form>

        <p className="mt-4 text-center text-sm text-muted-foreground">
          Chưa có tài khoản?{' '}
          <a href="/register" className="font-medium text-accent hover:underline">
            {t('auth.register')}
          </a>
        </p>
      </div>
    </div>
  )
}
