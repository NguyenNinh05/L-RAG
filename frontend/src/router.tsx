import { lazy, Suspense } from 'react'
import { createBrowserRouter, Navigate } from 'react-router-dom'
import { ProtectedRoute } from '@/components/auth/ProtectedRoute'

const LoginPage = lazy(() => import('@/pages/LoginPage').then((m) => ({ default: m.LoginPage })))
const RegisterPage = lazy(() =>
  import('@/pages/RegisterPage').then((m) => ({ default: m.RegisterPage })),
)

function Loading() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-background">
      <p className="text-sm text-muted-foreground">Loading...</p>
    </div>
  )
}

export const router = createBrowserRouter([
  {
    path: '/login',
    element: (
      <Suspense fallback={<Loading />}>
        <LoginPage />
      </Suspense>
    ),
  },
  {
    path: '/register',
    element: (
      <Suspense fallback={<Loading />}>
        <RegisterPage />
      </Suspense>
    ),
  },
  {
    element: <ProtectedRoute />,
    children: [
      {
        path: '/',
        element: (
          <Suspense fallback={<Loading />}>
            <Navigate to="/compare" replace />
          </Suspense>
        ),
      },
      {
        path: '/compare',
        element: (
          <Suspense fallback={<Loading />}>
            <div className="p-8"><h1 className="text-2xl">Compare — coming in M2</h1></div>
          </Suspense>
        ),
      },
      {
        path: '/report/:id',
        element: (
          <Suspense fallback={<Loading />}>
            <div className="p-8"><h1 className="text-2xl">Report — coming in M4</h1></div>
          </Suspense>
        ),
      },
      {
        path: '/library',
        element: (
          <Suspense fallback={<Loading />}>
            <div className="p-8"><h1 className="text-2xl">Library — coming in M5</h1></div>
          </Suspense>
        ),
      },
    ],
  },
])
