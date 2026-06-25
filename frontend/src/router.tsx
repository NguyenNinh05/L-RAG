import { lazy, Suspense } from 'react'
import { createBrowserRouter } from 'react-router-dom'
import { ProtectedRoute } from '@/components/auth/ProtectedRoute'
import { Layout } from '@/components/layout/Layout'

const LoginPage = lazy(() => import('@/pages/LoginPage').then((m) => ({ default: m.LoginPage })))
const RegisterPage = lazy(() =>
  import('@/pages/RegisterPage').then((m) => ({ default: m.RegisterPage })),
)
const CreateJobPage = lazy(() =>
  import('@/pages/CreateJobPage').then((m) => ({ default: m.CreateJobPage })),
)
const ReportPage = lazy(() => import('@/pages/ReportPage').then((m) => ({ default: m.ReportPage })))
const DashboardPage = lazy(() =>
  import('@/pages/DashboardPage').then((m) => ({ default: m.DashboardPage })),
)
const SettingsPage = lazy(() =>
  import('@/pages/SettingsPage').then((m) => ({ default: m.SettingsPage })),
)
const EvalDashboardPage = lazy(() =>
  import('@/pages/EvalDashboardPage').then((m) => ({ default: m.EvalDashboardPage })),
)
const EvalRunDetailPage = lazy(() =>
  import('@/pages/EvalRunDetailPage').then((m) => ({ default: m.EvalRunDetailPage })),
)
const EvalPairDetailPage = lazy(() =>
  import('@/pages/EvalPairDetailPage').then((m) => ({ default: m.EvalPairDetailPage })),
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
    element: (
      <ProtectedRoute />
    ),
    children: [
      {
        element: <Layout />,
        children: [
      {
        path: '/',
        element: (
          <Suspense fallback={<Loading />}>
            <DashboardPage />
          </Suspense>
        ),
      },
      {
        path: '/compare',
        element: (
          <Suspense fallback={<Loading />}>
            <CreateJobPage />
          </Suspense>
        ),
      },
      {
        path: '/report/:id',
        element: (
          <Suspense fallback={<Loading />}>
            <ReportPage />
          </Suspense>
        ),
      },
      {
        path: '/settings',
        element: (
          <Suspense fallback={<Loading />}>
            <SettingsPage />
          </Suspense>
        ),
      },
      {
        path: '/eval',
        element: (
          <Suspense fallback={<Loading />}>
            <EvalDashboardPage />
          </Suspense>
        ),
      },
      {
        path: '/eval/:slug',
        element: (
          <Suspense fallback={<Loading />}>
            <EvalRunDetailPage />
          </Suspense>
        ),
      },
      {
        path: '/eval/:slug/:pairName',
        element: (
          <Suspense fallback={<Loading />}>
            <EvalPairDetailPage />
          </Suspense>
        ),
      },
        ],
      },
    ],
  },
])
