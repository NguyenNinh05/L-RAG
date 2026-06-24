import { lazy, Suspense } from 'react'
import { createBrowserRouter } from 'react-router-dom'
import { ProtectedRoute } from '@/components/auth/ProtectedRoute'

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
const LibraryPage = lazy(() =>
  import('@/pages/LibraryPage').then((m) => ({ default: m.LibraryPage })),
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
        path: '/library',
        element: (
          <Suspense fallback={<Loading />}>
            <LibraryPage />
          </Suspense>
        ),
      },
    ],
  },
])
