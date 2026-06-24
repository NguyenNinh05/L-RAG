import { Link, NavLink, Outlet, useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { useAuthStore } from '@/stores/auth'
import { useThemeStore } from '@/stores/theme'

export function Layout() {
  const { t, i18n } = useTranslation()
  const logout = useAuthStore((s) => s.logout)
  const { theme, toggle } = useThemeStore()
  const navigate = useNavigate()
  const token = useAuthStore((s) => s.token)

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  const toggleLang = () => {
    const next = i18n.language === 'vi' ? 'en' : 'vi'
    i18n.changeLanguage(next)
  }

  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
      isActive ? 'bg-muted text-foreground' : 'text-muted-foreground hover:text-foreground'
    }`

  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-sticky border-b bg-white/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-2.5">
          <div className="flex items-center gap-6">
            <Link to="/" className="font-serif text-lg font-semibold tracking-tight">
              LegalDiff
            </Link>
            {token && (
              <nav className="flex gap-1" aria-label="Main navigation">
                <NavLink to="/" end className={linkClass}>
                  {t('nav.dashboard')}
                </NavLink>
                <NavLink to="/compare" className={linkClass}>
                  {t('nav.compare')}
                </NavLink>
                <NavLink to="/library" className={linkClass}>
                  {t('nav.library')}
                </NavLink>
              </nav>
            )}
          </div>

          <div className="flex items-center gap-2">
            {/* Language toggle */}
            <button
              onClick={toggleLang}
              className="rounded-md px-2 py-1.5 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors"
              aria-label="Toggle language"
            >
              {i18n.language === 'vi' ? 'EN' : 'VI'}
            </button>

            {/* Dark mode toggle */}
            <button
              onClick={toggle}
              className="rounded-md p-1.5 text-muted-foreground hover:text-foreground transition-colors"
              aria-label="Toggle dark mode"
            >
              {theme === 'dark' ? (
                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                </svg>
              ) : (
                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                </svg>
              )}
            </button>

            {/* Logout */}
            {token && (
              <button
                onClick={handleLogout}
                className="rounded-md px-2 py-1.5 text-xs font-medium text-muted-foreground hover:text-destructive transition-colors"
              >
                {t('nav.logout')}
              </button>
            )}
          </div>
        </div>
      </header>

      <main>
        <Outlet />
      </main>
    </div>
  )
}
