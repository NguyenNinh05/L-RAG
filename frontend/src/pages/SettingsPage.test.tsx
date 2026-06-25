import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createMemoryRouter, RouterProvider } from 'react-router-dom'
import { describe, it, expect, vi } from 'vitest'
import { SettingsPage } from './SettingsPage'

// Mock i18n — t() returns its key so we can assert on stable strings.
vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k: string) => k }),
}))

// Mock the settings API.
vi.mock('@/api/settings', () => ({
  getLLMSettings: vi.fn().mockResolvedValue({
    config: {
      llm_provider: 'local',
      llm_model_name: 'qwen2.5:14b',
      llm_base_url: 'http://localhost:11434/v1',
      llm_api_key: 'not-needed',
      llm_temperature_acu: 0.05,
      llm_temperature_summary: 0.3,
      max_tokens_acu: 8192,
      max_tokens_summary: 1024,
      max_concurrency: 1,
      llm_timeout_seconds: 300,
      llm_max_retries: 3,
      match_threshold: 0.6,
      skip_phase3: false,
      enable_second_pass: true,
      enable_number_enumeration: true,
      min_confidence_to_include: 0.2,
      max_comparison_pairs: null,
    },
    providers: ['local', 'deepseek'],
    models: ['qwen2.5:14b', 'qwen2.5:7b'],
  }),
  saveLLMSettings: vi.fn(),
  listModels: vi.fn().mockResolvedValue({ provider: 'local', models: ['qwen2.5:14b'] }),
}))

function renderPage() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const router = createMemoryRouter(
    [{ path: '/settings', element: <SettingsPage /> }],
    { initialEntries: ['/settings'] },
  )
  render(
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  )
}

describe('SettingsPage', () => {
  it('renders the heading', async () => {
    renderPage()
    expect(await screen.findByText('settings.title')).toBeInTheDocument()
  })

  it('loads the saved model name into the form', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getByDisplayValue('qwen2.5:14b')).toBeInTheDocument()
    })
  })

  it('lists the fetched models as suggestions', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getByText('settings.modelHintLive')).toBeInTheDocument()
    })
  })
})
