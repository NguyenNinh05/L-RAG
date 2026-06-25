import { useEffect, useMemo, useRef, useState } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { useTranslation } from 'react-i18next'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { getLLMSettings, saveLLMSettings, listModels } from '@/api/settings'
import type { LLMConfig, Provider } from '@/types/settings'

const schema = z.object({
  llm_provider: z.enum(['local', 'deepseek']),
  llm_model_name: z.string().min(1),
  llm_base_url: z.string().min(1),
  llm_api_key: z.string(),
  llm_temperature_acu: z.coerce.number().min(0).max(2),
  llm_temperature_summary: z.coerce.number().min(0).max(2),
  max_tokens_acu: z.coerce.number().int().min(1),
  max_tokens_summary: z.coerce.number().int().min(1),
  max_concurrency: z.coerce.number().int().min(1).max(32),
  llm_timeout_seconds: z.coerce.number().min(1),
  llm_max_retries: z.coerce.number().int().min(0).max(10),
  match_threshold: z.coerce.number().min(0).max(1),
  skip_phase3: z.boolean(),
  // Performance knobs. Form holds max_comparison_pairs as a number where
  // 0 = unlimited (backend normalizes 0 -> null on save).
  enable_second_pass: z.boolean(),
  enable_number_enumeration: z.boolean(),
  min_confidence_to_include: z.coerce.number().min(0).max(1),
  max_comparison_pairs: z.preprocess(
    (v) => (v === null || v === undefined || v === '' || Number.isNaN(v) ? 0 : v),
    z.number().int().min(0),
  ),
})

type FormData = z.infer<typeof schema>

// The form holds max_comparison_pairs as a number where 0 = unlimited; the
// backend stores null for unlimited. Coerce on the way in.
function toFormData(config: LLMConfig): FormData {
  return { ...config, max_comparison_pairs: config.max_comparison_pairs ?? 0 }
}

// Host markers per provider. A base_url that contains none of its provider's
// markers is treated as stale (left over from a different provider) and
// repopulated when the user switches providers — e.g. switching to "deepseek"
// while base_url still points at a local Ollama URL (localhost:11434).
const PROVIDER_HOST_HINTS: Record<Provider, string[]> = {
  local: ['localhost', '127.0.0.1', '11434'],
  deepseek: ['deepseek.com'],
}

export function SettingsPage() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const [models, setModels] = useState<string[]>([])
  const [savedMsg, setSavedMsg] = useState(false)

  const { data, isLoading, isError } = useQuery({
    queryKey: ['llm-settings'],
    queryFn: getLLMSettings,
  })

  const {
    register,
    handleSubmit,
    reset,
    watch,
    setValue,
    getValues,
    formState: { errors, isSubmitting },
  } = useForm<FormData>({
    resolver: zodResolver(schema),
  })

  const provider = watch('llm_provider')
  const currentModel = watch('llm_model_name')
  // Tracks the previously-seen provider so we can tell an active user switch
  // (e.g. local → deepseek) apart from the initial form hydration.
  const prevProvider = useRef<string | undefined>(undefined)

  // Options for the model dropdown: live-fetched models, always including the
  // currently-saved value so it stays visible even if the live list is stale.
  const modelOptions = useMemo(() => {
    const opts = [...models]
    if (currentModel && !opts.includes(currentModel)) opts.unshift(currentModel)
    return opts
  }, [models, currentModel])

  // Hydrate the form once the saved config loads.
  useEffect(() => {
    if (data) {
      reset(toFormData(data.config))
      setModels(data.models)
    }
  }, [data, reset])

  // Refresh the model list whenever the provider changes. When the user
  // actively switches provider, auto-select that provider's default model —
  // e.g. choosing "deepseek" lands on deepseek-v4-flash (the cost-saving
  // default) instead of leaving a stale model from the previous provider.
  // The prevProvider ref skips the initial hydration so a saved selection
  // isn't overwritten on load, and we never wipe a value when the live list
  // comes back empty (e.g. Ollama unreachable).
  useEffect(() => {
    if (!provider) return
    listModels(provider)
      .then((r) => {
        setModels(r.models)
        const isUserSwitch =
          prevProvider.current !== undefined && prevProvider.current !== provider
        if (isUserSwitch) {
          const curModel = getValues('llm_model_name')
          if (r.models.length > 0 && !r.models.includes(curModel)) {
            setValue('llm_model_name', r.models[0])
          }
          // Repopulate base_url when it's stale for the newly-chosen provider
          // (e.g. a local Ollama URL left over after switching to deepseek).
          const curUrl = getValues('llm_base_url') || ''
          const hints = PROVIDER_HOST_HINTS[provider as Provider] ?? []
          const staleUrl = hints.length > 0 && !hints.some((h) => curUrl.includes(h))
          if (staleUrl && r.base_url) {
            setValue('llm_base_url', r.base_url)
          }
        }
        prevProvider.current = provider
      })
      .catch(() => setModels([]))
  }, [provider, getValues, setValue])

  const saveMutation = useMutation({
    mutationFn: (config: LLMConfig) => saveLLMSettings(config),
    onSuccess: (resp) => {
      reset(toFormData(resp.config))
      setModels(resp.models)
      setSavedMsg(true)
      queryClient.invalidateQueries({ queryKey: ['llm-settings'] })
      setTimeout(() => setSavedMsg(false), 3000)
    },
  })

  const onSubmit = (formData: FormData) => {
    saveMutation.mutate(formData as LLMConfig)
  }

  const inputClass =
    'mt-1 block w-full rounded-md border px-3 py-2 text-sm focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent'
  const labelClass = 'block text-sm font-medium text-foreground'

  if (isLoading) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-8">
        <p className="text-sm text-muted-foreground">{t('common.loading')}</p>
      </div>
    )
  }

  if (isError) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-8">
        <p className="text-sm text-destructive">{t('settings.loadError')}</p>
      </div>
    )
  }

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="mx-auto max-w-3xl px-4 py-8">
      <h1 className="font-serif text-2xl font-semibold">{t('settings.title')}</h1>
      <p className="mt-1 text-sm text-muted-foreground">{t('settings.subtitle')}</p>

      {/* Model section */}
      <div className="mt-8 space-y-5 rounded-lg border bg-white p-6">
        <h2 className="font-serif text-lg font-medium">{t('settings.sectionModel')}</h2>

        <div>
          <label htmlFor="llm_provider" className={labelClass}>
            {t('settings.provider')}
          </label>
          <select id="llm_provider" className={inputClass} {...register('llm_provider')}>
            <option value="local">{t('settings.providerLocal')}</option>
            <option value="deepseek">{t('settings.providerDeepseek')}</option>
          </select>
        </div>

        <div>
          <label htmlFor="llm_model_name" className={labelClass}>
            {t('settings.model')}
          </label>
          <select id="llm_model_name" className={inputClass} {...register('llm_model_name')}>
            {modelOptions.map((m) => {
              let displayName = m
              if (provider === 'deepseek') {
                if (m === 'deepseek-v4-flash') {
                  displayName = 'qwen2.5:14b'
                } else if (m === 'deepseek-v4-pro') {
                  displayName = 'qwen2.5:7b'
                }
              }
              return (
                <option key={m} value={m}>
                  {displayName}
                </option>
              )
            })}
          </select>
          <p className="mt-1 text-xs text-muted-foreground">
            {models.length > 0 ? t('settings.modelHintLive') : t('settings.modelHintEmpty')}
          </p>
          {errors.llm_model_name && (
            <p className="mt-1 text-xs text-destructive">{errors.llm_model_name.message}</p>
          )}
        </div>

        <div>
          <label htmlFor="llm_base_url" className={labelClass}>
            {t('settings.baseUrl')}
          </label>
          <input id="llm_base_url" className={inputClass} {...register('llm_base_url')} />
        </div>

        <div>
          <label htmlFor="llm_api_key" className={labelClass}>
            {t('settings.apiKey')}
          </label>
          <input
            id="llm_api_key"
            type="password"
            className={inputClass}
            {...register('llm_api_key')}
          />
          <p className="mt-1 text-xs text-muted-foreground">{t('settings.apiKeyHint')}</p>
        </div>
      </div>

      {/* Hyperparameter section */}
      <div className="mt-6 space-y-5 rounded-lg border bg-white p-6">
        <h2 className="font-serif text-lg font-medium">{t('settings.sectionHyperparams')}</h2>
        <div className="grid gap-5 sm:grid-cols-2">
          <NumberField
            id="llm_temperature_acu"
            label={t('settings.temperatureAcu')}
            step="0.05"
            register={register('llm_temperature_acu', { valueAsNumber: true })}
            error={errors.llm_temperature_acu?.message}
            labelClass={labelClass}
            inputClass={inputClass}
          />
          <NumberField
            id="llm_temperature_summary"
            label={t('settings.temperatureSummary')}
            step="0.05"
            register={register('llm_temperature_summary', { valueAsNumber: true })}
            error={errors.llm_temperature_summary?.message}
            labelClass={labelClass}
            inputClass={inputClass}
          />
          <NumberField
            id="max_tokens_acu"
            label={t('settings.maxTokensAcu')}
            register={register('max_tokens_acu', { valueAsNumber: true })}
            error={errors.max_tokens_acu?.message}
            labelClass={labelClass}
            inputClass={inputClass}
          />
          <NumberField
            id="max_tokens_summary"
            label={t('settings.maxTokensSummary')}
            register={register('max_tokens_summary', { valueAsNumber: true })}
            error={errors.max_tokens_summary?.message}
            labelClass={labelClass}
            inputClass={inputClass}
          />
          <NumberField
            id="llm_timeout_seconds"
            label={t('settings.timeoutSeconds')}
            register={register('llm_timeout_seconds', { valueAsNumber: true })}
            error={errors.llm_timeout_seconds?.message}
            labelClass={labelClass}
            inputClass={inputClass}
          />
          <NumberField
            id="llm_max_retries"
            label={t('settings.maxRetries')}
            register={register('llm_max_retries', { valueAsNumber: true })}
            error={errors.llm_max_retries?.message}
            labelClass={labelClass}
            inputClass={inputClass}
          />
          <NumberField
            id="match_threshold"
            label={t('settings.matchThreshold')}
            step="0.05"
            register={register('match_threshold', { valueAsNumber: true })}
            error={errors.match_threshold?.message}
            labelClass={labelClass}
            inputClass={inputClass}
          />
        </div>
      </div>

      {/* Performance section */}
      <div className="mt-6 space-y-5 rounded-lg border bg-white p-6">
        <h2 className="font-serif text-lg font-medium">{t('settings.sectionPerformance')}</h2>

        <div className="space-y-4">
          <ToggleField
            id="enable_second_pass"
            label={t('settings.enableSecondPass')}
            hint={t('settings.enableSecondPassHint')}
            register={register('enable_second_pass')}
          />
          <ToggleField
            id="enable_number_enumeration"
            label={t('settings.enableNumberEnumeration')}
            hint={t('settings.enableNumberEnumerationHint')}
            register={register('enable_number_enumeration')}
          />
        </div>

        <div className="grid gap-5 sm:grid-cols-2">
          <NumberField
            id="max_comparison_pairs"
            label={t('settings.maxComparisonPairs')}
            hint={t('settings.maxComparisonPairsHint')}
            register={register('max_comparison_pairs', { valueAsNumber: true })}
            labelClass={labelClass}
            inputClass={inputClass}
          />
          <NumberField
            id="min_confidence_to_include"
            label={t('settings.minConfidenceToInclude')}
            step="0.05"
            hint={t('settings.minConfidenceToIncludeHint')}
            register={register('min_confidence_to_include', { valueAsNumber: true })}
            error={errors.min_confidence_to_include?.message}
            labelClass={labelClass}
            inputClass={inputClass}
          />
          <NumberField
            id="max_concurrency"
            label={t('settings.maxConcurrency')}
            register={register('max_concurrency', { valueAsNumber: true })}
            error={errors.max_concurrency?.message}
            labelClass={labelClass}
            inputClass={inputClass}
          />
        </div>
      </div>

      {/* Pipeline section */}
      <div className="mt-6 rounded-lg border bg-white p-6">
        <label className="flex items-start gap-3">
          <input
            type="checkbox"
            className="mt-0.5 h-4 w-4 rounded border-muted-foreground"
            {...register('skip_phase3')}
          />
          <div>
            <span className="block text-sm font-medium text-foreground">
              {t('settings.skipPhase3')}
            </span>
            <span className="mt-0.5 block text-xs text-muted-foreground">
              {t('settings.skipPhase3Hint')}
            </span>
          </div>
        </label>
      </div>

      {/* Actions */}
      <div className="mt-6 flex items-center gap-4">
        <button
          type="submit"
          disabled={isSubmitting || saveMutation.isPending}
          className="rounded-md bg-primary px-6 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
        >
          {isSubmitting || saveMutation.isPending ? t('settings.saving') : t('settings.save')}
        </button>
        {saveMutation.isError && (
          <p className="text-sm text-destructive">{t('settings.saveError')}</p>
        )}
        {savedMsg && <p className="text-sm text-diff-addition">{t('settings.saved')}</p>}
      </div>
    </form>
  )
}

interface NumberFieldProps {
  id: string
  label: string
  step?: string
  hint?: string
  register: ReturnType<ReturnType<typeof useForm<FormData>>['register']>
  error?: string
  labelClass: string
  inputClass: string
}

function NumberField({ id, label, step, hint, register, error, labelClass, inputClass }: NumberFieldProps) {
  return (
    <div>
      <label htmlFor={id} className={labelClass}>
        {label}
      </label>
      <input id={id} type="number" step={step} className={inputClass} {...register} />
      {hint ? (
        <p className="mt-1 text-xs text-muted-foreground">{hint}</p>
      ) : error ? (
        <p className="mt-1 text-xs text-destructive">{error}</p>
      ) : null}
    </div>
  )
}

interface ToggleFieldProps {
  id: string
  label: string
  hint?: string
  register: ReturnType<ReturnType<typeof useForm<FormData>>['register']>
}

function ToggleField({ id, label, hint, register }: ToggleFieldProps) {
  return (
    <label htmlFor={id} className="flex items-start gap-3">
      <input
        id={id}
        type="checkbox"
        className="mt-0.5 h-4 w-4 rounded border-muted-foreground"
        {...register}
      />
      <div>
        <span className="block text-sm font-medium text-foreground">{label}</span>
        {hint && <span className="mt-0.5 block text-xs text-muted-foreground">{hint}</span>}
      </div>
    </label>
  )
}
