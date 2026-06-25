// Per-user LLM settings — field names mirror backend LLMConfigSchema /
// PipelineRunConfig so they flow through config_overrides unchanged.

export type Provider = 'local' | 'deepseek'

export interface LLMConfig {
  llm_provider: Provider
  llm_model_name: string
  llm_base_url: string
  llm_api_key: string
  llm_temperature_acu: number
  llm_temperature_summary: number
  max_tokens_acu: number
  max_tokens_summary: number
  max_concurrency: number
  llm_timeout_seconds: number
  llm_max_retries: number
  match_threshold: number
  skip_phase3: boolean
  // Phase 3 speed knobs. max_comparison_pairs: null = unlimited (UI shows 0).
  enable_second_pass: boolean
  enable_number_enumeration: boolean
  min_confidence_to_include: number
  max_comparison_pairs: number | null
}

export interface LLMSettingsResponse {
  config: LLMConfig
  providers: Provider[]
  models: string[]
}

export interface AvailableModelsResponse {
  provider: string
  models: string[]
  base_url: string | null
}
