import { api } from '@/api/client'
import type { LLMConfig, LLMSettingsResponse, AvailableModelsResponse } from '@/types/settings'

export function getLLMSettings(): Promise<LLMSettingsResponse> {
  return api.get<LLMSettingsResponse>('/settings/llm')
}

export function saveLLMSettings(config: LLMConfig): Promise<LLMSettingsResponse> {
  return api.put<LLMSettingsResponse>('/settings/llm', config)
}

export function listModels(provider: string): Promise<AvailableModelsResponse> {
  return api.get<AvailableModelsResponse>(`/settings/models?provider=${encodeURIComponent(provider)}`)
}
