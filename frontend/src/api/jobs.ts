import { api } from '@/api/client'
import type { Job } from '@/types/document'

export interface CreateJobPayload {
  document_v1_id: string
  document_v2_id: string
  skip_phase3?: boolean
  config_overrides?: Record<string, unknown>
}

export function createJob(payload: CreateJobPayload): Promise<Job> {
  return api.post<Job>('/jobs', payload)
}

export function getJob(id: string): Promise<Job> {
  return api.get<Job>(`/jobs/${id}`)
}

interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  page_size: number
  total_pages: number
}

export async function listJobs(): Promise<Job[]> {
  const res = await api.get<PaginatedResponse<Job>>('/jobs')
  return res.items ?? []
}

export function cancelJob(id: string): Promise<void> {
  return api.post<void>(`/jobs/${id}/cancel`)
}
