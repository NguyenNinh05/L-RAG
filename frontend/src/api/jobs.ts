import { api } from '@/api/client'
import type { Job } from '@/types/document'

export interface CreateJobPayload {
  name: string
  doc_v1_id: string
  doc_v2_id: string
  metadata?: Record<string, string>
}

export function createJob(payload: CreateJobPayload): Promise<Job> {
  return api.post<Job>('/jobs', payload)
}

export function getJob(id: string): Promise<Job> {
  return api.get<Job>(`/jobs/${id}`)
}

export function listJobs(): Promise<Job[]> {
  return api.get<Job[]>('/jobs')
}
