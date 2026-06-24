import { api } from '@/api/client'
import type { DiffReport } from '@/types/diff'

export function getReport(jobId: string): Promise<DiffReport> {
  return api.get<DiffReport>(`/jobs/${jobId}/report`)
}
